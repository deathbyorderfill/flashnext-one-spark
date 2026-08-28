"""Fast-boot dump/restore for Qwen4Exp on single-device deployments.

Boot is processing-bound (~8 min of per-tensor NVFP4 prep vs 8 s of PLE I/O),
so: after the first fully processed load, serialize the model's final memory;
on later boots skip the shard loader (process_weights_after_loading still
runs, to recreate non-tensor module state) and restore tensors directly.

Format is ALIAS-AWARE (v2): many tensors are views into shared fused storage
(GDN qkvzba splits, tied weights); restoring them as independent tensors
breaks stride arithmetic and produces illegal memory accesses. So each unique
untyped storage is saved once, and every tensor is recorded as
(storage id, dtype, shape, stride, storage_offset) and rebuilt as a view onto
the shared device storage at restore.

Env: SGLANG_FASTBOOT_DIR=<dir>. Dumps land in <dir>/<ModelClassName>/.
Delete a model's dump dir after changing any flag that alters tensor layout.
UMA notes baked in: one-storage-at-a-time CPU staging, posix_fadvise DONTNEED
after each file read (page cache shares the pool with GPU allocations), and
free-before-load ordering (interleaved alloc/free fragments the allocator).
"""
import gc
import json
import logging
import os
import time

import torch

logger = logging.getLogger(__name__)

MANIFEST = "MANIFEST_V2.json"


def _iter_all_tensors(model):
    seen_names = set()
    sd = model.state_dict(keep_vars=True)
    for name, t in sd.items():
        seen_names.add(name)
        yield name, t
    for mod_name, mod in model.named_modules():
        cand = {}
        for k, v in getattr(mod, "_buffers", {}).items():
            if v is not None:
                cand[k] = v
        for k, v in vars(mod).items():
            if isinstance(v, torch.Tensor):
                cand[k] = v
        for k, v in cand.items():
            full = f"{mod_name}.{k}" if mod_name else k
            if full not in seen_names:
                seen_names.add(full)
                yield full, v


def dump_after(model, dirpath, key):
    d = os.path.join(dirpath, key)
    if os.path.isfile(os.path.join(d, MANIFEST)):
        return
    os.makedirs(d, exist_ok=True)
    t0 = time.time()
    storages = {}  # data_ptr -> sid
    files = {}  # sid -> filename
    tensors = {}
    total = 0
    for name, t in _iter_all_tensors(model):
        t = t.detach()
        st = t.untyped_storage()
        ptr = st.data_ptr()
        if ptr not in storages:
            sid = f"s{len(storages):05d}"
            storages[ptr] = sid
            u8 = torch.empty(0, dtype=torch.uint8, device=t.device)
            u8.set_(st)
            torch.save(u8.to("cpu", copy=True), os.path.join(d, sid + ".pt"))
            files[sid] = sid + ".pt"
            total += st.nbytes()
            del u8
        tensors[name] = [
            storages[ptr],
            str(t.dtype).replace("torch.", ""),
            list(t.shape),
            list(t.stride()),
            t.storage_offset(),
        ]
    with open(os.path.join(d, MANIFEST), "w") as f:
        json.dump({"storages": files, "tensors": tensors}, f)
    logger.info(
        "[fastboot] dumped %d storages / %d tensors (%.1f GB) for %s in %.0fs",
        len(files), len(tensors), total / 1e9, key, time.time() - t0,
    )


def _assign(model, name, t):
    parts = name.split(".")
    mod = model
    for p in parts[:-1]:
        mod = getattr(mod, p)
    leaf = parts[-1]
    params = getattr(mod, "_parameters", {})
    bufs = getattr(mod, "_buffers", {})
    if leaf in params and params[leaf] is not None:
        mod._parameters[leaf] = torch.nn.Parameter(t, requires_grad=False)
    elif leaf in bufs:
        mod._buffers[leaf] = t
    else:
        object.__setattr__(mod, leaf, t)


def try_restore(model, dirpath, key, target_device):
    d = os.path.join(dirpath, key)
    mf = os.path.join(d, MANIFEST)
    if not os.path.isfile(mf):
        return False
    t0 = time.time()
    with open(mf) as f:
        man = json.load(f)
    dev = (
        target_device
        if isinstance(target_device, torch.device)
        else torch.device(target_device)
    )

    # Free every tensor we are about to replace BEFORE loading (allocator
    # fragmentation otherwise strands ~17 GB on this box).
    for name in man["tensors"]:
        try:
            _assign(model, name, torch.empty(0, device="cpu"))
        except AttributeError:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    store_map = {}
    for sid, fn in man["storages"].items():
        with open(os.path.join(d, fn), "rb") as f:
            u8 = torch.load(f, map_location="cpu", weights_only=True).to(dev)
            try:
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            except (AttributeError, OSError):
                pass
        store_map[sid] = u8.untyped_storage()
        del u8

    restored = set()
    for name, (sid, dt, shape, stride, off) in man["tensors"].items():
        t = torch.empty(0, dtype=getattr(torch, dt), device=dev)
        t.set_(store_map[sid], off, shape, stride)
        _assign(model, name, t)
        restored.add(name)
    del store_map
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    stale = [
        n
        for n, _ in list(model.named_parameters()) + list(model.named_buffers())
        if n not in restored
    ]
    if stale:
        logger.warning(
            "[fastboot] %d constructed tensors absent from dump (first 8): %s",
            len(stale), stale[:8],
        )
    logger.info(
        "[fastboot] restored %d tensors (%d shared storages) for %s in %.0fs",
        len(restored), len(man["storages"]), key, time.time() - t0,
    )
    return True
