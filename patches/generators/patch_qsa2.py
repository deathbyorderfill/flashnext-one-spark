#!/usr/bin/env python3
"""Replace the QSA decode fallback's broken pack+FA path with a direct
torch gather + SDPA over topk positions (hole-tolerant, graph-safe)."""
import io
import py_compile

p = "/home/serverdestroyers/flashnext/qwen_sparse_attn_backend.py"
s = io.open(p, encoding="utf-8").read()

# Anchor: from the resolver call to the final return of the fallback branch.
start = s.index("        flash_attn_varlen_func = _resolve_flash_attn_varlen_func()")
end_marker = '        return output.reshape(q.shape[0], -1)\n\n\nclass QwenSparseMultiStepDraftBackend:'
end = s.index(end_marker) + len('        return output.reshape(q.shape[0], -1)\n')

NEW = '''        # --- local patch: direct torch sparse-decode attention ---
        # The image's _compact_kv triton kernel does NOT compact: valid rows
        # stay at their original column offsets while cu_seqlens/valid_count
        # promise a contiguous segment, so interleaved -1 indices leave
        # uninitialized (NaN) holes that the attention then reads. Untested
        # upstream because GB300 uses the trtllm decode path (sm100-only).
        # Gather + masked SDPA needs no packing and tolerates holes natively.
        import torch.nn.functional as _F

        batch, topk = topk_indices.shape
        sequence_lens = metadata.sequence_lengths
        req_pool_idx = (
            metadata.row_req_pool_indices
            if metadata.row_req_pool_indices is not None
            else forward_batch.req_pool_indices
        )
        positions = topk_indices.to(torch.long)
        valid = (positions >= 0) & (positions < sequence_lens.view(-1, 1))
        slots = self.req_to_token_pool.req_to_token[
            req_pool_idx.view(-1, 1).to(torch.long), positions.clamp_min(0)
        ].to(torch.long)
        k_g = k_buffer[slots]  # [B, topk, Hkv, D]
        v_g = v_buffer[slots]
        num_q_heads = q.shape[1] if q.dim() == 3 else q.shape[-1]
        if q.dim() == 2:
            head_dim = k_buffer.shape[2]
            qh = q.view(batch, -1, head_dim)
        else:
            qh = q
        Hq = qh.shape[1]
        Hkv = k_buffer.shape[1]
        kh = k_g.permute(0, 2, 1, 3)
        vh = v_g.permute(0, 2, 1, 3)
        if Hkv != Hq:
            rep = Hq // Hkv
            kh = kh.repeat_interleave(rep, dim=1)
            vh = vh.repeat_interleave(rep, dim=1)
        nvalid = valid.sum(-1)
        safe_mask = valid.clone()
        safe_mask[:, 0] |= nvalid == 0
        out = _F.scaled_dot_product_attention(
            qh.unsqueeze(2),
            kh,
            vh,
            attn_mask=safe_mask.view(batch, 1, 1, topk),
            scale=layer.scaling,
        ).squeeze(2)
        out = torch.where(nvalid.view(batch, 1, 1) > 0, out, torch.zeros_like(out))
        return out.reshape(q.shape[0], -1)
'''

s = s[:start] + NEW + s[end:]
io.open(p, "w", encoding="utf-8").write(s)
py_compile.compile(p, doraise=True)
print("QSA fallback replaced with direct torch gather+SDPA")
