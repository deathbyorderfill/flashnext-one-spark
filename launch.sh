#!/usr/bin/env bash
# Launch Qwen3.8-Flash-Next 180B on a single DGX Spark (GB10).
#
# Modes (pick one):
#   PLE_MODE=hashk  (default) -- 12.8 GB hash-compressed n-gram table.
#                    Requires the artifact from tools/build_hashk_ple.py.
#                    Frees enough memory for the MTP head -> NEXTN spec decode.
#   PLE_MODE=packed -- 28.8 GB load-time NVFP4-packed table (lossless-ish,
#                    group-16). No room for MTP; spec decode flags are dropped.
#
# Env overrides:
#   HF_CACHE      HuggingFace cache dir holding the RadixArk checkpoint
#                 (auto-downloads ~135 GB on first boot if absent)
#   PORT          serving port                        (default 30000)
#   MEM_FRACTION  --mem-fraction-static               (default 0.95)
#   CTX           --context-length                    (default 262144)
#   THINKING      server-default reasoning: "medium", "xhigh", "low", "off"
#                 (default medium; per-request chat_template_kwargs override)
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
PORT="${PORT:-30000}"
MEM_FRACTION="${MEM_FRACTION:-0.95}"
CTX="${CTX:-262144}"
THINKING="${THINKING:-medium}"
PLE_MODE="${PLE_MODE:-hashk}"
IMAGE="lmsysorg/sglang:qwen38flashnext"
HASHK_ARTIFACT="$REPO_DIR/ple_hashk_R4.pt"

if [ "$THINKING" = "off" ]; then
  KWARGS='{"enable_thinking": false}'
else
  KWARGS="{\"enable_thinking\": true, \"reasoning_effort\": \"$THINKING\"}"
fi

PLE_ENV=()
SPEC_FLAGS=()
if [ "$PLE_MODE" = "hashk" ]; then
  [ -f "$HASHK_ARTIFACT" ] || {
    echo "ERROR: $HASHK_ARTIFACT missing. Build it first:" >&2
    echo "  docker run --rm --gpus all -v $HF_CACHE:/root/.cache/huggingface:ro \\" >&2
    echo "    -v $REPO_DIR:/out --entrypoint python3 $IMAGE /out/tools/build_hashk_ple.py" >&2
    exit 1
  }
  PLE_ENV=(-e "SGLANG_QWEN4_PLE_HASHK=/patches/ple_hashk_R4.pt")
  SPEC_FLAGS=(--speculative-algorithm NEXTN --speculative-num-steps 3
              --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
else
  PLE_ENV=(-e "SGLANG_QWEN4_PLE_NVFP4=1")
fi

docker rm -f flashnext 2>/dev/null || true
docker run -d --name flashnext --gpus all --network host --ipc=host --shm-size 32g \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -v "$REPO_DIR":/patches \
  -v "$REPO_DIR/patches/qwen4_exp_nvfp4.py":/sgl-workspace/sglang/python/sglang/srt/models/qwen4_exp.py:ro \
  -v "$REPO_DIR/patches/flash_fwd.py":/usr/local/lib/python3.12/dist-packages/flash_attn/cute/flash_fwd.py:ro \
  -v "$REPO_DIR/patches/qwen_sparse_attn_backend.py":/sgl-workspace/sglang/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py:ro \
  -v "$REPO_DIR/patches/sparse_attn.py":/sgl-workspace/sglang/python/sglang/srt/layers/attention/qsa/sparse_attn.py:ro \
  "${PLE_ENV[@]}" \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --model-path RadixArk/Qwen3.8-Flash-Next-NVFP4 --trust-remote-code --language-only \
    --quantization modelopt_fp4 --fp4-gemm-backend flashinfer_cutlass \
    --kv-cache-dtype fp8_e4m3 --page-size 64 \
    --mamba-scheduler-strategy extra_buffer --mamba-track-interval 64 \
    --chunked-prefill-size 8192 --max-prefill-tokens 32768 --max-running-requests 8 --max-mamba-cache-size 24 --mamba-ssm-dtype bfloat16 \
    --context-length "$CTX" --mem-fraction-static "$MEM_FRACTION" \
    --default-chat-template-kwargs "$KWARGS" \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder --strip-thinking-cache \
    "${SPEC_FLAGS[@]}" \
    --host 0.0.0.0 --port "$PORT"

echo "Booting (~9 min warm, ~20 min first run incl. download)."
echo "Watch:  docker logs -f flashnext     Health:  curl localhost:$PORT/health"
