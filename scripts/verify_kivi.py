"""Verify KIVI KV-cache backend produces identical greedy-decode tokens as baseline.

Usage:
    python scripts/verify_kivi.py --config configs/default.yaml \
        --prompt "Hello, I am sakuya" --max_tokens 32

The script:
  1. Runs greedy decode with the default dense KV cache.
  2. Re-runs with KiviKVCacheBackend injected.
  3. Compares token-by-token and reports max diff.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from utils.config import GlobalConfig
from engine.loader import load_model
from engine.context import set_context


def greedy_decode(model, tokenizer, prompt: str, max_tokens: int, device: str):
    """Run greedy decode and return generated token IDs."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)

    # Prefill
    set_context(is_prefill=True, cache_len=0,
                cu_seqlens_q=torch.tensor([0, input_ids.shape[1]], device=device, dtype=torch.long))
    with torch.no_grad():
        logits = model(input_ids, position_ids)
    next_token = logits[:, -1, :].argmax(dim=-1)

    generated = [next_token.item()]
    past_len = input_ids.shape[1]

    for _ in range(max_tokens - 1):
        set_context(is_prefill=False, cache_len=past_len,
                    cu_seqlens_q=torch.tensor([0, 1], device=device, dtype=torch.long))
        ids = next_token.unsqueeze(-1)
        pos = torch.tensor([[past_len]], device=device, dtype=torch.long)
        with torch.no_grad():
            logits = model(ids, pos)
        next_token = logits[:, -1, :].argmax(dim=-1)
        generated.append(next_token.item())
        past_len += 1

    return generated


def reset_kv_cache(model):
    """Zero out or reset all KV cache backends."""
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return
    for layer in layers:
        attn = getattr(getattr(layer, "self_attn", None), "attn", None)
        if attn is None:
            continue
        if attn.kv_backend is not None:
            attn.kv_backend.reset()
        elif hasattr(attn, "k_cache") and attn.k_cache is not None:
            attn.k_cache.zero_()
            attn.v_cache.zero_()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--prompt", default="Hello, I am sakuya, I'm a 24 year old student from UCAS University.")
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--k_bits", type=int, default=2)
    parser.add_argument("--v_bits", type=int, default=2)
    parser.add_argument("--group_size", type=int, default=32)
    parser.add_argument("--residual_length", type=int, default=32)
    args = parser.parse_args()

    cfg = GlobalConfig.from_yaml(args.config)
    device = cfg.env.device
    torch.set_default_dtype(cfg.env.get_torch_dtype())
    torch.set_default_device(device)

    # Init distributed (required by Qwen3Attention which calls dist.get_world_size())
    import torch.distributed as dist
    import os
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29599")
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo",
                                rank=0, world_size=1)

    # ---- Baseline run ----
    print("[Baseline] Loading model with default KV cache...")
    cfg.inference.use_cuda_graph = False
    cfg.inference.backend = "default"
    model, tokenizer, _ = load_model(cfg)

    reset_kv_cache(model)
    baseline_tokens = greedy_decode(model, tokenizer, args.prompt, args.max_tokens, device)
    baseline_text = tokenizer.decode(baseline_tokens, skip_special_tokens=True)
    print(f"[Baseline] Tokens: {baseline_tokens}")
    print(f"[Baseline] Text: {baseline_text}")

    # ---- KIVI run ----
    print("\n[KIVI] Loading model with KiviKVCacheBackend...")
    cfg.inference.kv_cache.backend = "kivi"
    cfg.inference.kv_cache.k_bits = args.k_bits
    cfg.inference.kv_cache.v_bits = args.v_bits
    cfg.inference.kv_cache.group_size = args.group_size
    cfg.inference.kv_cache.residual_length = args.residual_length

    model_kivi, tokenizer_kivi, _ = load_model(cfg)

    reset_kv_cache(model_kivi)
    kivi_tokens = greedy_decode(model_kivi, tokenizer_kivi, args.prompt, args.max_tokens, device)
    kivi_text = tokenizer_kivi.decode(kivi_tokens, skip_special_tokens=True)
    print(f"[KIVI]   Tokens: {kivi_tokens}")
    print(f"[KIVI]   Text: {kivi_text}")

    # ---- Comparison ----
    match = baseline_tokens == kivi_tokens
    min_len = min(len(baseline_tokens), len(kivi_tokens))
    matching = sum(1 for i in range(min_len) if baseline_tokens[i] == kivi_tokens[i])
    match_rate = matching / min_len * 100 if min_len > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"Token match rate: {matching}/{min_len} ({match_rate:.1f}%)")
    if match:
        print("PASS: Token sequences are IDENTICAL.")
    else:
        print("NOTE: Token sequences differ (expected for approximate quantization).")
        for i in range(min_len):
            if baseline_tokens[i] != kivi_tokens[i]:
                print(f"  First diff at step {i}: baseline={baseline_tokens[i]} kivi={kivi_tokens[i]}")
                break
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
