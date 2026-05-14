#!/usr/bin/env python3
"""Verify megakernel correctness against mini-vllm baseline.

Usage:
    python scripts/verify_megakernel.py [--steps N] [--prompt "text"]
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer

from engine.eval_runner import setup_env, load_eval_model, BaselineRunner, MegakernelRunner


def greedy_decode(runner, tokenizer, prompt: str, num_steps: int):
    """Greedy decode: prefill + num_steps decode steps."""
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    prompt_len = input_ids.shape[1]
    ids = input_ids[0].tolist()

    # Prefill
    logits = runner.prefill(ids)
    next_token = torch.argmax(logits[:, -1, :]).item()
    tokens = [next_token]
    all_logits = [logits[0, -1, :].float().cpu()]

    # Decode
    for i in range(num_steps - 1):
        logits = runner.decode_step(next_token, prompt_len + i)
        next_token = torch.argmax(logits[:, -1, :]).item()
        tokens.append(next_token)
        all_logits.append(logits[0, -1, :].float().cpu())

    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return tokens, all_logits, text


def compare(baseline_runner, megakernel_runner, tokenizer, prompt, num_steps):
    baseline_tokens, baseline_logits, baseline_text = greedy_decode(
        baseline_runner, tokenizer, prompt, num_steps
    )
    megakernel_tokens, megakernel_logits, megakernel_text = greedy_decode(
        megakernel_runner, tokenizer, prompt, num_steps
    )

    print("\n" + "=" * 60)
    print("TOKEN COMPARISON")
    print("=" * 60)
    all_match = True
    for i, (bt, mt) in enumerate(zip(baseline_tokens, megakernel_tokens)):
        match = bt == mt
        mark = "OK" if match else "DIFF"
        bt_str = tokenizer.decode([bt])
        mt_str = tokenizer.decode([mt])
        print(f"  step {i}: baseline={bt_str!r} megakernel={mt_str!r} [{mark}]")
        if not match:
            all_match = False

    print("\n" + "=" * 60)
    print("LOGITS COMPARISON")
    print("=" * 60)
    max_diffs = []
    cos_sims = []
    for i, (bl, ml) in enumerate(zip(baseline_logits, megakernel_logits)):
        bl = bl.float()
        ml = ml.float()
        max_diff = (bl - ml).abs().max().item()
        cos_sim = F.cosine_similarity(bl.unsqueeze(0), ml.unsqueeze(0)).item()
        max_diffs.append(max_diff)
        cos_sims.append(cos_sim)
        print(f"  step {i}: max_diff={max_diff:.4f}  cos_sim={cos_sim:.6f}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  All tokens match: {all_match}")
    print(f"  Max logit diff (overall): {max(max_diffs):.4f}")
    print(f"  Min cos_sim (overall): {min(cos_sims):.6f}")
    print(f"\n  Baseline output: {baseline_text}")
    print(f"  Megakernel output: {megakernel_text}")

    if all_match and max(max_diffs) < 0.5 and min(cos_sims) > 0.999:
        print("\n  VERIFICATION PASSED")
        return 0
    else:
        print("\n  VERIFICATION FAILED")
        return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--prompt", default="The capital of France is")
    args = parser.parse_args()

    cfg = setup_env(args.config)
    tokenizer = AutoTokenizer.from_pretrained(cfg.path.model_path)

    baseline_model = load_eval_model(cfg, backend="default")
    baseline = BaselineRunner(baseline_model, use_cuda_graph=False)

    from model.qwen3_megakernel import Qwen3MegakernelForCausalLM
    megakernel_model = Qwen3MegakernelForCausalLM.from_model(baseline_model)
    megakernel = MegakernelRunner(megakernel_model)

    ret = compare(baseline, megakernel, tokenizer, args.prompt, args.steps)
    dist.destroy_process_group()
    return ret


if __name__ == "__main__":
    sys.exit(main())
