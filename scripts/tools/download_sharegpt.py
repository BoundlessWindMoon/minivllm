"""Download a subset of ShareGPT for batch inference testing.

ShareGPT contains real user-assistant conversations with naturally
varying prompt lengths — ideal for heterogeneous batch inference tests.

Output: assets/prompts/sharegpt.jsonl
Each line: {"prompt": "...", "max_new_tokens": N}

Usage:
  python scripts/tools/download_sharegpt.py [--num-prompts 200] [--max-tokens 128]

Requires: pip install datasets
"""

import argparse
import json
import os
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-prompts", type=int, default=200,
                        help="Number of prompts to extract")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Default max_new_tokens per prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="assets/prompts/sharegpt.jsonl")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install: pip install datasets")
        return

    print("Downloading ShareGPT dataset (this may take a moment)...")
    ds = load_dataset(
        "Aeala/ShareGPT_Vicuna_unfiltered",
        split="train",
        trust_remote_code=True,
    )

    rng = random.Random(args.seed)
    prompts = []
    for row in ds:
        convs = row.get("conversations") or []
        if not convs:
            continue
        first = convs[0]
        text = first.get("value", "").strip()
        if not text or len(text) < 10:
            continue
        # Keep reasonable length for testing (not too short, not too long)
        words = text.split()
        if len(words) < 5 or len(words) > 200:
            continue
        prompts.append(text)

    rng.shuffle(prompts)
    prompts = prompts[:args.num_prompts]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for p in prompts:
            json.dump({"prompt": p, "max_new_tokens": args.max_tokens}, f, ensure_ascii=False)
            f.write("\n")

    print(f"Saved {len(prompts)} prompts → {args.output}")
    # Show length distribution
    lengths = [len(p.split()) for p in prompts]
    print(f"Prompt word-count: min={min(lengths)} median={sorted(lengths)[len(lengths)//2]} max={max(lengths)}")


if __name__ == "__main__":
    main()
