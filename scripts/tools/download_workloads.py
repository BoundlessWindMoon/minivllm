"""Download and prepare workload prompt assets for scheduler benchmarks.

Creates three files under assets/prompts/:
  chat.jsonl         ShareGPT user turns — natural chat distribution
                     input: 50-800 tok, output: 100-500 tok
  long_context.jsonl MS-MARCO passage QA — long-input, short-output (RAG/agent proxy)
                     input: 800-3500 tok, output: 50-200 tok
  coding.jsonl       HumanEval-style Python tasks — medium input, long output
                     input: 100-500 tok, output: 200-800 tok

Usage:
  pip install datasets
  python scripts/tools/download_workloads.py [--num 200] [--seed 42]

Each line: {"prompt": "...", "max_new_tokens": N, "workload": "<type>",
            "est_input_tokens": N}
"""

import argparse
import json
import os
import random

OUT_DIR = "assets/prompts"


def _tok_approx(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 chars for English, 1.5 for CJK."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Chat workload  (ShareGPT)
# ---------------------------------------------------------------------------

def build_chat(num: int, seed: int) -> list[dict]:
    from datasets import load_dataset
    print("  Downloading ShareGPT...")
    ds = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train",
                      trust_remote_code=True)
    rng = random.Random(seed)
    items = []
    for row in ds:
        convs = row.get("conversations") or []
        if not convs:
            continue
        # Use the first human turn as the prompt
        human = next((c for c in convs if c.get("from") == "human"), None)
        if not human:
            continue
        text = human.get("value", "").strip()
        if not text:
            continue
        n_tok = _tok_approx(text)
        # Keep natural chat range: 30-600 input tokens
        if n_tok < 30 or n_tok > 600:
            continue
        items.append({
            "prompt":           text,
            "max_new_tokens":   300,
            "workload":         "chat",
            "est_input_tokens": n_tok,
        })
    rng.shuffle(items)
    result = items[:num]
    print(f"  chat: {len(result)} prompts, "
          f"input p50={sorted([i['est_input_tokens'] for i in result])[len(result)//2]} tok")
    return result


# ---------------------------------------------------------------------------
# Long-context workload  (MS-MARCO Passage Ranking → RAG/agent proxy)
# ---------------------------------------------------------------------------

def build_long_context(num: int, seed: int) -> list[dict]:
    from datasets import load_dataset
    print("  Downloading MS-MARCO (passage QA)...")
    # ms_marco v1.1 "passages" split is large; sample from train
    ds = load_dataset("microsoft/ms_marco", "v1.1", split="train",
                      trust_remote_code=True)
    rng = random.Random(seed)
    items = []
    for row in ds:
        query    = (row.get("query") or "").strip()
        passages = row.get("passages", {}).get("passage_text", [])
        if not query or not passages:
            continue
        # Concatenate top passages until we hit ~1000-3000 input tokens
        context = " ".join(p.strip() for p in passages[:5])
        prompt  = (f"Based on the following passages, answer the question.\n\n"
                   f"Passages:\n{context}\n\nQuestion: {query}\n\nAnswer:")
        n_tok = _tok_approx(prompt)
        if n_tok < 500 or n_tok > 3500:
            continue
        items.append({
            "prompt":           prompt,
            "max_new_tokens":   150,
            "workload":         "long_context",
            "est_input_tokens": n_tok,
        })
    rng.shuffle(items)
    result = items[:num]
    print(f"  long_context: {len(result)} prompts, "
          f"input p50={sorted([i['est_input_tokens'] for i in result])[len(result)//2]} tok")
    return result


# ---------------------------------------------------------------------------
# Coding workload  (HumanEval + MBPP style)
# ---------------------------------------------------------------------------

def build_coding(num: int, seed: int) -> list[dict]:
    from datasets import load_dataset
    print("  Downloading HumanEval + MBPP...")
    items = []

    # HumanEval
    try:
        he = load_dataset("openai/openai_humaneval", split="test",
                          trust_remote_code=True)
        for row in he:
            prompt  = row.get("prompt", "").strip()
            n_tok   = _tok_approx(prompt)
            if n_tok < 40:
                continue
            items.append({
                "prompt":           prompt,
                "max_new_tokens":   512,
                "workload":         "coding",
                "est_input_tokens": n_tok,
            })
    except Exception as e:
        print(f"  HumanEval skipped ({e})")

    # MBPP
    try:
        mbpp = load_dataset("google-research-datasets/mbpp", split="test", trust_remote_code=True)
        for row in mbpp:
            text   = row.get("text", "").strip()
            prompt = f"Write a Python function that {text}\n\ndef solution("
            n_tok  = _tok_approx(prompt)
            items.append({
                "prompt":           prompt,
                "max_new_tokens":   400,
                "workload":         "coding",
                "est_input_tokens": n_tok,
            })
    except Exception as e:
        print(f"  MBPP skipped ({e})")

    rng = random.Random(seed)
    rng.shuffle(items)
    result = items[:num]
    print(f"  coding: {len(result)} prompts, "
          f"input p50={sorted([i['est_input_tokens'] for i in result])[len(result)//2]} tok")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num",  type=int, default=200, help="Prompts per workload")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    workloads = [
        ("chat",         build_chat,         f"{OUT_DIR}/chat.jsonl"),
        ("long_context", build_long_context, f"{OUT_DIR}/long_context.jsonl"),
        ("coding",       build_coding,       f"{OUT_DIR}/coding.jsonl"),
    ]

    for name, fn, path in workloads:
        print(f"\n[{name}]")
        try:
            items = fn(args.num, args.seed)
            with open(path, "w", encoding="utf-8") as f:
                for item in items:
                    json.dump(item, f, ensure_ascii=False)
                    f.write("\n")
            print(f"  → {path}  ({len(items)} items)")
        except Exception as e:
            print(f"  FAILED: {e}")

    print("\nDone. Run benchmarks with:")
    print("  python tmp/bench_schedulers.py --workload chat")
    print("  python tmp/bench_schedulers.py --workload long_context")
    print("  python tmp/bench_schedulers.py --workload coding")


if __name__ == "__main__":
    main()
