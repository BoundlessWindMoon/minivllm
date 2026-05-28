#!/usr/bin/env python3
"""Run lm-evaluation-harness on a mini-vllm model.

Example:
    python -m eval.run \
        --config configs/qwen3_5.yaml \
        --tasks arc_easy,hellaswag \
        --output results.json

Local tasks (bypass HF Hub download):
    python -m eval.run \
        --config configs/qwen3_5.yaml \
        --tasks arc_easy_local \
        --include_path eval/tasks_local \
        --output results.json
"""

import argparse
import json
import os
import types

from lm_eval import evaluator
from lm_eval.tasks import TaskManager


def _clean_for_json(obj):
    """Recursively remove unserializable objects (functions, lambdas, etc.)."""
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_for_json(v) for v in obj]
    if isinstance(obj, (types.FunctionType, types.BuiltinFunctionType, types.MethodType)):
        return None
    return obj


def main():
    parser = argparse.ArgumentParser(description="Run lm-eval on mini-vllm")
    parser.add_argument("--config", required=True, help="Path to mini-vllm YAML config")
    parser.add_argument("--tasks", required=True, help="Comma-separated task names")
    parser.add_argument("--output", default="eval_results.json", help="Output JSON path")
    parser.add_argument("--num_fewshot", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--include_path", default=None, help="Extra path for custom task YAMLs")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from eval.lm_eval_minivllm import MiniVLLM

    model = MiniVLLM(config=args.config, device=args.device)

    task_manager = None
    if args.include_path:
        task_manager = TaskManager(include_path=args.include_path)

    results = evaluator.simple_evaluate(
        model=model,
        tasks=args.tasks.split(","),
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
        log_samples=True,
        task_manager=task_manager,
    )

    with open(args.output, "w") as f:
        json.dump(_clean_for_json(results), f, indent=2, ensure_ascii=False)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
