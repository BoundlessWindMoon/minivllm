#!/usr/bin/env python3
"""Run lm-evaluation-harness on a mini-vllm model.

Example:
    python -m eval.run \
        --config configs/qwen3_5.yaml \
        --tasks arc_easy,hellaswag \
        --output results.json

Quick slice (10 min, limit sample count):
    python -m eval.run \
        --config configs/qwen3_5.yaml \
        --tasks arc_easy,hellaswag \
        --limit 0.1 \
        --log log/eval_slice.log \
        --output results_slice.json

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
import sys
import types
import time
from contextlib import contextmanager
from datetime import datetime

from lm_eval import evaluator
from lm_eval.tasks import TaskManager


@contextmanager
def _tee_stdout_stderr(log_path: str | None):
    """Redirect stdout/stderr to both terminal and a log file."""
    if log_path is None:
        yield
        return

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")

    class _Tee:
        def __init__(self, streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    tee = _Tee([original_stdout, log_file])
    sys.stdout = tee
    sys.stderr = tee
    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


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
    parser.add_argument("--output", default="eval/results/eval_results.json", help="Output JSON path")
    parser.add_argument("--num_fewshot", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--limit", type=float, default=None, help="Limit samples per task (e.g. 0.1 for 10%%%, 100 for 100 samples)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--include_path", default=None, help="Extra path for custom task YAMLs")
    parser.add_argument("--log", default=None, help="Log file path (stdout/stderr tee)")
    parser.add_argument(
        "--time_limit_minutes",
        type=float,
        default=None,
        help="Soft time limit in minutes. Prints a warning when exceeded but does NOT kill lm-eval mid-flight.",
    )
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    with _tee_stdout_stderr(args.log):
        print(f"[{datetime.now().isoformat()}] Eval start")
        print(f"  config: {args.config}")
        print(f"  tasks:  {args.tasks}")
        print(f"  limit:  {args.limit}")
        print(f"  batch_size: {args.batch_size}")
        print(f"  output: {args.output}")
        if args.log:
            print(f"  log:    {args.log}")
        print("-" * 60)

        t0 = time.time()

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

        elapsed = time.time() - t0
        print("-" * 60)
        print(f"[{datetime.now().isoformat()}] Eval finished in {elapsed:.1f}s ({elapsed/60:.1f}min)")

        if args.time_limit_minutes and elapsed > args.time_limit_minutes * 60:
            print(f"[WARNING] Exceeded time limit of {args.time_limit_minutes} minutes!")

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(_clean_for_json(results), f, indent=2, ensure_ascii=False)
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
