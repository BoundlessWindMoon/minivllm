#!/usr/bin/env python3
"""Benchmark megakernel vs mini-vllm baseline (with CUDA Graph).

Usage:
    python scripts/bench_megakernel.py --backend both --input-len 32 --output-len 128
    python scripts/bench_megakernel.py --backend baseline --no-cuda-graph
    python scripts/bench_megakernel.py --config configs/default.yaml --backend megakernel
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

# Disable torch.compile/dynamo interference with CUDA Graph capture
torch._dynamo.config.suppress_errors = True
torch._dynamo.reset()

from utils.runner import setup_env, load_eval_model, BaselineRunner, MegakernelRunner
from utils.bench_harness import run_benchmark, print_results_table


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark decode throughput: baseline vs megakernel"
    )
    parser.add_argument(
        "--backend", choices=["baseline", "megakernel", "both"], default="both"
    )
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    parser.add_argument("--input-len", type=int, default=32)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--num-warmup", type=int, default=20)
    parser.add_argument("--num-runs", type=int, default=50)
    parser.add_argument(
        "--no-cuda-graph", action="store_true", help="Disable CUDA Graph for baseline"
    )
    args = parser.parse_args()

    setup_env(args.config)
    cfg = setup_env(args.config)

    tokenizer = AutoTokenizer.from_pretrained(cfg.path.model_path)
    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(0, vocab_size, (args.input_len,)).tolist()

    results = []
    baseline_model = None

    if args.backend in ("baseline", "both"):
        baseline_model = load_eval_model(cfg, backend="default")
        use_graph = not args.no_cuda_graph
        runner = BaselineRunner(baseline_model, use_cuda_graph=use_graph)
        label = "baseline (CUDA Graph)" if use_graph else "baseline (no graph)"
        results.append(
            run_benchmark(
                label, runner, input_ids, args.output_len, args.num_warmup, args.num_runs
            )
        )

    if args.backend in ("megakernel", "both"):
        if baseline_model is None:
            baseline_model = load_eval_model(cfg, backend="default")
        from model.qwen3_megakernel import Qwen3MegakernelForCausalLM
        megakernel_model = Qwen3MegakernelForCausalLM.from_model(baseline_model)
        megakernel_model.greedy_fast_path = True
        runner = MegakernelRunner(megakernel_model)
        results.append(
            run_benchmark(
                "megakernel", runner, input_ids, args.output_len, args.num_warmup, args.num_runs
            )
        )

    print_results_table(
        results, args.input_len, args.output_len, torch.cuda.get_device_name(0)
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
