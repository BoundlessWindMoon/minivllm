import argparse
import copy
import time
import uuid

import torch
import torch.distributed as dist

from utils.config import GlobalConfig, print_runtime_config, dump_config
from engine.loader import load_model, build_kv_pool
from engine.request import Request
from utils.batch_loader import load_batch_requests
from ui.batch_display import run_batch_with_display, print_metrics_summary, print_repeat_summary, print_sweep_summary
from engine.scheduler import Scheduler
from engine.batched_runner import BatchedModelRunner


def _run_once(model, tokenizer, cfg: GlobalConfig, pool, reqs_template, *, show_display: bool = True):
    """Run one batch inference pass. Returns (finished_reqs, elapsed_seconds)."""
    reqs = [
        Request(
            request_id       = str(uuid.uuid4())[:8],
            prompt_token_ids = r.prompt_token_ids,
            prompt_text      = r.prompt_text,
            sampling_params  = copy.copy(r.sampling_params),
        )
        for r in reqs_template
    ]

    pool.reset()
    model.attach_kv_pool(pool)

    scheduler = Scheduler(
        pool,
        max_batch_size          = cfg.batch.max_batch_size,
        admission_policy        = cfg.batch.admission_policy,
        max_num_batched_tokens  = cfg.batch.max_num_batched_tokens,
    )
    runner = BatchedModelRunner(model, tokenizer, pool, scheduler, cfg)

    for r in reqs:
        scheduler.add_request(r)

    t0 = time.perf_counter()
    if show_display:
        run_batch_with_display(runner, scheduler, pool, reqs, cfg, tokenizer,
                               timeout_seconds=cfg.batch.timeout_seconds)
    else:
        while scheduler.has_work():
            if cfg.batch.timeout_seconds and (time.perf_counter() - t0) > cfg.batch.timeout_seconds:
                break
            runner.step()
    elapsed = time.perf_counter() - t0

    return reqs, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/runs/batch.yaml")
    parser.add_argument("--model",       default=None, help="Override model layer, e.g. 'qwen3_5'")
    parser.add_argument("--dump-config", action="store_true")
    parser.add_argument("--repeat",      type=int, default=1,
                        help="Run the same batch N times; report mean ± std of metrics.")
    parser.add_argument("--sweep-policies", nargs="+", metavar="POLICY",
                        default=None,
                        help="Run each policy in sequence with the same workload. "
                             "Example: --sweep-policies fifo spf ljf random")
    args = parser.parse_args()

    cfg = GlobalConfig.from_yaml(args.config, model=args.model)
    if args.dump_config:
        print(dump_config(cfg))
        return

    cfg.model.backend           = "default"
    cfg.model.use_cuda_graph    = False
    cfg.model.kv_cache.backend  = "default"

    torch.set_default_dtype(cfg.env.get_torch_dtype())
    torch.set_default_device(cfg.env.device)
    dist.init_process_group(
        backend     = "gloo" if not torch.cuda.is_available() else cfg.env.distributed.backend,
        init_method = cfg.env.distributed.init_method,
        world_size  = cfg.env.distributed.world_size,
        rank        = cfg.env.distributed.rank,
    )

    print_runtime_config(cfg)

    model, tokenizer = load_model(cfg)
    pool = build_kv_pool(model, cfg)

    reqs_template = load_batch_requests(cfg, tokenizer)

    print("Warming up...")
    model.attach_kv_pool(pool)
    _s = Scheduler(pool, max_batch_size=cfg.batch.max_batch_size)
    _r = BatchedModelRunner(model, tokenizer, pool, _s, cfg)
    max_prompt_len = max(len(r.prompt_token_ids) for r in reqs_template)
    _r.warmup(prompt_tokens=max_prompt_len, decode_steps=3)
    pool.reset()
    print("Warmup done.\n")

    all_runs: list[tuple[list, float]] = []

    if args.sweep_policies:
        sweep: list[tuple[str, list[tuple[list, float]]]] = []
        for policy in args.sweep_policies:
            cfg.batch.admission_policy = policy
            policy_runs: list[tuple[list, float]] = []
            for i in range(args.repeat):
                print(f"── {policy}  {i + 1}/{args.repeat} {'─' * max(0, 50 - len(policy))}")
                show = (i == 0 and policy == args.sweep_policies[0])
                reqs, elapsed = _run_once(model, tokenizer, cfg, pool, reqs_template, show_display=show)
                policy_runs.append((reqs, elapsed))
                print_metrics_summary(reqs, elapsed)
            sweep.append((policy, policy_runs))
        print_sweep_summary(sweep)
    else:
        for i in range(args.repeat):
            if args.repeat > 1:
                print(f"\n── Repeat {i + 1}/{args.repeat} ──────────────────────────────")
            show = (i == 0) or (args.repeat == 1)
            reqs, elapsed = _run_once(model, tokenizer, cfg, pool, reqs_template, show_display=show)
            all_runs.append((reqs, elapsed))
            print_metrics_summary(reqs, elapsed)

        if args.repeat > 1:
            print_repeat_summary(all_runs)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
