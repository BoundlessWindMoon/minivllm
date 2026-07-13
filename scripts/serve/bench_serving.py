#!/usr/bin/env python3
"""LLM serving benchmark — framework-agnostic, targets OpenAI-compatible API.

Metrics
-------
TTFT   Time To First Token     : submit → first token received
TPOT   Time Per Output Token   : (last_token - first_token) / (output_tokens - 1)
E2E    End-to-End latency       : submit → last token received
Throughput : output_tokens / elapsed  (system-level, not per-request)

Modes
-----
fixed   Keep exactly N requests in-flight at all times (semaphore-based).
        Best for comparing optimisations with controlled concurrency.

poisson Requests arrive at rate R req/s following a Poisson process.
        Best for finding throughput saturation and tail-latency behaviour.

sweep   Run fixed mode across a list of concurrency levels and print a
        latency-vs-throughput table. Use this to build the baseline curve
        before and after each optimisation.

Prompt sources
--------------
synthetic  Random English-like text padded to a target token count.
sharegpt   Load real prompts from a ShareGPT JSON file
           (--dataset /path/to/ShareGPT_V3_unfiltered_cleaned_split.json).

Usage examples
--------------
# Sweep concurrency 1→16, synthetic prompts, 200 requests each level:
python scripts/serve/bench_serving.py --mode sweep --num-requests 200

# Poisson at 4 req/s for 60 seconds:
python scripts/serve/bench_serving.py --mode poisson --request-rate 4 --duration 60

# Fixed concurrency=8, real ShareGPT prompts:
python scripts/serve/bench_serving.py --mode fixed --concurrency 8 \\
    --dataset /data/ShareGPT_V3_unfiltered_cleaned_split.json

# Save results:
python scripts/serve/bench_serving.py --mode sweep --output results/bench_$(date +%Y%m%d).json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Iterator, Optional

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_URL          = "http://localhost:8000"
DEFAULT_CONCURRENCY  = 1
DEFAULT_NUM_REQUESTS = 100
DEFAULT_INPUT_LEN    = 512    # approx tokens (synthetic mode)
DEFAULT_OUTPUT_LEN   = 128    # max_tokens sent to server
DEFAULT_RATE         = 4.0    # req/s  (Poisson mode)
DEFAULT_DURATION     = 60     # seconds (Poisson mode)
WARMUP_REQUESTS      = 5      # discarded from stats

SWEEP_LEVELS = [1, 2, 4, 8, 16]

# Rough English vocabulary for synthetic prompt generation.
_VOCAB = (
    "the of and to a in that is was he for it with as his on be at by i this "
    "had not are but from or have an they which one you were her all she there "
    "we been their has would will no said each about up other into them its "
    "time two more write go see number way could people than first water been "
    "call who oil find long down day did get come made may part over new sound "
    "take only little work know place years live me back give most very after "
    "things our just name good sentence man think say great where help through "
    "much before line right too means old any same tell boy following came show "
    "also around form three small set put end does another well large need large "
    "hand high place year different move try kind hand picture again change play "
    "spell air away animal house point page letter mother answer found study "
    "still learn should America world high every near add food between own below"
).split()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    request_id:    str
    prompt_tokens: int          # approx; server doesn't return this
    output_tokens: int          # actual tokens received
    t_submit:      float        # epoch seconds
    t_first_token: float = 0.0
    t_last_token:  float = 0.0
    error:         Optional[str] = None

    @property
    def ttft(self) -> float:
        """Time to first token (seconds)."""
        return self.t_first_token - self.t_submit if self.t_first_token else float("nan")

    @property
    def tpot(self) -> float:
        """Time per output token (seconds). Undefined for single-token outputs."""
        if self.output_tokens <= 1 or not self.t_first_token:
            return float("nan")
        return (self.t_last_token - self.t_first_token) / (self.output_tokens - 1)

    @property
    def e2e(self) -> float:
        """End-to-end latency (seconds)."""
        return self.t_last_token - self.t_submit if self.t_last_token else float("nan")

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class BenchmarkResult:
    mode:          str
    concurrency:   Optional[int]
    request_rate:  Optional[float]
    num_requests:  int
    duration:      float           # measured wall time
    requests:      list[RequestResult] = field(default_factory=list)

    # computed after collection
    stats:         dict = field(default_factory=dict)

    def compute_stats(self) -> None:
        ok = [r for r in self.requests if r.ok]
        if not ok:
            self.stats = {"error": "all requests failed"}
            return

        def _pct(values: list[float], label: str) -> dict:
            if not values:
                return {}
            s = sorted(v for v in values if not math.isnan(v))
            if not s:
                return {}
            n = len(s)
            return {
                f"{label}_mean":  sum(s) / n,
                f"{label}_p50":   s[int(n * 0.50)],
                f"{label}_p95":   s[int(n * 0.95)],
                f"{label}_p99":   s[int(n * 0.99)],
                f"{label}_min":   s[0],
                f"{label}_max":   s[-1],
            }

        ttfts  = [r.ttft  for r in ok]
        tpots  = [r.tpot  for r in ok]
        e2es   = [r.e2e   for r in ok]
        out_tk = sum(r.output_tokens for r in ok)

        self.stats = {
            "requests_total":   len(self.requests),
            "requests_ok":      len(ok),
            "requests_failed":  len(self.requests) - len(ok),
            "output_tokens_total": out_tk,
            "throughput_tok_s": out_tk / self.duration if self.duration else 0.0,
            "throughput_req_s": len(ok) / self.duration if self.duration else 0.0,
            **_pct(ttfts, "ttft"),
            **_pct(tpots, "tpot"),
            **_pct(e2es,  "e2e"),
        }


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

def make_synthetic_prompt(target_tokens: int) -> str:
    """Build a random English-like prompt of approximately *target_tokens* tokens.

    Uses ~0.75 words-per-token as a rough approximation.
    """
    n_words = max(10, int(target_tokens * 0.75))
    body = " ".join(random.choices(_VOCAB, k=n_words))
    return body + "\n\nPlease respond in detail."


def load_sharegpt(path: str, n: int, seed: int = 42) -> list[str]:
    """Load up to *n* human turns from a ShareGPT JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    prompts = []
    for conv in data:
        for turn in conv.get("conversations", []):
            if turn.get("from") in ("human", "user"):
                prompts.append(turn["value"].strip())
    random.Random(seed).shuffle(prompts)
    return prompts[:n] if len(prompts) >= n else prompts * (n // len(prompts) + 1)[:n]


# ---------------------------------------------------------------------------
# HTTP request (blocking — runs in thread pool)
# ---------------------------------------------------------------------------

def send_request(
    base_url: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    result: RequestResult,
) -> RequestResult:
    """Send a streaming chat completion request and fill *result* in-place."""
    payload = {
        "model": "mini-vllm",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    result.t_submit = time.perf_counter()
    try:
        with requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=300,
        ) as resp:
            resp.raise_for_status()
            token_count = 0
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode() if isinstance(raw, bytes) else raw
                if line == "data: [DONE]":
                    break
                if not line.startswith("data: "):
                    continue
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0]["delta"].get("content") or ""
                if not delta:
                    continue
                now = time.perf_counter()
                if token_count == 0:
                    result.t_first_token = now
                result.t_last_token = now
                token_count += 1
            result.output_tokens = token_count
    except Exception as exc:
        result.error = str(exc)
    return result


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def run_fixed(
    base_url: str,
    prompts: list[str],
    concurrency: int,
    num_requests: int,
    max_tokens: int,
    temperature: float,
    warmup: int,
) -> BenchmarkResult:
    """Keep exactly *concurrency* requests in-flight using a semaphore."""
    total = num_requests + warmup
    prompt_cycle = (prompts * (total // len(prompts) + 1))[:total]

    results: list[RequestResult] = []
    lock = threading.Lock()

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = []
        for i, p in enumerate(prompt_cycle):
            r = RequestResult(request_id=str(i), prompt_tokens=0, output_tokens=0, t_submit=0)
            futures.append(pool.submit(send_request, base_url, p, max_tokens, temperature, r))
        for fut in as_completed(futures):
            with lock:
                results.append(fut.result())
    t_end = time.perf_counter()

    # sort by submit time, discard warmup
    results.sort(key=lambda r: r.t_submit)
    results = results[warmup:]

    br = BenchmarkResult(
        mode="fixed",
        concurrency=concurrency,
        request_rate=None,
        num_requests=len(results),
        duration=t_end - t_start,
        requests=results,
    )
    br.compute_stats()
    return br


def run_poisson(
    base_url: str,
    prompts: list[str],
    rate: float,
    duration: float,
    max_tokens: int,
    temperature: float,
    warmup: int,
) -> BenchmarkResult:
    """Submit requests following a Poisson process at *rate* req/s."""
    results: list[RequestResult] = []
    lock = threading.Lock()

    prompt_idx = [0]

    def _next_prompt() -> str:
        p = prompts[prompt_idx[0] % len(prompts)]
        prompt_idx[0] += 1
        return p

    t_start = time.perf_counter()
    t_deadline = t_start + duration

    # Use a large thread pool so threads never block on concurrency.
    with ThreadPoolExecutor(max_workers=256) as pool:
        req_id = [0]
        futures = []

        while True:
            now = time.perf_counter()
            if now >= t_deadline:
                break
            interval = random.expovariate(rate)
            next_t   = now + interval
            if next_t >= t_deadline:
                break

            sleep_for = max(0.0, next_t - time.perf_counter())
            time.sleep(sleep_for)

            i = req_id[0]; req_id[0] += 1
            r = RequestResult(request_id=str(i), prompt_tokens=0, output_tokens=0, t_submit=0)
            futures.append(pool.submit(send_request, base_url, _next_prompt(), max_tokens, temperature, r))

        for fut in as_completed(futures):
            with lock:
                results.append(fut.result())

    t_end = time.perf_counter()

    results.sort(key=lambda r: r.t_submit)
    results = results[warmup:]

    br = BenchmarkResult(
        mode="poisson",
        concurrency=None,
        request_rate=rate,
        num_requests=len(results),
        duration=t_end - t_start,
        requests=results,
    )
    br.compute_stats()
    return br


def run_sweep(
    base_url: str,
    prompts: list[str],
    levels: list[int],
    num_requests: int,
    max_tokens: int,
    temperature: float,
    warmup: int,
) -> list[BenchmarkResult]:
    results = []
    for c in levels:
        print(f"\n  concurrency={c} ({num_requests} requests) ...", flush=True)
        br = run_fixed(base_url, prompts, c, num_requests, max_tokens, temperature, warmup)
        results.append(br)
        _print_summary_row(br)
    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_HDR = f"{'concur':>7} {'ttft_p50':>10} {'ttft_p99':>10} {'tpot_p50':>10} {'tpot_p99':>10} {'e2e_p50':>10} {'tok/s':>9} {'ok/total':>10}"

def _print_summary_row(br: BenchmarkResult) -> None:
    s = br.stats
    c = str(br.concurrency) if br.concurrency else f"rate={br.request_rate}"
    def ms(k):
        v = s.get(k)
        return f"{v*1000:8.1f}" if v is not None else "     n/a"
    print(
        f"{c:>7} "
        f"{ms('ttft_p50')} ms "
        f"{ms('ttft_p99')} ms "
        f"{ms('tpot_p50')} ms "
        f"{ms('tpot_p99')} ms "
        f"{ms('e2e_p50')} ms "
        f"{s.get('throughput_tok_s', 0):8.1f} "
        f"{s.get('requests_ok', 0)}/{s.get('requests_total', 0)}"
    )


def print_header():
    print(f"\n{'concur':>7} {'ttft_p50':>12} {'ttft_p99':>12} {'tpot_p50':>12} {'tpot_p99':>12} {'e2e_p50':>12} {'tok/s':>9} {'ok/total':>10}")
    print("-" * 95)


def save_results(results: list[BenchmarkResult], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def _serialize(br: BenchmarkResult) -> dict:
        d = asdict(br)
        # add computed properties not in dataclass fields
        d["requests"] = [
            {**asdict(r), "ttft": r.ttft, "tpot": r.tpot, "e2e": r.e2e}
            for r in br.requests
        ]
        return d

    payload = {
        "timestamp": datetime.now().isoformat(),
        "runs": [_serialize(r) for r in results],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {path}")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def check_health(base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="mini-vllm serving benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--url",          default=DEFAULT_URL)
    p.add_argument("--mode",         choices=["fixed", "poisson", "sweep"], default="sweep")
    p.add_argument("--concurrency",  type=int,   default=DEFAULT_CONCURRENCY,  help="fixed/sweep mode")
    p.add_argument("--num-requests", type=int,   default=DEFAULT_NUM_REQUESTS, help="fixed/sweep mode")
    p.add_argument("--request-rate", type=float, default=DEFAULT_RATE,         help="req/s, Poisson mode")
    p.add_argument("--duration",     type=float, default=DEFAULT_DURATION,     help="seconds, Poisson mode")
    p.add_argument("--input-len",    type=int,   default=DEFAULT_INPUT_LEN,    help="synthetic prompt tokens")
    p.add_argument("--output-len",   type=int,   default=DEFAULT_OUTPUT_LEN,   help="max_tokens to server")
    p.add_argument("--temperature",  type=float, default=0.0)
    p.add_argument("--warmup",       type=int,   default=WARMUP_REQUESTS,      help="requests to discard")
    p.add_argument("--sweep-levels", type=int,   nargs="+", default=SWEEP_LEVELS)
    p.add_argument("--dataset",      default=None, help="ShareGPT JSON path")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--output",       default=None, help="save JSON results to this path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    # Health check
    if not check_health(args.url):
        print(f"[error] server not reachable at {args.url}", file=sys.stderr)
        sys.exit(1)
    print(f"Server OK: {args.url}")

    # Build prompt pool
    if args.dataset:
        need = max(args.num_requests + WARMUP_REQUESTS, 512)
        prompts = load_sharegpt(args.dataset, need, args.seed)
        print(f"Loaded {len(prompts)} prompts from {args.dataset}")
    else:
        need = max(args.num_requests + WARMUP_REQUESTS, 64)
        prompts = [make_synthetic_prompt(args.input_len) for _ in range(need)]
        print(f"Generated {len(prompts)} synthetic prompts (~{args.input_len} tokens each)")

    # Run
    all_results: list[BenchmarkResult] = []

    if args.mode == "sweep":
        print_header()
        all_results = run_sweep(
            args.url, prompts, args.sweep_levels,
            args.num_requests, args.output_len, args.temperature, args.warmup,
        )

    elif args.mode == "fixed":
        print_header()
        br = run_fixed(
            args.url, prompts, args.concurrency,
            args.num_requests, args.output_len, args.temperature, args.warmup,
        )
        all_results = [br]
        _print_summary_row(br)

    elif args.mode == "poisson":
        print(f"\nPoisson mode: rate={args.request_rate} req/s, duration={args.duration}s")
        br = run_poisson(
            args.url, prompts, args.request_rate, args.duration,
            args.output_len, args.temperature, args.warmup,
        )
        all_results = [br]
        print_header()
        _print_summary_row(br)

    # Save
    if args.output:
        save_results(all_results, args.output)
    else:
        # Auto-save with timestamp
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = f"log/bench/{args.mode}_{ts}.json"
        save_results(all_results, default_path)


if __name__ == "__main__":
    main()
