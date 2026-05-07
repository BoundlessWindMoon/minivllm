"""Shared benchmark harness for mini-vllm evaluation scripts.

Provides timing, statistics, and rich-formatted reporting.
"""

import time
import statistics
from dataclasses import dataclass, field

import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

console = Console()


@dataclass
class BenchmarkResult:
    """Container for prefill + decode benchmark metrics."""

    name: str
    prefill_times: list[float] = field(default_factory=list)
    decode_tok_per_sec: list[float] = field(default_factory=list)

    @property
    def prefill_avg(self) -> float:
        return statistics.mean(self.prefill_times)

    @property
    def prefill_med(self) -> float:
        return statistics.median(self.prefill_times)

    @property
    def prefill_min(self) -> float:
        return min(self.prefill_times)

    @property
    def prefill_max(self) -> float:
        return max(self.prefill_times)

    @property
    def decode_avg(self) -> float:
        return statistics.mean(self.decode_tok_per_sec)

    @property
    def decode_med(self) -> float:
        return statistics.median(self.decode_tok_per_sec)

    @property
    def decode_min(self) -> float:
        return min(self.decode_tok_per_sec)

    @property
    def decode_max(self) -> float:
        return max(self.decode_tok_per_sec)


def run_benchmark(
    name: str,
    runner,
    input_ids: list[int],
    output_len: int,
    num_warmup: int,
    num_runs: int,
) -> BenchmarkResult:
    """Run warmup + timed benchmark on a runner and return metrics."""
    result = BenchmarkResult(name=name)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        # Warmup
        task = progress.add_task(f"[cyan]Warming up {name}...", total=num_warmup)
        for _ in range(num_warmup):
            runner.prefill(input_ids)
            tok = 0
            for i in range(output_len):
                logits = runner.decode_step(tok, len(input_ids) + i)
                tok = torch.argmax(logits[:, -1, :]).item()
            runner.reset()
            progress.advance(task)

        # Timed runs
        task = progress.add_task(f"[green]Benchmarking {name}...", total=num_runs)
        for _ in range(num_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            runner.prefill(input_ids)
            torch.cuda.synchronize()
            result.prefill_times.append((time.perf_counter() - t0) * 1000)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            tok = 0
            for i in range(output_len):
                logits = runner.decode_step(tok, len(input_ids) + i)
                tok = torch.argmax(logits[:, -1, :]).item()
            torch.cuda.synchronize()
            decode_ms = (time.perf_counter() - t0) * 1000
            result.decode_tok_per_sec.append(output_len / (decode_ms / 1000))

            runner.reset()
            progress.advance(task)

    return result


def print_results_table(
    results: list[BenchmarkResult],
    input_len: int,
    output_len: int,
    device_name: str,
):
    """Render a rich table with benchmark results."""
    table = Table(
        title=(
            f"Benchmark Results\n"
            f"(input_len={input_len}, output_len={output_len}, device={device_name})"
        ),
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Backend", style="cyan", no_wrap=True)
    table.add_column("Prefill (ms)", justify="right")
    table.add_column("Decode (tok/s)", justify="right")
    table.add_column("Speedup", justify="right", style="bold green")

    baseline_decode = None
    for r in results:
        prefill_str = (
            f"{r.prefill_avg:.1f} "
            f"(med {r.prefill_med:.1f}, "
            f"min {r.prefill_min:.1f}, "
            f"max {r.prefill_max:.1f})"
        )
        decode_str = (
            f"{r.decode_avg:.1f} "
            f"(med {r.decode_med:.1f}, "
            f"min {r.decode_min:.1f}, "
            f"max {r.decode_max:.1f})"
        )

        if baseline_decode is None and "baseline" in r.name.lower():
            baseline_decode = r.decode_avg

        if baseline_decode and baseline_decode > 0:
            speedup = r.decode_avg / baseline_decode
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "—"

        table.add_row(r.name, prefill_str, decode_str, speedup_str)

    console.print()
    console.print(table)
    console.print()

    if len(results) >= 2:
        fastest = max(results, key=lambda r: r.decode_avg)
        slowest = min(results, key=lambda r: r.decode_avg)
        ratio = fastest.decode_avg / slowest.decode_avg
        panel = Panel(
            f"[bold green]{fastest.name}[/] is [bold]{ratio:.2f}x[/] "
            f"faster than {slowest.name} in decode throughput.",
            title="Summary",
            border_style="green",
        )
        console.print(panel)
