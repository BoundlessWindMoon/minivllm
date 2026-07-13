"""Rich live display for batched inference.

Provides:
  run_batch_with_display()  — inference loop wrapped in a Live panel
  make_panel()              — build the renderable for one refresh tick
  print_metrics_summary()   — per-request table + aggregate stats after run
"""

import os
import statistics
import time
import unicodedata


def _vis(s: str) -> int:
    """Visual display width: CJK full-width chars count as 2 columns."""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)


def _trunc(s: str, max_cols: int) -> str:
    """Truncate string so its visual width ≤ max_cols, appending '…' if cut."""
    cols, out = 0, []
    for ch in s:
        w = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if cols + w > max_cols - 1:   # -1 to leave room for '…'
            out.append("…")
            break
        out.append(ch)
        cols += w
    return "".join(out)

from rich.console import Console, Group
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from engine.request import Request
from utils.config import GlobalConfig

_STATUS_STYLE = {
    "WAITING":    "dim",
    "PREFILLING": "bold yellow",
    "DECODING":   "bold green",
    "FINISHED":   "dim",
}


def _short_prompt(prompt_text: str, max_cols: int = 45) -> str:
    return _trunc(" ".join(prompt_text.split()), max_cols)


def make_panel(
    reqs: list[Request],
    scheduler,
    pool,
    step: int,
    t0: float,
    total_toks: int,
    last_stats: dict,
    cfg: GlobalConfig,
    tokenizer,
) -> Panel:
    table = Table(show_header=True, box=None, padding=(0, 1), expand=True)
    table.add_column("ID",     style="cyan", width=9,  no_wrap=True)
    table.add_column("Status", width=11, no_wrap=True)
    table.add_column("Cache",  justify="right", width=6,  no_wrap=True)
    table.add_column("Gen",    justify="right", width=8,  no_wrap=True)
    table.add_column("Prompt", no_wrap=True, ratio=1)

    for req in reqs:
        sname  = req.status.name
        style  = _STATUS_STYLE.get(sname, "")
        cache  = str(req.cache_len) if req.cache_len > 0 else "—"
        gen    = f"{req.num_generated_tokens}/{req.sampling_params.max_new_tokens}"
        prompt = _trunc(" ".join((req.prompt_text or "").split()), 40)
        table.add_row(
            req.request_id[:8],
            Text(sname,  style=style),
            Text(cache,  style=style),
            Text(gen,    style=style),
            Text(prompt, style=style),
        )

    elapsed = time.perf_counter() - t0
    tps     = total_toks / elapsed if elapsed > 0 else 0.0
    nd, np_, pt = (
        last_stats.get("n_decode", 0),
        last_stats.get("n_prefill", 0),
        last_stats.get("prefill_tokens", 0),
    )
    if nd > 0 and np_ > 0:
        action = f"D×{nd} P×{np_}({pt}tok)"
    elif nd > 0:
        action = f"Decode ×{nd}"
    elif np_ > 0:
        action = f"Prefill ×{np_} ({pt} tok)"
    else:
        action = "—"

    n_done = sum(1 for r in reqs if r.is_finished)
    footer = Text(overflow="ellipsis", no_wrap=True)
    footer.append(f"Step {step:4d}")
    footer.append(f"  Running {scheduler.num_running}", style="green")
    footer.append(f"  Waiting {scheduler.num_waiting}", style="yellow")
    footer.append(f"  Free {pool.num_free_slots()}", style="cyan")
    footer.append(f"  Done {n_done}/{len(reqs)}", style="dim")
    footer.append(f"  {tps:5.1f} tok/s", style="bold")
    footer.append(f"  │ {action}", style="dim")

    model_name = os.path.basename(cfg.path.model_path.rstrip("/\\"))
    title = (
        f"[bold]Batch Inference[/bold]  [dim]{model_name}[/dim]  "
        f"slots [cyan]{cfg.batch.num_slots}[/cyan]  "
        f"batch [cyan]{cfg.batch.max_batch_size}[/cyan]"
    )
    return Panel(Group(table, footer), title=title, border_style="bright_blue")


def run_batch_with_display(
    runner, scheduler, pool, reqs, cfg, tokenizer,
    timeout_seconds: float | None = None,
) -> None:
    """Run the full batch inference loop with a rich Live panel.

    Prints each completed request above the panel as it finishes,
    then shows a final throughput summary.
    """
    t0         = time.perf_counter()
    step       = 0
    total_toks = 0

    def _panel():
        return make_panel(reqs, scheduler, pool, step, t0, total_toks,
                          runner.last_step_stats, cfg, tokenizer)

    with Live(_panel(), refresh_per_second=10, transient=False) as live:
        # Register callback: fires just before each prefill forward pass.
        # At that point requests are already PREFILLING, so _panel() shows
        # the correct state. PyTorch releases the GIL during CUDA ops, so
        # Rich's background render thread keeps refreshing the display for
        # the full duration of the prefill computation.
        runner.on_prefill_start = lambda _: live.update(_panel())

        while scheduler.has_work():
            if timeout_seconds and (time.perf_counter() - t0) > timeout_seconds:
                print(f"\n[yellow]Timeout ({timeout_seconds:.0f}s): stopping early.[/yellow]")
                break
            finished    = runner.step()
            total_toks += sum(r.num_generated_tokens for r in finished)
            step       += 1

            for req in finished:
                out    = " ".join(tokenizer.decode(req.generated_ids,    skip_special_tokens=True).split())
                prompt = " ".join((req.prompt_text or "").split())

                w = live.console.width
                # Fixed columns used by id / reason / tok / prompt / separator
                prompt_trunc  = _trunc(prompt, 30)
                prefix_cols   = (_vis(req.request_id) + 1 + _vis(req.finish_reason) + 1
                                 + len(str(req.num_generated_tokens)) + 4   # "tok  "
                                 + _vis(prompt_trunc) + 5)                  # "  →  "
                out_budget    = max(10, w - prefix_cols)
                out_trunc     = _trunc(out, out_budget)

                reason_style  = "green" if req.finish_reason == "eos" else "yellow"
                live.console.print(
                    f"[cyan]{req.request_id}[/cyan] "
                    f"[{reason_style}]{req.finish_reason}[/{reason_style}] "
                    f"[dim]{req.num_generated_tokens}tok  {escape(prompt_trunc)}  →[/dim] "
                    f"{escape(out_trunc)}"
                )

            live.update(_panel())

    elapsed = time.perf_counter() - t0
    print(
        f"\nDone: {len(reqs)} requests, {total_toks} tokens, "
        f"{elapsed:.2f}s, {total_toks / elapsed:.1f} tok/s"
    )


def print_repeat_summary(runs: list[tuple[list, float]]) -> None:
    """Print mean ± std across multiple repeats of the same configuration.

    runs: list of (reqs, elapsed) from each repeat.
    """
    import statistics as _stats

    console = Console()
    console.print(Rule("[bold]Repeat Summary[/bold]", style="bright_blue"))

    def _collect(metric_fn):
        per_run = []
        for reqs, _ in runs:
            vals = [metric_fn(r) for r in reqs if metric_fn(r) is not None]
            if vals:
                per_run.append(_stats.mean(vals) * 1000)
        return per_run

    ttfts  = _collect(lambda r: r.ttft)
    tpots  = _collect(lambda r: r.tpot)
    queues = _collect(lambda r: r.time_in_queue)
    thrpts = [
        sum(r.num_generated_tokens for r in reqs) / elapsed
        for reqs, elapsed in runs
    ]

    def _fmt(label, values, unit="ms"):
        if len(values) < 2:
            return f"[bold]{label}[/bold]  [cyan]{values[0]:.1f}{unit}[/cyan]  [dim](n=1, no std)[/dim]"
        mean = _stats.mean(values)
        std  = _stats.stdev(values)
        cv   = std / mean * 100 if mean else 0
        color = "green" if cv < 5 else ("yellow" if cv < 15 else "red")
        return (f"[bold]{label}[/bold]  "
                f"mean [cyan]{mean:.1f}{unit}[/cyan]  "
                f"± [{color}]{std:.1f}{unit}[/{color}]  "
                f"[dim]CV={cv:.1f}%[/dim]")

    n = len(runs)
    lines = [
        _fmt("TTFT (mean/req)  ", ttfts),
        _fmt("In Queue (mean)  ", queues),
        _fmt("TPOT (mean/req)  ", tpots),
        _fmt("Throughput       ", thrpts, unit=" tok/s"),
        f"[dim]n={n} repeats · identical prompt set · CV<5% = stable[/dim]",
    ]
    console.print(Panel("\n".join(lines), border_style="bright_blue", padding=(0, 1)))


def _pct(values: list[float], p: int) -> float:
    """Return the p-th percentile (0-100) of a sorted-or-unsorted list (ms)."""
    if not values:
        return 0.0
    if len(values) < 2:
        return values[0]
    return statistics.quantiles(values, n=100)[p - 1]


def print_metrics_summary(reqs: list[Request], elapsed: float) -> None:
    """Print per-request metrics table and aggregate statistics.

    Args:
        reqs:    Completed requests (all must have finished_at set).
        elapsed: Total wall-clock time for the whole batch (seconds).
    """
    console = Console()
    console.print(Rule("[bold]Metrics[/bold]", style="bright_blue"))

    # ── per-request table ─────────────────────────────────────────
    table = Table(show_header=True, box=None, padding=(0, 1), expand=True)
    table.add_column("ID",      style="cyan",  width=10, no_wrap=True)
    table.add_column("TTFT",    justify="right", width=8)
    table.add_column("In Queue",justify="right", width=9)
    table.add_column("Prefill", justify="right", width=9)
    table.add_column("TPOT",    justify="right", width=8)
    table.add_column("E2E",     justify="right", width=8)
    table.add_column("Tokens",  justify="right", width=7)
    table.add_column("Reason",  width=7)

    for req in reqs:
        reason_style = "green" if req.finish_reason == "eos" else "yellow"
        table.add_row(
            req.request_id[:8],
            f"{req.ttft*1000:.0f}ms"         if req.ttft          else "—",
            f"{req.time_in_queue*1000:.0f}ms" if req.time_in_queue else "—",
            f"{req.prefill_time*1000:.0f}ms"  if req.prefill_time  else "—",
            f"{req.tpot*1000:.0f}ms"          if req.tpot          else "—",
            f"{req.e2e_latency*1000:.0f}ms"   if req.e2e_latency   else "—",
            str(req.num_generated_tokens),
            f"[{reason_style}]{req.finish_reason}[/{reason_style}]",
        )
    console.print(table)

    # ── aggregate stats ───────────────────────────────────────────
    ttfts   = [r.ttft          * 1000 for r in reqs if r.ttft]
    tpots   = [r.tpot          * 1000 for r in reqs if r.tpot]
    queues  = [r.time_in_queue * 1000 for r in reqs if r.time_in_queue]
    total_tokens = sum(r.num_generated_tokens for r in reqs)
    throughput   = total_tokens / elapsed if elapsed > 0 else 0.0

    def _row(label: str, values: list[float], unit: str = "ms") -> str:
        if not values:
            return f"[dim]{label}: —[/dim]"
        p50, p90, p99 = _pct(values, 50), _pct(values, 90), _pct(values, 99)
        return (f"[bold]{label}[/bold]  "
                f"P50 [cyan]{p50:.0f}{unit}[/cyan]  "
                f"P90 [yellow]{p90:.0f}{unit}[/yellow]  "
                f"P99 [red]{p99:.0f}{unit}[/red]")

    summary = "\n".join([
        _row("TTFT      ", ttfts),
        _row("In Queue  ", queues),
        _row("TPOT      ", tpots),
        f"[bold]Throughput[/bold]  [green]{throughput:.1f} tok/s[/green]  "
        f"[dim]({len(reqs)} reqs · {total_tokens} tokens · {elapsed:.1f}s)[/dim]",
    ])
    console.print(Panel(summary, border_style="bright_blue", padding=(0, 1)))


def print_sweep_summary(
    sweep: list[tuple[str, list[tuple[list, float]]]],
) -> None:
    """Print a横向 policy comparison table from a sweep run.

    sweep: list of (policy_name, runs) where runs is list of (reqs, elapsed).
    Each cell shows mean±std across repeats.  Best value per column is bold+green.
    A '*' suffix marks cells where the mean differs from the worst policy by > 2σ
    of the worst policy's distribution (practically significant difference).
    """
    import statistics as _stats

    console = Console()
    console.print(Rule("[bold]Sweep Summary[/bold]", style="bright_blue"))

    # ── collect per-policy aggregate stats ───────────────────────────
    # Each metric is a list of per-repeat means (one float per repeat).
    Metrics = dict  # policy -> {metric: [float per repeat]}

    def _per_repeat_mean(runs, fn):
        out = []
        for reqs, elapsed in runs:
            vals = [fn(r, elapsed) for r in reqs]
            vals = [v for v in vals if v is not None]
            out.append(_stats.mean(vals) * 1000 if vals else None)
        return [v for v in out if v is not None]

    def _throughput_per_repeat(runs):
        return [
            sum(r.num_generated_tokens for r in reqs) / elapsed
            for reqs, elapsed in runs
        ]

    policy_stats: list[tuple[str, dict]] = []
    for policy, runs in sweep:
        s = {
            "ttft_p50":   _per_repeat_mean(runs, lambda r, _: r.ttft),
            "ttft_p90":   _per_repeat_mean(runs, lambda r, _: r.ttft),  # same raw, pct applied later
            "tpot_p50":   _per_repeat_mean(runs, lambda r, _: r.tpot),
            "tpot_p90":   _per_repeat_mean(runs, lambda r, _: r.tpot),
            "queue_p50":  _per_repeat_mean(runs, lambda r, _: r.time_in_queue),
            "throughput": _throughput_per_repeat(runs),
        }
        policy_stats.append((policy, s))

    # For P50/P90 we collect all finished-request values across repeats and
    # compute percentiles on the pooled sample, then get std across repeats.
    def _pct_across_repeats(runs, fn, pct):
        per_repeat = []
        for reqs, _ in runs:
            vals = sorted(v * 1000 for r in reqs for v in [fn(r)] if v is not None)
            if not vals:
                continue
            per_repeat.append(_pct(vals, pct))
        return per_repeat

    policy_stats2: list[tuple[str, dict]] = []
    for policy, runs in sweep:
        s = {
            "TTFT P50":  _pct_across_repeats(runs, lambda r: r.ttft, 50),
            "TTFT P90":  _pct_across_repeats(runs, lambda r: r.ttft, 90),
            "TPOT P50":  _pct_across_repeats(runs, lambda r: r.tpot, 50),
            "TPOT P90":  _pct_across_repeats(runs, lambda r: r.tpot, 90),
            "Queue P50": _pct_across_repeats(runs, lambda r: r.time_in_queue, 50),
            "Throughput": _throughput_per_repeat(runs),
        }
        policy_stats2.append((policy, s))

    metrics_order = ["TTFT P50", "TTFT P90", "TPOT P50", "TPOT P90", "Queue P50", "Throughput"]
    # lower is better for latency metrics, higher is better for throughput
    higher_is_better = {"Throughput"}

    # ── find best mean per metric ─────────────────────────────────────
    best: dict[str, float] = {}
    for m in metrics_order:
        vals = []
        for _, s in policy_stats2:
            v = s[m]
            if v:
                vals.append(_stats.mean(v))
        if vals:
            best[m] = min(vals) if m not in higher_is_better else max(vals)

    # ── build table ───────────────────────────────────────────────────
    table = Table(show_header=True, box=None, padding=(0, 1), expand=True)
    table.add_column("Policy", style="cyan", width=10, no_wrap=True)
    for m in metrics_order:
        unit = " t/s" if m == "Throughput" else "ms"
        table.add_column(f"{m}\n({unit})", justify="right", width=13, no_wrap=True)

    n_repeats = max(len(runs) for _, runs in sweep)

    for policy, s in policy_stats2:
        cells = []
        for m in metrics_order:
            vals = s[m]
            if not vals:
                cells.append(Text("—", style="dim"))
                continue
            mean = _stats.mean(vals)
            std  = _stats.stdev(vals) if len(vals) > 1 else 0.0
            unit = "" if m == "Throughput" else ""

            is_best = abs(mean - best.get(m, mean)) < 1e-9

            # significance: diff from worst > 2σ of this distribution
            all_means = []
            for _, s2 in policy_stats2:
                v2 = s2[m]
                if v2:
                    all_means.append(_stats.mean(v2))
            if all_means:
                worst = min(all_means) if m in higher_is_better else max(all_means)
                sig = abs(mean - worst) > 2 * (std if std > 0 else 0.1)
            else:
                sig = False

            label = f"{mean:.1f}±{std:.1f}{'*' if sig and not is_best else ''}"
            style = "bold green" if is_best else ("green" if sig else "")
            cells.append(Text(label, style=style))

        table.add_row(Text(policy, style="cyan"), *cells)

    console.print(table)
    n = n_repeats
    console.print(
        f"[dim]n={n} repeat{'s' if n > 1 else ''} per policy · "
        f"best per column = [bold green]green[/bold green] · "
        f"* = differs from worst by >2σ[/dim]"
    )
