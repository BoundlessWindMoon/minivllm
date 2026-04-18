from __future__ import annotations
from typing import Optional

from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    ProgressColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from rich.live import Live
from rich.console import Group, Console
from rich.panel import Panel
from rich.text import Text

CONSOLE = Console()


class _StreamText:
    """流式输出文本容器"""

    def __init__(self):
        self.prompt = ""
        self.generated = ""

    def __rich__(self):
        t = Text()
        t.append(self.prompt)
        if self.generated:
            t.append(self.generated, style="bold green")
        t.append("▌", style="bold cyan")
        return t


class _TokenThroughputColumn(ProgressColumn):
    """tok/s 列"""

    def render(self, task):
        if task.elapsed and task.completed:
            return f"{task.completed / task.elapsed:.1f} tok/s"
        return ""


class ModelLoadProgress:
    """
    用法:
        with ModelLoadProgress(data_path) as pbar:
            for tensor in tensors:
                load(tensor)
                pbar.step()
    """

    def __init__(self, data_path: str, enabled: bool = True):
        self.enabled = enabled
        if not enabled:
            return

        from safetensors import safe_open
        from glob import glob
        import os

        self._files = sorted(glob(os.path.join(data_path, "*.safetensors")))
        self._total = 0
        for f in self._files:
            with safe_open(f, "pt", "cpu") as sf:
                self._total += len(sf.keys())

        self._progress = Progress(
            TextColumn("[bold blue]📦 Loading model"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            TextColumn("• {task.completed}/{task.total} tensors"),
            "•",
            TimeElapsedColumn(),
            console=CONSOLE,
        )
        self._task = self._progress.add_task("weights", total=self._total)

    def step(self, n: int = 1):
        if self.enabled:
            self._progress.update(self._task, advance=n)

    def __enter__(self):
        if self.enabled:
            self._progress.__enter__()
        return self

    def __exit__(self, *exc):
        if self.enabled:
            self._progress.__exit__(*exc)


class InferenceProgress:
    """
    用法:
        with InferenceProgress(tokenizer, max_new_tokens, prompt) as pbar:
            # prefill
            logits = model(input_ids, position_ids)
            pbar.end_prefill(first_token_id)

            # decode
            pbar.start_decode()
            while ...:
                logits = model(...)
                pbar.step_decode(next_token_id)
    """

    def __init__(
        self,
        tokenizer,
        max_new_tokens: int,
        prompt_text: str = "",
        enabled: bool = True,
    ):
        self._tokenizer = tokenizer
        self._max_new_tokens = max_new_tokens
        self._enabled = enabled
        self._generated_ids: list[int] = []

        if not enabled:
            return

        self._progress = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=40),
            "[progress.percentage]{task.percentage:>3.0f}%",
            _TokenThroughputColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=CONSOLE,
        )

        self._stream = _StreamText()
        self._stream.prompt = prompt_text

        self._live = Live(
            Group(
                Panel(
                    self._stream,
                    title="[bold]Output",
                    border_style="green",
                    padding=(0, 2),
                ),
                self._progress,
            ),
            console=CONSOLE,
            refresh_per_second=10,
            vertical_overflow="visible",
        )

        self._prefill_task = None
        self._decode_task = None

    def start_prefill(self):
        """开始 prefill（显示 spinner）"""
        if self._enabled:
            self._prefill_task = self._progress.add_task("[cyan]Prefill", total=None)

    def end_prefill(self, first_token_id: int | float):
        """prefill 完成，传入首个生成的 token id"""
        tid = int(first_token_id)  # 统一转 int，解决 Number → int 类型问题
        self._generated_ids.append(tid)
        if self._enabled:
            assert self._prefill_task is not None  # 告诉类型检查器：这里绝不可能是 None
            self._progress.update(self._prefill_task, total=1, completed=1)
            self._stream.generated = self._tokenizer.decode(
                self._generated_ids, skip_special_tokens=True
            )

    def start_decode(self):
        """开始 decode 阶段"""
        if self._enabled:
            self._decode_task = self._progress.add_task(
                "[green]Decode", total=self._max_new_tokens
            )

    def step_decode(self, token_id: int):
        """每生成一个 token 调一次"""
        self._generated_ids.append(token_id)
        if self._enabled:
            self._stream.generated = self._tokenizer.decode(
                self._generated_ids, skip_special_tokens=True
            )
            self._progress.update(self._decode_task, advance=1)

    def __enter__(self):
        if self._enabled:
            self._live.__enter__()
        return self

    def __exit__(self, *exc):
        if self._enabled:
            self._live.__exit__(*exc)


class _NoOpProgress:
    """enabled=False 时的空操作替代，零开销"""

    def __init__(self, *a, **kw):
        pass

    def step(self, n=1):
        pass

    def start_prefill(self):
        pass

    def end_prefill(self, first_token_id):
        pass

    def start_decode(self):
        pass

    def step_decode(self, token_id):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass
