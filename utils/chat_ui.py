"""Rich-based UI helpers for the multi-turn chat CLI.

This module is intentionally UI-only: it knows nothing about models,
tokenizers, or KV caches.
"""

from __future__ import annotations

import re
import sys

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


def extract_cot(text: str) -> tuple[str | None, str]:
    """Extract ``<think>...</think>`` block from assistant output.

    Returns:
        ``(cot_content, actual_response)``.  If no ``<think>`` block is found,
        ``cot_content`` is *None* and ``actual_response`` is the original text.
    """
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if not match:
        return None, text

    cot = match.group(1).strip()
    actual = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL, count=1).strip()
    return cot, actual


class ChatUI:
    """Console UI for multi-turn dialogue."""

    def __init__(self) -> None:
        self.console = Console()

    # ------------------------------------------------------------------ #
    # Header / meta
    # ------------------------------------------------------------------ #

    def print_header(self, model_name: str = "") -> None:
        """Print the welcome header."""
        title = "[bold green]mini-vllm 多轮对话[/bold green]"
        if model_name:
            title += f" [dim]({model_name})[/dim]"
        self.console.print(
            Panel(
                f"{title}\n"
                "[dim]命令: /exit 退出, /clear 清空历史, "
                "/think on|off 开关思考模式[/dim]",
                border_style="green",
            )
        )

    # ------------------------------------------------------------------ #
    # Turns
    # ------------------------------------------------------------------ #

    def print_user(self, content: str) -> None:
        """Render user input.

        In an interactive TTY the user already sees what they typed right
        after the prompt, so we only echo the line when stdin is piped.
        """
        if not sys.stdin.isatty():
            self.console.print(f"[bold blue]User:[/bold blue] {content}\n")
        else:
            self.console.print()

    def print_assistant(
        self,
        content: str,
        turn: int,
        elapsed_ms: float | None = None,
        show_cot: bool = True,
    ) -> None:
        """Render an assistant turn.

        If *show_cot* is *True* and the content contains a ``<think>`` block,
        the thinking trace is printed in dim/italic style before the actual
        response, separated by a blank line.
        """
        title = f"[bold green]Assistant (Turn {turn})[/bold green]"
        if elapsed_ms is not None:
            title += f" [dim]({elapsed_ms:.0f} ms)[/dim]"
        self.console.print(title)

        cot, actual = extract_cot(content)
        if cot and show_cot:
            # Render CoT as a dim/italic paragraph; no prefix so that Rich
            # soft-wrap works naturally without leaving a dangling prompt.
            self.console.print(Text(cot, style="italic dim"))
            self.console.print()

        self.console.print(actual)
        self.console.print()

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #

    def print_info(self, message: str) -> None:
        """Print an info line."""
        self.console.print(f"[yellow]ℹ {message}[/yellow]")

    def print_error(self, message: str) -> None:
        """Print an error line."""
        self.console.print(f"[bold red]✗ {message}[/bold red]")

    def input_prompt(self) -> str:
        """Read a line of user input.

        We use plain ``input()`` (rather than ``Console.input()``) to avoid
        ANSI escape sequences interfering with terminal backspace handling.
        The coloured label is printed separately via ``console.print``.
        """
        if sys.stdin.isatty():
            self.console.print("[bold blue]User:[/bold blue] ", end="")
        try:
            return input()
        except EOFError:
            raise
