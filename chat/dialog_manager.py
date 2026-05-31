"""Dialog manager for multi-turn conversation.

Tracks message history and KV-cache bookkeeping so that ModelRunner can
perform incremental prefill across turns.
"""

from __future__ import annotations

import torch


class DialogManager:
    """Manages conversation state and incremental-prefill metadata.

    Responsibilities:
        - Maintain the message list (user / assistant / system turns).
        - Build ``input_ids`` from the full history via the tokenizer's chat template.
        - Track ``cached_len`` so the engine can skip already-computed KV entries.
        - Guard against sequence-length overflow and handle re-tokenization mismatches
          safely (falling back to full-prefill when necessary).

    The manager is intentionally decoupled from ``ModelRunner`` and UI code.
    It only depends on a HuggingFace-compatible tokenizer and a device spec.
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len: int,
        device: torch.device | str,
        system_prompt: str | None = None,
        use_thinking: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.device = device
        self.use_thinking = use_thinking
        self._prev_thinking = use_thinking

        self.messages: list[dict] = []
        self.cached_len: int = 0
        self._last_input_ids: torch.Tensor | None = None

        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def add_user(self, content: str) -> None:
        """Append a user message."""
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        """Append an assistant message (typically the decoded output)."""
        self.messages.append({"role": "assistant", "content": content})

    def prepare_input(self) -> torch.Tensor:
        """Build input_ids from the complete message history.

        Returns:
            Long tensor of shape ``[1, seq_len]`` on ``self.device``.

        Raises:
            ValueError: If the conversation is empty.
        """
        if not self.messages:
            raise ValueError("Empty conversation; add a user message first.")

        kwargs = {
            "add_generation_prompt": True,
            "return_tensors": "pt",
        }
        if self.use_thinking is not None:
            kwargs["enable_thinking"] = self.use_thinking

        result = self.tokenizer.apply_chat_template(self.messages, **kwargs)
        # Some tokenizers return a BatchEncoding, others a plain tensor.
        if hasattr(result, "input_ids"):
            result = result.input_ids

        self._last_input_ids = result.to(self.device)
        return self._last_input_ids

    def get_generation_input(self) -> tuple[torch.Tensor, int]:
        """Return the full input_ids and the safe ``cached_len`` for generation.

        The returned ``cached_len`` may be smaller than ``self.cached_len`` if the
        history was re-tokenized to a shorter length (e.g. special-token round-trip
        differences).  In that edge case we fall back to ``0`` so correctness is
        preserved at the cost of a full prefill.

        Raises:
            ValueError: If the sequence exceeds ``max_seq_len``.
        """
        input_ids = self.prepare_input()
        seq_len = input_ids.shape[1]

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        # Safety 1: if re-tokenization shrunk the history, fall back to full prefill.
        # Safety 2: if thinking mode changed mid-conversation, the tokenization of
        # history may differ (e.g. <think> tags added/removed). Fall back to full
        # prefill to guarantee correctness.
        thinking_changed = self._prev_thinking != self.use_thinking
        if thinking_changed:
            effective_cache = 0
        else:
            effective_cache = self.cached_len if seq_len >= self.cached_len else 0
        self._prev_thinking = self.use_thinking
        return input_ids, effective_cache

    def update_cache(self, output_ids: torch.Tensor) -> None:
        """Update ``cached_len`` after a successful generation step.

        Args:
            output_ids: Tensor of shape ``[1, num_new_tokens]``.
        """
        if self._last_input_ids is None:
            raise RuntimeError("update_cache() called before prepare_input()")
        self.cached_len = self._last_input_ids.shape[1] + output_ids.shape[1]

    def clear(self) -> None:
        """Reset conversation state while preserving the system prompt (if any)."""
        system = [m for m in self.messages if m.get("role") == "system"]
        self.messages = system.copy()
        self.cached_len = 0
        self._last_input_ids = None

    def rollback_last_user(self) -> None:
        """Remove the last user message; useful when input validation fails."""
        if self.messages and self.messages[-1].get("role") == "user":
            self.messages.pop()
