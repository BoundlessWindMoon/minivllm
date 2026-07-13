"""Per-request lifecycle state.

A Request is the unit of work tracked by the Scheduler and mutated by
BatchedModelRunner.  It holds both the static inputs (prompt tokens,
sampling config) and the mutable runtime state (cache_len, generated
tokens, assigned slot, status).

Ownership:
  Created by     — caller / AsyncLLMEngine
  Slot assigned  — Scheduler.schedule()
  State mutated  — BatchedModelRunner._sample_and_update()
  Slot freed     — Scheduler.on_request_finished()
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable
import time

from engine.schema import SamplingParams


class RequestStatus(Enum):
    WAITING = auto()  # queued, no KV slot yet
    PREFILLING = auto()  # slot assigned, first forward pass in progress
    DECODING = auto()  # prefill done, generating tokens step by step
    FINISHED = auto()  # hit EOS or max_new_tokens; slot will be freed


@dataclass
class Request:
    # ---- Immutable inputs ----------------------------------------
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    prompt_text: str = ""          # human-readable original prompt for display

    # ---- Assigned by Scheduler -----------------------------------
    slot_id: int = -1

    # ---- Mutated by BatchedModelRunner ---------------------------
    cache_len: int = 0
    prefilled_len: int = 0         # tokens of the prompt already in KV cache
    generated_ids: list[int] = field(default_factory=list)
    status: RequestStatus = field(default=RequestStatus.WAITING)
    finish_reason: str | None = None

    # ---- Latency timestamps (wall-clock seconds, perf_counter) ---
    # enqueued_at:        set automatically on construction
    # first_scheduled_at: set by Scheduler when a KV slot is first assigned
    # first_token_at:     set by runner when the 1st output token is sampled
    # finished_at:        set by mark_finished()
    #
    # Derived metrics:
    #   TTFT            = first_token_at - enqueued_at        (user-perceived)
    #   time_in_queue   = first_scheduled_at - enqueued_at    (scheduler wait)
    #   prefill_time    = first_token_at - first_scheduled_at (GPU prefill)
    #   TPOT            = (finished_at - first_token_at) / (n_tokens - 1)
    #   e2e_latency     = finished_at - enqueued_at
    enqueued_at:        float      = field(default_factory=time.perf_counter)
    first_scheduled_at: float | None = None
    first_token_at:     float | None = None
    finished_at:        float | None = None

    # ---- Optional streaming callback -----------------------------
    on_token: Callable[[str, int, bool], None] | None = None

    # ------------------------------------------------------------------

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_generated_tokens(self) -> int:
        return len(self.generated_ids)

    @property
    def is_finished(self) -> bool:
        return self.status == RequestStatus.FINISHED

    @property
    def ttft(self) -> float | None:
        """Time To First Token: includes queue wait + prefill."""
        if self.first_token_at is None:
            return None
        return self.first_token_at - self.enqueued_at

    @property
    def time_in_queue(self) -> float | None:
        """Time spent waiting for a KV slot (scheduler queue delay)."""
        if self.first_scheduled_at is None:
            return None
        return self.first_scheduled_at - self.enqueued_at

    @property
    def prefill_time(self) -> float | None:
        """Actual GPU prefill duration (slot assigned → first token)."""
        if self.first_scheduled_at is None or self.first_token_at is None:
            return None
        return self.first_token_at - self.first_scheduled_at

    @property
    def tpot(self) -> float | None:
        """Time Per Output Token (average, wall-clock, includes decode stalls)."""
        if self.finished_at is None or self.first_token_at is None:
            return None
        n = self.num_generated_tokens - 1
        if n <= 0:
            return None
        return (self.finished_at - self.first_token_at) / n

    @property
    def e2e_latency(self) -> float | None:
        """End-to-end latency from enqueue to completion."""
        if self.finished_at is None:
            return None
        return self.finished_at - self.enqueued_at

    def mark_finished(self, reason: str) -> None:
        self.status = RequestStatus.FINISHED
        self.finish_reason = reason
        self.finished_at = time.perf_counter()
