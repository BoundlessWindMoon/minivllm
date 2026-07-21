"""Scheduler: decides which requests enter the running batch each step.

vLLM-aligned design
--------------------
schedule() returns (prefill_chunks, decode_reqs):

  prefill_chunks  list of (Request, chunk_tokens) — each pair says
                  "process *chunk_tokens* tokens of this request's prompt".
                  Multiple requests can have chunks in the same step.

  decode_reqs     list of DECODING requests (each contributes 1 token).

Token budget (max_num_batched_tokens)
--------------------------------------
When set, the total tokens per step = decode_tokens + sum(chunk_tokens)
is bounded by max_num_batched_tokens.  This matches vLLM's
--enable-chunked-prefill with max_num_batched_tokens.

When None (default), each admitted request's full remaining prompt is
processed in one step — equivalent to disabling chunked prefill.

Admission policies
------------------
fifo    First-come first-served.
spf     Shortest-Prompt-First: admit the waiting request with the
        fewest prompt tokens.
ljf     Longest-Job-First: admit the waiting request with the most
        prompt tokens.  Favours throughput over tail latency.
random  Random admission order.  Useful as a baseline.
"""

from __future__ import annotations
import random as _random
import time
from typing import TYPE_CHECKING

from engine.request import Request, RequestStatus

if TYPE_CHECKING:
    from engine.kv_pool import PagedKVPool
    from engine.prefix_cache import PrefixCache


class Scheduler:

    def __init__(
        self,
        kv_pool: "PagedKVPool",
        max_batch_size: int,
        admission_policy: str = "fifo",
        max_num_batched_tokens: int | None = None,
        prefix_cache: "PrefixCache | None" = None,
    ) -> None:
        self._pool                  = kv_pool
        self.max_batch_size         = max_batch_size
        self.admission_policy       = admission_policy
        self.max_num_batched_tokens = max_num_batched_tokens
        self._prefix_cache          = prefix_cache

        self._waiting: list[Request] = []
        self._running: list[Request] = []
        # Track which pages each running request borrowed from prefix cache.
        self._req_prefix_pages: dict[str, list[int]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_request(self, request: Request) -> None:
        self._waiting.append(request)

    def schedule(self) -> tuple[list[tuple[Request, int]], list[Request]]:
        """Return (prefill_chunks, decode_reqs) for the next step.

        prefill_chunks — list of (request, chunk_tokens).  All requests in
                         this list are in PREFILLING status.  chunk_tokens is
                         how many prompt tokens to process this step; it may
                         be less than the remaining prompt length when a token
                         budget is active.

        decode_reqs    — all currently DECODING requests.

        Algorithm
        ---------
        1. Collect all DECODING requests.
        2. Collect ongoing PREFILLING requests (partially-prefilled from
           previous steps).
        3. Admit new WAITING requests up to max_batch_size.
        4. Allocate token chunks:
             - If max_num_batched_tokens is None: each request processes
               its entire remaining prompt.
             - Otherwise: distribute the token budget greedily among all
               prefill requests (ongoing + newly admitted) in admission
               order.
        """
        decode_reqs: list[Request] = [
            r for r in self._running if r.status == RequestStatus.DECODING
        ]
        ongoing_prefill: list[Request] = [
            r for r in self._running if r.status == RequestStatus.PREFILLING
        ]

        # Admit new requests to fill remaining batch slots.
        new_budget = max(0, self.max_batch_size - len(decode_reqs) - len(ongoing_prefill))
        admitted: list[Request] = []
        while new_budget > 0 and self._waiting and self._pool.num_free_slots() > 0:
            req = self._pop_next_waiting()

            # Prefix cache lookup: find matching physical pages.
            prefix_pages: list[int] = []
            if self._prefix_cache is not None:
                prefix_pages, req.cached_prefix_len = self._prefix_cache.lookup(
                    req.prompt_token_ids, self._pool.page_size,
                )

            req.slot_id            = self._pool.allocate(req.request_id, prefix_pages)
            req.status             = RequestStatus.PREFILLING
            req.first_scheduled_at = time.perf_counter()
            # Skip already-cached tokens: prefill starts from cached_prefix_len.
            req.prefilled_len      = req.cached_prefix_len
            req.cache_len          = req.cached_prefix_len
            self._running.append(req)
            self._req_prefix_pages[req.request_id] = prefix_pages
            admitted.append(req)
            new_budget -= 1

        all_prefill = ongoing_prefill + admitted
        if not all_prefill:
            return [], decode_reqs

        # Allocate token chunks.
        if self.max_num_batched_tokens is None:
            # Unlimited: process each request's full remaining prompt.
            chunks = [
                (req, req.num_prompt_tokens - req.prefilled_len)
                for req in all_prefill
            ]
        else:
            # Distribute token budget evenly across all prefill requests so
            # no single long prompt monopolises the whole budget.
            decode_tokens  = len(decode_reqs)
            prefill_budget = max(1, self.max_num_batched_tokens - decode_tokens)
            n              = len(all_prefill)
            per_req        = max(1, prefill_budget // n)   # floor division
            chunks: list[tuple[Request, int]] = []
            for req in all_prefill:
                take = min(per_req, req.num_prompt_tokens - req.prefilled_len)
                if take > 0:
                    chunks.append((req, take))

        return chunks, decode_reqs

    def on_request_finished(self, request: Request) -> None:
        self._running = [r for r in self._running if r.request_id != request.request_id]

        if self._prefix_cache is not None:
            pool = self._pool
            if request.slot_id >= 0:
                pages = pool.pages_for(request.request_id)
                self._prefix_cache.insert(
                    request.prompt_token_ids, pages, pool.page_size,
                )
            prefix_pages = self._req_prefix_pages.pop(request.request_id, [])
            self._prefix_cache.release(prefix_pages)
        else:
            self._req_prefix_pages.pop(request.request_id, None)

        if request.slot_id >= 0:
            self._pool.free(request.slot_id)

    def has_work(self) -> bool:
        return bool(self._waiting or self._running)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pop_next_waiting(self) -> Request:
        if self.admission_policy == "spf":
            idx = min(range(len(self._waiting)),
                      key=lambda i: self._waiting[i].num_prompt_tokens)
            req = self._waiting[idx]
            del self._waiting[idx]
            return req
        if self.admission_policy == "ljf":
            idx = max(range(len(self._waiting)),
                      key=lambda i: self._waiting[i].num_prompt_tokens)
            req = self._waiting[idx]
            del self._waiting[idx]
            return req
        if self.admission_policy == "random":
            idx = _random.randrange(len(self._waiting))
            req = self._waiting[idx]
            del self._waiting[idx]
            return req
        return self._waiting.pop(0)   # fifo

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    @property
    def num_running(self) -> int:
        return len(self._running)
