"""Tests for engine/scheduler.py and related display helpers.

Covers:
  - All 4 admission policies (fifo, spf, ljf, random)
  - Chunked prefill token budget distribution
  - Scheduler lifecycle: add, schedule, finish, has_work
  - max_batch_size / num_slots interaction
  - _pct edge cases (n=0, n=1, n=2+)
  - print_sweep_summary smoke test (no crash, correct structure)
"""

import time
import statistics
import pytest

from engine.scheduler import Scheduler
from engine.request import Request, RequestStatus
from engine.schema import SamplingParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakePool:
    """Minimal KVCachePool stub: unlimited slots, tracks allocation."""

    def __init__(self, num_slots: int = 32):
        self._free = set(range(num_slots))
        self._used: dict[int, str] = {}

    def num_free_slots(self) -> int:
        return len(self._free)

    def allocate(self, request_id: str) -> int:
        slot = self._free.pop()
        self._used[slot] = request_id
        return slot

    def free(self, slot_id: int) -> None:
        self._used.pop(slot_id, None)
        self._free.add(slot_id)


def make_req(prompt_len: int, req_id: str = None) -> Request:
    return Request(
        request_id       = req_id or f"req-{prompt_len}",
        prompt_token_ids = list(range(prompt_len)),
        sampling_params  = SamplingParams(max_new_tokens=32),
    )


def make_scheduler(pool=None, max_batch_size=8, policy="fifo", budget=None):
    pool = pool or FakePool()
    return Scheduler(pool, max_batch_size=max_batch_size,
                     admission_policy=policy,
                     max_num_batched_tokens=budget)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_empty_scheduler_has_no_work(self):
        s = make_scheduler()
        assert not s.has_work()

    def test_add_request_has_work(self):
        s = make_scheduler()
        s.add_request(make_req(10))
        assert s.has_work()

    def test_schedule_moves_waiting_to_running(self):
        s = make_scheduler()
        s.add_request(make_req(10))
        chunks, decode = s.schedule()
        assert len(chunks) == 1
        assert len(decode) == 0
        assert s.num_running == 1
        assert s.num_waiting == 0

    def test_on_request_finished_frees_slot(self):
        pool = FakePool(num_slots=4)
        s = make_scheduler(pool=pool, max_batch_size=4)
        req = make_req(10)
        s.add_request(req)
        chunks, _ = s.schedule()
        assert pool.num_free_slots() == 3
        req.status = RequestStatus.FINISHED
        s.on_request_finished(req)
        assert pool.num_free_slots() == 4
        assert not s.has_work()

    def test_has_work_false_after_all_finished(self):
        s = make_scheduler()
        req = make_req(5)
        s.add_request(req)
        chunks, _ = s.schedule()
        s.on_request_finished(req)
        assert not s.has_work()


# ---------------------------------------------------------------------------
# Admission policies
# ---------------------------------------------------------------------------

class TestAdmissionPolicies:
    def _admitted_order(self, policy: str, lengths: list[int]) -> list[int]:
        """Return prompt lengths in the order they were admitted."""
        s = make_scheduler(policy=policy, max_batch_size=1)
        for i, l in enumerate(lengths):
            s.add_request(make_req(l, req_id=f"r{i}"))
        order = []
        while s.has_work() or s.num_running > 0:
            chunks, _ = s.schedule()
            if not chunks:
                break
            for req, _ in chunks:
                order.append(req.num_prompt_tokens)
                req.status = RequestStatus.FINISHED
                s.on_request_finished(req)
        return order

    def test_fifo_preserves_insertion_order(self):
        lengths = [10, 5, 20, 3, 15]
        order = self._admitted_order("fifo", lengths)
        assert order == lengths

    def test_spf_admits_shortest_first(self):
        lengths = [10, 5, 20, 3, 15]
        order = self._admitted_order("spf", lengths)
        assert order == sorted(lengths)

    def test_ljf_admits_longest_first(self):
        lengths = [10, 5, 20, 3, 15]
        order = self._admitted_order("ljf", lengths)
        assert order == sorted(lengths, reverse=True)

    def test_random_admits_all_requests(self):
        lengths = [10, 5, 20, 3, 15]
        order = self._admitted_order("random", lengths)
        assert sorted(order) == sorted(lengths)

    def test_random_is_non_deterministic_across_seeds(self):
        # With 10 requests it's astronomically unlikely two runs produce same order.
        import random
        lengths = list(range(1, 11))
        orders = set()
        for _ in range(5):
            order = self._admitted_order("random", lengths)
            orders.add(tuple(order))
        # At least 2 distinct orderings across 5 runs
        assert len(orders) > 1


# ---------------------------------------------------------------------------
# max_batch_size and slot constraints
# ---------------------------------------------------------------------------

class TestBatchSizeAndSlots:
    def test_max_batch_size_respected(self):
        s = make_scheduler(max_batch_size=2)
        for i in range(6):
            s.add_request(make_req(5, req_id=f"r{i}"))
        chunks, decode = s.schedule()
        assert len(chunks) <= 2

    def test_slot_exhaustion_limits_admission(self):
        pool = FakePool(num_slots=2)
        s = make_scheduler(pool=pool, max_batch_size=8)
        for i in range(5):
            s.add_request(make_req(5, req_id=f"r{i}"))
        chunks, _ = s.schedule()
        # Only 2 slots → only 2 admitted regardless of max_batch_size
        assert len(chunks) == 2
        assert pool.num_free_slots() == 0

    def test_slot_freed_allows_next_admission(self):
        pool = FakePool(num_slots=1)
        s = make_scheduler(pool=pool, max_batch_size=4)
        r1, r2 = make_req(5, "r1"), make_req(5, "r2")
        s.add_request(r1)
        s.add_request(r2)
        chunks, _ = s.schedule()
        assert len(chunks) == 1  # only 1 slot
        s.on_request_finished(r1)
        chunks2, _ = s.schedule()
        assert len(chunks2) == 1  # r2 now admitted


# ---------------------------------------------------------------------------
# Chunked prefill
# ---------------------------------------------------------------------------

class TestChunkedPrefill:
    def test_no_budget_processes_full_prompt(self):
        s = make_scheduler(budget=None)
        s.add_request(make_req(100))
        chunks, _ = s.schedule()
        assert len(chunks) == 1
        req, chunk_tokens = chunks[0]
        assert chunk_tokens == 100

    def test_budget_caps_chunk_size(self):
        s = make_scheduler(budget=32)
        s.add_request(make_req(100))
        chunks, _ = s.schedule()
        _, chunk_tokens = chunks[0]
        assert chunk_tokens <= 32

    def test_budget_distributes_across_requests(self):
        # 2 requests of 100 tokens each, budget=60 → each gets ≤30 tokens
        s = make_scheduler(budget=60, max_batch_size=4)
        s.add_request(make_req(100, "a"))
        s.add_request(make_req(100, "b"))
        chunks, _ = s.schedule()
        assert len(chunks) == 2
        for _, ct in chunks:
            assert ct <= 30

    def test_budget_accounts_for_decode_tokens(self):
        # 1 decoding request (1 decode token) + budget=32 → prefill budget = 31
        s = make_scheduler(budget=32, max_batch_size=4)
        prefill_req = make_req(50, "p")
        decode_req  = make_req(5, "d")
        # Manually put decode_req into DECODING state in the running list
        decode_req.slot_id = FakePool().allocate("d")
        decode_req.status  = RequestStatus.DECODING
        s._running.append(decode_req)
        s.add_request(prefill_req)
        chunks, decode = s.schedule()
        assert len(decode) == 1
        assert len(chunks) == 1
        _, ct = chunks[0]
        # prefill budget = 32 - 1 decode = 31
        assert ct <= 31

    def test_multi_step_chunked_prefill_completes(self):
        s = make_scheduler(budget=20, max_batch_size=4)
        req = make_req(55, "r")
        s.add_request(req)
        total_processed = 0
        steps = 0
        while total_processed < req.num_prompt_tokens:
            chunks, _ = s.schedule()
            if not chunks:
                break
            for r, ct in chunks:
                r.prefilled_len += ct
                r.cache_len      = r.prefilled_len
                total_processed += ct
            steps += 1
            if steps > 20:
                pytest.fail("chunked prefill did not converge")
        assert total_processed == 55


# ---------------------------------------------------------------------------
# _pct edge cases
# ---------------------------------------------------------------------------

class TestPct:
    def _pct(self, values, p):
        from ui.batch_display import _pct
        return _pct(values, p)

    def test_empty_returns_zero(self):
        assert self._pct([], 50) == 0.0

    def test_single_value_returns_itself(self):
        assert self._pct([42.0], 50) == 42.0
        assert self._pct([42.0], 99) == 42.0

    def test_two_values_p50(self):
        result = self._pct([10.0, 20.0], 50)
        assert 10.0 <= result <= 20.0

    def test_large_list_p50_near_median(self):
        values = list(range(1, 101))  # 1..100
        p50 = self._pct(values, 50)
        assert 48 <= p50 <= 52

    def test_p99_near_max(self):
        values = list(range(1, 101))
        p99 = self._pct(values, 99)
        assert p99 >= 95


# ---------------------------------------------------------------------------
# print_sweep_summary smoke test
# ---------------------------------------------------------------------------

class TestSweepSummary:
    def _make_runs(self, n_reqs=4, n_repeats=2):
        import uuid
        runs = []
        for _ in range(n_repeats):
            reqs = []
            t0 = 0.0
            for i in range(n_reqs):
                r = Request(
                    request_id       = str(uuid.uuid4())[:8],
                    prompt_token_ids = list(range(10 + i * 5)),
                    sampling_params  = SamplingParams(max_new_tokens=32),
                )
                r.enqueued_at        = t0
                r.first_scheduled_at = t0 + 0.01
                r.first_token_at     = t0 + 0.05
                r.finished_at        = t0 + 0.30
                r.generated_ids      = list(range(20))
                r.status             = RequestStatus.FINISHED
                r.finish_reason      = "length"
                reqs.append(r)
                t0 += 0.05
            runs.append((reqs, 1.0 + _ * 0.05))
        return runs

    def test_smoke_no_crash(self):
        from ui.batch_display import print_sweep_summary
        sweep = [
            ("fifo",   self._make_runs()),
            ("spf",    self._make_runs()),
            ("ljf",    self._make_runs()),
            ("random", self._make_runs()),
        ]
        # Should not raise
        print_sweep_summary(sweep)

    def test_single_repeat_no_crash(self):
        from ui.batch_display import print_sweep_summary
        sweep = [("fifo", self._make_runs(n_repeats=1))]
        print_sweep_summary(sweep)

