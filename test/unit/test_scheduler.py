"""Unit tests for engine/scheduler.py -- no GPU, no real model."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
from engine.scheduler import Scheduler
from engine.request import Request
from engine.schema import SamplingParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakePool:
    """Minimal pool stub: tracks slot alloc/free, never runs out unless told to."""
    def __init__(self, num_slots=32):
        self._free = set(range(num_slots))
        self._used = {}

    def num_free_slots(self): return len(self._free)

    def allocate(self, request_id):
        slot = self._free.pop()
        self._used[slot] = request_id
        return slot

    def free(self, slot_id):
        self._used.pop(slot_id, None)
        self._free.add(slot_id)


def make_req(prompt_len, req_id=None):
    return Request(
        request_id=req_id or f"r{prompt_len}",
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_new_tokens=8),
    )


def make_sched(pool=None, max_batch_size=8, policy="fifo", budget=None):
    return Scheduler(
        pool or FakePool(),
        max_batch_size=max_batch_size,
        admission_policy=policy,
        max_num_batched_tokens=budget,
    )


def finish(scheduler, req):
    """Complete a request through the public API."""
    req.mark_finished("length")
    scheduler.on_request_finished(req)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def test_empty_has_no_work():
    assert not make_sched().has_work()


def test_add_request_creates_work():
    s = make_sched()
    s.add_request(make_req(10))
    assert s.has_work()


def test_schedule_moves_waiting_to_running():
    s = make_sched()
    s.add_request(make_req(10))
    chunks, decode = s.schedule()
    assert len(chunks) == 1
    assert len(decode) == 0
    assert s.num_running == 1
    assert s.num_waiting == 0


def test_finish_frees_slot():
    pool = FakePool(4)
    s = make_sched(pool=pool, max_batch_size=4)
    r = make_req(10)
    s.add_request(r)
    s.schedule()
    assert pool.num_free_slots() == 3
    finish(s, r)
    assert pool.num_free_slots() == 4
    assert not s.has_work()


def test_no_work_after_all_finished():
    s = make_sched()
    r = make_req(5)
    s.add_request(r)
    s.schedule()
    finish(s, r)
    assert not s.has_work()


# ---------------------------------------------------------------------------
# Admission policies
# ---------------------------------------------------------------------------

def _admitted_order(policy, lengths):
    """Return prompt_lengths in the order they were admitted, one slot at a time."""
    s = make_sched(max_batch_size=1, policy=policy)
    for i, l in enumerate(lengths):
        s.add_request(make_req(l, req_id=f"r{i}"))
    order = []
    while s.has_work():
        chunks, _ = s.schedule()
        if not chunks:
            break
        for req, _ in chunks:
            order.append(req.num_prompt_tokens)
            finish(s, req)
    return order


def test_fifo_preserves_insertion_order():
    lengths = [10, 5, 20, 3, 15]
    assert _admitted_order("fifo", lengths) == lengths


def test_spf_admits_shortest_first():
    lengths = [10, 5, 20, 3, 15]
    assert _admitted_order("spf", lengths) == sorted(lengths)


def test_ljf_admits_longest_first():
    lengths = [10, 5, 20, 3, 15]
    assert _admitted_order("ljf", lengths) == sorted(lengths, reverse=True)


def test_random_admits_all_requests():
    lengths = [10, 5, 20, 3, 15]
    assert sorted(_admitted_order("random", lengths)) == sorted(lengths)


@pytest.mark.parametrize("policy", ["fifo", "spf", "ljf", "random"])
def test_all_policies_complete_without_leak(policy):
    """Every policy must finish all requests and leave the pool clean."""
    pool = FakePool(8)
    s = make_sched(pool=pool, max_batch_size=4, policy=policy)
    reqs = [make_req(l, f"r{i}") for i, l in enumerate([10, 5, 20, 3, 15])]
    for r in reqs:
        s.add_request(r)
    while s.has_work():
        chunks, _ = s.schedule()
        if not chunks:
            break
        for req, _ in chunks:
            finish(s, req)
    assert pool.num_free_slots() == 8


# ---------------------------------------------------------------------------
# max_batch_size and slot constraints
# ---------------------------------------------------------------------------

def test_max_batch_size_respected():
    s = make_sched(max_batch_size=2)
    for i in range(6):
        s.add_request(make_req(5, f"r{i}"))
    chunks, _ = s.schedule()
    assert len(chunks) <= 2


def test_slot_exhaustion_caps_admission():
    pool = FakePool(2)
    s = make_sched(pool=pool, max_batch_size=8)
    for i in range(5):
        s.add_request(make_req(5, f"r{i}"))
    chunks, _ = s.schedule()
    assert len(chunks) == 2
    assert pool.num_free_slots() == 0


def test_freed_slot_allows_next_admission():
    pool = FakePool(1)
    s = make_sched(pool=pool, max_batch_size=4)
    r1, r2 = make_req(5, "r1"), make_req(5, "r2")
    s.add_request(r1)
    s.add_request(r2)
    chunks, _ = s.schedule()
    assert len(chunks) == 1
    finish(s, r1)
    chunks2, _ = s.schedule()
    assert len(chunks2) == 1


# ---------------------------------------------------------------------------
# Chunked prefill
# ---------------------------------------------------------------------------

def test_no_budget_processes_full_prompt():
    s = make_sched(budget=None)
    s.add_request(make_req(100))
    chunks, _ = s.schedule()
    _, chunk_tokens = chunks[0]
    assert chunk_tokens == 100


def test_budget_caps_chunk_size():
    s = make_sched(budget=32)
    s.add_request(make_req(100))
    chunks, _ = s.schedule()
    _, chunk_tokens = chunks[0]
    assert chunk_tokens <= 32


def test_budget_distributes_across_requests():
    s = make_sched(budget=60, max_batch_size=4)
    s.add_request(make_req(100, "a"))
    s.add_request(make_req(100, "b"))
    chunks, _ = s.schedule()
    assert len(chunks) == 2
    for _, ct in chunks:
        assert ct <= 30


def test_chunked_prefill_converges():
    """Simulate the runner driving chunked prefill to completion."""
    s = make_sched(budget=20, max_batch_size=4)
    r = make_req(55)
    s.add_request(r)
    total = 0
    for _ in range(20):
        chunks, _ = s.schedule()
        if not chunks:
            break
        for req, ct in chunks:
            req.prefilled_len += ct
            req.cache_len = req.prefilled_len
            total += ct
        if total >= r.num_prompt_tokens:
            break
    assert total == r.num_prompt_tokens
