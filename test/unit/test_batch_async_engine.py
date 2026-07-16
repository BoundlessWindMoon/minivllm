"""Unit tests for engine/batch_async_engine.py.

All tests run with stubbed runner/scheduler/tokenizer — no GPU, no real model.
The step loop thread is exercised directly by controlling what the stubs return.
Uses asyncio.run() so no async pytest plugin is required.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import asyncio
import pytest
from unittest.mock import MagicMock
from engine.batch_async_engine import BatchAsyncEngine
from engine.request import Request, RequestStatus
from engine.schema import SamplingParams, GenerationOutput


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

def _make_req(req_id, prompt_len=4, max_new_tokens=4):
    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
    )


class StubTokenizer:
    """Returns token_id as a decimal string (deterministic, easy to assert)."""
    def decode(self, token_ids, skip_special_tokens=True):
        return str(token_ids[0])


class StepSequence:
    """Controls what runner.step() returns over multiple calls.

    Each entry in `steps` is (new_tokens: dict, finished_ids: list[str]).
    After exhausting steps, has_work() returns False.
    """

    def __init__(self, steps: list[tuple[dict, list[str]]], reqs: dict[str, Request]):
        self._steps = list(steps)
        self._reqs = reqs
        self._idx = 0

    def has_work(self):
        return self._idx < len(self._steps)

    def step(self):
        new_tokens, fin_ids = self._steps[self._idx]
        self._idx += 1
        finished = []
        for rid in fin_ids:
            req = self._reqs[rid]
            req.mark_finished("eos")
            finished.append(req)
        return finished, new_tokens

    def add_request(self, req):
        self._reqs[req.request_id] = req

    def schedule(self):
        return [], []

    def on_request_finished(self, req):
        pass


def _make_engine(steps, req_ids=None):
    """Build a BatchAsyncEngine with a fully stubbed runner."""
    req_ids = req_ids or []
    reqs = {rid: _make_req(rid) for rid in req_ids}
    seq = StepSequence(steps, reqs)

    runner = MagicMock()
    runner.step.side_effect = seq.step

    engine = BatchAsyncEngine(runner, seq, StubTokenizer())
    return engine, seq


# ---------------------------------------------------------------------------
# Basic token delivery
# ---------------------------------------------------------------------------

def test_single_request_receives_all_tokens():
    async def _run():
        steps = [
            ({"r1": 10}, []),
            ({"r1": 11}, []),
            ({"r1": 2},  ["r1"]),
        ]
        engine, _ = _make_engine(steps, req_ids=["r1"])
        tokens = []
        async for out in engine.generate([1, 2, 3], SamplingParams(max_new_tokens=4), "r1"):
            tokens.append(out)
        engine.shutdown()
        return tokens

    tokens = asyncio.run(_run())
    assert len(tokens) == 3
    assert [t.token_id for t in tokens] == [10, 11, 2]
    assert tokens[-1].is_finished
    assert tokens[-1].finish_reason == "eos"
    assert not tokens[0].is_finished


def test_text_delta_matches_token_decode():
    async def _run():
        steps = [({"r1": 42}, ["r1"])]
        engine, _ = _make_engine(steps, req_ids=["r1"])
        outputs = []
        async for out in engine.generate([1], SamplingParams(max_new_tokens=1), "r1"):
            outputs.append(out)
        engine.shutdown()
        return outputs

    outputs = asyncio.run(_run())
    assert outputs[0].text_delta == "42"


# ---------------------------------------------------------------------------
# Concurrent requests
# ---------------------------------------------------------------------------

def test_two_concurrent_requests_get_independent_tokens():
    async def _run():
        steps = [
            ({"r1": 10, "r2": 20}, []),
            ({"r1": 11, "r2": 21}, ["r1", "r2"]),
        ]
        engine, _ = _make_engine(steps, req_ids=["r1", "r2"])

        async def collect(rid, prompt):
            toks = []
            async for out in engine.generate(prompt, SamplingParams(max_new_tokens=4), rid):
                toks.append(out.token_id)
            return rid, toks

        result = await asyncio.gather(collect("r1", [1, 2]), collect("r2", [3, 4]))
        engine.shutdown()
        return result

    (_, toks1), (_, toks2) = asyncio.run(_run())
    assert toks1 == [10, 11]
    assert toks2 == [20, 21]


def test_requests_finish_at_different_times():
    async def _run():
        steps = [
            ({"r1": 5, "r2": 50}, ["r1"]),
            ({"r2": 51}, []),
            ({"r2": 52}, ["r2"]),
        ]
        engine, _ = _make_engine(steps, req_ids=["r1", "r2"])

        async def collect(rid, prompt):
            toks = []
            async for out in engine.generate(prompt, SamplingParams(max_new_tokens=8), rid):
                toks.append(out.token_id)
            return rid, toks

        result = await asyncio.gather(collect("r1", [1]), collect("r2", [2]))
        engine.shutdown()
        return result

    (_, toks1), (_, toks2) = asyncio.run(_run())
    assert toks1 == [5]
    assert toks2 == [50, 51, 52]


# ---------------------------------------------------------------------------
# Queue cleanup: no leak after request finishes
# ---------------------------------------------------------------------------

def test_queue_cleaned_up_after_finish():
    async def _run():
        steps = [({"r1": 1}, ["r1"])]
        engine, _ = _make_engine(steps, req_ids=["r1"])
        async for _ in engine.generate([1], SamplingParams(max_new_tokens=1), "r1"):
            pass
        await asyncio.sleep(0.05)
        with engine._lock:
            leaked = "r1" in engine._queues
        engine.shutdown()
        return leaked

    assert not asyncio.run(_run())


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

def test_shutdown_stops_thread():
    engine, _ = _make_engine([])
    assert engine._thread.is_alive()
    engine.shutdown()
    assert not engine._thread.is_alive()


# ---------------------------------------------------------------------------
# GenerationOutput fields
# ---------------------------------------------------------------------------

def test_generation_output_fields_are_correct():
    async def _run():
        steps = [({"r1": 99}, ["r1"])]
        engine, _ = _make_engine(steps, req_ids=["r1"])
        outputs = []
        async for out in engine.generate([1], SamplingParams(max_new_tokens=1), "r1"):
            outputs.append(out)
        engine.shutdown()
        return outputs

    outputs = asyncio.run(_run())
    o = outputs[0]
    assert o.request_id == "r1"
    assert o.token_id == 99
    assert o.is_finished is True
    assert o.finish_reason == "eos"
    assert o.error is None
