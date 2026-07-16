"""Unit tests for BatchedModelRunner.step() new return value.

Verifies the (finished, new_tokens) tuple contract without GPU or a real model.
A minimal stub replaces the scheduler and sampler so the test is CPU-only.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
from unittest.mock import MagicMock, patch
from engine.request import Request, RequestStatus
from engine.schema import SamplingParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_req(req_id, prompt_len=4, max_new_tokens=8):
    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
    )


def make_runner_with_stub_step(decode_reqs, prefill_chunks, sampled_tokens, eos_ids=None):
    """Build a BatchedModelRunner whose internals are fully stubbed.

    decode_reqs     : list[Request] returned as decode batch by scheduler
    prefill_chunks  : list[(Request, int)] returned as prefill batch
    sampled_tokens  : list[int] — one token per request that gets sampled
    eos_ids         : set of token ids treated as EOS (default empty)
    """
    import torch
    from engine.batched_runner import BatchedModelRunner

    # Stub scheduler: returns fixed schedule, has_work() always True initially.
    sched = MagicMock()
    sched.schedule.return_value = (prefill_chunks, decode_reqs)

    # Stub KV pool.
    pool = MagicMock()
    pool.num_free_slots.return_value = 8

    # Stub model.
    model = MagicMock()
    vocab = 32
    total_reqs = len(decode_reqs) + len([r for r, _ in prefill_chunks])
    fake_logits = torch.zeros(max(total_reqs, 1), 1, vocab)
    model.return_value = fake_logits
    model.iter_attention_modules.return_value = []

    # Stub tokenizer.
    tokenizer = MagicMock()

    # Stub config.
    cfg = MagicMock()
    cfg.env.device = "cpu"
    cfg.generation.sampling.sample_method = "greedy"
    cfg.generation.sampling.temperature = 1.0
    cfg.generation.sampling.topk = 1
    cfg.generation.sampling.topp = 1.0
    cfg.model.use_cuda_graph = False
    cfg.model.cuda_graph_max_batch_size = None
    cfg.batch.max_batch_size = 8

    runner = BatchedModelRunner.__new__(BatchedModelRunner)
    runner.model = model
    runner.tokenizer = tokenizer
    runner.kv_pool = pool
    runner.scheduler = sched
    runner.cfg = cfg
    runner.device = "cpu"
    runner._graph_manager = None
    runner.on_prefill_start = None
    runner.on_graph_capture_start = None
    runner.on_graph_capture_step = None
    runner.last_step_stats = {}

    # Inject EOS ids and sampler.
    from engine.sampler import Sampler
    runner._eos_ids = set(eos_ids or [])

    token_iter = iter(sampled_tokens)

    def fake_sample(logits):
        tokens = []
        for _ in range(logits.shape[0]):
            t = next(token_iter, 99)
            tokens.append([t])
        return torch.tensor(tokens)

    runner.sampler = MagicMock()
    runner.sampler.sample.side_effect = fake_sample

    return runner


# ---------------------------------------------------------------------------
# step() return type
# ---------------------------------------------------------------------------

def test_step_returns_tuple():
    req = make_req("r1")
    req.status = RequestStatus.DECODING
    req.generated_ids = [5]
    req.cache_len = 4

    runner = make_runner_with_stub_step(
        decode_reqs=[req],
        prefill_chunks=[],
        sampled_tokens=[7],
    )

    with patch.object(runner, "_run_decode", return_value=__import__("torch").zeros(1, 1, 32)):
        with patch.object(runner, "_sample_and_update", wraps=lambda reqs, logits: _fake_sample_update(reqs, [7])):
            result = runner.step()

    assert isinstance(result, tuple) and len(result) == 2
    finished, new_tokens = result
    assert isinstance(finished, list)
    assert isinstance(new_tokens, dict)


def _fake_sample_update(reqs, tokens):
    for req, tok in zip(reqs, tokens):
        req.generated_ids.append(tok)


# ---------------------------------------------------------------------------
# new_tokens populated for decode requests
# ---------------------------------------------------------------------------

def test_decode_token_appears_in_new_tokens():
    req = make_req("r1", max_new_tokens=100)
    req.status = RequestStatus.DECODING
    req.generated_ids = [5]
    req.cache_len = 4

    import torch
    runner = make_runner_with_stub_step(
        decode_reqs=[req],
        prefill_chunks=[],
        sampled_tokens=[42],
    )

    with patch.object(runner, "_run_decode", return_value=torch.zeros(1, 1, 32)):
        finished, new_tokens = runner.step()

    assert "r1" in new_tokens
    assert new_tokens["r1"] == req.generated_ids[-1]


def test_multiple_decode_requests_each_get_token():
    import torch
    reqs = [make_req(f"r{i}", max_new_tokens=100) for i in range(3)]
    for r in reqs:
        r.status = RequestStatus.DECODING
        r.generated_ids = [5]
        r.cache_len = 4

    runner = make_runner_with_stub_step(
        decode_reqs=reqs,
        prefill_chunks=[],
        sampled_tokens=[10, 20, 30],
    )

    with patch.object(runner, "_run_decode", return_value=torch.zeros(3, 1, 32)):
        finished, new_tokens = runner.step()

    assert set(new_tokens.keys()) == {"r0", "r1", "r2"}


# ---------------------------------------------------------------------------
# finished list and new_tokens agree
# ---------------------------------------------------------------------------

def test_eos_request_in_both_finished_and_new_tokens():
    """An EOS token must appear in new_tokens AND the request in finished."""
    import torch
    req = make_req("r_eos", max_new_tokens=100)
    req.status = RequestStatus.DECODING
    req.generated_ids = [5]
    req.cache_len = 4

    EOS = 2
    runner = make_runner_with_stub_step(
        decode_reqs=[req],
        prefill_chunks=[],
        sampled_tokens=[EOS],
        eos_ids={EOS},
    )
    runner.scheduler.on_request_finished = MagicMock()

    with patch.object(runner, "_run_decode", return_value=torch.zeros(1, 1, 32)):
        finished, new_tokens = runner.step()

    assert "r_eos" in new_tokens
    assert new_tokens["r_eos"] == EOS
    assert any(r.request_id == "r_eos" for r in finished)


def test_non_finished_request_not_in_finished():
    import torch
    req = make_req("r1", max_new_tokens=100)
    req.status = RequestStatus.DECODING
    req.generated_ids = [5]
    req.cache_len = 4

    runner = make_runner_with_stub_step(
        decode_reqs=[req],
        prefill_chunks=[],
        sampled_tokens=[7],  # not EOS
    )

    with patch.object(runner, "_run_decode", return_value=torch.zeros(1, 1, 32)):
        finished, new_tokens = runner.step()

    assert finished == []
    assert "r1" in new_tokens


# ---------------------------------------------------------------------------
# Empty step
# ---------------------------------------------------------------------------

def test_empty_step_returns_empty():
    runner = make_runner_with_stub_step(
        decode_reqs=[],
        prefill_chunks=[],
        sampled_tokens=[],
    )
    finished, new_tokens = runner.step()
    assert finished == []
    assert new_tokens == {}
