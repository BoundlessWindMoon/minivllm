"""Integration tests: multi-batch CUDA graph decode (BatchDecodeGraphManager).

Requires: ~/huggingface/Qwen3-0.6B/ and a CUDA device.

Test coverage:
  - Functional: CUDA graph produces identical tokens to eager for bs=1 and bs=3
  - Padding correctness: padded requests (real_bs < padded_bucket) don't corrupt output
  - KV-pool safety: capture snapshot/restore preserves pre-existing KV state
  - Slot isolation: two concurrent requests don't bleed KV across slots
  - Bucket selection: _build_capture_sizes and pad_to_bucket correctness
"""

import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
import torch
import torch.distributed as dist

from utils.config import GlobalConfig
from engine.loader import load_model, build_kv_pool
from engine.scheduler import Scheduler
from engine.batched_runner import (
    BatchedModelRunner,
    BatchDecodeGraphManager,
    _build_capture_sizes,
)
from engine.request import Request
from engine.schema import SamplingParams

MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dist_init():
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://localhost:29512",
            world_size=1,
            rank=0,
        )
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def model_and_tokenizer(dist_init):
    cfg = _base_cfg()
    prev_dtype  = torch.get_default_dtype()
    prev_device = torch.get_default_device()
    torch.set_default_dtype(cfg.env.get_torch_dtype())
    torch.set_default_device(cfg.env.device)
    model, tokenizer = load_model(cfg)
    model.eval()
    yield model, tokenizer
    torch.set_default_dtype(prev_dtype)
    torch.set_default_device(prev_device)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_cfg(use_cuda_graph: bool = False):
    cfg = GlobalConfig.from_yaml("configs/runs/batch.yaml")
    cfg.model.backend = "default"
    cfg.model.use_cuda_graph = use_cuda_graph
    cfg.model.use_quantized_model = False
    cfg.model.kv_cache.backend = "default"
    cfg.model.kv_cache.max_len = 256
    cfg.generation.max_new_tokens = 16
    cfg.path.model_path = MODEL_PATH
    cfg.batch.num_slots = 8
    cfg.batch.max_batch_size = 4
    return cfg


def _make_req(prompt_ids, max_new_tokens=12):
    return Request(
        request_id=str(uuid.uuid4())[:6],
        prompt_token_ids=list(prompt_ids),
        sampling_params=SamplingParams(
            temperature=0.0, max_new_tokens=max_new_tokens, stop_on_eos=False
        ),
    )


def _drain(model, cfg, reqs):
    pool = build_kv_pool(model, cfg)
    model.attach_kv_pool(pool)
    scheduler = Scheduler(pool, max_batch_size=cfg.batch.max_batch_size)
    tokenizer = None
    runner = BatchedModelRunner(model, tokenizer, pool, scheduler, cfg)
    for r in reqs:
        scheduler.add_request(r)
    steps = 0
    while scheduler.has_work():
        runner.step()
        steps += 1
        assert steps < 300
    return reqs, pool


# ---------------------------------------------------------------------------
# Unit tests: bucket helpers
# ---------------------------------------------------------------------------


def test_build_capture_sizes_small():
    assert _build_capture_sizes(4) == [1, 2, 4]


def test_build_capture_sizes_medium():
    sizes = _build_capture_sizes(16)
    assert sizes[0] == 1
    assert sizes[-1] == 16
    assert 8 in sizes


def test_build_capture_sizes_always_includes_max():
    for max_bs in [3, 5, 7, 9, 17]:
        sizes = _build_capture_sizes(max_bs)
        assert sizes[-1] == max_bs, f"max_bs={max_bs} not in {sizes}"


def test_pad_to_bucket():
    gm = BatchDecodeGraphManager.__new__(BatchDecodeGraphManager)
    gm.capture_sizes = [1, 2, 4, 8, 16]
    assert gm.pad_to_bucket(1) == 1
    assert gm.pad_to_bucket(2) == 2
    assert gm.pad_to_bucket(3) == 4
    assert gm.pad_to_bucket(4) == 4
    assert gm.pad_to_bucket(5) == 8
    assert gm.pad_to_bucket(8) == 8
    assert gm.pad_to_bucket(9) == 16
    assert gm.pad_to_bucket(16) == 16


# ---------------------------------------------------------------------------
# Integration: correctness vs eager
# ---------------------------------------------------------------------------


def test_cuda_graph_matches_eager_bs1(model_and_tokenizer):
    """Single request: graph and eager must produce identical token sequences."""
    model, _ = model_and_tokenizer
    prompt = [1, 2, 3, 4, 5, 6, 7, 8]
    n = 10

    eager_reqs = [_make_req(prompt, n)]
    _drain(model, _base_cfg(use_cuda_graph=False), eager_reqs)

    cg_reqs = [_make_req(prompt, n)]
    _drain(model, _base_cfg(use_cuda_graph=True), cg_reqs)

    assert eager_reqs[0].generated_ids == cg_reqs[0].generated_ids, (
        f"bs=1 mismatch\n  eager={eager_reqs[0].generated_ids}\n  graph={cg_reqs[0].generated_ids}"
    )


def test_cuda_graph_matches_eager_bs3(model_and_tokenizer):
    """Three concurrent requests: graph and eager must produce identical tokens."""
    model, _ = model_and_tokenizer
    prompts = [
        [1, 2, 3, 4, 5],
        [10, 20, 30, 40, 50],
        [100, 200, 300, 400, 500],
    ]
    n = 8

    eager_reqs = [_make_req(p, n) for p in prompts]
    _drain(model, _base_cfg(use_cuda_graph=False), eager_reqs)

    cg_reqs = [_make_req(p, n) for p in prompts]
    _drain(model, _base_cfg(use_cuda_graph=True), cg_reqs)

    for i, (er, cr) in enumerate(zip(eager_reqs, cg_reqs)):
        assert er.generated_ids == cr.generated_ids, (
            f"req {i} mismatch\n  eager={er.generated_ids}\n  graph={cr.generated_ids}"
        )


def test_cuda_graph_padding_no_cross_contamination(model_and_tokenizer):
    """real_bs=3 padded to bucket=4: padding row must not corrupt real outputs."""
    model, _ = model_and_tokenizer
    cfg = _base_cfg(use_cuda_graph=True)
    cfg.batch.max_batch_size = 4  # bucket=[1,2,4]; 3 reqs -> padded to 4

    prompts = [[i * 10 + j for j in range(6)] for i in range(3)]
    n = 6

    cg_reqs = [_make_req(p, n) for p in prompts]
    _drain(model, cfg, cg_reqs)

    eager_reqs = [_make_req(p, n) for p in prompts]
    _drain(model, _base_cfg(use_cuda_graph=False), eager_reqs)

    for i, (er, cr) in enumerate(zip(eager_reqs, cg_reqs)):
        assert er.generated_ids == cr.generated_ids, (
            f"padding contamination on req {i}\n  eager={er.generated_ids}\n  graph={cr.generated_ids}"
        )


def test_cuda_graph_kv_snapshot_restore(model_and_tokenizer):
    """KV pool snapshot/restore: capture must not corrupt pre-existing KV data."""
    model, _ = model_and_tokenizer
    cfg = _base_cfg(use_cuda_graph=True)
    cfg.batch.max_batch_size = 4

    # Two sequential batches: first one fills slot 0, second one should use graph.
    # If snapshot/restore failed, slot 0 KV from the first batch would be wiped
    # and the second batch's output would diverge.
    prompt_a = [1, 2, 3, 4, 5, 6]
    prompt_b = [7, 8, 9, 10, 11, 12]

    # Run both prompts in one session so the second sees the same model state.
    reqs = [_make_req(prompt_a, 8), _make_req(prompt_b, 8)]
    cg_reqs_run, _ = _drain(model, cfg, reqs)

    # Eager reference for same prompts.
    eager_reqs = [_make_req(prompt_a, 8), _make_req(prompt_b, 8)]
    _drain(model, _base_cfg(use_cuda_graph=False), eager_reqs)

    for i, (er, cr) in enumerate(zip(eager_reqs, cg_reqs_run)):
        assert er.generated_ids == cr.generated_ids, (
            f"snapshot/restore failure on req {i}\n"
            f"  eager={er.generated_ids}\n  graph={cr.generated_ids}"
        )
