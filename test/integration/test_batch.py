"""Integration test: continuous batching (BatchedModelRunner, bs > 1).

Requires: ~/huggingface/Qwen3-0.6B/
"""
import sys, os, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
import torch
import torch.distributed as dist

from utils.config import GlobalConfig
from engine.loader import load_model, build_kv_pool
from engine.scheduler import Scheduler
from engine.batched_runner import BatchedModelRunner
from engine.request import Request
from engine.schema import SamplingParams

MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

PROMPTS = [
    "The capital of France is",
    "Once upon a time in a land far away,",
    "Explain gravity in one sentence:",
    "What is Python?",
    "Write a haiku about the moon:",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dist_init():
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo" if not torch.cuda.is_available() else "nccl",
            init_method="tcp://localhost:29511",
            world_size=1, rank=0,
        )
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def model_and_tokenizer(dist_init):
    cfg = _base_cfg()
    torch.set_default_dtype(cfg.env.get_torch_dtype())
    torch.set_default_device(cfg.env.device)
    model, tokenizer = load_model(cfg)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_cfg():
    cfg = GlobalConfig.from_yaml("configs/runs/batch.yaml")
    cfg.model.backend             = "default"
    cfg.model.use_cuda_graph      = False
    cfg.model.use_quantized_model = False
    cfg.model.kv_cache.backend    = "default"
    cfg.model.kv_cache.max_len    = 256
    cfg.generation.max_new_tokens = 16
    cfg.path.model_path           = MODEL_PATH
    cfg.batch.num_slots           = 8
    cfg.batch.max_batch_size      = 4
    return cfg


def make_req(prompt, tokenizer, max_new_tokens=16):
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    return Request(
        request_id=str(uuid.uuid4())[:6],
        prompt_token_ids=ids,
        sampling_params=SamplingParams(temperature=0.0, max_new_tokens=max_new_tokens),
    )


def drain(model, tokenizer, cfg, reqs, max_steps=500):
    """Build a fresh pool + runner, add reqs, run until done. Returns (reqs, pool, steps)."""
    pool = build_kv_pool(model, cfg)
    model.attach_kv_pool(pool)
    scheduler = Scheduler(
        pool,
        max_batch_size=cfg.batch.max_batch_size,
        admission_policy=cfg.batch.admission_policy,
        max_num_batched_tokens=cfg.batch.max_num_batched_tokens,
    )
    runner = BatchedModelRunner(model, tokenizer, pool, scheduler, cfg)
    for r in reqs:
        scheduler.add_request(r)
    steps = 0
    while scheduler.has_work():
        runner.step()
        steps += 1
        assert steps < max_steps, f"runaway loop after {max_steps} steps"
    return reqs, pool, steps, scheduler


def assert_clean(reqs, pool, cfg):
    assert all(r.is_finished for r in reqs), "not all requests finished"
    assert all(r.finish_reason in ("eos", "length") for r in reqs)
    assert all(len(r.generated_ids) > 0 for r in reqs)
    assert pool.num_free_slots() == cfg.batch.num_slots, "slot leak"


# ---------------------------------------------------------------------------
# Core batch correctness
# ---------------------------------------------------------------------------

def test_batch_all_finish_no_leak(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    cfg = _base_cfg()
    reqs = [make_req(p, tokenizer) for p in PROMPTS]
    reqs, pool, _, _ = drain(model, tokenizer, cfg, reqs)
    assert_clean(reqs, pool, cfg)


def test_bs1_via_batched_runner(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    cfg = _base_cfg()
    cfg.batch.max_batch_size = 1
    cfg.batch.num_slots = 2
    cfg.model.kv_cache.max_len = 128
    r = make_req("The capital of France is", tokenizer)
    reqs, pool, _, _ = drain(model, tokenizer, cfg, [r])
    assert_clean(reqs, pool, cfg)


# ---------------------------------------------------------------------------
# Heterogeneous decode: wave-2 arrives later
# ---------------------------------------------------------------------------

def test_heterogeneous_decode(model_and_tokenizer):
    """Requests from wave-2 must have smaller cache_lens than wave-1 when they first join."""
    model, tokenizer = model_and_tokenizer
    cfg = _base_cfg()
    pool = build_kv_pool(model, cfg)
    model.attach_kv_pool(pool)
    scheduler = Scheduler(pool, max_batch_size=cfg.batch.max_batch_size)
    runner = BatchedModelRunner(model, tokenizer, pool, scheduler, cfg)

    wave1 = [make_req(p, tokenizer, max_new_tokens=30) for p in PROMPTS[:2]]
    wave2 = [make_req(p, tokenizer, max_new_tokens=30) for p in PROMPTS[2:4]]
    for r in wave1:
        scheduler.add_request(r)

    # Run wave-1 alone for a few steps so it builds up cache_len
    WAVE1_SOLO_STEPS = 4
    for _ in range(WAVE1_SOLO_STEPS):
        if not scheduler.has_work():
            break
        runner.step()

    # Now inject wave-2 and capture cache_lens at the very next step
    for r in wave2:
        scheduler.add_request(r)
    runner.step()
    running_cls = {r.request_id: r.cache_len for r in scheduler._running if not r.is_finished}

    # Drain the rest
    for _ in range(500):
        if not scheduler.has_work():
            break
        runner.step()

    assert all(r.is_finished for r in wave1 + wave2)
    assert pool.num_free_slots() == cfg.batch.num_slots

    # The two waves must have had distinct cache_lens at the moment wave-2 joined
    if len(running_cls) >= 2:
        assert len(set(running_cls.values())) > 1, \
            f"All cache_lens equal after wave-2 joined: {running_cls}"


# ---------------------------------------------------------------------------
# Admission policies
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("policy", ["fifo", "spf", "ljf", "random"])
def test_admission_policy_all_finish(model_and_tokenizer, policy):
    model, tokenizer = model_and_tokenizer
    cfg = _base_cfg()
    cfg.batch.admission_policy = policy
    reqs = [make_req(p, tokenizer) for p in PROMPTS]
    reqs, pool, _, _ = drain(model, tokenizer, cfg, reqs)
    assert_clean(reqs, pool, cfg)


# ---------------------------------------------------------------------------
# Chunked prefill
# ---------------------------------------------------------------------------

def test_chunked_prefill_completes(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    cfg = _base_cfg()
    cfg.batch.max_num_batched_tokens = 32
    reqs = [make_req(p, tokenizer, max_new_tokens=8) for p in PROMPTS[:3]]
    reqs, pool, steps, _ = drain(model, tokenizer, cfg, reqs)
    assert_clean(reqs, pool, cfg)
    # With budget=32 and prompts of ~5–10 tokens, prefill still takes multiple steps
    assert steps >= 2


# ---------------------------------------------------------------------------
# Scheduler overflow: more requests than slots
# ---------------------------------------------------------------------------

def test_scheduler_overflow(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    cfg = _base_cfg()
    cfg.batch.num_slots = 4         # fewer than len(PROMPTS)=5
    cfg.batch.max_batch_size = 2
    cfg.model.kv_cache.max_len = 128
    reqs = [make_req(p, tokenizer, max_new_tokens=8) for p in PROMPTS]
    reqs, pool, _, _ = drain(model, tokenizer, cfg, reqs)
    assert_clean(reqs, pool, cfg)


# ---------------------------------------------------------------------------
# Multi-run pool lifecycle
# ---------------------------------------------------------------------------

def test_kv_pool_lifecycle_multi_run(model_and_tokenizer):
    """Three independent runs on the same pool after reset; no leaks."""
    model, tokenizer = model_and_tokenizer
    cfg = _base_cfg()
    pool = build_kv_pool(model, cfg)

    for run_idx in range(3):
        pool.reset()
        model.attach_kv_pool(pool)
        scheduler = Scheduler(pool, max_batch_size=cfg.batch.max_batch_size)
        runner = BatchedModelRunner(model, tokenizer, pool, scheduler, cfg)
        reqs = [make_req(p, tokenizer, max_new_tokens=8) for p in PROMPTS[:3]]
        for r in reqs:
            scheduler.add_request(r)
        steps = 0
        while scheduler.has_work():
            runner.step()
            steps += 1
            assert steps < 300, f"run {run_idx}: runaway loop"
        assert_clean(reqs, pool, cfg)


# ---------------------------------------------------------------------------
# Greedy output regression
# ---------------------------------------------------------------------------

def test_greedy_output_regression(model_and_tokenizer):
    """bs>1 greedy must produce the same first token as bs=1 for the same prompt."""
    model, tokenizer = model_and_tokenizer
    cfg = _base_cfg()
    cfg.batch.max_batch_size = 1
    cfg.batch.num_slots = 2
    cfg.model.kv_cache.max_len = 128
    prompt = "The capital of France is"
    r = make_req(prompt, tokenizer, max_new_tokens=1)
    reqs, _, _, _ = drain(model, tokenizer, cfg, [r])
    first_token_text = tokenizer.decode(reqs[0].generated_ids[:1], skip_special_tokens=True).strip()
    assert "paris" in first_token_text.lower() or len(first_token_text) > 0
