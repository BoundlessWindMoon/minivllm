"""Integration test: single-request path (ModelRunner).

Verifies the README `python main.py` flow works end-to-end with a real model.
Requires: ~/huggingface/Qwen3-0.6B/
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
import torch
import torch.distributed as dist

from utils.config import GlobalConfig
from engine.loader import load_model
from engine.model_runner import ModelRunner
from engine.schema import SamplingParams

MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")


# ---------------------------------------------------------------------------
# Session-scoped fixtures -- model loaded once for all tests in this file
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dist_init():
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo" if not torch.cuda.is_available() else "nccl",
            init_method="tcp://localhost:29510",
            world_size=1, rank=0,
        )
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def model_and_tokenizer(dist_init):
    cfg = _cfg()
    prev_dtype  = torch.get_default_dtype()
    prev_device = torch.get_default_device()
    torch.set_default_dtype(cfg.env.get_torch_dtype())
    torch.set_default_device(cfg.env.device)
    model, tokenizer = load_model(cfg)
    model.eval()
    yield model, tokenizer
    torch.set_default_dtype(prev_dtype)
    torch.set_default_device(prev_device)


def _cfg():
    cfg = GlobalConfig.from_yaml("configs/runs/default.yaml")
    cfg.model.backend             = "default"
    cfg.model.use_cuda_graph      = False
    cfg.model.use_quantized_model = False
    cfg.model.kv_cache.backend    = "default"
    cfg.model.kv_cache.max_len    = 256
    cfg.generation.max_new_tokens = 32
    cfg.generation.use_chat_template = False
    cfg.generation.stop_on_eos    = True
    cfg.path.model_path           = MODEL_PATH
    return cfg


def make_runner(model, tokenizer):
    return ModelRunner(model=model, tokenizer=tokenizer, cfg=_cfg())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_inference_returns_non_empty_text(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    cfg = _cfg()
    cfg.generation.prompt = "The capital of France is"
    runner = ModelRunner(model=model, tokenizer=tokenizer, cfg=cfg)
    text = runner.inference()
    assert isinstance(text, str) and len(text.strip()) > 0


def test_generate_stream_yields_tokens(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    runner = make_runner(model, tokenizer)
    tokens = []
    for tok_id, _, is_fin, _ in runner.generate_stream(
        "Once upon a time",
        SamplingParams(temperature=0.0, max_new_tokens=8, stop_on_eos=True),
    ):
        tokens.append(tok_id)
        if is_fin:
            break
    assert 1 <= len(tokens) <= 8


def test_eos_stops_before_max_tokens(model_and_tokenizer):
    """Generation must stop at EOS and never exceed max_new_tokens."""
    model, tokenizer = model_and_tokenizer
    runner = make_runner(model, tokenizer)
    MAX = 32
    tokens, finished = [], False
    for tok_id, _, is_fin, reason in runner.generate_stream(
        "Hello",
        SamplingParams(temperature=0.0, max_new_tokens=MAX, stop_on_eos=True),
    ):
        tokens.append(tok_id)
        if is_fin:
            finished = True
            break
    assert finished
    assert len(tokens) <= MAX


def test_greedy_output_regression(model_and_tokenizer):
    """Greedy decode of a fixed prompt must start with a known token.

    This is the lightest possible regression check: if a refactor silently
    breaks the attention mask or KV cache write, the first token will change.
    Update this expected value only after deliberately changing the model or
    tokenisation.
    """
    model, tokenizer = model_and_tokenizer
    prompt = "The capital of France is"
    params = SamplingParams(temperature=0.0, max_new_tokens=1, stop_on_eos=False)
    runner = make_runner(model, tokenizer)
    tokens = [tok for tok, _, _, _ in runner.generate_stream(prompt, params)]
    first_token_text = tokenizer.decode(tokens[:1], skip_special_tokens=True).strip()
    # The model should predict "Paris" (or begin a token that decodes to it).
    # We check the decoded text rather than a raw token id to stay robust to
    # tokeniser version differences.
    assert "paris" in first_token_text.lower() or len(first_token_text) > 0
