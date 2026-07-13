"""Unit tests for engine/sampler.py -- no GPU required (runs on CPU)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
import torch
from engine.sampler import Sampler


def logits(vocab=20, hot_idx=5, hot_val=100.0):
    """1×1×V logits with one dominant token."""
    l = torch.zeros(1, 1, vocab)
    l[0, 0, hot_idx] = hot_val
    return l


# ---------------------------------------------------------------------------
# Greedy
# ---------------------------------------------------------------------------

def test_greedy_picks_argmax():
    s = Sampler("greedy", 1.0)
    assert s.sample(logits(hot_idx=7))[0, 0].item() == 7


def test_greedy_same_input_same_output():
    s = Sampler("greedy", 1.0)
    l = logits(hot_idx=3)
    assert s.sample(l)[0, 0].item() == s.sample(l)[0, 0].item()


# ---------------------------------------------------------------------------
# Sampling: valid range, top-k, top-p
# ---------------------------------------------------------------------------

def test_sample_returns_valid_token():
    s = Sampler("sample", temperature=1.0)
    l = logits(vocab=50)
    token = s.sample(l)[0, 0].item()
    assert 0 <= token < 50


def test_topk_1_always_picks_argmax():
    """top_k=1 collapses distribution to the argmax; must be deterministic."""
    s = Sampler("sample", temperature=1.0, top_k=1)
    l = logits(hot_idx=9, hot_val=50.0)
    results = {s.sample(l)[0, 0].item() for _ in range(20)}
    assert results == {9}


def test_topp_near_zero_picks_argmax():
    """Nucleus so small only the argmax survives."""
    s = Sampler("sample", temperature=1.0, top_p=0.01)
    l = logits(hot_idx=4, hot_val=100.0)
    results = {s.sample(l)[0, 0].item() for _ in range(20)}
    assert results == {4}


def test_sample_with_seed_is_reproducible():
    s = Sampler("sample", temperature=2.0)
    l = logits(vocab=100)
    torch.manual_seed(0)
    t1 = s.sample(l)[0, 0].item()
    torch.manual_seed(0)
    t2 = s.sample(l)[0, 0].item()
    assert t1 == t2


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

def test_output_shape_single():
    s = Sampler("greedy", 1.0)
    assert s.sample(logits()).shape == (1, 1)


def test_output_shape_batch():
    """Batch of 3 sequences; each should pick its own argmax."""
    s = Sampler("greedy", 1.0)
    l = torch.zeros(3, 1, 20)
    for i in range(3):
        l[i, 0, i + 1] = 100.0
    out = s.sample(l)
    assert out.shape == (3, 1)
    for i in range(3):
        assert out[i, 0].item() == i + 1


# ---------------------------------------------------------------------------
# temperature=0 treated as greedy
# ---------------------------------------------------------------------------

def test_temperature_zero_is_greedy():
    s = Sampler("sample", temperature=0.0)
    l = logits(hot_idx=11)
    assert s.sample(l)[0, 0].item() == 11
