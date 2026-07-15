"""Unit tests for PagedKVPool slot/page management -- runs on CPU."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
import torch
from engine.kv_pool import PagedKVPool, PAGE_SIZE
from engine.context import set_context


NUM_SLOTS = 4
NUM_LAYERS = 2
NUM_KV_HEADS = 2
# Must be a multiple of PAGE_SIZE (256) for real FA2; use 256 for unit tests.
MAX_SEQ_LEN = PAGE_SIZE
HEAD_DIM = 8


def make_pool(**kwargs):
    defaults = dict(
        num_seqs=NUM_SLOTS, num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS, max_seq_len=MAX_SEQ_LEN,
        head_dim=HEAD_DIM, device="cpu", dtype=torch.float32,
    )
    defaults.update(kwargs)
    return PagedKVPool(**defaults)


# ---------------------------------------------------------------------------
# Slot lifecycle
# ---------------------------------------------------------------------------

def test_all_slots_free_initially():
    assert make_pool().num_free_slots() == NUM_SLOTS


def test_allocate_reduces_free_count():
    p = make_pool()
    p.allocate("r1")
    assert p.num_free_slots() == NUM_SLOTS - 1


def test_free_restores_slot():
    p = make_pool()
    slot = p.allocate("r1")
    p.free(slot)
    assert p.num_free_slots() == NUM_SLOTS


def test_allocate_all_then_raises():
    p = make_pool(num_seqs=2)
    p.allocate("r1"); p.allocate("r2")
    with pytest.raises(RuntimeError):
        p.allocate("r3")


def test_free_and_reallocate_same_slot():
    p = make_pool(num_seqs=1)
    slot = p.allocate("r1")
    p.free(slot)
    assert p.allocate("r2") == slot


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def test_reset_returns_all_slots():
    p = make_pool()
    p.allocate("r1"); p.allocate("r2")
    p.reset()
    assert p.num_free_slots() == NUM_SLOTS


def test_reset_preserves_tensor_capacity():
    """After reset the cache tensors still hold the original page pool."""
    p = make_pool()
    p.allocate("r1")
    p.reset()
    k = p.k_caches[0]
    # shape: (total_pages + 1, page_size, kv_heads, head_dim)
    expected_pages = p.total_pages + 1  # +1 dummy page
    assert k.shape[0] == expected_pages
    assert k.shape[1] == PAGE_SIZE
    assert k.shape[2] == NUM_KV_HEADS
    assert k.shape[3] == HEAD_DIM


# ---------------------------------------------------------------------------
# PagedKVLayer store / load correctness
# ---------------------------------------------------------------------------

def _set_ctx(is_prefill, slot_ids, cache_lens, pool=None):
    if pool is not None:
        bt = pool.block_table_for(slot_ids)
    else:
        bt = None
    set_context(
        is_prefill=is_prefill,
        cache_len=int(cache_lens[0].item()),
        slot_ids=slot_ids,
        cache_lens=cache_lens,
        block_tables=bt,
    )


def test_prefill_store_then_load_round_trip():
    p = make_pool(num_seqs=2, num_layers=1, num_kv_heads=2,
                  max_seq_len=PAGE_SIZE, head_dim=4)
    slot = p.allocate("r1")
    layer = p.get_layer_view(0)

    k = torch.randn(1, 2, 6, 4)   # (batch, kv_heads, seq_len, head_dim)
    v = torch.randn_like(k)
    slot_ids = torch.tensor([slot])
    cache_lens = torch.tensor([0])

    p.ensure_pages("r1", 0)
    _set_ctx(True, slot_ids, cache_lens, p)
    layer.store_kv(k, v, cache_len=0, is_prefill=True)

    _set_ctx(True, slot_ids, cache_lens, p)
    k_out, v_out = layer.load_kv_for_sdpa(total_len=6)

    assert k_out.shape[-2] == 6
    assert k_out.shape[-1] == 4
    assert k_out.dtype == k.dtype


def test_decode_store_extends_cache():
    """After prefill + one decode step, load_kv_for_sdpa must return seq_len+1 tokens."""
    p = make_pool(num_seqs=2, num_layers=1, num_kv_heads=2,
                  max_seq_len=PAGE_SIZE, head_dim=4)
    slot = p.allocate("r1")
    layer = p.get_layer_view(0)
    slot_ids = torch.tensor([slot])

    # Prefill 4 tokens
    k_pf = torch.randn(1, 2, 4, 4)
    p.ensure_pages("r1", 0)
    _set_ctx(True, slot_ids, torch.tensor([0]), p)
    layer.store_kv(k_pf, torch.randn_like(k_pf), cache_len=0, is_prefill=True)

    # Decode step (cache_len = 4)
    k_dec = torch.randn(1, 2, 1, 4)
    p.ensure_pages("r1", 4)
    _set_ctx(False, slot_ids, torch.tensor([4]), p)
    layer.store_kv(k_dec, torch.randn_like(k_dec), cache_len=4, is_prefill=False)

    _set_ctx(False, slot_ids, torch.tensor([4]), p)
    k_out, _ = layer.load_kv_for_sdpa(total_len=5)
    assert k_out.shape[-2] == 5


def test_multi_slot_isolation():
    """Data written to slot A must not appear when reading slot B."""
    p = make_pool(num_seqs=4, num_layers=1, num_kv_heads=1,
                  max_seq_len=PAGE_SIZE, head_dim=4)
    slot_a = p.allocate("a")
    slot_b = p.allocate("b")
    layer = p.get_layer_view(0)

    k_a = torch.ones(1, 1, 2, 4) * 7.0
    p.ensure_pages("a", 0)
    _set_ctx(True, torch.tensor([slot_a]), torch.tensor([0]), p)
    layer.store_kv(k_a, torch.zeros_like(k_a), cache_len=0, is_prefill=True)

    # Slot B has no pages and has never been written.
    _set_ctx(True, torch.tensor([slot_b]), torch.tensor([0]), p)
    k_b_out, _ = layer.load_kv_for_sdpa(total_len=2)

    # Slot B reads from dummy_page which is initialized to zeros.
    assert not torch.any(k_b_out == 7.0)
