"""Unit tests for KiviKVCacheBackend -- no GPU required.

Validates store/load round-trips and the residual-window logic.
Numerical precision is relaxed because KIVI is lossy; we check that
dequantised values are close to the originals within a tolerance,
and that structural invariants (total_len, residual boundaries) hold.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
import torch
from layers.kv_cache import KiviKVCacheBackend


BATCH = 1
KV_HEADS = 2
HEAD_DIM = 32   # must be divisible by group_size
GROUP_SIZE = 16
RESIDUAL = 16
MAX_SEQ = 128


def make_kivi(k_bits=4, v_bits=4, residual=RESIDUAL):
    return KiviKVCacheBackend(
        batch_size=BATCH, num_kv_heads=KV_HEADS,
        max_seq_len=MAX_SEQ, head_dim=HEAD_DIM,
        k_bits=k_bits, v_bits=v_bits,
        group_size=GROUP_SIZE, residual_length=residual,
        device="cpu", dtype=torch.float16,
    )


def rand_kv(seq_len):
    k = torch.randn(BATCH, KV_HEADS, seq_len, HEAD_DIM).half()
    v = torch.randn_like(k)
    return k, v


def store(be, k, v, cache_len=0, is_prefill=True):
    """Wrapper: KiviKVCacheBackend.store_kv uses a positional _cache_len param."""
    be.store_kv(k, v, cache_len, is_prefill)


# ---------------------------------------------------------------------------
# Prefill: short prompt (≤ residual) stays in full-precision
# ---------------------------------------------------------------------------

def test_short_prefill_no_quantisation():
    be = make_kivi()
    k, v = rand_kv(RESIDUAL // 2)          # half the residual window
    store(be, k, v, cache_len=0, is_prefill=True)
    assert be._k_quant_len == 0             # nothing quantised yet
    assert be._k_full_len == RESIDUAL // 2


# ---------------------------------------------------------------------------
# Prefill: long prompt (> residual) quantises older tokens
# ---------------------------------------------------------------------------

def test_long_prefill_quantises_prefix():
    be = make_kivi()
    seq = RESIDUAL * 2                       # clearly above residual
    k, v = rand_kv(seq)
    store(be, k, v, cache_len=0, is_prefill=True)
    assert be._k_quant_len > 0              # something was quantised
    assert be._k_full_len <= RESIDUAL       # residual window respected


# ---------------------------------------------------------------------------
# Round-trip: load_kv reconstructs the right number of tokens
# ---------------------------------------------------------------------------

def test_prefill_load_kv_total_len():
    be = make_kivi()
    seq = RESIDUAL + GROUP_SIZE             # triggers quantisation
    k, v = rand_kv(seq)
    store(be, k, v, cache_len=0, is_prefill=True)
    k_out, v_out = be.load_kv(total_len=seq)
    assert k_out.shape == (BATCH, KV_HEADS, seq, HEAD_DIM)
    assert v_out.shape == (BATCH, KV_HEADS, seq, HEAD_DIM)


# ---------------------------------------------------------------------------
# Decode: residual window slides correctly
# ---------------------------------------------------------------------------

def test_decode_fills_residual_then_quantises():
    """K flush triggers when _k_full_len reaches residual_length exactly.

    group_size must divide residual_length for quantisation to actually fire.
    With residual=GROUP_SIZE*2: prefill GROUP_SIZE tokens (full_len=GROUP_SIZE),
    then decode GROUP_SIZE more steps fills to residual, triggering flush.
    """
    residual = GROUP_SIZE * 2
    be = make_kivi(residual=residual)
    k_pf, v_pf = rand_kv(GROUP_SIZE)
    store(be, k_pf, v_pf, cache_len=0, is_prefill=True)
    assert be._k_quant_len == 0

    for i in range(GROUP_SIZE):          # fills exactly to residual
        k_dec, v_dec = rand_kv(1)
        store(be, k_dec, v_dec, cache_len=GROUP_SIZE + i, is_prefill=False)

    # Flush must have happened: quant_len > 0, full_len reset
    assert be._k_quant_len > 0
    assert be._k_full_len == 0


def test_decode_load_kv_grows_by_one_per_step():
    be = make_kivi(residual=64)   # big residual so nothing is quantised
    k_pf, v_pf = rand_kv(4)
    store(be, k_pf, v_pf, cache_len=0, is_prefill=True)

    for step in range(3):
        k_dec, v_dec = rand_kv(1)
        store(be, k_dec, v_dec, cache_len=4 + step, is_prefill=False)
        expected = 4 + step + 1
        k_out, v_out = be.load_kv(total_len=expected)
        assert k_out.shape[2] == expected


# ---------------------------------------------------------------------------
# Reset clears state
# ---------------------------------------------------------------------------

def test_reset_clears_all_lengths():
    be = make_kivi()
    k, v = rand_kv(RESIDUAL * 3)
    store(be, k, v, cache_len=0, is_prefill=True)
    assert be._total_len > 0
    be.reset()
    assert be._total_len == 0
    assert be._k_quant_len == 0
    assert be._v_quant_len == 0
    assert be._k_full_len == 0
    assert be._v_full_len == 0


# ---------------------------------------------------------------------------
# Quantisation precision: dequantised values within tolerable range
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bits", [2, 4])
def test_dequant_within_tolerance(bits):
    """Dequantised K must be close to the original within a generous tolerance."""
    be = make_kivi(k_bits=bits, v_bits=bits, residual=GROUP_SIZE)
    seq = GROUP_SIZE * 3   # enough tokens to trigger quantisation
    k, v = rand_kv(seq)
    store(be, k, v, cache_len=0, is_prefill=True)
    k_out, _ = be.load_kv(total_len=seq)
    # 2-bit is highly lossy; 4-bit is moderate. Tolerances reflect expected KIVI accuracy.
    tol = 0.40 if bits == 2 else 0.15
    rel_err = (k_out.float() - k.float()).abs().mean() / (k.float().abs().mean() + 1e-6)
    assert rel_err < tol, f"{bits}-bit round-trip relative error too large: {rel_err:.3f}"
