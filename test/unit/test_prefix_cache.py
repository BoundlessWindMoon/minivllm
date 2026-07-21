"""Unit tests for engine/prefix_cache.py -- no GPU, no model."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
from engine.prefix_cache import PrefixCache

PAGE = 4   # tiny page size for tests (not bound by FA2 constraint here)


def make_tokens(n, start=0):
    return list(range(start, start + n))


# ---------------------------------------------------------------------------
# Basic lookup / insert
# ---------------------------------------------------------------------------

def test_empty_cache_returns_no_match():
    pc = PrefixCache(max_pages=16)
    pages, matched = pc.lookup(make_tokens(PAGE * 3), PAGE)
    assert pages == []
    assert matched == 0


def test_insert_then_lookup_hits():
    pc = PrefixCache(max_pages=16)
    tokens = make_tokens(PAGE * 2)
    phys = [10, 11]
    pc.insert(tokens, phys, PAGE)
    pages, matched = pc.lookup(tokens, PAGE)
    pc.release(pages)
    assert matched == PAGE * 2
    assert pages == phys


def test_partial_match_stops_at_first_miss():
    """Cache has pages 0 and 1; page 2 is missing → only first two match."""
    pc = PrefixCache(max_pages=16)
    tokens = make_tokens(PAGE * 3)
    pc.insert(tokens[:PAGE * 2], [10, 11], PAGE)
    # Lookup full sequence: only the first two pages are cached.
    pages, matched = pc.lookup(tokens, PAGE)
    pc.release(pages)
    assert matched == PAGE * 2
    assert pages == [10, 11]


def test_tail_page_not_cached():
    """Insert with a tail < PAGE tokens — that tail must NOT be cached."""
    pc = PrefixCache(max_pages=16)
    # 2.5 pages worth of tokens; only 2 full pages should be inserted.
    tokens = make_tokens(int(PAGE * 2.5))
    pc.insert(tokens, [10, 11, 12], PAGE)
    pages, matched = pc.lookup(tokens, PAGE)
    pc.release(pages)
    assert matched == PAGE * 2
    assert len(pages) == 2


def test_lookup_with_no_full_page_returns_empty():
    pc = PrefixCache(max_pages=16)
    pc.insert(make_tokens(PAGE - 1), [99], PAGE)
    pages, matched = pc.lookup(make_tokens(PAGE - 1), PAGE)
    pc.release(pages)
    assert matched == 0
    assert pages == []


# ---------------------------------------------------------------------------
# Chain hash: different prefixes → different blocks
# ---------------------------------------------------------------------------

def test_same_block_different_prefix_no_collision():
    """Same block tokens but different preceding context must not collide."""
    pc = PrefixCache(max_pages=16)
    block = make_tokens(PAGE)

    # Sequence A: [0..PAGE) + block
    seq_a = make_tokens(PAGE) + block
    # Sequence B: [100..100+PAGE) + block  (different prefix)
    seq_b = make_tokens(PAGE, start=100) + block

    pc.insert(seq_a, [10, 20], PAGE)
    pc.insert(seq_b, [30, 40], PAGE)

    pages_a, matched_a = pc.lookup(seq_a, PAGE)
    pc.release(pages_a)
    pages_b, matched_b = pc.lookup(seq_b, PAGE)
    pc.release(pages_b)

    assert matched_a == PAGE * 2
    assert matched_b == PAGE * 2
    assert pages_a == [10, 20]
    assert pages_b == [30, 40]


# ---------------------------------------------------------------------------
# Reference counting
# ---------------------------------------------------------------------------

def test_ref_count_incremented_on_lookup():
    pc = PrefixCache(max_pages=16)
    tokens = make_tokens(PAGE)
    pc.insert(tokens, [5], PAGE)
    pages, _ = pc.lookup(tokens, PAGE)
    assert pc._ref[5] == 1
    pc.release(pages)
    assert pc._ref.get(5, 0) == 0


def test_multiple_lookups_accumulate_refs():
    pc = PrefixCache(max_pages=16)
    tokens = make_tokens(PAGE)
    pc.insert(tokens, [5], PAGE)
    p1, _ = pc.lookup(tokens, PAGE)
    p2, _ = pc.lookup(tokens, PAGE)
    assert pc._ref[5] == 2
    pc.release(p1)
    assert pc._ref[5] == 1
    pc.release(p2)
    assert pc._ref.get(5, 0) == 0


def test_release_does_not_go_below_zero():
    pc = PrefixCache(max_pages=16)
    pc.release([42])   # no ref for page 42 — should not crash


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------

def test_lru_evicts_least_recently_used():
    """Cache capacity=2; inserting a third entry evicts the LRU page."""
    pc = PrefixCache(max_pages=2)
    t0 = make_tokens(PAGE, start=0)
    t1 = make_tokens(PAGE, start=PAGE)
    t2 = make_tokens(PAGE, start=PAGE * 2)
    pc.insert(t0, [10], PAGE)
    pc.insert(t1, [11], PAGE)
    # Access t0 to make it MRU.
    p, _ = pc.lookup(t0, PAGE)
    pc.release(p)
    # Insert t2: t1 (LRU, ref=0) should be evicted.
    pc.insert(t2, [12], PAGE)
    assert pc.num_cached_pages == 2
    _, m1 = pc.lookup(t1, PAGE)  # should miss
    _, m2 = pc.lookup(t2, PAGE)  # should hit
    pc.release([])
    assert m1 == 0
    assert m2 == PAGE


def test_lru_does_not_evict_referenced_page():
    """If all pages are referenced, eviction is skipped and insert is a no-op."""
    pc = PrefixCache(max_pages=1)
    tokens = make_tokens(PAGE)
    pc.insert(tokens, [7], PAGE)
    held, _ = pc.lookup(tokens, PAGE)   # ref_count = 1
    # Try to insert another entry — cache is full and only page is in use.
    pc.insert(make_tokens(PAGE, start=PAGE), [8], PAGE)
    assert pc.num_cached_pages == 1     # no eviction happened
    pc.release(held)


# ---------------------------------------------------------------------------
# Insert idempotency
# ---------------------------------------------------------------------------

def test_double_insert_does_not_duplicate():
    pc = PrefixCache(max_pages=16)
    tokens = make_tokens(PAGE)
    pc.insert(tokens, [10], PAGE)
    pc.insert(tokens, [10], PAGE)   # second insert of same tokens
    assert pc.num_cached_pages == 1


# ---------------------------------------------------------------------------
# Integration: lookup → skip prefix → insert on finish
# ---------------------------------------------------------------------------

def test_second_request_hits_first_requests_pages():
    """Simulate: req1 finishes → pages cached → req2 lookup hits."""
    pc = PrefixCache(max_pages=16)
    system_prompt = make_tokens(PAGE * 2)   # 2 full pages

    # req1 finishes: insert its pages.
    phys_req1 = [50, 51]
    pc.insert(system_prompt, phys_req1, PAGE)

    # req2 arrives with same system prompt + extra user tokens.
    user_tokens = make_tokens(PAGE, start=1000)
    req2_tokens = system_prompt + user_tokens
    pages, matched = pc.lookup(req2_tokens, PAGE)
    pc.release(pages)

    assert matched == PAGE * 2
    assert pages == phys_req1


# ---------------------------------------------------------------------------
# Full-cache silent skip: skipped entry must not appear in cache
# ---------------------------------------------------------------------------

def test_full_cache_skipped_entry_not_cached():
    """When cache is full and all pages are referenced, skipped entries must not appear."""
    pc = PrefixCache(max_pages=1)
    tokens_a = make_tokens(PAGE, start=0)
    tokens_b = make_tokens(PAGE, start=PAGE)

    pc.insert(tokens_a, [7], PAGE)
    held, _ = pc.lookup(tokens_a, PAGE)   # ref_count = 1, cannot evict

    pc.insert(tokens_b, [8], PAGE)        # should be silently skipped

    # tokens_b must NOT be in cache.
    pages_b, matched_b = pc.lookup(tokens_b, PAGE)
    pc.release(pages_b)
    assert matched_b == 0

    pc.release(held)


# ---------------------------------------------------------------------------
# PagedKVPool: shared pages are not returned to free pool on free()
# ---------------------------------------------------------------------------

def test_paged_kv_pool_shared_pages_not_freed():
    """Pages borrowed from prefix cache (prefix_pages) must not go back to _free_pages."""
    import torch
    from engine.kv_pool import PagedKVPool, PAGE_SIZE

    pool = PagedKVPool(
        num_seqs=4, num_layers=1, num_kv_heads=1,
        max_seq_len=PAGE_SIZE, head_dim=4,
        device="cpu", dtype=torch.float32,
    )

    # Allocate slot A independently to get a real physical page.
    slot_a = pool.allocate("req_a")
    pool.ensure_pages("req_a", 0)
    shared_page = pool._block_tables["req_a"][0]

    # Simulate prefix cache: slot B borrows that page.
    slot_b = pool.allocate("req_b", prefix_pages=[shared_page])

    free_before = len(pool._free_pages)
    pool.free(slot_b)
    free_after = len(pool._free_pages)

    # Freeing slot B must NOT return the shared page to the free pool.
    assert free_after == free_before
    assert shared_page not in pool._free_pages

    # Clean up slot A normally.
    pool.free(slot_a)


def test_paged_kv_pool_owned_pages_freed_correctly():
    """Exclusively-owned pages (not shared) must be returned to _free_pages on free()."""
    import torch
    from engine.kv_pool import PagedKVPool, PAGE_SIZE

    pool = PagedKVPool(
        num_seqs=4, num_layers=1, num_kv_heads=1,
        max_seq_len=PAGE_SIZE, head_dim=4,
        device="cpu", dtype=torch.float32,
    )

    slot = pool.allocate("req")
    pool.ensure_pages("req", 0)
    owned_page = pool._block_tables["req"][0]
    free_before = len(pool._free_pages)

    pool.free(slot)

    assert len(pool._free_pages) == free_before + 1
    assert owned_page in pool._free_pages

