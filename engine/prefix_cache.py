"""Block-level prefix cache for KV reuse across requests.

Design
------
Granularity: one cache entry = one physical page worth of tokens.
A block is identified by the hash of its token_ids AND the hash of the
preceding block (chain hash), so two sequences only share a block if
every token up to that point is identical — same as vLLM's design.

Lifecycle
---------
1. lookup(token_ids, page_size)
   Returns (matched_pages: list[int], matched_len: int).
   matched_len is always a multiple of page_size (only full pages are cached).
   Matched pages have their ref_count incremented; caller owns a reference.

2. insert(token_ids, pages, page_size)
   Registers newly computed full pages into the cache.
   Partial (tail) pages are ignored — they may still be in flight.
   If the cache is full, LRU pages with ref_count == 0 are evicted first.

3. release(pages)
   Decrements ref_count for each page. Pages with ref_count == 0 become
   eviction candidates (they stay in the cache until capacity forces eviction).

Concurrency
-----------
All operations are called from the scheduler (single thread), so no locking
is needed.
"""

from __future__ import annotations

from collections import OrderedDict


class PrefixCache:

    def __init__(self, max_pages: int) -> None:
        """
        max_pages: maximum number of physical pages the cache may hold
                   (typically = PagedKVPool.total_pages, but can be smaller).
        """
        self._max_pages = max_pages
        # block_hash -> phys_page_id  (LRU order: MRU at end)
        self._cache: OrderedDict[int, int] = OrderedDict()
        # phys_page_id -> ref_count
        self._ref: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, token_ids: list[int], page_size: int) -> tuple[list[int], int]:
        """Return (matched_pages, matched_len) for the longest cached prefix.

        Only contiguous matches from the start are returned — as soon as one
        block is not in the cache the search stops (prefix semantics).
        matched_len is always a multiple of page_size.
        """
        matched: list[int] = []
        prev_hash = 0  # chain seed
        n_full = len(token_ids) // page_size

        for i in range(n_full):
            block_tokens = token_ids[i * page_size: (i + 1) * page_size]
            bh = self._block_hash(block_tokens, prev_hash)
            if bh not in self._cache:
                break
            phys = self._cache[bh]
            # Move to MRU position
            self._cache.move_to_end(bh)
            self._ref[phys] = self._ref.get(phys, 0) + 1
            matched.append(phys)
            prev_hash = bh

        return matched, len(matched) * page_size

    def insert(self, token_ids: list[int], pages: list[int], page_size: int) -> None:
        """Register full pages produced for token_ids into the cache.

        pages[i] is the physical page id for token_ids[i*page_size:(i+1)*page_size].
        The tail (len(token_ids) % page_size tokens) is intentionally ignored.
        Pages already in the cache are skipped (another request beat us to it).
        """
        prev_hash = 0
        n_full = min(len(token_ids) // page_size, len(pages))

        for i in range(n_full):
            block_tokens = token_ids[i * page_size: (i + 1) * page_size]
            bh = self._block_hash(block_tokens, prev_hash)
            prev_hash = bh
            if bh in self._cache:
                continue  # already cached by a concurrent/earlier request
            if not self._maybe_evict():
                break     # all pages in use; skip remaining inserts
            self._cache[bh] = pages[i]
            # ref_count starts at 0: no one holds a reference yet
            if pages[i] not in self._ref:
                self._ref[pages[i]] = 0

    def release(self, pages: list[int]) -> None:
        """Decrement ref_count for each page. Call when a request finishes."""
        for phys in pages:
            if phys in self._ref:
                self._ref[phys] = max(0, self._ref[phys] - 1)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def num_cached_pages(self) -> int:
        return len(self._cache)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _block_hash(block_tokens: list[int], prev_hash: int) -> int:
        return hash((prev_hash, tuple(block_tokens)))

    def _maybe_evict(self) -> bool:
        """Evict the LRU page with ref_count == 0 if at capacity.

        Returns True if there is space for a new entry (either we evicted one,
        or the cache was already below capacity). Returns False if all pages
        are referenced and cannot be evicted.
        """
        if len(self._cache) < self._max_pages:
            return True
        # Walk from LRU end, find first evictable entry.
        for bh, phys in list(self._cache.items()):
            if self._ref.get(phys, 0) == 0:
                del self._cache[bh]
                self._ref.pop(phys, None)
                return True
        # All pages are referenced — cannot evict.
        return False
