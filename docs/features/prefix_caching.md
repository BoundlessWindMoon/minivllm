# Prefix Caching

<!-- code-ref: engine/prefix_cache.py -->
<!-- code-ref: engine/scheduler.py -->
<!-- code-ref: engine/kv_pool.py -->
Prefix caching reuses KV cache blocks across requests that share a common prefix,
skipping the prefill computation for the shared portion.

## When it helps

- **Shared system prompts** — multiple users hitting the same system prompt pay prefill cost once.
- **Multi-turn chat** — growing conversation history is only prefilled incrementally.
- **Document QA** — the same long document queried with different questions is processed once.

Prefix caching only reduces prefill latency. If your workload is decode-bound (long output,
short input), the benefit is small.

## Enabling

Pass a `PrefixCache` to the `Scheduler`:

```python
from engine.prefix_cache import PrefixCache
from engine.scheduler import Scheduler

prefix_cache = PrefixCache(max_pages=pool.total_pages)
scheduler = Scheduler(pool, max_batch_size=8, prefix_cache=prefix_cache)
```

The HTTP server enables it automatically when started via `start_server.py`.

## Benchmarking

Use `--shared-prefix-len` to simulate a shared system prompt. The value must produce
at least one full KV page after tokenization (page_size = 256 tokens); use 512+ to be safe.

```bash
# Run twice: first populates the cache, second hits it
python scripts/serve/bench_serving.py \
  --mode fixed --concurrency 4 --num-requests 20 \
  --input-len 64 --shared-prefix-len 512 \
  --output-len 64 --temperature 0.0
```

## Limits

- Only **full pages** are cached. A prefix of 300 tokens with page_size=256 caches
  one page (256 tokens); the remaining 44 tokens are always recomputed.
- Decode tokens are never cached; only prompt tokens enter the cache.
- Cache capacity is bounded by `max_pages`. When full, LRU pages with no active
  references are evicted.
