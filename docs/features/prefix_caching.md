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

Prefix caching is enabled automatically when using the HTTP server:

```bash
python scripts/serve/start_server.py --config configs/runs/batch.yaml
```

No additional configuration is required.

## Benchmarking

Use `--shared-prefix-len` to simulate a shared system prompt across all requests.
The value must produce at least one full KV page after tokenization — use 512+ words
to be safe (page_size = 256 tokens).

Run the benchmark twice: the first run populates the cache, the second hits it.

```bash
python scripts/serve/bench_serving.py \
  --mode fixed --concurrency 4 --num-requests 20 \
  --input-len 64 --shared-prefix-len 512 \
  --output-len 64 --temperature 0.0
```

## Limits

- Only **full pages** are cached (page_size = 256 tokens). A prefix of 300 tokens
  caches one page (256 tokens); the remaining 44 tokens are always recomputed.
- Decode tokens are never cached; only prompt tokens enter the cache.
- When the cache is full, LRU pages with no active references are evicted first.
  If all pages are in use, new entries are silently skipped until a page is freed.

