# Documentation

## Structure

```
docs/
├── features/      How to use specific capabilities
├── design/        Why things work the way they do (implementation deep-dives)
└── flash-attention/  Reading notes on FA2/FA3 kernel internals
```

**features/** answers: what does this do, when should I use it, what are the limits?

**design/** answers: why this design, how does it work internally?

## Feature guides

- [HTTP Serving](features/serving.md) — OpenAI-compatible server, streaming, concurrent requests
- [Prefix Caching](features/prefix_caching.md) — KV cache reuse across requests with the same prefix

## Design documents

- [Continuous Batching](design/continuous_batching.md) — scheduler, batching mechanics, admission policies
- [Chunked Prefill](design/chunked_prefill.md) — interleaving prefill and decode to reduce stalls

## Reading notes

[flash-attention/](flash-attention/README.md) — deep-dives into FA2/FA3 kernel source. Not required for using mini-vllm.
