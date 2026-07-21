# mini-vllm — AI Agent Instructions

A lightweight LLM inference engine for studying core serving systems:
paged KV cache, continuous batching, prefix caching, CUDA graphs, and HTTP serving.

## Quick reference

```bash
# Run tests
python -m pytest test/unit/     -q   # fast, no GPU needed
python -m pytest test/integration/ -q   # requires CUDA

# Check doc references are not stale
make check-docs

# Start HTTP server
python scripts/serve/start_server.py --config configs/runs/batch.yaml

# Benchmark (run twice to see prefix cache effect)
python scripts/serve/bench_serving.py --mode fixed --concurrency 4 \
  --num-requests 20 --input-len 64 --shared-prefix-len 512 \
  --output-len 64 --temperature 0.0

# Lint
python -m ruff check . --fix
python -m ruff format .
```

## Project layout

```
engine/          # Core inference: scheduler, batched runner, KV pool,
                 #   prefix cache, async engine bridge, decode graph
layers/          # Attention, MLP, RMSNorm, KV cache backends
model/           # Qwen3 / Qwen3.5 architectures + megakernel variants
kernels/         # Triton/CUDA kernels (AWQ, KIVI)
quantization/    # AWQ calibration + quantized layers
utils/           # Config, logging, model loader, chat UI, batch display
server.py        # FastAPI HTTP server (OpenAI-compatible)
configs/         # YAML configs — runs/, models/, base.yaml
scripts/serve/   # start_server.py, bench_serving.py, chat_client.py
test/unit/       # CPU-only unit tests, no real model
test/integration/# GPU integration tests, require a loaded model
```

## Two inference paths

mini-vllm has two independent runtime paths. Do not mix them.

### Single-request path
Entry point: `main.py`, `chat_cli.py`, `eval/`

```
main.py → ModelRunner → (optional) AsyncLLMEngine (single-request HTTP only)
```

- Supports megakernel backend, KIVI KV quantization, multimodal
- No continuous batching; one request at a time
- Use for: local interactive use, lm-eval, megakernel benchmarks

### Batch serving path
Entry point: `batch_main.py`, `server.py`

```
server.py / batch_main.py → BatchedModelRunner → BatchAsyncEngine → HTTP
```

- Continuous batching: all in-flight requests share one GPU forward pass per step
- Paged KV cache, prefix caching, CUDA graph batched decode
- Use for: serving, throughput benchmarks, concurrent request testing

**When in doubt, the serving path (`server.py`) is the main path.**
The single-request path exists for backward compatibility and special use cases.


## Config system

Configs use `_base` inheritance. A run config points to a model config which
points to `base.yaml`:

```
configs/runs/batch.yaml
  → _base: ../models/qwen3.yaml
    → _base: ../base.yaml
```

Override any field inline; later values win. Always use `GlobalConfig.from_yaml(path)`
to load — do not parse YAML manually.

**All config fields and their defaults are documented in `configs/base.yaml`.**
That file is the single source of truth for config schema — do not duplicate
field names or defaults in other docs.

## Testing rules

**Unit tests** (`test/unit/`):
- No GPU, no real model, no network. Must run on CPU in < 5 s total.
- Use stubs/fakes instead of mocks where possible (prefer `FakePool` over
  `MagicMock()` for pool-shaped objects).
- Assert observable behavior through public APIs — do not reach into `_private`
  attributes unless unavoidable, and add a comment explaining why.
- Use `asyncio.run()` for async tests; do not add `pytest-asyncio` as a dependency.
- Follow the existing pattern: `sys.path.insert` at top of file (conftest handles
  root, but individual test files also do it for robustness).

**Integration tests** (`test/integration/`):
- Require CUDA. Each test file sets `torch.set_default_device` in its fixture
  and restores it on teardown — safe to run in any order.
- Run unit and integration suites separately: `pytest test/unit/` then
  `pytest test/integration/`.

## Key invariants and sharp edges

**PagedKVPool**
- `page_size` must be divisible by 256 (FA2 alignment requirement).
- `dummy_page_id = total_pages` (the last page is a sink for padding writes).
- Call `ensure_pages(req_id, token_pos)` before any `store_kv` for that position.
- Shared pages (from prefix cache) are tracked in `_shared_pages` and are NOT
  returned to `_free_pages` on `free()`. Only call `pool.pages_for(req_id)` to
  read block tables — do not access `_block_tables` directly.

**Prefix cache**
- Only full pages are cached (tail < page_size is always skipped).
- Chain hash: block identity = hash(prev_block_hash, token_ids). Two sequences
  only share a block if every preceding token is identical.
- `lookup()` increments ref counts. Always pair with `release()` when done.
- `PrefixCache` is not thread-safe; call only from the scheduler thread.

**BatchAsyncEngine**
- One background thread owns the GPU exclusively. Never call model forward
  passes from any other thread.
- `generate(prompt, sampling_params, request_id)` accepts either a `str` (tokenized
  internally via the engine's tokenizer) or a `list[int]` (used as-is). Callers
  that have already applied a chat template should pass token ids to avoid
  double-tokenization.
- The event loop is captured lazily on the first `generate()` call. The step
  loop waits for `_loop` to be set before dispatching tokens — this is
  intentional to avoid a race where the thread runs a step before the first
  HTTP handler has set up its queue.

**CUDA graph + kv_backend conflict**
- `use_cuda_graph_bucket=True` and a custom `kv_backend` on the same attention
  layer are incompatible. The Attention layer raises `ValueError` at init.
- The batched decode CUDA graph (`BatchDecodeGraphManager`) captures at bucket
  sizes `[1, 2, 4, 8, ...]`. Padding rows write to `dummy_slot_id`, not slot 0.

**Attention backends**
- `flash_attn` backend for batch decode uses `flash_attn_with_kvcache` directly
  with the paged block table — FA never gathers the full KV tensor.
- `sdpa` backend calls `load_kv_for_sdpa()` which does a vectorized gather to
  reconstruct a dense `(B, kv_h, total_len, d)` tensor. This is correct but
  slower than the paged FA path.

## Adding a new feature checklist

1. Write unit tests first (`test/unit/`) — they must pass with no GPU.
2. If the feature touches the engine step loop, verify the current `step()`
   return type in `engine/batched_runner.py` before updating callers.
3. If the feature adds a new config field, add it to `base.yaml` and document
   the default.
4. Run `python -m pytest test/unit/ -q` before committing.
5. If it affects serving throughput, run `bench_serving.py` and note the delta.
6. If adding a new `docs/features/` or `docs/design/` file, add `<!-- code-ref: path/to/module.py -->` annotations for the source files it describes. Run `make check-docs` to verify.

## Documentation sync rules

**Before writing or updating any doc: verify against the code first.**
Do not rely on memory or prior context. Check the actual source file for
flag names, defaults, return types, and behavior. If what you find contradicts
the existing doc, the code wins — fix the doc.

Docs live in `docs/`. Two kinds:

- **features/** — what it does, how to enable, limits. Update when behavior changes.
  Write only what users observe (CLI commands, API schema, config keys).
  Do not name internal classes (`BatchAsyncEngine`, `PagedKVPool`, etc.) or
  implementation details — those belong in `design/` or code comments.
- **design/** — why this design. Update only when the architecture changes, not on every refactor.

**When to update docs:**

| Change | Doc to update |
|--------|--------------|
| New user-facing feature | `docs/features/<feature>.md` (create if missing) |
| Changed CLI arg or config field | `docs/features/` + `CLAUDE.md` quick reference |
| Changed `step()` signature or `BatchAsyncEngine.generate()` params | `CLAUDE.md` invariants section |
| Changed two-path architecture | `CLAUDE.md` "Two inference paths" section |
| Pure refactor, no behavior change | No doc update needed |

**Source of truth:** code is always authoritative. If a doc contradicts the code,
fix the doc. Never add a workaround to make code match stale docs.

