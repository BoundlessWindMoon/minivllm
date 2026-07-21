# HTTP Serving

<!-- code-ref: server.py -->
<!-- code-ref: engine/batch_async_engine.py -->
<!-- code-ref: engine/batched_runner.py -->
<!-- code-ref: scripts/serve/start_server.py -->
<!-- code-ref: scripts/serve/bench_serving.py -->
<!-- code-ref: scripts/serve/chat_client.py -->
mini-vllm exposes an OpenAI-compatible `/v1/chat/completions` endpoint backed by
continuous batching. Concurrent requests are automatically grouped into batches.

## Starting the server

```bash
python scripts/serve/start_server.py --config configs/runs/batch.yaml
```

Options: `--host` (default `0.0.0.0`), `--port` (default `8000`).

## API

### `POST /v1/chat/completions`

```json
{
  "model": "mini-vllm",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 128,
  "temperature": 0.0,
  "stream": false
}
```

Set `"stream": true` to receive Server-Sent Events (SSE). Each chunk follows the
OpenAI streaming format; the final event is `data: [DONE]`.

### `GET /health`

Returns `{"status": "ok"}` when the engine is ready.

## Chat client

```bash
python scripts/serve/chat_client.py --url http://localhost:8000
```

Supports multi-turn conversation, `/think on` / `/think off`, `/clear`, `/exit`.

## Benchmarking

```bash
# Fixed concurrency
python scripts/serve/bench_serving.py --mode fixed --concurrency 4 --num-requests 40

# Sweep across concurrency levels 1→16
python scripts/serve/bench_serving.py --mode sweep

# Poisson arrivals at 4 req/s for 60 s
python scripts/serve/bench_serving.py --mode poisson --request-rate 4 --duration 60
```

Metrics reported: TTFT p50/p99, TPOT p50/p99, E2E p50, throughput (tok/s).
