"""BatchAsyncEngine: bridges BatchedModelRunner with async FastAPI handlers.

Design
------
- One background thread owns the GPU and runs the continuous-batching step loop.
- HTTP handlers call add_request() to enqueue work, then iterate the returned
  AsyncGenerator to receive tokens one by one as they are produced.
- The step loop calls batched_runner.step() which processes ALL in-flight
  requests in a single forward pass (real batching), then distributes new
  tokens to each request's asyncio.Queue via loop.call_soon_threadsafe.
- The thread sleeps briefly (IDLE_SLEEP_S) when the scheduler has no work to
  avoid burning CPU in a tight loop.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from typing import AsyncIterator

from utils.logger import logger
from engine.schema import SamplingParams, GenerationOutput
from engine.request import Request

IDLE_SLEEP_S = 0.002   # 2 ms sleep when scheduler queue is empty


class BatchAsyncEngine:
    def __init__(self, runner, scheduler, tokenizer):
        self._runner = runner
        self._scheduler = scheduler
        self._tokenizer = tokenizer

        # per-request queues: request_id -> asyncio.Queue[GenerationOutput]
        self._queues: dict[str, asyncio.Queue] = {}
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

        self._thread = threading.Thread(
            target=self._step_loop, name="batch-inference-worker", daemon=True
        )
        self._running = True
        self._thread.start()
        logger.info("BatchAsyncEngine started.")

    # ------------------------------------------------------------------
    # Public API (called from asyncio / HTTP handlers)
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        request_id: str | None = None,
    ) -> AsyncIterator[GenerationOutput]:
        """Generate tokens for a prompt.

        Args:
            prompt: Either a raw string (tokenized internally) or a list of
                token ids (used as-is). Callers that have already tokenized
                (e.g. server.py after applying a chat template) should pass
                token ids to avoid double-tokenization.
            sampling_params: Sampling configuration.
            request_id: Optional request identifier; auto-generated if None.
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        if self._loop is None:
            self._loop = asyncio.get_running_loop()

        if isinstance(prompt, str):
            has_tmpl = getattr(self._tokenizer, "chat_template", None) is not None
            if has_tmpl:
                token_ids = self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    enable_thinking=sampling_params.enable_thinking,
                    return_tensors="pt",
                )["input_ids"][0].tolist()
            else:
                token_ids = self._tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
        else:
            token_ids = prompt

        output_queue: asyncio.Queue[GenerationOutput] = asyncio.Queue()
        req = Request(
            request_id=request_id,
            prompt_token_ids=token_ids,
            sampling_params=sampling_params,
        )

        with self._lock:
            self._queues[request_id] = output_queue
            self._scheduler.add_request(req)

        while True:
            output = await output_queue.get()
            yield output
            if output.is_finished:
                break

    def shutdown(self):
        self._running = False
        self._thread.join(timeout=5)

    # ------------------------------------------------------------------
    # Background step loop (runs in dedicated thread)
    # ------------------------------------------------------------------

    def _step_loop(self):
        while self._running:
            if not self._scheduler.has_work():
                time.sleep(IDLE_SLEEP_S)
                continue

            try:
                finished, new_tokens = self._runner.step()
            except Exception as exc:
                logger.exception(f"BatchAsyncEngine step error: {exc}")
                time.sleep(IDLE_SLEEP_S)
                continue

            if not new_tokens:
                continue

            # Wait until the event loop is available (set by the first generate() call).
            while self._loop is None and self._running:
                time.sleep(0.001)
            if not self._running:
                break

            finished_map = {r.request_id: r for r in finished}

            outputs: list[tuple[str, GenerationOutput]] = []
            for req_id, token_id in new_tokens.items():
                fin_req = finished_map.get(req_id)
                text_delta = self._tokenizer.decode(
                    [token_id], skip_special_tokens=True
                )
                outputs.append((req_id, GenerationOutput(
                    request_id=req_id,
                    token_id=token_id,
                    text_delta=text_delta,
                    is_finished=fin_req is not None,
                    finish_reason=fin_req.finish_reason if fin_req else None,
                )))

            # Push and clean up under a single lock acquisition.
            with self._lock:
                for req_id, out in outputs:
                    q = self._queues.get(req_id)
                    if q is not None:
                        self._loop.call_soon_threadsafe(q.put_nowait, out)
                for req in finished:
                    self._queues.pop(req.request_id, None)
