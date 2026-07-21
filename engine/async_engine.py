"""AsyncLLMEngine: single-request async bridge for main.py and chat_cli.py.

This module belongs to the single-request inference path (main.py → ModelRunner).
For the batch serving path used by server.py, see engine/batch_async_engine.py.

Design:
- A single dedicated inference thread owns the GPU; requests are serialised
  through a threading.Queue so we never have concurrent forward passes.
- Each request gets its own asyncio.Queue for token-by-token streaming.
- The thread puts results back into the event loop via loop.call_soon_threadsafe,
  which is the standard safe cross-thread asyncio bridge.
"""

import asyncio
import queue
import threading
import uuid
from typing import AsyncIterator

from utils.logger import logger
from engine.schema import SamplingParams, GenerationOutput
from engine.model_runner import ModelRunner


class AsyncLLMEngine:
    def __init__(self, runner: ModelRunner):
        self._runner = runner
        self._thread_queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._inference_thread, name="inference-worker", daemon=True
        )
        self._thread.start()
        logger.info("AsyncLLMEngine started (inference worker thread ready).")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str | None = None,
    ) -> AsyncIterator[GenerationOutput]:
        if request_id is None:
            request_id = str(uuid.uuid4())

        loop = asyncio.get_event_loop()
        output_queue: asyncio.Queue[GenerationOutput] = asyncio.Queue()

        self._thread_queue.put((request_id, prompt, sampling_params, output_queue, loop))

        while True:
            output = await output_queue.get()
            yield output
            if output.is_finished:
                break

    def shutdown(self):
        self._thread_queue.put(None)
        self._thread.join(timeout=5)

    # ------------------------------------------------------------------
    # Inference thread
    # ------------------------------------------------------------------

    def _inference_thread(self):
        while True:
            item = self._thread_queue.get()
            if item is None:
                break
            request_id, prompt, params, output_queue, loop = item
            self._step(request_id, prompt, params, output_queue, loop)

    def _step(self, request_id, prompt, params, output_queue, loop):
        """Process a single request. Future: accept list[request] for batching."""
        try:
            for token_id, text_delta, is_finished, finish_reason in self._runner.generate_stream(
                prompt, params
            ):
                out = GenerationOutput(
                    request_id=request_id,
                    token_id=token_id,
                    text_delta=text_delta,
                    is_finished=is_finished,
                    finish_reason=finish_reason,
                )
                loop.call_soon_threadsafe(output_queue.put_nowait, out)
        except Exception as exc:
            logger.exception(f"Inference error for request {request_id}: {exc}")
            out = GenerationOutput(
                request_id=request_id,
                token_id=0,
                text_delta="",
                is_finished=True,
                finish_reason="error",
                error=str(exc),
            )
            loop.call_soon_threadsafe(output_queue.put_nowait, out)
