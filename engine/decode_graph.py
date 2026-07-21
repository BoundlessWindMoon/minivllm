"""Single-request decode CUDA graph manager.

Owns the static input buffers and per-step graph captures used by ModelRunner
during the decode phase of single-request inference.

Contrast with BatchDecodeGraphManager (engine/batched_runner.py), which handles
batched decode with bucket-padded static buffers for multiple concurrent requests.
"""

from __future__ import annotations

import torch

from engine.context import set_context
from utils.logger import logger


class SingleRequestDecodeGraphManager:
    """Manages on-demand CUDA graph capture and replay for single-request decode.

    Each unique cache_len (or bucket, when bucket_size > 1) gets its own graph.
    The first call for a new cache_len triggers a capture; subsequent calls replay.
    """

    def __init__(
        self,
        model,
        device: str,
        bucket_size: int = 1,
        cu_seqlens_q: torch.Tensor | None = None,
    ) -> None:
        self._model = model
        self._device = device
        self._bucket_size = max(1, bucket_size)

        self._input_ids     = torch.zeros((1, 1), device=device, dtype=torch.long)
        self._position_ids  = torch.zeros((1, 1), device=device, dtype=torch.long)
        self._cu_seqlens_q  = cu_seqlens_q or torch.tensor(
            [0, 1], dtype=torch.long, device=device
        )
        self._pool          = torch.cuda.graph_pool_handle()
        self._graphs:  dict[int, torch.cuda.CUDAGraph] = {}
        self._outputs: dict[int, torch.Tensor]         = {}

        if bucket_size > 1:
            logger.info(
                f"[CUDA Graph] Single-request decode enabled, "
                f"bucket_size={bucket_size}."
            )
        else:
            logger.info("[CUDA Graph] Single-request decode enabled (exact per-step).")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def to_bucket(self, cache_len: int) -> int:
        """Map cache_len to its bucket key."""
        if self._bucket_size <= 1:
            return cache_len
        if cache_len == 0:
            return 0
        return ((cache_len - 1) // self._bucket_size + 1) * self._bucket_size

    def ensure(self, cache_len: int, restore_state: bool = True) -> None:
        """Capture a graph for cache_len if not already done (on-demand)."""
        bucket = self.to_bucket(cache_len)
        if bucket in self._graphs:
            return

        capture_len = bucket if self._bucket_size > 1 else cache_len

        set_context(
            is_prefill=False,
            cache_len=capture_len,
            cu_seqlens_q=self._cu_seqlens_q,
        )
        self._position_ids[0, 0] = capture_len

        snapshot = None
        if restore_state and hasattr(self._model, "_snapshot_cuda_graph_state"):
            snapshot = self._model._snapshot_cuda_graph_state()

        warmup_iters = 3 if not self._graphs else 1
        for _ in range(warmup_iters):
            self._model(self._input_ids, self._position_ids, decode_position=capture_len)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self._pool):
            out = self._model(self._input_ids, self._position_ids, decode_position=capture_len)
        torch.cuda.synchronize()

        if restore_state and snapshot is not None and hasattr(self._model, "_restore_cuda_graph_state"):
            self._model._restore_cuda_graph_state(snapshot)

        self._graphs[bucket]  = g
        self._outputs[bucket] = out

    def precapture_all(self, start: int, num_tokens: int, profiler=None, pbar=None) -> None:
        """Pre-capture graphs for all decode steps before the loop starts."""
        end = start + num_tokens - 1
        if self._bucket_size <= 1:
            n_unique = num_tokens
        else:
            n_unique = (self.to_bucket(end) - self.to_bucket(start)) // self._bucket_size + 1

        logger.info(
            f"[CUDA Graph] Pre-capturing {n_unique} graphs "
            f"(cache_len {start}~{end}, bucket_size={self._bucket_size}) ..."
        )

        snapshot = None
        has_snap = hasattr(self._model, "_snapshot_cuda_graph_state")
        if has_snap:
            snapshot = self._model._snapshot_cuda_graph_state()

        if profiler:
            profiler.pause()

        try:
            for i in range(num_tokens):
                self.ensure(start + i, restore_state=False)
                if pbar:
                    pbar.step_warmup(1)
        finally:
            if has_snap and snapshot is not None:
                self._model._restore_cuda_graph_state(snapshot)
            if profiler:
                profiler.resume()

        logger.info(f"[CUDA Graph] Pre-captured {len(self._graphs)} graphs.")

    def replay(self, next_token: torch.Tensor, past_len: int) -> torch.Tensor:
        """Copy inputs into static buffers and replay the graph for past_len."""
        bucket = self.to_bucket(past_len)
        self._input_ids.copy_(next_token.reshape(1, 1))
        self._position_ids[0, 0] = past_len
        self.ensure(past_len)
        self._graphs[bucket].replay()
        return self._outputs[bucket]
