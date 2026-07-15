"""BatchedModelRunner: drives inference for a heterogeneous batch of requests."""

from __future__ import annotations
import bisect
import copy
import time
import uuid
from typing import TYPE_CHECKING, Callable

import torch

from engine.context import set_context
from engine.request import Request, RequestStatus
from engine.schema import SamplingParams
from engine.sampler import Sampler
from utils.config import GlobalConfig
from utils.logger import logger

if TYPE_CHECKING:
    from engine.kv_pool import KVCachePool
    from engine.scheduler import Scheduler


def _build_capture_sizes(max_bs: int) -> list[int]:
    """Build the bucket list for decode CUDA graph capture.

    Mirrors vllm's default: [1, 2, 4] + multiples-of-8 up to max_bs.
    Ensures max_bs itself is always included.
    """
    sizes = [1, 2, 4]
    sizes += list(range(8, max_bs, 8))
    if not sizes or sizes[-1] != max_bs:
        sizes.append(max_bs)
    return sorted(set(s for s in sizes if s <= max_bs))


class BatchDecodeGraphManager:
    """Owns static decode buffers and the per-bucket CUDA graphs.

    Design (vllm FULL-mode style):
    - Static buffers allocated to max_bs; capture slices [:bs] for each bucket.
    - At runtime, real_bs is rounded up to the nearest bucket with bisect.
    - Padding rows get a dummy slot_id (0) and cache_len (0); their KV writes
      are suppressed via ctx.num_real_reqs in kv_pool / attention layers.
    - Graph key = padded_bs (int).
    """

    def __init__(self, model, max_bs: int, device: str, dummy_slot_id: int = 0,
                 kv_pool=None) -> None:
        self.model = model
        self.max_bs = max_bs
        self.device = device
        self.dummy_slot_id = dummy_slot_id
        self.kv_pool = kv_pool
        self.capture_sizes = _build_capture_sizes(max_bs)

        # Static input buffers sized to max_bs.
        self._input_ids    = torch.zeros(max_bs, 1, dtype=torch.long,  device=device)
        self._position_ids = torch.zeros(max_bs, 1, dtype=torch.long,  device=device)
        self._slot_ids     = torch.full((max_bs,), dummy_slot_id, dtype=torch.long, device=device)
        self._cache_lens   = torch.zeros(max_bs,    dtype=torch.long,  device=device)

        # Static buffers for paged KV (populated when kv_pool is a PagedKVPool).
        pages_per_seq = getattr(kv_pool, 'pages_per_seq', 0) if kv_pool else 0
        self._pages_per_seq = pages_per_seq
        if pages_per_seq > 0:
            dummy_pid = getattr(kv_pool, 'dummy_page_id', 0)
            self._block_table = torch.full(
                (max_bs, pages_per_seq), dummy_pid, dtype=torch.int32, device=device
            )
            self._seq_lens_buf = torch.zeros(max_bs, dtype=torch.int32, device=device)
        else:
            self._block_table = None
            self._seq_lens_buf = None

        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._outputs: dict[int, torch.Tensor] = {}
        self._pool_handle = torch.cuda.graph_pool_handle()

        logger.info(
            f"[BatchCUDAGraph] Initialized: max_bs={max_bs}, "
            f"capture_sizes={self.capture_sizes}, dummy_slot={dummy_slot_id}"
        )

    def pad_to_bucket(self, real_bs: int) -> int:
        idx = bisect.bisect_left(self.capture_sizes, real_bs)
        return self.capture_sizes[idx]

    def _capture_one(self, bs: int) -> None:
        """Capture the decode graph for a given bs bucket."""
        set_context(
            is_prefill=False,
            cache_len=0,
            slot_ids=self._slot_ids[:bs],
            cache_lens=self._cache_lens[:bs],
            num_real_reqs=bs,
            block_tables=self._block_table[:bs] if self._block_table is not None else None,
            seq_lens=self._seq_lens_buf[:bs] if self._seq_lens_buf is not None else None,
        )

        # Warmup — more iterations for the first capture to let CUDA allocate.
        warmup_iters = 3 if not self._graphs else 1
        for _ in range(warmup_iters):
            self.model(self._input_ids[:bs], self._position_ids[:bs])
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self._pool_handle):
            out = self.model(self._input_ids[:bs], self._position_ids[:bs])
        torch.cuda.synchronize()

        self._graphs[bs] = g
        self._outputs[bs] = out
        logger.info(f"[BatchCUDAGraph] Captured bs={bs}")

    def capture_all(self, kv_pool=None, on_captured=None) -> None:
        """Pre-capture graphs for all bucket sizes (largest first for pool reuse).

        kv_pool: if provided, snapshot its caches before capture and restore
        afterwards so that warmup/capture forward passes don't corrupt live KV.
        on_captured: optional fn(bs: int) called after each bucket is captured.
        """
        logger.info(
            f"[BatchCUDAGraph] Capturing {len(self.capture_sizes)} graphs ..."
        )
        snapshots = None
        if kv_pool is not None:
            snapshots = (
                [c.cpu() for c in kv_pool.k_caches],
                [c.cpu() for c in kv_pool.v_caches],
            )

        try:
            for bs in reversed(self.capture_sizes):
                self._capture_one(bs)
                if on_captured:
                    on_captured(bs)
        finally:
            if snapshots is not None:
                k_snaps, v_snaps = snapshots
                for i, (ks, vs) in enumerate(zip(k_snaps, v_snaps)):
                    kv_pool.k_caches[i].copy_(ks)
                    kv_pool.v_caches[i].copy_(vs)
                logger.info("[BatchCUDAGraph] KV pool restored after capture.")

        logger.info(
            f"[BatchCUDAGraph] Done. Captured {len(self._graphs)} graphs."
        )

    def run(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        slot_ids: torch.Tensor,
        cache_lens: torch.Tensor,
        real_bs: int,
        requests: list | None = None,
    ) -> torch.Tensor:
        """Copy real inputs into static buffers, replay graph, return logits."""
        padded_bs = self.pad_to_bucket(real_bs)

        # Ensure pages before graph replay — store_kv inside the graph cannot
        # call Python ensure_pages at replay time.
        if self.kv_pool is not None and hasattr(self.kv_pool, 'ensure_pages') and requests:
            for req in requests:
                self.kv_pool.ensure_pages(req.request_id, req.cache_len)

        # Fill real rows.
        self._input_ids[:real_bs].copy_(input_ids)
        self._position_ids[:real_bs].copy_(position_ids)
        self._slot_ids[:real_bs].copy_(slot_ids)
        self._cache_lens[:real_bs].copy_(cache_lens)

        # Zero out padding rows; point their slot_ids to the dummy sink slot
        # so any KV writes land in the dedicated dummy slot instead of slot 0.
        if padded_bs > real_bs:
            self._input_ids[real_bs:padded_bs].zero_()
            self._position_ids[real_bs:padded_bs].zero_()
            self._slot_ids[real_bs:padded_bs].fill_(self.dummy_slot_id)
            self._cache_lens[real_bs:padded_bs].zero_()

        # Copy paged KV metadata for real rows.
        if self._block_table is not None and self.kv_pool is not None:
            bt = self.kv_pool.block_table_for(slot_ids)  # (real_bs, pages_per_seq)
            self._block_table[:real_bs].copy_(bt)
            if padded_bs > real_bs:
                dummy_pid = getattr(self.kv_pool, 'dummy_page_id', 0)
                self._block_table[real_bs:padded_bs].fill_(dummy_pid)
            seq_lens_real = (cache_lens + 1).to(torch.int32)
            self._seq_lens_buf[:real_bs].copy_(seq_lens_real)
            if padded_bs > real_bs:
                self._seq_lens_buf[real_bs:padded_bs].zero_()

        set_context(
            is_prefill=False,
            cache_len=0,
            slot_ids=self._slot_ids[:padded_bs],
            cache_lens=self._cache_lens[:padded_bs],
            num_real_reqs=real_bs,
            block_tables=self._block_table[:padded_bs] if self._block_table is not None else None,
            seq_lens=self._seq_lens_buf[:padded_bs] if self._seq_lens_buf is not None else None,
        )

        self._graphs[padded_bs].replay()
        return self._outputs[padded_bs][:real_bs]


class BatchedModelRunner:

    def __init__(
        self,
        model,
        tokenizer,
        kv_pool: "KVCachePool",
        scheduler: "Scheduler",
        cfg: GlobalConfig,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.kv_pool = kv_pool
        self.scheduler = scheduler
        self.cfg = cfg
        self.device = cfg.env.device

        sampling_cfg = cfg.generation.sampling
        self.sampler = Sampler(
            sampling_cfg.sample_method,
            sampling_cfg.temperature,
            top_k=sampling_cfg.topk,
            top_p=sampling_cfg.topp,
        )
        self._eos_ids = self._collect_eos_ids()
        self.last_step_stats: dict = {"n_decode": 0, "n_prefill": 0, "prefill_tokens": 0}
        self.on_prefill_start: Callable[[list[Request]], None] | None = None
        # Called once before graph capture starts: fn(capture_sizes: list[int])
        self.on_graph_capture_start: Callable[[list[int]], None] | None = None
        # Called after each bucket is captured: fn(bs: int)
        self.on_graph_capture_step: Callable[[int], None] | None = None

        # CUDA graph setup for batched decode.
        self._graph_manager: BatchDecodeGraphManager | None = None
        use_cg = cfg.model.use_cuda_graph and torch.cuda.is_available()
        if use_cg:
            max_bs = cfg.model.cuda_graph_max_batch_size or cfg.batch.max_batch_size
            if not getattr(model, "supports_cuda_graph", True):
                logger.warning(
                    f"{type(model).__name__} does not support CUDA Graph, disabling."
                )
            else:
                self._graph_manager = BatchDecodeGraphManager(
                    model, max_bs, self.device,
                    dummy_slot_id=getattr(self.kv_pool, 'dummy_page_id',
                                          getattr(self.kv_pool, 'dummy_slot_id', 0)),
                    kv_pool=self.kv_pool,
                )

    @torch.inference_mode()
    def step(self) -> list[Request]:
        """Run one inference step; return requests that finished this step."""
        prefill_chunks, decode_reqs = self.scheduler.schedule()
        self.last_step_stats = {
            "n_decode":       len(decode_reqs),
            "n_prefill":      len(prefill_chunks),
            "prefill_tokens": sum(c for _, c in prefill_chunks),
        }
        finished: list[Request] = []

        if decode_reqs:
            # Lazy-capture on the first decode step so the KV pool and model
            # are fully warmed up before we touch the CUDA graph.
            if self._graph_manager is not None and not self._graph_manager._graphs:
                if self.on_graph_capture_start:
                    self.on_graph_capture_start(self._graph_manager.capture_sizes)
                self._graph_manager.capture_all(
                    kv_pool=self.kv_pool,
                    on_captured=self.on_graph_capture_step,
                )

            decode_logits = self._run_decode(decode_reqs)
            self._sample_and_update(decode_reqs, decode_logits)
            for req in decode_reqs:
                if req.is_finished:
                    self.scheduler.on_request_finished(req)
                    finished.append(req)

        if prefill_chunks:
            completed, first_tok_logits = self._run_prefill(prefill_chunks)
            if completed:
                self._sample_and_update(completed, first_tok_logits)
                for req in completed:
                    if req.is_finished:
                        self.scheduler.on_request_finished(req)
                        finished.append(req)

        return finished

    def _ensure_pages_for_prefill(self, requests, offsets, chunk_lens):
        """Pre-allocate physical pages for all tokens in this prefill chunk.

        Only allocates at page boundaries — at most ceil(chunk_len / page_size)
        allocations per request instead of one per token.
        """
        if not hasattr(self.kv_pool, 'ensure_pages'):
            return
        page_size = getattr(self.kv_pool, 'page_size', 1)
        for req, start, chunk_len in zip(requests, offsets, chunk_lens):
            req_id = req.request_id
            first_page = start // page_size
            last_page  = (start + chunk_len - 1) // page_size
            for logical_page in range(first_page, last_page + 1):
                self.kv_pool.ensure_pages(req_id, logical_page * page_size)

    def _run_prefill(
        self, chunks: list[tuple[Request, int]]
    ) -> tuple[list[Request], torch.Tensor | None]:
        """Prefill a batch of requests; return those that finished and their first-token logits."""
        requests    = [r for r, _ in chunks]
        chunk_sizes = [c for _, c in chunks]

        if self.on_prefill_start:
            self.on_prefill_start(requests)

        (input_ids, position_ids, slot_ids,
         cu_seqlens, offsets, actual_lens) = self._build_prefill_inputs(requests, chunk_sizes)

        # Ensure pages exist for every token in this chunk before the forward pass.
        self._ensure_pages_for_prefill(requests, offsets, actual_lens)

        cache_lens_t = torch.tensor(offsets, dtype=torch.long, device=self.device)

        # block_tables for paged KV prefill attention
        block_tables = (
            self.kv_pool.block_table_for(slot_ids)
            if hasattr(self.kv_pool, 'block_table_for') else None
        )
        set_context(
            is_prefill=True,
            cache_len=int(max(offsets)) if offsets else 0,
            slot_ids=slot_ids,
            cache_lens=cache_lens_t,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max(actual_lens),
            max_seqlen_k=max(o + l for o, l in zip(offsets, actual_lens)),
            attn_mask=None,
            block_tables=block_tables,
        )
        logits_all = self.model(input_ids, position_ids)

        completed: list[Request] = []
        completed_logits: list[torch.Tensor] = []

        for i, req in enumerate(requests):
            new_prefilled     = offsets[i] + actual_lens[i]
            req.prefilled_len = new_prefilled
            req.cache_len     = new_prefilled

            if new_prefilled >= req.num_prompt_tokens:
                req.prefilled_len = req.num_prompt_tokens
                req.cache_len     = req.num_prompt_tokens
                req.status        = RequestStatus.DECODING
                completed.append(req)
                completed_logits.append(logits_all[i, actual_lens[i] - 1, :])

        if completed:
            return completed, torch.stack(completed_logits)
        return [], None

    def _run_decode(self, requests: list[Request]) -> torch.Tensor:
        """Decode one token per request; return logits of shape (batch, vocab)."""
        input_ids, position_ids, slot_ids, cache_lens_t = self._build_decode_inputs(requests)

        # Ensure a page exists for the new token position (cache_len) for each req.
        if hasattr(self.kv_pool, 'ensure_pages'):
            for req in requests:
                self.kv_pool.ensure_pages(req.request_id, req.cache_len)

        block_tables = (
            self.kv_pool.block_table_for(slot_ids)
            if hasattr(self.kv_pool, 'block_table_for') else None
        )
        # seq_lens = cache_len + 1 (the new token written this step is included)
        seq_lens = (cache_lens_t + 1).to(torch.int32) if block_tables is not None else None

        if self._graph_manager is not None:
            real_bs = len(requests)
            logits_all = self._graph_manager.run(
                input_ids, position_ids, slot_ids, cache_lens_t, real_bs,
                requests=requests,
            )
        else:
            set_context(
                is_prefill=False,
                cache_len=int(cache_lens_t[0].item()),
                slot_ids=slot_ids,
                cache_lens=cache_lens_t,
                block_tables=block_tables,
                seq_lens=seq_lens,
            )
            logits_all = self.model(input_ids, position_ids)

        for req in requests:
            req.cache_len += 1

        return logits_all[:, 0, :]

    # ------------------------------------------------------------------
    # Input construction
    # ------------------------------------------------------------------

    def _build_prefill_inputs(
        self,
        requests: list[Request],
        chunk_sizes: list[int] | int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[int]]:
        """Build padded input tensors and cu_seqlens for a prefill batch."""
        batch  = len(requests)
        pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0

        offsets: list[int] = []
        chunks:  list[list[int]] = []
        for i, req in enumerate(requests):
            start = req.prefilled_len
            if chunk_sizes is None:
                end = req.num_prompt_tokens
            elif isinstance(chunk_sizes, int):
                end = min(start + chunk_sizes, req.num_prompt_tokens)
            else:
                end = min(start + chunk_sizes[i], req.num_prompt_tokens)
            offsets.append(start)
            chunks.append(req.prompt_token_ids[start:end])

        chunk_lens = [len(c) for c in chunks]
        max_chunk  = max(chunk_lens)

        input_ids    = torch.full((batch, max_chunk), pad_id, dtype=torch.long,  device=self.device)
        position_ids = torch.zeros((batch, max_chunk),        dtype=torch.long,  device=self.device)
        for i, (chunk, offset) in enumerate(zip(chunks, offsets)):
            L = len(chunk)
            input_ids[i, :L]    = torch.tensor(chunk, dtype=torch.long, device=self.device)
            position_ids[i, :L] = torch.arange(offset, offset + L, device=self.device)

        slot_ids = torch.tensor([r.slot_id for r in requests], dtype=torch.long, device=self.device)

        cu_seqlens = torch.zeros(batch + 1, dtype=torch.int32, device=self.device)
        for i, l in enumerate(chunk_lens):
            cu_seqlens[i + 1] = cu_seqlens[i] + l

        return input_ids, position_ids, slot_ids, cu_seqlens, offsets, chunk_lens

    def _build_decode_inputs(
        self, requests: list[Request]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build input tensors for a decode step (one token per request)."""
        cache_lens_list = [r.cache_len for r in requests]

        input_ids = torch.tensor(
            [r.generated_ids[-1] for r in requests],
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(1)
        position_ids = torch.tensor(
            cache_lens_list, dtype=torch.long, device=self.device
        ).unsqueeze(1)
        slot_ids = torch.tensor(
            [r.slot_id for r in requests], dtype=torch.long, device=self.device
        )
        cache_lens_t = torch.tensor(
            cache_lens_list, dtype=torch.long, device=self.device
        )

        return input_ids, position_ids, slot_ids, cache_lens_t

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_and_update(self, requests: list[Request], logits: torch.Tensor) -> None:
        next_tokens = self.sampler.sample(logits.unsqueeze(1))

        for i, req in enumerate(requests):
            token_id    = int(next_tokens[i, 0].item())
            max_new_tok = req.sampling_params.max_new_tokens
            req.generated_ids.append(token_id)
            if len(req.generated_ids) == 1:
                req.first_token_at = time.perf_counter()

            if req.on_token is not None:
                is_last = (token_id in self._eos_ids
                           or len(req.generated_ids) >= max_new_tok)
                req.on_token(req.request_id, token_id, is_last)

            if token_id in self._eos_ids:
                req.mark_finished("eos")
            elif len(req.generated_ids) >= max_new_tok:
                req.mark_finished("length")

    def _collect_eos_ids(self) -> set[int]:
        """Union EOS token ids from model config, tokenizer, and generation config."""
        ids: set[int] = set()
        cfg_eos = getattr(getattr(self.model, "config", None), "eos_token_id", None)
        if isinstance(cfg_eos, (list, tuple)):
            ids.update(int(x) for x in cfg_eos if x is not None)
        elif cfg_eos is not None:
            ids.add(int(cfg_eos))
        tok_eos = getattr(self.tokenizer, "eos_token_id", None)
        if tok_eos is not None:
            ids.add(int(tok_eos))
        return ids

    def warmup(self, prompt_tokens: int = 8, decode_steps: int = 3) -> None:
        """Run a short dummy request to warm up CUDA kernels before timed inference."""
        dummy = Request(
            request_id       = "__warmup__",
            prompt_token_ids = list(range(prompt_tokens)),
            sampling_params  = SamplingParams(
                temperature    = 0.0,
                max_new_tokens = decode_steps,
                stop_on_eos    = False,
            ),
        )
        self.scheduler.add_request(dummy)
        while self.scheduler.has_work():
            self.step()
