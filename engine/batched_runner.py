"""BatchedModelRunner: drives inference for a heterogeneous batch of requests."""

from __future__ import annotations
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

        cache_lens_t = torch.tensor(offsets, dtype=torch.long, device=self.device)
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
        set_context(
            is_prefill=False,
            cache_len=int(cache_lens_t[0].item()),
            slot_ids=slot_ids,
            cache_lens=cache_lens_t,
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
