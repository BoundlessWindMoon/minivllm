"""ModelRunner: orchestrates prefill, decode, CUDA graph, and sampling."""

import os
from typing import Iterator, Optional

import torch

from utils.logger import logger
from engine.context import get_context, set_context
from engine.decode_graph import SingleRequestDecodeGraphManager
from engine.progress import InferenceProgress, _NoOpProgress
from engine.profiler import build_profiler
from engine.sampler import Sampler
from engine.schema import SamplingParams, GenerationOutput
from utils.config import GlobalConfig


class ModelRunner:
    def __init__(
        self,
        model,
        tokenizer,
        cfg: GlobalConfig,
        processor=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.processor = processor

        self.device = cfg.env.device
        self.max_new_tokens = cfg.generation.max_new_tokens
        self.use_kvcache = cfg.model.use_kvcache
        self.stop_on_eos = cfg.generation.stop_on_eos
        self.eos_token_ids = self._collect_eos_ids() if self.stop_on_eos else set()

        self.use_progress = True
        self._multimodal_kwargs = {}

        bucket_size = max(1, cfg.model.cuda_graph_bucket_size)
        use_cg = cfg.model.use_cuda_graph

        if use_cg and not getattr(model, "supports_cuda_graph", True):
            logger.warning(f"{type(model).__name__} does not support CUDA Graph, disabling.")
            use_cg = False

        if use_cg and not torch.cuda.is_available():
            logger.warning("CUDA Graph requested but CUDA not available, disabling.")
            use_cg = False

        self.use_cuda_graph = use_cg
        self._cu_seqlens_q_decode = torch.tensor([0, 1], dtype=torch.long, device=self.device)
        self._setup_attention_bucket_mode(use_cg and bucket_size > 1)

        self._graph_mgr: SingleRequestDecodeGraphManager | None = None
        if use_cg:
            self._graph_mgr = SingleRequestDecodeGraphManager(
                model=model,
                device=self.device,
                bucket_size=bucket_size,
                cu_seqlens_q=self._cu_seqlens_q_decode,
            )

        sampling_cfg = cfg.generation.sampling
        self.sampler = Sampler(
            sampling_cfg.sample_method,
            sampling_cfg.temperature,
            top_k=sampling_cfg.topk,
            top_p=sampling_cfg.topp,
        )

        self.profiler = build_profiler(cfg)
        if not self.use_kvcache:
            logger.error("KVCACHE is not enabled, please enable it for better performance")
            assert self.use_kvcache

    def _setup_attention_bucket_mode(self, use_bucket: bool):
        """Configure all attention layers for CUDA Graph bucketing."""
        for attn_module in self.model.iter_attention_modules():
            if hasattr(attn_module, "use_cuda_graph_bucket"):
                attn_module.use_cuda_graph_bucket = use_bucket

    def _update_attention_masks(self, past_len: int):
        """Update _write_pos and _attn_mask for all attention layers."""
        seq_len = past_len + 1
        for attn_module in self.model.iter_attention_modules():
            if not getattr(attn_module, "use_cuda_graph_bucket", False):
                continue
            inner_attn = getattr(attn_module, "attn", attn_module)
            if hasattr(inner_attn, "_write_pos"):
                inner_attn._write_pos[0] = past_len
            if hasattr(inner_attn, "_attn_mask"):
                inner_attn._attn_mask.fill_(float("-inf"))
                inner_attn._attn_mask[..., :seq_len] = 0

    def _prepare_multimodal_inputs(self, multimodal_cfg, use_thinking: bool):
        from PIL import Image

        image_path = os.path.expanduser(multimodal_cfg.image_path)
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
            logger.info(f"Loaded image from {image_path}")
        else:
            image = Image.new("RGB", (224, 224), color=(100, 150, 200))
            logger.warning(
                f"Multimodal image not found at {image_path}; using synthetic test image."
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=use_thinking,
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        self.input_ids = inputs["input_ids"]
        self.position_ids = torch.arange(
            self.input_ids.shape[1], device=self.device
        ).unsqueeze(0)

        self._multimodal_kwargs = {
            k: v
            for k, v in inputs.items()
            if k in ("pixel_values", "image_grid_thw", "mm_token_type_ids")
        }
        pv = self._multimodal_kwargs.get('pixel_values')
        logger.info(
            f"Multimodal input shape: {self.input_ids.shape}, "
            f"pixel_values: {pv.shape if pv is not None else None}"
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        cached_len: int = 0,
        max_new_tokens: int = None,
        pbar=None,
    ) -> torch.Tensor:
        if pbar is None:
            pbar = _NoOpProgress()

        max_new_tokens = max_new_tokens or self.max_new_tokens
        prompt_seq_len = input_ids.shape[1]
        new_seq_len = prompt_seq_len - cached_len
        past_len = prompt_seq_len
        generated_ids = []

        position_ids = torch.arange(
            cached_len, prompt_seq_len, device=self.device
        ).unsqueeze(0)

        new_input_ids = input_ids[:, cached_len:]

        # ==========================================
        # 1. Prefill
        # ==========================================
        pbar.start_prefill()
        cu_seqlens_q_prefill = torch.tensor(
            [0, new_seq_len], dtype=torch.long, device=self.device
        )
        set_context(
            is_prefill=True,
            cache_len=cached_len,
            cu_seqlens_q=cu_seqlens_q_prefill,
        )
        logits = self.run(new_input_ids, position_ids, **self._multimodal_kwargs)

        next_token = self.sampler.sample(logits)
        generated_ids.append(next_token.item())
        pbar.end_prefill(next_token.item())

        # ==========================================
        # 2. Decode
        # ==========================================

        if self.use_cuda_graph and self._graph_mgr is not None:
            self._graph_mgr._input_ids.copy_(next_token.reshape(1, 1))

            can_precapture = hasattr(self.model, "_snapshot_cuda_graph_state")
            if can_precapture:
                pbar.start_warmup(total=max_new_tokens + 2)
                self._graph_mgr.precapture_all(
                    past_len, max_new_tokens, profiler=self.profiler, pbar=pbar
                )
                pbar.end_warmup()

        for _, token_id, is_finished, _ in self._run_decode_loop(
            next_token, past_len, max_new_tokens, self.sampler, self.eos_token_ids, pbar
        ):
            generated_ids.append(token_id)
            if is_finished:
                break

        return torch.tensor([generated_ids], device=self.device, dtype=torch.long)

    @torch.inference_mode()
    def inference(self, prompt: Optional[str] = None) -> str:
        prompt = prompt or self.cfg.generation.prompt
        use_chat = self.cfg.generation.use_chat_template
        use_thinking = self.cfg.generation.use_thinking
        has_tmpl = getattr(self.tokenizer, "chat_template", None) is not None

        multimodal_cfg = self.cfg.generation.multimodal
        multimodal_enabled = (
            multimodal_cfg is not None
            and multimodal_cfg.enabled
            and self.processor is not None
        )

        if multimodal_enabled:
            self._prepare_multimodal_inputs(multimodal_cfg, use_thinking)
            input_ids = self.input_ids
        else:
            if use_chat and has_tmpl:
                chat_inputs = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    enable_thinking=use_thinking,
                    return_tensors="pt",
                )
                input_ids = chat_inputs["input_ids"].to(self.device)
            else:
                if use_chat and not has_tmpl:
                    logger.warning(
                        "use_chat_template=True but tokenizer has no chat_template; "
                        "falling back to raw tokenization."
                    )
                input_ids = self.tokenizer(prompt, return_tensors="pt")[
                    "input_ids"
                ].to(self.device)

        max_new_tokens = self.max_new_tokens
        tokenizer = self.tokenizer
        prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        with InferenceProgress(tokenizer, max_new_tokens, prompt_text) as pbar:
            with self.profiler.scope("decode"):
                output_ids = self.generate(
                    input_ids,
                    cached_len=0,
                    max_new_tokens=max_new_tokens,
                    pbar=pbar,
                )
                generated_ids = output_ids[0].tolist()

        text = tokenizer.decode(
            input_ids[0], skip_special_tokens=True
        ) + tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text

    def _tokenize_prompt(self, prompt: str, enable_thinking: Optional[bool] = None) -> torch.Tensor:
        use_chat = self.cfg.generation.use_chat_template
        if enable_thinking is None:
            enable_thinking = self.cfg.generation.use_thinking
        has_tmpl = getattr(self.tokenizer, "chat_template", None) is not None
        if use_chat and has_tmpl:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                return_tensors="pt",
            )["input_ids"].to(self.device)
        return self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)

    def _run_decode_loop(
        self,
        first_token: torch.Tensor,
        past_len: int,
        max_new_tokens: int,
        sampler: Sampler,
        eos_ids: set,
        pbar=None,
    ) -> Iterator[tuple]:
        """Shared decode loop. Yields (next_token_tensor, token_id, is_finished, finish_reason).

        Caller is responsible for prefill and CUDA graph pre-capture before calling this.
        """
        if pbar is None:
            pbar = _NoOpProgress()

        next_token = first_token
        pbar.start_decode()
        for step in range(max_new_tokens):
            logits = self.run_decode(next_token, past_len)
            next_token = sampler.sample(logits)
            token_id = next_token.item()
            past_len += 1
            pbar.step_decode(token_id)
            is_eos = token_id in eos_ids
            is_last = step == max_new_tokens - 1
            is_finished = is_eos or is_last
            finish_reason = "eos" if is_eos else ("length" if is_last else None)
            yield next_token, token_id, is_finished, finish_reason
            if is_finished:
                break

    @torch.inference_mode()
    def generate_stream(
        self,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> Iterator[tuple]:
        """Thin wrapper: tokenise → prefill → _run_decode_loop → yield formatted tokens.

        Yields (token_id, text_delta, is_finished, finish_reason) per token.
        """
        input_ids = self._tokenize_prompt(prompt, sampling_params.enable_thinking)
        sampler = Sampler(
            sample_method="greedy" if sampling_params.temperature == 0.0 else "sample",
            temperature=sampling_params.temperature,
            top_k=sampling_params.top_k,
            top_p=sampling_params.top_p,
        )
        max_new_tokens = sampling_params.max_new_tokens
        eos_ids = self.eos_token_ids if sampling_params.stop_on_eos else set()
        prompt_len = input_ids.shape[1]

        # Prefill
        position_ids = torch.arange(prompt_len, device=self.device).unsqueeze(0)
        cu_seqlens_q = torch.tensor([0, prompt_len], dtype=torch.long, device=self.device)
        set_context(is_prefill=True, cache_len=0, cu_seqlens_q=cu_seqlens_q)
        logits = self.run(input_ids, position_ids)
        first_token = sampler.sample(logits)
        token_id = first_token.item()
        text_delta = self.tokenizer.decode([token_id], skip_special_tokens=True)
        is_eos = token_id in eos_ids
        yield token_id, text_delta, is_eos, ("eos" if is_eos else None)
        if is_eos:
            return

        # Decode
        for _, token_id, is_finished, finish_reason in self._run_decode_loop(
            first_token, prompt_len, max_new_tokens - 1, sampler, eos_ids
        ):
            text_delta = self.tokenizer.decode([token_id], skip_special_tokens=True)
            yield token_id, text_delta, is_finished, finish_reason
            if is_finished:
                break

    @torch.inference_mode()
    def run_decode(self, next_token: torch.Tensor, past_len: int) -> torch.Tensor:
        """Run a single decode step; uses CUDA Graph when enabled (on-demand capture)."""
        if not self.use_cuda_graph or self._graph_mgr is None:
            set_context(
                is_prefill=False,
                cache_len=past_len,
                cu_seqlens_q=self._cu_seqlens_q_decode,
            )
            decoder_input_ids = next_token.reshape(1, 1)
            decoder_position_ids = torch.tensor(
                [[past_len]], device=self.device, dtype=torch.long
            )
            return self.model(
                decoder_input_ids, decoder_position_ids, decode_position=past_len
            )

        if self._graph_mgr._bucket_size > 1:
            self._update_attention_masks(past_len)
        return self._graph_mgr.replay(next_token, past_len)

    @torch.inference_mode()
    def run(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.model(input_ids, position_ids, **kwargs)

    def _collect_eos_ids(self) -> set[int]:
        """Union EOS ids from model.config.eos_token_id and tokenizer.eos_token_id."""
        ids: set[int] = set()
        cfg_eos = getattr(getattr(self.model, "config", None), "eos_token_id", None)
        if isinstance(cfg_eos, (list, tuple)):
            ids.update(int(x) for x in cfg_eos if x is not None)
        elif cfg_eos is not None:
            ids.add(int(cfg_eos))
        tok_eos = getattr(self.tokenizer, "eos_token_id", None)
        if tok_eos is not None:
            ids.add(int(tok_eos))
        gen = getattr(self.model, "generation_config", None)
        gen_eos = getattr(gen, "eos_token_id", None) if gen is not None else None
        if isinstance(gen_eos, (list, tuple)):
            ids.update(int(x) for x in gen_eos if x is not None)
        elif gen_eos is not None:
            ids.add(int(gen_eos))
        return ids
