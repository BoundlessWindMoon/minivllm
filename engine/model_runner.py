"""ModelRunner: orchestrates prefill, decode, CUDA graph, and sampling."""

import os
import torch

from utils.logger import logger
from engine.context import get_context, set_context
from engine.progress import InferenceProgress, _NoOpProgress
from engine.profiler import build_profiler
from engine.sampler import Sampler
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
        self.max_new_tokens = cfg.inference.max_new_tokens
        self.use_kvcache = cfg.inference.use_kvcache
        self.stop_on_eos = getattr(cfg.inference, "stop_on_eos", True)
        self.eos_token_ids = self._collect_eos_ids() if self.stop_on_eos else set()

        self.use_progress = True

        self.prompt = cfg.inference.prompt
        use_chat = getattr(cfg.inference, "use_chat_template", False)
        use_thinking = getattr(cfg.inference, "use_thinking", True)
        has_tmpl = getattr(self.tokenizer, "chat_template", None) is not None

        multimodal_cfg = getattr(cfg.inference, "multimodal", None)
        self._multimodal_enabled = (
            multimodal_cfg is not None
            and multimodal_cfg.enabled
            and processor is not None
        )
        self._multimodal_kwargs = {}

        if self._multimodal_enabled:
            logger.info("Multimodal inference enabled.")
            self._prepare_multimodal_inputs(multimodal_cfg, use_thinking)
        else:
            if use_chat and has_tmpl:
                chat_inputs = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": self.prompt}],
                    add_generation_prompt=True,
                    enable_thinking=use_thinking,
                    return_tensors="pt",
                )
                self.input_ids = chat_inputs["input_ids"].to(self.device)
            else:
                if use_chat and not has_tmpl:
                    logger.warning(
                        "use_chat_template=True but tokenizer has no chat_template; "
                        "falling back to raw tokenization."
                    )
                inputs = self.tokenizer(self.prompt, return_tensors="pt").to(
                    self.device
                )
                self.input_ids = inputs["input_ids"].to(self.device)
            self.position_ids = torch.arange(
                self.input_ids.shape[1], device=self.device
            ).unsqueeze(0)

        self.use_cuda_graph = getattr(cfg.inference, "use_cuda_graph", False)
        self._cuda_graph_bucket_size = getattr(
            cfg.inference, "cuda_graph_bucket_size", 1
        )
        if self._cuda_graph_bucket_size < 1:
            self._cuda_graph_bucket_size = 1
        self._setup_attention_bucket_mode(
            self.use_cuda_graph and self._cuda_graph_bucket_size > 1
        )
        self._cuda_graphs = {}
        self._cuda_graph_outputs = {}
        self._decode_input_ids = None
        self._decode_position_ids = None
        self._cu_seqlens_q_decode = torch.tensor(
            [0, 1], dtype=torch.long, device=self.device
        )

        if self.use_cuda_graph and not getattr(model, "supports_cuda_graph", True):
            logger.warning(
                f"{type(model).__name__} does not support CUDA Graph, disabling."
            )
            self.use_cuda_graph = False

        if self.use_cuda_graph:
            if not torch.cuda.is_available():
                logger.warning(
                    "CUDA Graph requested but CUDA not available, disabling."
                )
                self.use_cuda_graph = False
            else:
                self._decode_input_ids = torch.zeros(
                    (1, 1), device=self.device, dtype=torch.long
                )
                self._decode_position_ids = torch.zeros(
                    (1, 1), device=self.device, dtype=torch.long
                )
                self._cuda_graph_pool = torch.cuda.graph_pool_handle()
                if self._cuda_graph_bucket_size > 1:
                    logger.info(
                        f"[CUDA Graph] Enabled for decode phase with "
                        f"bucket_size={self._cuda_graph_bucket_size}. "
                        f"Note: bucketing requires kernel support for padded lengths."
                    )
                else:
                    logger.info(
                        "[CUDA Graph] Enabled for decode phase (exact per-step)."
                    )

        sampling_cfg = cfg.inference.sampling
        self.sampler = Sampler(
            sampling_cfg.sample_method,
            sampling_cfg.temperature,
            top_k=sampling_cfg.topk,
            top_p=sampling_cfg.topp,
        )

        self.profiler = build_profiler(cfg)
        if not self.use_kvcache:
            logger.error(
                "KVCACHE is not enabled, please enable it for better performance"
            )
            assert self.use_kvcache == True

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
        current_tokens = 0
        stopped_by_eos = next_token.item() in self.eos_token_ids

        if self.use_cuda_graph:
            self._decode_input_ids.copy_(next_token.reshape(1, 1))

            can_precapture = hasattr(self.model, "_snapshot_cuda_graph_state")
            if can_precapture:
                pbar.start_warmup(total=max_new_tokens + 2)
                self._capture_all_decode_graphs(past_len, max_new_tokens, pbar)
                pbar.end_warmup()

        pbar.start_decode()
        while current_tokens < max_new_tokens and not stopped_by_eos:
            logits = self.run_decode(next_token, past_len)
            next_token = self.sampler.sample(logits)
            generated_ids.append(next_token.item())
            past_len += 1
            current_tokens += 1
            pbar.step_decode(next_token.item())
            if next_token.item() in self.eos_token_ids:
                stopped_by_eos = True
                break

        return torch.tensor([generated_ids], device=self.device, dtype=torch.long)

    @torch.inference_mode()
    def inference(self) -> str:
        max_new_tokens = self.max_new_tokens
        tokenizer = self.tokenizer
        input_ids = self.input_ids
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

    def _cache_len_to_bucket(self, cache_len: int) -> int:
        """Map a cache_len to its bucket key for CUDA Graph sharing.

        bucket_size=1  -> exact per-step (default, safe for all models).
        bucket_size>1  -> round up to next multiple; reduces graph count
        but requires the underlying kernels to handle padded lengths.
        """
        if self._cuda_graph_bucket_size <= 1:
            return cache_len

        if cache_len == 0:
            return 0
        return (
            (cache_len - 1) // self._cuda_graph_bucket_size + 1
        ) * self._cuda_graph_bucket_size

    def _ensure_decode_graph(self, cache_len: int, restore_state: bool = True):
        """On-demand capture a decode graph for the given cache_len.

        This avoids the long upfront wait of pre-capturing all max_new_tokens graphs.
        The first call for a new cache_len triggers capture (~50-200ms); subsequent
        calls are instant replays.
        """
        bucket = self._cache_len_to_bucket(cache_len)
        if bucket in self._cuda_graphs:
            return

        capture_len = bucket if self._cuda_graph_bucket_size > 1 else cache_len

        set_context(
            is_prefill=False,
            cache_len=capture_len,
            cu_seqlens_q=self._cu_seqlens_q_decode,
        )
        self._decode_position_ids[0, 0] = capture_len

        state_snapshot = None
        if restore_state and hasattr(self.model, "_snapshot_cuda_graph_state"):
            state_snapshot = self.model._snapshot_cuda_graph_state()

        warmup_iters = 3 if len(self._cuda_graphs) == 0 else 1
        for _ in range(warmup_iters):
            _ = self.model(
                self._decode_input_ids,
                self._decode_position_ids,
                decode_position=capture_len,
            )
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self._cuda_graph_pool):
            static_logits = self.model(
                self._decode_input_ids,
                self._decode_position_ids,
                decode_position=capture_len,
            )

        if (
            restore_state
            and state_snapshot is not None
            and hasattr(self.model, "_restore_cuda_graph_state")
        ):
            self.model._restore_cuda_graph_state(state_snapshot)

        self._cuda_graphs[bucket] = g
        self._cuda_graph_outputs[bucket] = static_logits

    def _capture_all_decode_graphs(self, start_cache_len: int, num_tokens: int, pbar):
        """Pre-capture all decode graphs before the decode loop starts."""
        end_cache_len = start_cache_len + num_tokens - 1
        start_bucket = self._cache_len_to_bucket(start_cache_len)
        end_bucket = self._cache_len_to_bucket(end_cache_len)
        if self._cuda_graph_bucket_size <= 1:
            unique_buckets = num_tokens
        else:
            unique_buckets = (
                end_bucket - start_bucket
            ) // self._cuda_graph_bucket_size + 1

        logger.info(
            f"[CUDA Graph] Pre-capturing {unique_buckets} graphs "
            f"(cache_len {start_cache_len} ~ {end_cache_len}, bucket_size={self._cuda_graph_bucket_size}) ..."
        )

        has_custom_snapshot = hasattr(self.model, "_snapshot_cuda_graph_state")
        state_snapshot = (
            self.model._snapshot_cuda_graph_state() if has_custom_snapshot else None
        )

        self.profiler.pause()
        logger.info("[CUDA Graph] Profiler paused for capture.")

        try:
            for i in range(num_tokens):
                cache_len = start_cache_len + i
                self._ensure_decode_graph(cache_len, restore_state=False)
                pbar.step_warmup(1)
        finally:
            if has_custom_snapshot and hasattr(self.model, "_restore_cuda_graph_state"):
                self.model._restore_cuda_graph_state(state_snapshot)
            logger.info("[CUDA Graph] State restored to pre-capture state.")

            self.profiler.resume()
            logger.info("[CUDA Graph] Profiler resumed after capture.")

        logger.info(f"[CUDA Graph] Pre-captured {len(self._cuda_graphs)} graphs.")

    @torch.inference_mode()
    def run_decode(self, next_token: torch.Tensor, past_len: int) -> torch.Tensor:
        """Run a single decode step; uses CUDA Graph when enabled (on-demand capture)."""
        if not self.use_cuda_graph:
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

        cache_len = past_len
        bucket = self._cache_len_to_bucket(cache_len)

        if self._cuda_graph_bucket_size > 1:
            self._update_attention_masks(past_len)

        self._decode_input_ids.copy_(next_token.reshape(1, 1))
        self._decode_position_ids[0, 0] = past_len
        self._ensure_decode_graph(cache_len)
        self._cuda_graphs[bucket].replay()
        return self._cuda_graph_outputs[bucket]

    @torch.inference_mode()
    def run(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.model(input_ids, position_ids, **kwargs)

    def post_process(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        new_input_ids = torch.cat([input_ids, next_token], dim=-1)
        new_position_ids = position_ids[:, -1:] + 1
        new_position_ids = torch.cat([position_ids, new_position_ids], dim=-1)
        return new_input_ids, new_position_ids

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
