"""ModelRunner: orchestrates prefill, decode, CUDA graph, sampling, and verification."""

import os
import torch
import warnings

from utils.logger import logger
from engine.context import get_context, set_context
from engine.progress import InferenceProgress
from engine.sampler import Sampler

try:
    from utils.verifier import Verifier

    VERIFIER_AVAILABLE = True
except:
    VERIFIER_AVAILABLE = False
    warnings.warn("Verifier is not available, some features may not work.")
from typing import Optional
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
        self.check_correction = cfg.inference.check_correction
        self.use_profile = cfg.inference.use_profile
        self.use_kvcache = cfg.inference.use_kvcache
        self.profile_dir = cfg.path.profile_dir
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
                    logger.info("[CUDA Graph] Enabled for decode phase (exact per-step).")

        sampling_cfg = cfg.inference.sampling
        self.sampler = Sampler(
            sampling_cfg.sample_method,
            sampling_cfg.temperature,
            top_k=sampling_cfg.topk,
            top_p=sampling_cfg.topp,
        )

        if sampling_cfg.sample_method != "greedy":
            self.check_correction = False
            logger.warning(
                "Only greedy sampling method supports correction, so correction is disabled."
            )

        self.verifier = None
        self.verification_results = {}
        if self.check_correction:
            if not VERIFIER_AVAILABLE:
                raise RuntimeError("Verifier is not available.")
            if cfg.path.baseline_model_path is None:
                raise ValueError(
                    "baseline_model_path must be provided for verification"
                )
            self.verifier = Verifier(
                baseline_model_path=cfg.path.baseline_model_path,
                baseline_model_dtype=cfg.env.get_torch_dtype(),
                tokenizer=tokenizer,
                device=self.device,
            )

        self.prof = None
        if self.use_profile:
            schedule = torch.profiler.schedule(wait=0, warmup=2, active=20, repeat=1)
            self.prof = torch.profiler.profile(
                schedule=schedule,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    self.profile_dir
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            logger.info(
                f"[Profiler] Enabled (decode-only). Trace will be saved to {self.profile_dir}"
            )
        if not self.use_kvcache:
            logger.error(
                "KVCACHE is not enabled, please enable it for better performance"
            )
            assert self.use_kvcache == True

    def _setup_attention_bucket_mode(self, use_bucket: bool):
        """Configure all attention layers for CUDA Graph bucketing."""
        model = self.model
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            layers = getattr(model.model.language_model, "layers", None)
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = getattr(model.model, "layers", None)
        else:
            layers = None

        if layers is None:
            return

        for layer in layers:
            attn_module = None
            if hasattr(layer, "self_attn"):
                if hasattr(layer.self_attn, "attn"):
                    attn_module = layer.self_attn.attn
                else:
                    attn_module = layer.self_attn
            if attn_module is not None and hasattr(attn_module, "use_cuda_graph_bucket"):
                attn_module.use_cuda_graph_bucket = use_bucket

    def _update_attention_masks(self, past_len: int):
        """Update _write_pos and _attn_mask for all attention layers."""
        model = self.model
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            layers = getattr(model.model.language_model, "layers", None)
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = getattr(model.model, "layers", None)
        else:
            return

        seq_len = past_len + 1
        for layer in layers:
            attn_module = None
            if hasattr(layer, "self_attn"):
                if hasattr(layer.self_attn, "attn"):
                    attn_module = layer.self_attn.attn
                else:
                    attn_module = layer.self_attn
            if attn_module is None or not getattr(attn_module, "use_cuda_graph_bucket", False):
                continue
            if hasattr(attn_module, "_write_pos"):
                attn_module._write_pos[0] = past_len
            if hasattr(attn_module, "_attn_mask"):
                attn_module._attn_mask.fill_(float("-inf"))
                attn_module._attn_mask[..., :seq_len] = 0

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

        # Terminal image rendering removed — too low resolution for useful display.
        # Use `utils.terminal_image.print_image(image)` if you need it.

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
    def inference(self) -> str:
        max_new_tokens = self.max_new_tokens
        tokenizer = self.tokenizer
        input_ids = self.input_ids
        position_ids = self.position_ids
        prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        from contextlib import nullcontext

        with InferenceProgress(tokenizer, max_new_tokens, prompt_text) as pbar:
            with self.prof if self.prof else nullcontext():
                prompt_seq_len = input_ids.shape[1]
                past_len = 0
                generated_ids = []

                # ==========================================
                # 1. Prefill 阶段 & PPL 验证
                # ==========================================
                pbar.start_prefill()
                cu_seqlens_q_prefill = torch.tensor(
                    [0, prompt_seq_len], dtype=torch.long, device=self.device
                )
                set_context(
                    is_prefill=True,
                    cache_len=past_len,
                    cu_seqlens_q=cu_seqlens_q_prefill,
                )
                logits = self.run(input_ids, position_ids, **self._multimodal_kwargs)

                if self.check_correction and self.verifier is not None:
                    logger.info("[ModelRunner] 计算 baseline PPL...")
                    self.verifier.compute_baseline_ppl(self.prompt)
                    logger.info("[ModelRunner] 比对 baseline model PPL...")
                    if logits.shape[1] == input_ids.shape[1]:
                        ppl_result = self.verifier.verify_ppl(logits, input_ids)
                        self.verification_results["ppl"] = ppl_result
                    else:
                        logger.warning(
                            "模型在 Prefill 阶段仅返回了最后一个 token 的 logits, 无法计算 PPL。"
                        )

                next_token = self.sampler.sample(logits)

                generated_ids.append(next_token.item())
                past_len += prompt_seq_len
                pbar.end_prefill(next_token.item())
                # ==========================================
                # 2. Decode 阶段 & 贪婪解码验证
                # ==========================================
                if self.check_correction and self.verifier is not None:
                    logger.info("[ModelRunner] 生成 baseline 的 greedy decode 结果...")
                    self.verifier.generate_baseline_greedy(self.prompt, max_new_tokens)
                    logger.info("[ModelRunner] 开始逐 Token 验证 Decode...")

                current_tokens = 0
                decode_pass = False
                stopped_by_eos = next_token.item() in self.eos_token_ids

                if self.use_cuda_graph:
                    self._decode_input_ids.copy_(next_token.reshape(1, 1))
                    pbar.start_warmup(total=max_new_tokens + 2)
                    # Pre-capture graphs for models that support it.
                    # Models without pre-capture support fall back to on-demand
                    # capture in run_decode.
                    can_precapture = hasattr(
                        self.model, "_snapshot_cuda_graph_state"
                    )
                    if can_precapture:
                        self._capture_all_decode_graphs(past_len, max_new_tokens, pbar)
                    pbar.end_warmup()

                pbar.start_decode()
                ncu_decode = os.environ.get("MINI_VLLM_NCU_DECODE") == "1"
                if ncu_decode:
                    torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStart()
                while current_tokens < max_new_tokens and not stopped_by_eos:
                    logits = self.run_decode(next_token, past_len)

                    if self.prof:
                        self.prof.step()

                    next_token = self.sampler.sample(logits)
                    if (
                        self.check_correction
                        and self.verifier is not None
                        and not decode_pass
                    ):
                        is_match, details = self.verifier.verify_decode_step(
                            logits[:, -1, :], next_token.squeeze(), current_tokens + 1
                        )

                        if not is_match:
                            if details["max_prob_diff"] < 0.1:
                                logger.warning(
                                    f"\n[Warning] Step {current_tokens}: baseline 选 '{details['baseline_text']}'({details['prob_baseline_tok_in_baseline']:.4f}), "
                                    f"\ncustom 选 '{details['test_text']}'({details['prob_test_tok_in_test']:.4f})"
                                    f"\nTop-K 分布一致，视为浮点精度问题, Decode 验证通过！"
                                )

                                details["is_match"] = True
                                self.verification_results["decode_diverge"] = details
                                decode_pass = True
                            else:
                                logger.error(
                                    f"\n[fatal error] Step {current_tokens}: Token 严重发散！最大概率差: {details['max_prob_diff']:.4f}"
                                )
                                self.verification_results["decode_diverge"] = details
                                self.verifier.print_verification_report(
                                    self.verification_results
                                )
                                break
                    generated_ids.append(next_token.item())
                    past_len += 1
                    current_tokens += 1
                    pbar.step_decode(next_token.item())
                    if next_token.item() in self.eos_token_ids:
                        stopped_by_eos = True
                        break

                if ncu_decode:
                    torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStop()

                if (
                    (current_tokens == max_new_tokens or stopped_by_eos)
                    and self.check_correction
                    and self.verifier is not None
                ):
                    if "decode_diverge" not in self.verification_results:
                        self.verification_results["decode_diverge"] = {"is_match": True}
                    self.verifier.print_verification_report(self.verification_results)
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
        # Round up to next multiple of bucket_size.
        # cache_len 0 is kept as 0 (first decode step after prefill).
        if cache_len == 0:
            return 0
        return ((cache_len - 1) // self._cuda_graph_bucket_size + 1) * self._cuda_graph_bucket_size

    def _ensure_decode_graph(self, cache_len: int, restore_state: bool = True):
        """On-demand capture a decode graph for the given cache_len.

        This avoids the long upfront wait of pre-capturing all max_new_tokens graphs.
        The first call for a new cache_len triggers capture (~50-200ms); subsequent
        calls are instant replays.
        """
        bucket = self._cache_len_to_bucket(cache_len)
        if bucket in self._cuda_graphs:
            return

        # When bucketing, capture with the bucket upper bound so that the
        # same graph can be replayed for any cache_len within the bucket.
        capture_len = bucket if self._cuda_graph_bucket_size > 1 else cache_len

        if self._cuda_graph_bucket_size > 1:
            self._update_attention_masks(capture_len)

        set_context(
            is_prefill=False,
            cache_len=capture_len,
            cu_seqlens_q=self._cu_seqlens_q_decode,
        )
        self._decode_position_ids[0, 0] = capture_len

        # Snapshot state for models with non-idempotent decode state
        # (e.g. megakernel linear attention recurrent states)
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

        # Restore state so replay starts from the correct pre-step state
        if restore_state and state_snapshot is not None and hasattr(
            self.model, "_restore_cuda_graph_state"
        ):
            self.model._restore_cuda_graph_state(state_snapshot)

        self._cuda_graphs[bucket] = g
        self._cuda_graph_outputs[bucket] = static_logits

    def _capture_all_decode_graphs(self, start_cache_len: int, num_tokens: int, pbar):
        """Pre-capture all decode graphs before the decode loop starts."""
        end_cache_len = start_cache_len + num_tokens - 1
        start_bucket = self._cache_len_to_bucket(start_cache_len)
        end_bucket = self._cache_len_to_bucket(end_cache_len)
        # Count unique buckets in the range.
        if self._cuda_graph_bucket_size <= 1:
            unique_buckets = num_tokens
        else:
            unique_buckets = (end_bucket - start_bucket) // self._cuda_graph_bucket_size + 1

        logger.info(
            f"[CUDA Graph] Pre-capturing {unique_buckets} graphs "
            f"(cache_len {start_cache_len} ~ {end_cache_len}, bucket_size={self._cuda_graph_bucket_size}) ..."
        )

        # Use model-level snapshot if available (handles linear attention, etc.)
        has_custom_snapshot = hasattr(self.model, "_snapshot_cuda_graph_state")
        state_snapshot = (
            self.model._snapshot_cuda_graph_state() if has_custom_snapshot else None
        )

        prof_was_running = False
        if self.prof is not None:
            self.prof.stop()
            prof_was_running = True
            logger.info("[CUDA Graph] Profiler paused for capture.")

        try:
            for i in range(num_tokens):
                cache_len = start_cache_len + i
                self._ensure_decode_graph(cache_len, restore_state=False)
                pbar.step_warmup(1)
        finally:
            if has_custom_snapshot and hasattr(
                self.model, "_restore_cuda_graph_state"
            ):
                self.model._restore_cuda_graph_state(state_snapshot)
            logger.info("[CUDA Graph] State restored to pre-capture state.")

            if prof_was_running:
                self.prof.start()
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
