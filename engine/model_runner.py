import torch
import warnings

from utils.logger import logger
from utils.context import get_context, set_context
from utils.progress import InferenceProgress
from utils.sampler import Sampler

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
        # max_new_tokens,
        # prompt="",
        # check_correction=False,
        # use_profile=False,
        # use_kvcache=True,
        # use_progress=True,
        # sample_method: Optional[str] = "greedy",
        # temperature: Optional[float] = 1.0,
        # topk: int = 1,
        # topp: float = 1.0,
        # baseline_model_path: Optional[str] = None,
        # baseline_model_dtype: Optional[torch.dtype] = torch.bfloat16,
        # device="cuda:0",
        # profile_dir: str = "./log/profile/",
        cfg: GlobalConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg

        self.device = cfg.env.device
        self.max_new_tokens = cfg.inference.max_new_tokens
        self.check_correction = cfg.inference.check_correction
        self.use_profile = cfg.inference.use_profile
        self.use_kvcache = cfg.inference.use_kvcache
        self.profile_dir = cfg.path.profile_dir

        self.use_progress = True

        self.prompt = cfg.inference.prompt
        inputs = self.tokenizer(self.prompt, return_tensors="pt").to(self.device)
        self.input_ids = inputs["input_ids"].to(self.device)
        self.position_ids = torch.arange(
            self.input_ids.shape[1], device=self.device
        ).unsqueeze(0)

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
            schedule = torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=2)
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
                f"[Profiler] Enabled. Trace will be saved to {self.profile_dir}"
            )
        if not self.use_kvcache:
            logger.error(
                "KVCACHE is not enabled, please enable it for better performance"
            )
            assert self.use_kvcache == True

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
                logits = self.run(input_ids, position_ids)

                if self.prof:
                    self.prof.step()

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

                pbar.start_decode()
                current_tokens = 0
                decode_pass = False
                cu_seqlens_q_decode = torch.tensor(
                    [0, 1], dtype=torch.long, device=self.device
                )
                while current_tokens < max_new_tokens:
                    set_context(
                        is_prefill=False,
                        cache_len=past_len,
                        cu_seqlens_q=cu_seqlens_q_decode,
                    )

                    decoder_input_ids = next_token
                    decoder_position_ids = torch.tensor(
                        [[past_len]], device=self.device, dtype=torch.long
                    )

                    logits = self.run(decoder_input_ids, decoder_position_ids)

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

                if (
                    current_tokens == max_new_tokens
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

    @torch.inference_mode()
    def run(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, position_ids)

    def post_process(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        new_input_ids = torch.cat([input_ids, next_token], dim=-1)
        new_position_ids = position_ids[:, -1:] + 1
        new_position_ids = torch.cat([position_ids, new_position_ids], dim=-1)
        return new_input_ids, new_position_ids
