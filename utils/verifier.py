"""Numerical correctness verifier: PPL and greedy-decode match."""

import os
import json
import math
import hashlib
import torch
from utils.logger import logger
from typing import Dict, Any, Optional, Tuple
from safetensors.torch import save_file, load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import warnings

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "tmp", "baseline_cache"
)
os.makedirs(CACHE_DIR, exist_ok=True)


class Verifier:

    def __init__(
        self,
        baseline_model_path: str,
        tokenizer: Any,
        device: str = "cuda:0",
        baseline_model_dtype: torch.dtype = torch.bfloat16,
    ):
        self.baseline_model_path = baseline_model_path
        self.baseline_model_dtype = baseline_model_dtype
        self.tokenizer = tokenizer
        self.device = device

        self.config = AutoConfig.from_pretrained(baseline_model_path)

        self.baseline_cache = {}
        self.baseline_meta = {}

    def _get_cache_hash(self, text: str, suffix: str = "") -> str:
        """生成缓存哈希键"""
        components = [
            self.baseline_model_path,
            text,
            str(self.baseline_model_dtype),
            suffix,
        ]
        hash_str = "_".join(components)
        return hashlib.md5(hash_str.encode()).hexdigest()

    def _save_to_cache(
        self, cache_key: str, data: Dict[str, torch.Tensor], meta: Dict[str, Any]
    ):
        cache_path = os.path.join(CACHE_DIR, f"{cache_key}.safetensors")
        meta_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
        cpu_data = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in data.items()
        }
        save_file(cpu_data, cache_path)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    def _load_from_cache(
        self, cache_key: str
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[Dict[str, Any]]]:
        cache_path = os.path.join(CACHE_DIR, f"{cache_key}.safetensors")
        meta_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
        if os.path.exists(cache_path) and os.path.exists(meta_path):
            try:
                data = load_file(cache_path)
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                return data, meta
            except Exception as e:
                warnings.warn(f"加载缓存失败: {e}")
        return None, None

    @staticmethod
    def calculate_ppl(
        logits: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[float, torch.Tensor]:
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = labels[:, 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        avg_loss = per_token_loss.mean().item()
        ppl = math.exp(avg_loss)
        return ppl, per_token_loss

    def compute_baseline_ppl(self, text: str) -> Dict[str, Any]:
        cache_key = self._get_cache_hash(text, suffix="ppl")
        cached_data, cached_meta = self._load_from_cache(cache_key)
        if cached_data is not None:
            logger.info(f"[Verifier] 加载 PPL baseline cache: {cache_key}")
            self.baseline_cache = cached_data
            self.baseline_meta = cached_meta
            return self.baseline_cache

        logger.info("[Verifier] 计算 HuggingFace baseline PPL...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            self.baseline_model_path,
            torch_dtype=self.baseline_model_dtype,
            device_map="auto" if self.device.startswith("cuda") else None,
        ).eval()

        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        with torch.inference_mode():
            outputs = hf_model(input_ids)
            logits = outputs.logits

        baseline_ppl, per_token_loss = self.calculate_ppl(logits, input_ids)
        self.baseline_cache = {
            "input_ids": input_ids.cpu(),
            "per_token_loss": per_token_loss.cpu(),
        }
        self.baseline_meta = {
            "model_path": self.baseline_model_path,
            "text_length": input_ids.shape[1],
            "baseline_ppl": baseline_ppl,
            "hash": cache_key,
        }
        self._save_to_cache(cache_key, self.baseline_cache, self.baseline_meta)
        del hf_model
        torch.cuda.empty_cache()
        return self.baseline_cache

    def verify_ppl(
        self, test_logits: torch.Tensor, input_ids: torch.Tensor
    ) -> Dict[str, Any]:
        if "per_token_loss" not in self.baseline_cache:
            raise ValueError("请先调用 compute_baseline_ppl() 生成基线")
        test_ppl, test_per_token_loss = self.calculate_ppl(test_logits, input_ids)
        baseline_ppl = self.baseline_meta["baseline_ppl"]
        baseline_per_token_loss = self.baseline_cache["per_token_loss"].to(
            test_per_token_loss.device
        )
        loss_mae = (baseline_per_token_loss - test_per_token_loss).abs().mean().item()
        ppl_diff_pct = abs(baseline_ppl - test_ppl) / baseline_ppl * 100
        is_close = ppl_diff_pct < 5
        return {
            "baseline_ppl": baseline_ppl,
            "test_ppl": test_ppl,
            "ppl_diff_pct": ppl_diff_pct,
            "loss_mae": loss_mae,
            "is_close": is_close,
        }

    def generate_baseline_greedy(
        self, prompt: str, max_new_tokens: int
    ) -> Dict[str, Any]:

        cache_key = self._get_cache_hash(prompt, suffix=f"greedy_{max_new_tokens}")
        cached_data, cached_meta = self._load_from_cache(cache_key)
        if cached_data is not None:
            logger.info(f"[Verifier] 加载 baseline greedy decode cache: {cache_key}")
            self.greedy_cache = cached_data
            return self.greedy_cache

        logger.warning("[Verifier] 生成 HuggingFace greedy decode cache...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            self.baseline_model_path,
            torch_dtype=self.baseline_model_dtype,
            device_map="cuda:0" if self.device.startswith("cuda") else None,
        ).eval()

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        generated_tokens = []
        step_logits = []

        current_ids = input_ids
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                outputs = hf_model(current_ids)
                logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1)

                generated_tokens.append(next_token.item())
                step_logits.append(logits.squeeze(0).cpu())

                current_ids = torch.cat([current_ids, next_token.unsqueeze(-1)], dim=-1)

        self.greedy_cache = {
            "greedy_tokens": torch.tensor(generated_tokens, dtype=torch.long),
            "greedy_logits": torch.stack(step_logits, dim=0),
        }
        self._save_to_cache(cache_key, self.greedy_cache, {})
        del hf_model
        torch.cuda.empty_cache()
        return self.greedy_cache

    def verify_decode_step(
        self,
        test_logits: torch.Tensor,
        test_token: torch.Tensor,
        step_idx: int,
        tolerance: float = 0.1,
    ) -> Tuple[bool, Dict[str, Any]]:

        if not hasattr(self, "greedy_cache"):
            raise ValueError("请先调用 generate_baseline_greedy() 生成基线")
        baseline_token = self.greedy_cache["greedy_tokens"][step_idx].to(
            test_token.device
        )
        baseline_logits = self.greedy_cache["greedy_logits"][step_idx].to(
            test_logits.device
        )

        test_logits_sq = test_logits.squeeze()  # (vocab_size,)
        is_match = test_token.item() == baseline_token.item()
        details = {
            "step": step_idx,
            "is_match": is_match,
            "baseline_token_id": baseline_token.item(),
            "test_token_id": test_token.item(),
            "baseline_text": self.tokenizer.decode(baseline_token),
            "test_text": self.tokenizer.decode(test_token),
            "is_critical_diverge": False,
        }
        if not is_match:
            baseline_probs = torch.softmax(baseline_logits.float(), dim=-1)
            test_probs = torch.softmax(test_logits_sq.float(), dim=-1)
            max_prob_diff = (baseline_probs - test_probs).abs().max().item()
            details["max_prob_diff"] = max_prob_diff

            b_tok = baseline_token.item()
            t_tok = test_token.item()
            details["prob_baseline_tok_in_baseline"] = baseline_probs[b_tok].item()
            details["prob_baseline_tok_in_test"] = test_probs[b_tok].item()
            details["prob_test_tok_in_baseline"] = baseline_probs[t_tok].item()
            details["prob_test_tok_in_test"] = test_probs[t_tok].item()

            prob_drop = baseline_probs[b_tok].item() - test_probs[b_tok].item()
            if prob_drop < tolerance and max_prob_diff < tolerance:
                details["is_critical_diverge"] = False
            else:
                details["is_critical_diverge"] = True

            def get_top5(probs):
                values, indices = torch.topk(probs, 5)
                return [
                    (self.tokenizer.decode(idx), f"{prob:.4f}")
                    for idx, prob in zip(indices.tolist(), values.tolist())
                ]

            details["baseline_top5"] = get_top5(baseline_probs)
            details["test_top5"] = get_top5(test_probs)
        return is_match, details

    def print_verification_report(self, verification_results: Dict[str, Any]):

        if "ppl" in verification_results:
            ppl_res = verification_results["ppl"]
            status = "✓ PASS" if ppl_res["is_close"] else "✗ FAIL"
            ("[Prefill 验证] PPL 差异:")
            logger.success(
                f"\n状态: {status}"
                f"  HuggingFace baseline PPL: {ppl_res['baseline_ppl']:.4f}"
                f"  custom model PPL:        {ppl_res['test_ppl']:.4f}"
                f"  PPL 相对差异:        {ppl_res['ppl_diff_pct']:.4f}%"
            )

        if "decode_diverge" in verification_results:
            div_res = verification_results["decode_diverge"]
            if div_res["is_match"]:
                logger.success("\n[Decode 验证] greedy decode 完全一致: ✓ PASS")
            else:
                logger.error(
                    f"\n[Decode 验证] 检测到发散: ✗ FAIL"
                    f"  首次发散步数:     {div_res['step']}"
                    f"  baseline model Token:       '{div_res['baseline_text']}' (ID: {div_res['baseline_token_id']})"
                    f"  custom model Token:       '{div_res['test_text']}' (ID: {div_res['test_token_id']})"
                    f"  最大概率差:       {div_res['max_prob_diff']:.6f}"
                    f"  --- 基线 Top-5 ---"
                    f"  {div_res['baseline_top5']}"
                    f"  --- 测试 Top-5 ---"
                    f"  {div_res['test_top5']}"
                )
