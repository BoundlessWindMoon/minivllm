"""lm-evaluation-harness wrapper for mini-vllm models.

Usage:
    lm_eval --model minivllm \
            --model_args config=configs/qwen3_5.yaml \
            --tasks mmlu,gsm8k \
            --batch_size 1
"""

import torch
from lm_eval.api.model import TemplateLM
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model

from utils.config import GlobalConfig
from engine.loader import load_model
from engine.runtime_setup import apply_runtime_patches
from engine.processor import load_processor
from engine.context import set_context


@register_model("minivllm")
class MiniVLLM(TemplateLM):
    """lm-eval TemplateLM adapter for mini-vllm inference engine."""

    _DEFAULT_CONFIG = "configs/default.yaml"

    def __init__(self, config: str = _DEFAULT_CONFIG, device: str | None = None, **kwargs):
        super().__init__()
        self.cfg = GlobalConfig.from_yaml(config)
        if device is not None:
            self.cfg.env.device = device

        # Force default backend: megakernel does not return standard logits
        # and greedy_fast_path breaks loglikelihood scoring.
        self.cfg.inference.backend = "default"
        self.cfg.inference.use_cuda_graph = False

        torch.set_default_dtype(self.cfg.env.get_torch_dtype())
        torch.set_default_device(self.cfg.env.device)

        import os
        import torch.distributed as dist
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "29600")
            dist.init_process_group(
                "nccl" if torch.cuda.is_available() else "gloo",
                rank=0, world_size=1,
            )

        self.model, self.tokenizer = load_model(self.cfg)
        self.model = apply_runtime_patches(self.model, self.cfg)
        self.processor = load_processor(self.cfg)
        self.model.eval()
        # Disable any fast path that bypasses logits
        if hasattr(self.model, "greedy_fast_path"):
            self.model.greedy_fast_path = False
        self._device = self.cfg.env.device
        self._rank = 0
        self._world_size = 1

    # ------------------------------------------------------------------
    # TemplateLM contract
    # ------------------------------------------------------------------
    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self) -> int:
        return self.tokenizer.bos_token_id or self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        return getattr(
            self.model.config, "max_position_embeddings", 32768
        )

    @property
    def max_gen_toks(self) -> int:
        return self.cfg.inference.max_new_tokens

    @property
    def batch_size(self) -> int:
        return 1

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def tokenizer_name(self) -> str:
        return getattr(self.tokenizer, "name_or_path", "minivllm")

    def tok_encode(self, string: str, add_special_tokens: bool | None = None, **kwargs) -> list[int]:
        if add_special_tokens is None:
            add_special_tokens = False
        return self.tokenizer.encode(string, add_special_tokens=add_special_tokens)

    def tok_decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _reset_kv_cache(self) -> None:
        """Clear KV cache and linear attention state across all layers."""
        if hasattr(self.model, "reset"):
            self.model.reset()
            return
        # Fallback for models without reset()
        layers = getattr(getattr(self.model, "model", None), "layers", None)
        if layers is None:
            return
        for layer in layers:
            attn = getattr(getattr(layer, "self_attn", None), "attn", None)
            if attn is None:
                continue
            if hasattr(attn, "k_cache") and attn.k_cache is not None:
                attn.k_cache.zero_()
                attn.v_cache.zero_()

    def _prefill(self, input_ids: list[int]) -> torch.Tensor:
        """Run prefill and return logits (1, seq_len, vocab_size)."""
        ids = torch.tensor([input_ids], device=self._device, dtype=torch.long)
        pos = torch.arange(len(input_ids), device=self._device, dtype=torch.long).unsqueeze(0)
        cu = torch.tensor([0, len(input_ids)], device=self._device, dtype=torch.long)
        set_context(is_prefill=True, cache_len=0, cu_seqlens_q=cu)
        with torch.no_grad():
            return self.model(ids, pos)

    def _decode_step(self, token_id: int, past_len: int) -> torch.Tensor:
        """Run single decode step and return logits (1, 1, vocab_size)."""
        ids = torch.tensor([[token_id]], device=self._device, dtype=torch.long)
        pos = torch.tensor([[past_len]], device=self._device, dtype=torch.long)
        cu = torch.tensor([0, 1], device=self._device, dtype=torch.long)
        set_context(is_prefill=False, cache_len=past_len, cu_seqlens_q=cu)
        with torch.no_grad():
            return self.model(ids, pos)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        disable_tqdm: bool = False,
    ) -> list[tuple[float, bool]]:
        from tqdm import tqdm

        res: list[tuple[float, bool]] = []
        for (_, _), context_enc, continuation_enc in tqdm(requests, disable=disable_tqdm):
            self._reset_kv_cache()

            input_ids = context_enc + continuation_enc
            logits = self._prefill(input_ids)  # (1, seq_len, vocab_size)
            logits = logits[0]  # (seq_len, vocab_size)

            cont_len = len(continuation_enc)
            start_idx = 0 if len(context_enc) == 0 else len(context_enc) - 1

            cont_logits = logits[start_idx : start_idx + cont_len]
            logprobs = torch.log_softmax(cont_logits, dim=-1)

            cont_tokens = torch.tensor(continuation_enc, device=self._device)
            token_logprobs = logprobs.gather(1, cont_tokens.unsqueeze(-1)).squeeze(-1)
            total_logprob = token_logprobs.sum().item()

            greedy_tokens = cont_logits.argmax(dim=-1)
            is_greedy = bool((greedy_tokens == cont_tokens).all().item())

            res.append((total_logprob, is_greedy))

        return res

    def loglikelihood_rolling(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[float]:
        """Rolling-window log-likelihood for perplexity tasks."""
        from tqdm import tqdm

        res: list[float] = []
        for req in tqdm(requests, disable=disable_tqdm):
            (string,) = req.args
            self._reset_kv_cache()

            token_ids = self.tok_encode(string, add_special_tokens=False)
            max_len = self.max_length

            if len(token_ids) <= max_len:
                logits = self._prefill(token_ids)
                logprobs = torch.log_softmax(logits[0], dim=-1)
                targets = torch.tensor(token_ids, device=self._device)
                total = logprobs[:-1].gather(1, targets[1:].unsqueeze(-1)).sum().item()
                res.append(total)
                continue

            # Chunked rolling
            total_logprob = 0.0
            for i in range(0, len(token_ids), max_len):
                self._reset_kv_cache()
                chunk = token_ids[i : i + max_len]
                logits = self._prefill(chunk)
                logprobs = torch.log_softmax(logits[0], dim=-1)
                targets = torch.tensor(chunk, device=self._device)
                # Score all but first token (conditioned on context within chunk)
                total_logprob += logprobs[:-1].gather(1, targets[1:].unsqueeze(-1)).sum().item()

            res.append(total_logprob)

        return res

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate_until(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[str]:
        from tqdm import tqdm

        res: list[str] = []
        for req in tqdm(requests, disable=disable_tqdm):
            context, gen_kwargs = req.args
            self._reset_kv_cache()

            input_ids = self.tok_encode(context)
            logits = self._prefill(input_ids)

            until = gen_kwargs.get("until", [])
            if isinstance(until, str):
                until = [until]
            max_tokens = gen_kwargs.get("max_gen_toks", gen_kwargs.get("max_length", self.max_gen_toks))
            temperature = gen_kwargs.get("temperature", 1.0)
            top_p = gen_kwargs.get("top_p", 1.0)

            generated: list[int] = []
            past_len = len(input_ids)
            next_token = logits[0, -1, :].argmax().item()
            generated.append(next_token)

            for _ in range(max_tokens - 1):
                logits = self._decode_step(next_token, past_len)
                next_token = logits[0, -1, :].argmax().item()
                generated.append(next_token)
                past_len += 1

                text = self.tok_decode(generated)
                if any(text.endswith(u) for u in until):
                    break

            res.append(self.tok_decode(generated))

        return res
