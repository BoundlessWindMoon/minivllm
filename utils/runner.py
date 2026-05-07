"""Shared model runners and environment setup for evaluation scripts.

Provides:
  - setup_env(config_path=None) → init distributed, device, dtype
  - load_eval_model(config_or_path, backend="default") → model instance
  - BaselineRunner / MegakernelRunner — thin wrappers with unified API
"""

import os
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoConfig

from utils.config import GlobalConfig
from utils.model_loader import ModelLoader
from utils.context import set_context
from model.qwen3 import Qwen3ForCausalLM
from model.qwen3_megakernel import Qwen3MegakernelForCausalLM


DEFAULT_PORT = "29599"


def setup_env(config_path: str | None = None):
    """Initialize distributed, device, and default dtype.

    If *config_path* is provided, the config file drives device / dtype /
    distributed settings.  Otherwise sensible defaults are used.
    """
    if config_path is not None:
        cfg = GlobalConfig.from_yaml(config_path)
        torch.set_default_dtype(cfg.env.get_torch_dtype())
        torch.set_default_device(cfg.env.device)
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", cfg.env.distributed.init_method.split(":")[-1])
        if not dist.is_initialized():
            dist.init_process_group(
                backend=cfg.env.distributed.backend if torch.cuda.is_available() else "gloo",
                init_method=cfg.env.distributed.init_method,
                world_size=cfg.env.distributed.world_size,
                rank=cfg.env.distributed.rank,
            )
        return cfg

    # Defaults for standalone scripts
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", DEFAULT_PORT)
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=0, world_size=1)
    torch.set_default_device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    return None


def load_eval_model(config_or_path: str | GlobalConfig | None = None, backend: str = "default"):
    """Load a model for evaluation.

    Args:
        config_or_path: Path to YAML config, a GlobalConfig instance, or None.
                        When None, loads from ``configs/default.yaml``.
        backend:        ``"default"`` | ``"megakernel_cuda"``

    Returns:
        nn.Module with eval() already called.
    """
    if config_or_path is None:
        config_or_path = "configs/default.yaml"

    if isinstance(config_or_path, str):
        cfg = GlobalConfig.from_yaml(config_or_path)
    else:
        cfg = config_or_path

    data_path = cfg.path.data_path or cfg.path.model_path
    config = AutoConfig.from_pretrained(cfg.path.model_path)
    config.use_sdpa = cfg.inference.use_sdpa
    device = cfg.env.device

    loader = ModelLoader(data_path)
    model_skeleton = Qwen3ForCausalLM(config).to(device)
    model = loader.inject_data(model_skeleton)

    if backend == "megakernel_cuda":
        model = Qwen3MegakernelForCausalLM.from_model(model)

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Unified runners
# ---------------------------------------------------------------------------

class _BaseRunner:
    """Common interface for baseline and megakernel runners."""

    def reset(self):
        raise NotImplementedError

    def prefill(self, input_ids: list[int]) -> torch.Tensor:
        raise NotImplementedError

    def decode_step(self, next_token: int, past_len: int) -> torch.Tensor:
        raise NotImplementedError


class BaselineRunner(_BaseRunner):
    """Wraps Qwen3ForCausalLM with optional CUDA Graph for fair benchmarking."""

    def __init__(self, model, use_cuda_graph: bool = True):
        self.model = model
        self.device = next(model.parameters()).device
        self.use_cuda_graph = use_cuda_graph and torch.cuda.is_available()
        self._cuda_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._cuda_graph_outputs: dict[int, torch.Tensor] = {}
        self._decode_input_ids = None
        self._decode_position_ids = None
        self._cu_seqlens_q_decode = torch.tensor(
            [0, 1], dtype=torch.long, device=self.device
        )

        if self.use_cuda_graph:
            self._decode_input_ids = torch.zeros(
                (1, 1), device=self.device, dtype=torch.long
            )
            self._decode_position_ids = torch.zeros(
                (1, 1), device=self.device, dtype=torch.long
            )

    def _ensure_decode_graph(self, cache_len: int):
        """On-demand CUDA Graph capture for a given cache_len."""
        if cache_len in self._cuda_graphs:
            return

        set_context(
            is_prefill=False,
            cache_len=cache_len,
            cu_seqlens_q=self._cu_seqlens_q_decode,
        )
        self._decode_position_ids[0, 0] = cache_len

        warmup = 3 if len(self._cuda_graphs) == 0 else 1
        for _ in range(warmup):
            _ = self.model(self._decode_input_ids, self._decode_position_ids)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_logits = self.model(
                self._decode_input_ids, self._decode_position_ids
            )

        self._cuda_graphs[cache_len] = g
        self._cuda_graph_outputs[cache_len] = static_logits

    def reset(self):
        # NOTE: keep CUDA graphs alive across runs so timed loops only
        # measure replay, not re-capture.
        set_context(is_prefill=False, cache_len=0)

    def prefill(self, input_ids: list[int]) -> torch.Tensor:
        prompt_len = len(input_ids)
        input_ids_t = torch.tensor([input_ids], device=self.device, dtype=torch.long)
        position_ids = torch.arange(0, prompt_len, device=self.device).unsqueeze(0)
        set_context(is_prefill=True, cache_len=0)
        with torch.no_grad():
            return self.model(input_ids_t, position_ids)

    @torch.inference_mode()
    def decode_step(self, next_token: int, past_len: int) -> torch.Tensor:
        if not self.use_cuda_graph:
            set_context(
                is_prefill=False,
                cache_len=past_len,
                cu_seqlens_q=self._cu_seqlens_q_decode,
            )
            decoder_input_ids = torch.tensor(
                [[next_token]], device=self.device, dtype=torch.long
            )
            decoder_position_ids = torch.tensor(
                [[past_len]], device=self.device, dtype=torch.long
            )
            return self.model(decoder_input_ids, decoder_position_ids)

        self._decode_input_ids[0, 0] = next_token
        self._decode_position_ids[0, 0] = past_len
        self._ensure_decode_graph(past_len)
        self._cuda_graphs[past_len].replay()
        return self._cuda_graph_outputs[past_len]


class MegakernelRunner(_BaseRunner):
    """Thin wrapper around Qwen3MegakernelForCausalLM."""

    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device

    def reset(self):
        self.model.reset()

    def prefill(self, input_ids: list[int]) -> torch.Tensor:
        input_ids_t = torch.tensor([input_ids], device=self.device, dtype=torch.long)
        position_ids = torch.arange(0, len(input_ids), device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.model(input_ids_t, position_ids)

    def decode_step(self, next_token: int, past_len: int) -> torch.Tensor:
        decoder_input_ids = torch.tensor(
            [[next_token]], device=self.device, dtype=torch.long
        )
        decoder_position_ids = torch.tensor(
            [[past_len]], device=self.device, dtype=torch.long
        )
        with torch.no_grad():
            return self.model(decoder_input_ids, decoder_position_ids)
