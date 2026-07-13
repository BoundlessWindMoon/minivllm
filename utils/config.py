"""YAML-backed nested dataclass configuration.

Config hierarchy
----------------
GlobalConfig
├── env         EnvironmentConfig   device, dtype, distributed
├── model       ModelConfig         backend, attention, cuda_graph, kv_cache
│   └── kv_cache  KVCacheConfig
├── generation  GenerationConfig    prompt, max_new_tokens, sampling, chat_template
├── batch       BatchConfig         num_slots, max_batch_size, prompts asset
│   └── prompts   PromptsConfig
├── path        PathConfig
├── quant       QuantConfig
└── profiling   ProfilingConfig

Config file inheritance
-----------------------
Any YAML can declare  ``_base: relative/path/to/base.yaml``
to inherit all keys from the base and override only what differs.
Nested dicts are merged recursively; scalar/list values are replaced wholesale.
"""

import os
import torch
import yaml
from dacite import from_dict
from dataclasses import dataclass, field
from typing import List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


# ---------------------------------------------------------------------------
# env
# ---------------------------------------------------------------------------

@dataclass
class DistributedConfig:
    backend: str = "nccl"
    init_method: str = "tcp://localhost:29500"
    world_size: int = 1
    rank: int = 0


@dataclass
class EnvironmentConfig:
    device: str = "cuda:0"
    default_dtype: str = "bfloat16"
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

    def get_torch_dtype(self):
        return getattr(torch, self.default_dtype)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

@dataclass
class KVCacheConfig:
    backend: str = "default"          # "default" | "kivi"
    max_len: int = 4096               # max sequence length per slot
    k_bits: int = 2
    v_bits: int = 2
    group_size: int = 32
    residual_length: int = 32


@dataclass
class ModelConfig:
    backend: str = "default"                      # "default" | "megakernel_cuda"
    megakernel_variant: str = "default"
    attention_backend: str = "sdpa"               # "sdpa" | "flash_attn" | "naive"
    use_cuda_graph: bool = False
    cuda_graph_bucket_size: int = 1
    use_quantized_model: bool = False
    use_kvcache: bool = True
    check_correction: bool = False
    cpu_offload_modules: List[str] = field(default_factory=list)
    linear_attn_prefill_backend: str = "torch"
    linear_attn_decode_backend: str = "fla"
    kv_cache: KVCacheConfig = field(default_factory=KVCacheConfig)


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

@dataclass
class SamplingConfig:
    sample_method: str = "greedy"
    temperature: float = 1.0
    topk: int = 1
    topp: float = 1.0


@dataclass
class MultimodalConfig:
    enabled: bool = False
    image_path: str = ""
    enable_thinking: bool = False


@dataclass
class GenerationConfig:
    prompt: str = "Hello, I am sakuya, I'm a 24 year old student from UCAS University."
    max_new_tokens: int = 128
    stop_on_eos: bool = True
    use_chat_template: bool = False
    use_thinking: bool = True
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    multimodal: MultimodalConfig = field(default_factory=MultimodalConfig)


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------

@dataclass
class PromptsConfig:
    asset: str = "assets/prompts/default.jsonl"
    random: bool = False
    seed: int = 42
    num_requests: int = 8              # random=True: sample N; False: take first N


@dataclass
class BatchConfig:
    num_slots: int = 8
    max_batch_size: int = 4
    # Admission policy: "fifo" | "spf" | "ljf" | "random"
    admission_policy: str = "fifo"
    # Total token budget per step (prefill + decode).  None = unlimited.
    # Setting this enables chunked prefill (vLLM-style).
    max_num_batched_tokens: int | None = None
    timeout_seconds: float | None = None
    prompts: PromptsConfig = field(default_factory=PromptsConfig)


# ---------------------------------------------------------------------------
# path
# ---------------------------------------------------------------------------

@dataclass
class PathConfig:
    model_path: str = "~/huggingface/Qwen3-0.6B/"
    data_path: str = ""
    profile_dir: str = "./log/profile/"
    quantized_model_path: str = ""
    baseline_model_path: str = ""

    def __post_init__(self):
        for attr in ("model_path", "data_path", "profile_dir",
                     "quantized_model_path", "baseline_model_path"):
            val = getattr(self, attr)
            if val:
                setattr(self, attr, os.path.expanduser(val))


# ---------------------------------------------------------------------------
# quant
# ---------------------------------------------------------------------------

@dataclass
class CalibConfig:
    data: str = "pileval"
    n_samples: int = 32
    max_seq_length: int = 512
    split: str = "train"
    text_column: str = "text"


@dataclass
class QuantConfig:
    quant_method: str = "AWQ"
    quant_bits: int = 4
    quant_targets: List[str] = field(default_factory=lambda: ["MLP", "ATTENTION"])
    group_size: int = 128
    has_zero_point: bool = True
    apply_clip: bool = True
    export_compatible: bool = False
    backend: str = "gemm"
    layout: str = "Wt"
    pack_order: str = "sequential"
    max_chunk_memory: int = 1024 * 1024 * 1024
    calibration: CalibConfig = field(default_factory=CalibConfig)

    def to_dict(self):
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# profiling
# ---------------------------------------------------------------------------

@dataclass
class TorchProfilerConfig:
    enabled: bool = False
    profile_dir: str = "./log/profile/"


@dataclass
class SwanLabConfig:
    enabled: bool = False
    project: str = "mini-vllm"
    experiment_name: str | None = None


@dataclass
class JSONProfilerConfig:
    enabled: bool = False
    path: str = "profile.json"


@dataclass
class ProfilingConfig:
    torch_profiler: TorchProfilerConfig = field(default_factory=TorchProfilerConfig)
    swanlab: SwanLabConfig = field(default_factory=SwanLabConfig)
    json: JSONProfilerConfig = field(default_factory=JSONProfilerConfig)


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

@dataclass
class GlobalConfig:
    env:        EnvironmentConfig = field(default_factory=EnvironmentConfig)
    model:      ModelConfig       = field(default_factory=ModelConfig)
    generation: GenerationConfig  = field(default_factory=GenerationConfig)
    batch:      BatchConfig       = field(default_factory=BatchConfig)
    path:       PathConfig        = field(default_factory=PathConfig)
    quant:      QuantConfig       = field(default_factory=QuantConfig)
    profiling:  ProfilingConfig   = field(default_factory=ProfilingConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str, model: str | None = None) -> "GlobalConfig":
        """Load config from *yaml_path*.

        Args:
            yaml_path:  Path to a run config (e.g. configs/runs/default.yaml).
            model:      Optional model name or path.  When given, any
                        ``configs/models/*.yaml`` file in the ``_base`` chain
                        is replaced with this model config so you can switch
                        models without editing run files.
                        Examples: "qwen3_5"  or  "configs/models/qwen3_5.yaml"
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        model_abs = cls._resolve_model_override(yaml_path, model)
        yaml_dict = cls._load_with_base(yaml_path, model_abs)
        yaml_dict = cls._migrate_legacy(yaml_dict)
        return from_dict(data_class=cls, data=yaml_dict)

    @classmethod
    def _resolve_model_override(cls, yaml_path: str, model: str | None) -> str | None:
        """Return absolute path of the override model yaml, or None."""
        if model is None:
            return None
        if model.endswith(".yaml"):
            return os.path.abspath(model)
        # "qwen3_5" → look for configs/models/qwen3_5.yaml next to the
        # configs/ directory that contains yaml_path
        configs_dir = os.path.dirname(os.path.dirname(os.path.abspath(yaml_path)))
        candidate = os.path.join(configs_dir, "configs", "models", f"{model}.yaml")
        if os.path.exists(candidate):
            return candidate
        # Also try relative to the configs dir itself
        candidate2 = os.path.join(os.path.dirname(os.path.abspath(yaml_path)),
                                   "..", "models", f"{model}.yaml")
        candidate2 = os.path.normpath(candidate2)
        if os.path.exists(candidate2):
            return candidate2
        raise FileNotFoundError(
            f"Model config not found for '{model}'. "
            f"Tried:\n  {candidate}\n  {candidate2}"
        )

    @classmethod
    def _load_with_base(cls, yaml_path: str,
                         model_override: str | None = None) -> dict:
        """Load YAML, recursively applying _base inheritance.

        When *model_override* is set and the ``_base`` path resolves to a file
        inside a ``models/`` directory, that file is replaced with the override.
        This lets callers swap the model layer without touching run configs.
        """
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        base_path = data.pop("_base", None)
        if base_path:
            base_abs = os.path.normpath(
                os.path.join(os.path.dirname(os.path.abspath(yaml_path)), base_path)
            )
            if model_override is not None:
                # Detect whether this _base points into a models/ directory
                parts = base_abs.replace("\\", "/").split("/")
                if "models" in parts:
                    base_abs = model_override

            base_data = cls._load_with_base(base_abs, model_override)
            data = _deep_merge(base_data, data)

        return data

    @staticmethod
    def _migrate_legacy(data: dict) -> dict:
        """Rewrite old-style ``inference:`` keys into the new structure."""
        inf = data.pop("inference", {})
        if not inf:
            return data

        model_keys = {
            "backend", "megakernel_variant", "attention_backend",
            "use_cuda_graph", "cuda_graph_bucket_size", "use_quantized_model",
            "use_kvcache", "check_correction", "cpu_offload_modules",
            "linear_attn_prefill_backend", "linear_attn_decode_backend",
            "kv_cache",
        }
        gen_keys = {
            "prompt", "max_new_tokens", "stop_on_eos",
            "use_chat_template", "use_thinking", "sampling", "multimodal",
        }

        model_patch: dict = {}
        gen_patch: dict = {}

        for k, v in inf.items():
            if k == "kv_cache_max_len":
                # flatten into model.kv_cache.max_len
                model_patch.setdefault("kv_cache", {})["max_len"] = v
            elif k in model_keys:
                model_patch[k] = v
            elif k in gen_keys:
                gen_patch[k] = v
            # else: drop unknown legacy keys silently

        # Handle old use_flash_attn / use_sdpa flags
        if "use_flash_attn" in inf or "use_sdpa" in inf:
            use_fa = inf.get("use_flash_attn", False)
            use_sdpa = inf.get("use_sdpa", True)
            model_patch["attention_backend"] = (
                "flash_attn" if use_fa else ("sdpa" if use_sdpa else "naive")
            )

        if model_patch:
            data["model"] = _deep_merge(data.get("model", {}), model_patch)
        if gen_patch:
            data["generation"] = _deep_merge(data.get("generation", {}), gen_patch)

        return data


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def resolve_data_path(cfg: GlobalConfig) -> str:
    if cfg.model.use_quantized_model:
        if not cfg.path.quantized_model_path:
            raise RuntimeError(
                "use_quantized_model is True but quantized_model_path is not set"
            )
        return cfg.path.quantized_model_path
    if cfg.path.data_path:
        return cfg.path.data_path
    return cfg.path.model_path


def is_running_quantized(cfg: GlobalConfig) -> bool:
    return cfg.model.use_quantized_model


def print_runtime_config(cfg: GlobalConfig):
    table = Table(box=box.SIMPLE_HEAVY, show_header=False, padding=(0, 2), expand=True)
    table.add_column("Key", style="cyan", no_wrap=True, ratio=1)
    table.add_column("Value", style="white", ratio=2)

    data_path = resolve_data_path(cfg)
    quantized = is_running_quantized(cfg)
    backend = cfg.model.backend

    table.add_row("data_path", data_path + (" (quantized)" if quantized else " (fp/bf16)"))
    table.add_row("device",           cfg.env.device)
    table.add_row("dtype",            cfg.env.default_dtype)
    table.add_row("backend",          backend)
    if backend == "megakernel_cuda":
        env_variant = os.environ.get("MINI_VLLM_MK_VARIANT")
        variant = env_variant or cfg.model.megakernel_variant
        suffix = " (via env)" if env_variant else ""
        table.add_row("megakernel_variant", variant + suffix)
    table.add_row("attention_backend", cfg.model.attention_backend)
    table.add_row("use_cuda_graph",    str(cfg.model.use_cuda_graph))
    table.add_row("use_kvcache",       str(cfg.model.use_kvcache))
    table.add_row("use_chat_template", str(cfg.generation.use_chat_template))
    if cfg.generation.use_chat_template:
        table.add_row("use_thinking", str(cfg.generation.use_thinking))
    table.add_row("max_new_tokens",    str(cfg.generation.max_new_tokens))
    table.add_row("sample_method",     cfg.generation.sampling.sample_method)
    table.add_row("temperature",       str(cfg.generation.sampling.temperature))
    if quantized:
        table.add_row("quant_bits",    str(cfg.quant.quant_bits))
        table.add_row("quant_backend", cfg.quant.backend)

    Console().print(Panel(
        table,
        title="[bold bright_blue]Runtime Configuration[/bold bright_blue]",
        border_style="bright_blue",
        padding=(1, 2),
    ))


def dump_config(cfg: GlobalConfig) -> str:
    """Return the fully-merged config as a YAML string (for --dump-config)."""
    import dataclasses
    return yaml.dump(dataclasses.asdict(cfg), allow_unicode=True, sort_keys=False)

