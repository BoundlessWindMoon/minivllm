"""YAML-backed nested dataclass configuration."""

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


@dataclass
class PathConfig:
    model_path: str = "~/huggingface/Qwen3-0.6B/"
    data_path: str = ""
    profile_dir: str = "./log/profile/"
    quantized_model_path: str = ""

    def __post_init__(self):
        if self.model_path:
            self.model_path = os.path.expanduser(self.model_path)
        if self.data_path:
            self.data_path = os.path.expanduser(self.data_path)
        if self.profile_dir:
            self.profile_dir = os.path.expanduser(self.profile_dir)
        if self.quantized_model_path:
            self.quantized_model_path = os.path.expanduser(self.quantized_model_path)


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
class KVCacheConfig:
    backend: str = "default"  # "default" | "kivi"
    k_bits: int = 2
    v_bits: int = 2
    group_size: int = 32
    residual_length: int = 32


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


@dataclass
class InferenceConfig:
    prompt: str = (
        "Hello, I am sakuya, I'm a 24 year old student from UCAS University. I like LLM and Infra"
    )
    max_new_tokens: int = 128
    use_kvcache: bool = True
    attention_backend: str = "sdpa"  # [flash_attn, sdpa, naive]
    use_cuda_graph: bool = True
    cuda_graph_bucket_size: int = 1
    use_profile: bool = False
    backend: str = "default"
    megakernel_variant: str = "default"
    use_quantized_model: bool = False
    stop_on_eos: bool = True
    use_chat_template: bool = False
    use_thinking: bool = True
    linear_attn_prefill_backend: str = "torch"
    linear_attn_decode_backend: str = "fla"
    cpu_offload_modules: List[str] = field(default_factory=list)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    multimodal: MultimodalConfig = field(default_factory=MultimodalConfig)
    kv_cache_max_len: int | None = None
    kv_cache: KVCacheConfig = field(default_factory=KVCacheConfig)


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


@dataclass
class GlobalConfig:
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    path: PathConfig = field(default_factory=PathConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "GlobalConfig":
        """从 YAML 文件加载配置，自动映射到嵌套 Dataclass"""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_dict = yaml.safe_load(f)

        # Backward-compat: migrate old boolean flags to attention_backend
        inference = yaml_dict.get("inference", {})
        if "use_flash_attn" in inference or "use_sdpa" in inference:
            use_fa = inference.pop("use_flash_attn", False)
            use_sdpa = inference.pop("use_sdpa", True)
            if use_fa:
                inference["attention_backend"] = "flash_attn"
            elif use_sdpa:
                inference["attention_backend"] = "sdpa"
            else:
                inference["attention_backend"] = "naive"
            import warnings
            warnings.warn(
                "Config fields 'use_sdpa' and 'use_flash_attn' are deprecated. "
                f"Use 'attention_backend: {inference['attention_backend']}' instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        config = from_dict(data_class=cls, data=yaml_dict)
        return config


def resolve_data_path(cfg) -> str:
    """Resolve the effective data path.

    Priority:
      1. quantized_model_path  (if use_quantized_model is True)
      2. data_path             (if set and non-empty)
      3. model_path            (fallback)
    """
    if cfg.inference.use_quantized_model:
        if not cfg.path.quantized_model_path:
            raise RuntimeError(
                "use_quantized_model is True but quantized_model_path is not set"
            )
        return cfg.path.quantized_model_path
    if cfg.path.data_path:
        return cfg.path.data_path
    return cfg.path.model_path


def is_running_quantized(cfg) -> bool:
    """Check whether the resolved data path points to a quantized model."""
    return cfg.inference.use_quantized_model


def print_runtime_config(cfg):
    """Print a rich-formatted summary of the runtime configuration."""
    table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=False,
        padding=(0, 2),
        expand=True,
    )
    table.add_column("Key", style="cyan", no_wrap=True, ratio=1)
    table.add_column("Value", style="white", ratio=2)

    data_path = resolve_data_path(cfg)
    quantized = is_running_quantized(cfg)
    backend = getattr(cfg.inference, "backend", "default")

    # Path
    path_label = " (quantized)" if quantized else " (fp16/bf16)"
    table.add_row("data_path", data_path + path_label)

    # Dtype / Device
    table.add_row("device", cfg.env.device)
    table.add_row("dtype", cfg.env.default_dtype)

    # Inference
    table.add_row("backend", backend)
    if backend == "megakernel_cuda":
        env_variant = os.environ.get("MINI_VLLM_MK_VARIANT")
        variant = env_variant or getattr(cfg.inference, "megakernel_variant", "default")
        suffix = " (via MINI_VLLM_MK_VARIANT)" if env_variant else ""
        table.add_row("megakernel_variant", variant + suffix)
    table.add_row("attention_backend", str(cfg.inference.attention_backend))
    table.add_row("use_cuda_graph", str(cfg.inference.use_cuda_graph))
    table.add_row("use_kvcache", str(cfg.inference.use_kvcache))
    table.add_row("use_chat_template", str(cfg.inference.use_chat_template))
    if cfg.inference.use_chat_template:
        table.add_row("use_thinking", str(cfg.inference.use_thinking))
    table.add_row("max_new_tokens", str(cfg.inference.max_new_tokens))

    # Sampling
    table.add_row("sample_method", cfg.inference.sampling.sample_method)
    table.add_row("temperature", str(cfg.inference.sampling.temperature))
    table.add_row("topk", str(cfg.inference.sampling.topk))
    table.add_row("topp", str(cfg.inference.sampling.topp))

    # Quantization
    if quantized:
        table.add_row("quant_bits", str(cfg.quant.quant_bits))
        table.add_row("quant_backend", cfg.quant.backend)
        table.add_row("group_size", str(cfg.quant.group_size))

    panel = Panel(
        table,
        title="[bold bright_blue]Runtime Configuration[/bold bright_blue]",
        border_style="bright_blue",
        padding=(1, 2),
    )
    console = Console()
    console.print(panel)
