import os
import torch
import torch.nn as nn
import yaml
from dacite import from_dict
from dataclasses import dataclass, field
from typing import Optional, List
from layers.activation import SiluAndMul

allowed_norms = [
    nn.LayerNorm,
]

allowed_act_fns = [
    SiluAndMul,
]


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
    baseline_model_path: str = "~/huggingface/baseline/"
    data_path: str = ""
    profile_dir: str = "./log/profile/"

    def __post_init__(self):
        if self.model_path:
            self.model_path = os.path.expanduser(self.model_path)
        if self.baseline_model_path:
            self.baseline_model_path = os.path.expanduser(self.baseline_model_path)
        if self.data_path:
            self.data_path = os.path.expanduser(self.data_path)
        if self.profile_dir:
            self.profile_dir = os.path.expanduser(self.profile_dir)


@dataclass
class SamplingConfig:
    sample_method: str = "greedy"
    temperature: float = 1.0
    topk: int = 1
    topp: float = 1.0


@dataclass
class InferenceConfig:
    prompt: str = (
        "Hello, I am sakuya, I'm a 24 year old student from UCAS University. I like LLM and Infra"
    )
    max_new_tokens: int = 128
    use_kvcache: bool = True
    check_correction: bool = False
    use_profile: bool = False
    sampling: SamplingConfig = field(default_factory=SamplingConfig)


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

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "GlobalConfig":
        """从 YAML 文件加载配置，自动映射到嵌套 Dataclass"""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_dict = yaml.safe_load(f)

        config = from_dict(data_class=cls, data=yaml_dict)
        return config
