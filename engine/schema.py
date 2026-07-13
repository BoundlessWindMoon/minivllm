from dataclasses import dataclass
from typing import Optional


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    max_new_tokens: int = 128
    stop_on_eos: bool = True
    enable_thinking: Optional[bool] = None  # None = 从 server config 读


@dataclass
class GenerationOutput:
    request_id: str
    token_id: int
    text_delta: str
    is_finished: bool
    finish_reason: Optional[str] = None  # "eos" | "length" | "error"
    error: Optional[str] = None
