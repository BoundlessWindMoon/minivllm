from typing import Protocol, Any
import torch
from torch import nn


class CausalLMProtocol(Protocol):

    config: Any

    supports_cuda_graph: bool = False

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor | None = None
    ) -> torch.Tensor: ...

    def reset(self) -> None: ...


class BaseCausalLM(nn.Module):
    supports_cuda_graph: bool = False

    def reset(self) -> None:
        pass

    def iter_attention_modules(self):
        """Yield attention modules for inspection / runtime patching.

        Subclasses must override this so that model runners can access
        attention layers without hard-coding attribute paths.
        """
        raise NotImplementedError
