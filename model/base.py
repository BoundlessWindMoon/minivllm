from typing import Protocol, Any, TYPE_CHECKING
import torch
from torch import nn

if TYPE_CHECKING:
    from engine.kv_pool import KVCachePool


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

    def attach_kv_pool(self, pool: "KVCachePool") -> None:
        """Wire KVCacheLayer backends from pool into every Attention module.

        Called once after model creation, before any forward pass.
        For each layer i, sets:
            attention_module.kv_backend = pool.get_layer_view(i)

        This replaces the Attention module's internal cache with the pool-
        backed view.  After this call the model no longer owns any KV tensors;
        all storage lives in the pool.

        Subclasses that iterate attention modules via iter_attention_modules()
        get a default implementation here; override only for non-standard
        layer structures (e.g. MoE, hybrid layers).
        """
        for layer_idx, attn_module in enumerate(self.iter_attention_modules()):
            if pool is None:
                attn_module.kv_backend = None
            else:
                attn_module.kv_backend = pool.get_layer_view(layer_idx)
                # Disable the internal cache buffers so memory is not wasted.
                attn_module.k_cache = None
                attn_module.v_cache = None
