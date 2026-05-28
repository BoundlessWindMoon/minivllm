from __future__ import annotations

from typing import Callable, Any
import torch
from transformers import PretrainedConfig

from model.base import BaseCausalLM, CausalLMProtocol

_REGISTRY: dict[str, dict[str, Callable | None]] = {}


def register_model(
    model_type: str,
    torch_factory: Callable[[Any, str], BaseCausalLM],
    megakernel_factory: Callable[[BaseCausalLM, str], CausalLMProtocol] | None = None,
) -> None:
    _REGISTRY[model_type] = {
        "torch": torch_factory,
        "megakernel": megakernel_factory,
    }


def _default_torch_factory(model_cls: type[BaseCausalLM]):
    """Build a simple ``(config, device) -> model`` wrapper around a class."""

    def _factory(config: PretrainedConfig, device: str) -> BaseCausalLM:
        return model_cls(config).to(device)

    return _factory


def create_base_model(
    config: PretrainedConfig,
    device: str | torch.device,
    attention_backend: str = "sdpa",
) -> BaseCausalLM:
    """Create a *native PyTorch* model from a HF config."""
    model_type = getattr(config, "model_type", None)
    if model_type not in _REGISTRY:
        known = sorted(_REGISTRY.keys())
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Registered families: {known}. "
            f"Did you forget to import the model module so its ``register_model()`` runs?"
        )

    config.attention_backend = attention_backend
    factory = _REGISTRY[model_type]["torch"]
    return factory(config, str(device))


def create_megakernel_model(
    base_model: BaseCausalLM,
    variant: str | None = None,
) -> CausalLMProtocol:
    """Wrap a base model with the CUDA megakernel backend.

    Raises:
        RuntimeError: If the model family has no megakernel support registered.
    """
    config = base_model.config
    model_type = getattr(config, "model_type", None)
    if model_type not in _REGISTRY:
        raise ValueError(f"Unknown model_type '{model_type}'.")

    mk_factory = _REGISTRY[model_type]["megakernel"]
    if mk_factory is None:
        raise RuntimeError(
            f"Model family '{model_type}' does not have a megakernel backend yet."
        )

    return mk_factory(base_model, variant)


def has_megakernel_support(model_type: str) -> bool:
    """Return whether a megakernel backend exists for *model_type*."""
    entry = _REGISTRY.get(model_type)
    return entry is not None and entry["megakernel"] is not None


def list_registered() -> list[str]:
    """Return all registered ``model_type`` strings."""
    return sorted(_REGISTRY.keys())


def _register_qwen3() -> None:
    from model.qwen3 import Qwen3ForCausalLM
    from model.qwen3_megakernel import Qwen3MegakernelForCausalLM

    register_model(
        "qwen3",
        torch_factory=_default_torch_factory(Qwen3ForCausalLM),
        megakernel_factory=lambda base, variant: Qwen3MegakernelForCausalLM.from_model(
            base, variant=variant
        ),
    )


def _register_qwen3_5() -> None:
    from model.qwen3_5 import Qwen3_5ForCausalLM
    from model.qwen3_5_multimodal import Qwen3_5MultimodalForCausalLM
    from model.qwen3_5_megakernel import Qwen3_5MegakernelForCausalLM

    def _qwen3_5_factory(config, device):
        # HF Qwen3_5Config is multimodal and nests text params in ``text_config``.
        full_config = config
        if hasattr(config, "text_config") and config.text_config is not None:
            text_cfg = config.text_config
            # Copy text params to top level so Qwen3_5Config picks them up
            for key, value in vars(text_cfg).items():
                if not key.startswith("_") and not hasattr(config, key):
                    setattr(config, key, value)
            # Ensure rope_theta is set
            if getattr(config, "rope_theta", None) is None:
                for key in ("rope_parameters", "rope_scaling"):
                    rope_cfg = getattr(config, key, None)
                    if isinstance(rope_cfg, dict) and "rope_theta" in rope_cfg:
                        config.rope_theta = rope_cfg["rope_theta"]
                        break

        if getattr(config, "vision_config", None) is not None:
            return _default_torch_factory(Qwen3_5MultimodalForCausalLM)(config, device)
        return _default_torch_factory(Qwen3_5ForCausalLM)(config, device)

    register_model(
        "qwen3_5",
        torch_factory=_qwen3_5_factory,
        megakernel_factory=lambda base, variant: Qwen3_5MegakernelForCausalLM.from_model(
            base, variant=variant
        ),
    )


_register_qwen3()
_register_qwen3_5()
