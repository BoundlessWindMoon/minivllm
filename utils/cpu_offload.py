"""CPU-offload utilities for hybrid CPU/GPU inference.

Decoupled from model definitions: operates on any `nn.Module` by submodule
path. Two independent pieces:

1. `materialize_with_offload`: turn a meta-device skeleton into real tensors
   with per-module device routing — offloaded paths land on CPU, the rest
   on the compute device. Non-meta tensors are left alone (so quantized
   stubs already allocated on the compute device keep their device).

2. `apply_cpu_offload`: register forward hooks so an offloaded module runs
   on CPU transparently — pre-hook moves args/kwargs to CPU, post-hook
   moves the output back to the compute device. State_dict key paths are
   unchanged, so `load_state_dict` works without any awareness of offload.

Typical use::

    with torch.device("meta"):
        model = MyModel(config)
    prepare_quantized_layers(model, ...)              # meta stubs
    materialize_with_offload(model, "cuda:0", paths)
    apply_cpu_offload(model, paths, "cuda:0")
    load_quantized_weights(model, ...)                # copy_ handles device
"""
from __future__ import annotations
from typing import Iterable, Union

import torch
import torch.nn as nn


DeviceLike = Union[str, torch.device]


def _is_under(name: str, roots: Iterable[str]) -> bool:
    for r in roots:
        if name == r or name.startswith(r + "."):
            return True
    return False


def _move(obj, device: torch.device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device) if obj.device != device else obj
    if isinstance(obj, tuple):
        return tuple(_move(x, device) for x in obj)
    if isinstance(obj, list):
        return [_move(x, device) for x in obj]
    if isinstance(obj, dict):
        return {k: _move(v, device) for k, v in obj.items()}
    return obj


def materialize_with_offload(
    model: nn.Module,
    compute_device: DeviceLike,
    offload_paths: Iterable[str],
) -> None:
    """Materialize meta tensors with per-module device routing.

    Walks every submodule. Meta parameters/buffers under any path in
    `offload_paths` are allocated on CPU; everything else on
    `compute_device`. Existing non-meta tensors are left untouched.
    """
    compute_device = torch.device(compute_device)
    cpu = torch.device("cpu")
    offload_paths = list(offload_paths)

    for module_name, module in model.named_modules():
        target = cpu if _is_under(module_name, offload_paths) else compute_device

        for param_name, param in list(module._parameters.items()):
            if param is None or param.device.type != "meta":
                continue
            new_p = nn.Parameter(
                torch.empty_like(param, device=target),
                requires_grad=param.requires_grad,
            )
            module._parameters[param_name] = new_p

        for buf_name, buf in list(module._buffers.items()):
            if buf is None or buf.device.type != "meta":
                continue
            module._buffers[buf_name] = torch.empty_like(buf, device=target)

        _fixup_module_attrs(module, target)


def _fixup_module_attrs(module: nn.Module, device: torch.device) -> None:
    # WHY: some modules hold tensors outside state_dict (plain attrs, or
    # persistent=False buffers); they need rebuilding after the meta-skeleton
    # materialization loop above. Implementers expose _post_materialize_fixup.
    hook = getattr(module, "_post_materialize_fixup", None)
    if callable(hook):
        hook(device)


def apply_cpu_offload(
    model: nn.Module,
    offload_paths: Iterable[str],
    compute_device: DeviceLike,
) -> None:
    """Install forward hooks so listed modules run on CPU transparently.

    Pre-hook moves args/kwargs to CPU; post-hook moves the output back
    to `compute_device`. Parameters and buffers stay on CPU; only the
    per-call I/O crosses PCIe.
    """
    compute_device = torch.device(compute_device)
    cpu = torch.device("cpu")

    def pre_hook(_module, args, kwargs):
        return _move(args, cpu), _move(kwargs, cpu)

    def post_hook(_module, _args, _kwargs, output):
        return _move(output, compute_device)

    for path in offload_paths:
        if not path:
            continue
        sub = model.get_submodule(path)
        sub.register_forward_pre_hook(pre_hook, with_kwargs=True)
        sub.register_forward_hook(post_hook, with_kwargs=True)
