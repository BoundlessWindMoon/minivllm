"""Module inspection helpers for quantization targets."""

import torch.nn as nn
from layers.linear import LinearBase


def get_op_name(module, op):
    """get the name of the op relative to the module"""
    for name, m in module.named_modules():
        if m is op:
            return name
    raise ValueError(f"Cannot find op {op} in module {module}")


def get_op_by_name(module, op_name):
    """get the op by its name relative to the module"""
    for name, m in module.named_modules():
        if name == op_name:
            return m
    raise ValueError(f"Cannot find op {op_name} in module {module}")


def set_op_by_name(layer, name, new_module):
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


def set_layer_by_name(model, name, layer):
    """parent_path be like:  ['model', 'layers', '0', 'self_attn']"""
    """child_attr be like: qkv_proj"""
    *parent_path, child_attr = name.split(".")

    parent = model
    for attr in parent_path:
        parent = parent[int(attr)] if attr.isdigit() else getattr(parent, attr)
    setattr(parent, child_attr, layer)


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, LinearBase)}


def append_str_prefix(x, prefix):
    if isinstance(x, str):
        return prefix + x
    elif isinstance(x, tuple):
        return tuple([append_str_prefix(y, prefix) for y in x])
    elif isinstance(x, list):
        return [append_str_prefix(y, prefix) for y in x]
    else:
        return x
