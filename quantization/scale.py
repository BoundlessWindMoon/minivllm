"""Scale tensor creation and per-channel scaling helpers."""

import torch
import torch.nn as nn

from layers.activation import SiluAndMul
from layers.linear import LinearBase
from quantization.module_ops import get_op_by_name


# Whitelist of modules eligible for AWQ scale application.
allowed_norms = [nn.LayerNorm]
allowed_act_fns = [SiluAndMul]


class ScaledActivation(nn.Module):
    def __init__(self, module, scales):
        super().__init__()
        self.act = module
        self.scales = nn.Parameter(scales.data)

    @torch.compile
    def forward(self, x):
        return self.act(x) / self.scales.view(1, 1, -1).to(x.device)


@torch.no_grad()
def scale_fc_fcs(fc1: LinearBase, fcs, scales: torch.Tensor):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(fc1.weight.device)

    fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_fc_fc(fc1: LinearBase, fc2: LinearBase, scales: torch.Tensor):
    assert isinstance(fc1, LinearBase)
    assert isinstance(fc2, LinearBase)

    scales = scales.to(fc1.weight.device)

    fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_ln_fcs(ln: nn.Linear, fcs, scales: torch.Tensor):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device)

    # GemmaRMSNorm is different from Llama's in that it multiplies
    # (1 + weight) to the output, instead of just weight.
    ln.weight.div_(scales)

    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_gelu_fc(gelu, fc: LinearBase, scales: torch.Tensor):
    assert any(isinstance(gelu, t) for t in allowed_act_fns)
    assert isinstance(fc, LinearBase)

    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))

    for p in fc.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def apply_scale(module, scales_list, input_feat_dict=None):
    for prev_op_name, layer_names, scales in scales_list:
        prev_op = get_op_by_name(module, prev_op_name)
        layers = [get_op_by_name(module, name) for name in layer_names]

        if (
            isinstance(prev_op, LinearBase)
            and isinstance(layers, list)
            and isinstance(layers[0], LinearBase)
        ):
            scale_fc_fcs(prev_op, layers, scales)

        elif isinstance(prev_op, LinearBase):
            assert len(layers) == 1
            scale_fc_fc(prev_op, layers[0], scales)

        elif (
            any(isinstance(prev_op, t) for t in allowed_norms)
            or "rmsnorm" in str(prev_op.__class__).lower()
        ):
            scale_ln_fcs(prev_op, layers, scales)

        elif any(isinstance(prev_op, t) for t in allowed_act_fns):
            prev_op.scales = nn.Parameter(scales.data)
            scale_gelu_fc(prev_op, layers[0], scales)

        else:
            raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")

        # apply the scaling to input feat if given; prepare it for clipping
        if input_feat_dict is not None:
            for layer_name in layer_names:
                # Skip the modules that are not quantized
                if layer_name in input_feat_dict:
                    inp = input_feat_dict[layer_name]
                    inp.div_(scales.view(1, -1).to(inp.device))
