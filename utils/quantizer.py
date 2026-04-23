import torch
import torch.nn as nn
import functools

from utils.logger import logger
from utils.calib_data import get_calib_dataset
from typing import Dict, List, Optional, Tuple, Union

from tqdm import tqdm
from collections import defaultdict

from layers.attention import Attention
from layers.linear import (
    QKVParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    LinearBase,
)
from layers.quanted_linear import WQLinear_GEMM
from model.qwen3 import (
    Qwen3DecoderLayer,
    Qwen3Attention,
    Qwen3MLP,
)

# Import decoupled utilities
from utils.config import QuantConfig
from utils import model_utils
from utils import quantize_utils
from utils import scale_utils


class Quantizer:
    def __init__(
        self,
        model,
        tokenizer,
        quant_config: QuantConfig,
        device="cuda:0",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.quant_config = quant_config
        self.device = device
        self.target_layers = self._get_target_layers()

        # hard code for temporary
        self.calib_data = 'pileval'
        self.calib_n_samples = 32
        self.calib_max_seq_length = 512
        self.calib_split = 'train'
        self.calib_text_column = 'text'
        self._apply_clip = True
        self.export_compatible = False
        self.max_chunk_memory = 1024 * 1024 * 1024
        self.linear_version = 'gemm'

        self.modules, self.layer_kwargs, self.inps = self._init_quant(
            nsamples=self.calib_n_samples, max_seq_len=self.calib_max_seq_length
        )

    def _calibrate(self):
        if self.quant_config.quant_method == "AWQ":
            pass
        pass

    def _get_target_layers(self):
        target_layers = []
        for target in self.quant_config.quant_targets:
            if target == "MLP":
                target_layers.append(MergedColumnParallelLinear)
                target_layers.append(RowParallelLinear)
                pass
            elif target == "ATTENTION":
                target_layers.append(QKVParallelLinear)
                target_layers.append(RowParallelLinear)
            else:
                logger.error(f"Unknown quantization target: {target}")
                raise ValueError("quantization target not supported")
        return tuple(dict.fromkeys(target_layers))

    def _quantize_layer(self, layer):
        from layers.linear_quantized import QuantizedLinearWrapper

        quantized_layer = QuantizedLinearWrapper(
            original_layer=layer, group_size=self.quant_config.group_size
        )
        return quantized_layer

    def _get_model_layers(self, model):
        raise NotImplementedError

    def _init_quant(self, nsamples, max_seq_len):
        modules = self.model.model.layers

        samples = get_calib_dataset(
            data=self.calib_data,
            tokenizer=self.tokenizer,
            n_samples=nsamples,
            max_seq_len=max_seq_len,
            split=self.calib_split,
            text_column=self.calib_text_column,
        )
        samples = torch.cat(samples, dim=0)
        inps = []
        layer_kwargs = {}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, positions, hidden_states, residual=None):
                # assume first input to forward is hidden states
                inps.append(hidden_states)
                layer_kwargs['positions'] = positions
                layer_kwargs['residual'] = None
                raise ValueError  # early exit to break later inference

        modules[0] = Catcher(modules[0])
        try:
            self.model(samples.to(self.device))
        except ValueError:
            pass
        modules[0] = modules[0].module  # restore

        del samples
        inps = inps[0]

        return modules, layer_kwargs, inps

    @torch.no_grad()
    def _module_forward(
        self, x: torch.Tensor, module: torch.nn.Module, module_kwargs: dict
    ) -> torch.Tensor:
        # 1. 整个 DecoderLayer
        if isinstance(module, Qwen3DecoderLayer):
            positions = module_kwargs.get('positions')
            if positions is None:
                seq_len = x.shape[1]
                positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)

            hidden_states_out, residual_out = module(
                positions=positions,
                hidden_states=x,
                residual=module_kwargs.get('residual'),
                use_cache=False,
            )

            module_kwargs['residual'] = residual_out
            return hidden_states_out

        elif isinstance(module, Qwen3Attention):
            positions = module_kwargs.get('positions')
            if positions is None:
                seq_len = x.shape[1]
                positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
            return module(
                hidden_states=x,
                positions=positions,
                use_cache=False,
            )

        elif isinstance(module, Qwen3MLP):
            return module(x)

        else:
            return module(x)

    def _get_input_features(self, layer, named_linears, module_kwargs):
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            # x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []

        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )

        self.inps = self._module_forward(self.inps, layer, module_kwargs)
        for h in handles:
            h.remove()

        # now solve for scaling and clipping
        def cat_and_assert(k, v):
            x = torch.cat(v, dim=0)
            assert x.shape[0] != 0, (
                f"{k} has a zero dimension. This can happen if no data was passed through (e.g. an expert in MoE not being activated). "
                "Try increasing max_calib_samples (warning: this can significantly increase quantization time and memory usage.)"
            )
            return x

        input_feat = {k: cat_and_assert(k, v) for k, v in input_feat.items()}

        return input_feat

    def get_layers_for_scaling(self, module, input_feat, module_kwargs):
        layers = []

        layers.append(
            dict(
                prev_op=module.input_layernorm,
                layers=[module.self_attn.qkv_proj],
                inp=input_feat["self_attn.qkv_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )

        layers.append(
            dict(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_up_proj],
                inp=input_feat["mlp.gate_up_proj"],
                module2inspect=module.mlp,
            )
        )

        layers.append(
            dict(
                prev_op=module.mlp.act_fn,
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
            )
        )
        return layers

    @torch.no_grad()
    def _compute_best_scale(
        self,
        x: torch.Tensor,
        w_mean: torch.Tensor,
        x_mean: torch.Tensor,
        module2inspect: torch.nn.Module,
        linears2scale: List[nn.Linear],
        fp16_output: torch.Tensor,
        kwargs: Dict = {},
    ):
        """
        Compute loss and select best scales

        L(s) = || Q(W * s) (s^-1 * X) - W * X ||
        Q: weight quantization function | pseudo_quantize_tensor(W * s)
        X: inputs from calib dataset    | X
        W: original weights in FP16     | layer
        s: per channel scaling factor   | s^-1 * X
        """
        n_grid = 20
        history = []
        best_ratio = -1
        best_scales = None
        best_error = float("inf")

        # org_sd = {k: v.cpu() for k, v in module2inspect.state_dict().items()}
        org_sd = {k: v.clone() for k, v in module2inspect.state_dict().items()}

        device = x.device
        x_mean = x_mean.view(-1).to(device)
        w_mean = w_mean.view(-1).to(device)

        for ratio in range(n_grid):
            # create new scales
            ratio = ratio / n_grid

            # NOTE: s^-1 * x is fused here, according to paper
            scales = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(
                min=1e-4
            )

            scales = scales / (scales.max() * scales.min()).sqrt()
            scales_view = scales.view(1, -1).to(device)

            # avoid scaling values that overflow
            scales[torch.isinf(scales)] = 1
            scales[torch.isnan(scales)] = 1

            # Q(W * s)
            for fc in linears2scale:
                fc.weight.mul_(scales_view)
                fc.weight.data = (
                    quantize_utils.pseudo_quantize_tensor(
                        fc.weight.data,
                        self.quant_config.quant_bits,
                        self.quant_config.group_size,
                        self.quant_config.has_zero_point,
                    )[0]
                    / scales_view
                )

            # W * X
            int_w_output = self._module_forward(x, module2inspect, kwargs)
            int_w_output = int_w_output.clip(
                torch.finfo(int_w_output.dtype).min, torch.finfo(int_w_output.dtype).max
            )

            # compute mean squared error (L2 norm)
            loss = quantize_utils.compute_loss(
                fp16_output, int_w_output, device, self.max_chunk_memory
            )

            history.append(loss)
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scales = scales.clone()
            module2inspect.load_state_dict(org_sd)

        if best_ratio == -1:
            logger.debug(history)
            raise Exception

        assert torch.isnan(best_scales).sum() == 0, best_scales

        return best_scales.detach()

    @torch.no_grad()
    def _search_best_scale(
        self,
        module,
        prev_op,
        layers,
        inp: torch.Tensor,
        module2inspect=None,
        kwargs={},
    ):
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        weight = torch.cat([_m.weight for _m in layers], dim=0)
        org_shape = weight.shape
        weight = weight.view(-1, self.quant_config.group_size)
        w_scale = weight.abs() / (weight.abs().amax(dim=1, keepdim=True) + 1e-6)

        w_scale = w_scale.view(org_shape)
        w_mean = w_scale.mean(0)

        inp_flat = inp.abs().view(-1, inp.shape[-1])
        num_elements = inp_flat.size(0)
        num_channels = inp_flat.size(1)
        element_size_bytes = inp_flat.element_size() * 2

        chunk_size = int(self.max_chunk_memory // (element_size_bytes * num_channels))
        chunk_size = min(chunk_size, num_elements)

        x_sum = torch.zeros(num_channels, dtype=torch.float32, device=inp.device)
        for i in range(0, num_elements, chunk_size):
            end = min(i + chunk_size, num_elements)
            chunk_sum = inp_flat[i:end].to(torch.float32).sum(dim=0)
            x_sum += chunk_sum.to(inp.device)

        x_mean = (x_sum / num_elements).to(inp.dtype)

        with torch.no_grad():
            fp16_output = self._module_forward(inp, module2inspect, kwargs)
            fp16_output = fp16_output.clip(
                torch.finfo(fp16_output.dtype).min, torch.finfo(fp16_output.dtype).max
            )

        best_scales = self._compute_best_scale(
            inp, w_mean, x_mean, module2inspect, layers, fp16_output, kwargs
        )

        return (
            model_utils.get_op_name(module, prev_op),
            tuple([model_utils.get_op_name(module, m) for m in layers]),
            best_scales,
        )

    @torch.no_grad()
    def _search_best_clip(self, layer, named_linears, input_feat):
        clip_list = []
        avoid_clipping = ["q_", "k_", "query", "key", "Wqkv", "qkv_proj"]

        for name in named_linears:
            # due to qk bmm, it is hard to clip precisely
            if any([_ in name for _ in avoid_clipping]):
                continue

            max_val = quantize_utils.compute_best_clip(
                named_linears[name].weight,
                input_feat[name],
                self.quant_config.quant_bits,
                self.quant_config.group_size,
                self.quant_config.has_zero_point,
            )
            clip_list.append((name, max_val))

        return clip_list

    def _apply_quant(self, module, named_linears: Dict[str, nn.Linear]):
        for name, linear_layer in named_linears.items():

            linear_layer.weight.data, scales, zeros = (
                quantize_utils.pseudo_quantize_tensor(
                    linear_layer.weight.data,
                    self.quant_config.quant_bits,
                    self.quant_config.group_size,
                    self.quant_config.has_zero_point,
                )
            )

            if self.linear_version == "gemm":
                scales = scales.t().contiguous()
                if zeros is not None:
                    zeros = zeros.t().contiguous()
                q_linear_module = WQLinear_GEMM
            else:
                raise ValueError(f"Unknown version {self.linear_version}")

            q_linear = q_linear_module.from_linear(
                linear=linear_layer,
                w_bit=self.quant_config.quant_bits,
                group_size=self.quant_config.group_size,
                init_only=False,
                scales=scales,
                zeros=zeros,
            )
            model_utils.set_op_by_name(module, name, q_linear)

    def _quantize_and_replace(self):

        for i in tqdm(range(len(self.modules)), desc="AWQ"):
            self.inps = self.inps.to(self.device)

            # [STEP 1]: Get layer, extract linear modules, extract input features
            named_linears = model_utils.get_named_linears(self.modules[i])

            input_feat = self._get_input_features(
                self.modules[i], named_linears, self.layer_kwargs
            )

            # [STEP 2]: Compute and apply scale list
            module_config = self.get_layers_for_scaling(
                self.modules[i], input_feat, self.layer_kwargs
            )

            scales_list = [
                self._search_best_scale(self.modules[i], **layer)
                for layer in module_config
            ]

            scale_utils.apply_scale(
                self.modules[i], scales_list, input_feat_dict=input_feat
            )

            scales_list = model_utils.append_str_prefix(
                scales_list, model_utils.get_op_name(self.model, self.modules[i]) + "."
            )

            # [STEP 3]: Compute and apply clipping list
            if self._apply_clip:
                clip_list = self._search_best_clip(
                    self.modules[i], named_linears, input_feat
                )
                quantize_utils.apply_clip(
                    self.modules[i], clip_list, model_utils.get_op_by_name
                )
                clip_list = model_utils.append_str_prefix(
                    clip_list,
                    model_utils.get_op_name(self.model, self.modules[i]) + ".",
                )

            # [STEP 4]: Quantize weights
            if not self.export_compatible:
                self._apply_quant(self.modules[i], named_linears)

    def run(self):
        self._quantize_and_replace()
        # quantized_model = self._save_quantized_model()

        return self.model
