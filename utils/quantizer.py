import torch
import torch.nn as nn
import functools


from utils.logger import logger
from utils.calib_data import get_calib_dataset
from typing import Dict, List, Optional, Tuple, Union

from collections import defaultdict

from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.console import Console

CONSOLE = Console()

from layers.attention import Attention
from layers.linear import (
    QKVParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    LinearBase,
)
from layers.embed_head import ParallelLMHead
from layers.quanted_linear import WQLinear_W
from layers.quanted_linear_wt import WQLinear_Wt
from model.qwen3 import (
    Qwen3DecoderLayer,
    Qwen3Attention,
    Qwen3MLP,
)

# Import decoupled utilities
from utils.config import QuantConfig, EnvironmentConfig, CalibConfig
from utils import model_utils
from utils import quantize_utils
from utils import scale_utils


class Quantizer:
    def __init__(
        self,
        model,
        tokenizer,
        quant_config: QuantConfig,
        env_config: EnvironmentConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.quant_config = quant_config
        self.env_config = env_config

        self.calib_config = self.quant_config.calibration

        self.device = env_config.device
        self.target_layers = self._get_target_layers()

        # quant_config
        self._apply_clip = quant_config.apply_clip
        self.export_compatible = quant_config.export_compatible
        self.max_chunk_memory = quant_config.max_chunk_memory
        self.backend = quant_config.backend

        self.modules, self.layer_kwargs, self.inps = self._init_quant(
            nsamples=self.calib_config.n_samples,
            max_seq_len=self.calib_config.max_seq_length,
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
            elif target == "ATTENTION":
                target_layers.append(QKVParallelLinear)
                target_layers.append(RowParallelLinear)
            elif target == "LM_HEAD":
                target_layers.append(ParallelLMHead)
            else:
                logger.error(f"Unknown quantization target: {target}")
                raise ValueError("quantization target not supported")
        return tuple(dict.fromkeys(target_layers))

    def _get_model_layers(self, model):
        raise NotImplementedError

    def _init_quant(self, nsamples, max_seq_len):
        modules = self.model.model.layers
        calib_cfg = self.calib_config

        samples = get_calib_dataset(
            data=calib_cfg.data,
            tokenizer=self.tokenizer,
            n_samples=nsamples,
            max_seq_len=max_seq_len,
            split=calib_cfg.split,
            text_column=calib_cfg.text_column,
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

            layout = getattr(self.quant_config, 'layout', 'Wt')
            pack_order = getattr(self.quant_config, 'pack_order', 'sequential')

            if layout == 'W':
                q_linear_module = WQLinear_W
            elif layout == 'Wt':
                q_linear_module = WQLinear_Wt
            else:
                raise ValueError(f"Unknown layout: {layout}")

            q_linear = q_linear_module.from_linear(
                linear=linear_layer,
                w_bit=self.quant_config.quant_bits,
                group_size=self.quant_config.group_size,
                init_only=False,
                scales=scales,
                zeros=zeros,
                backend=self.backend,
                layout=layout,
                pack_order=pack_order,
            )
            model_utils.set_op_by_name(module, name, q_linear)

    def _quantize_and_replace(self):

        with Progress(
            TextColumn("[bold blue]🛠️  Quantizing"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            TextColumn("• {task.completed}/{task.total} layers"),
            "•",
            TimeElapsedColumn(),
            "<",
            TimeRemainingColumn(),
            console=CONSOLE,
        ) as progress:
            quant_task = progress.add_task("AWQ", total=len(self.modules))

            for i in range(len(self.modules)):
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
                    scales_list,
                    model_utils.get_op_name(self.model, self.modules[i]) + ".",
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

                progress.update(quant_task, advance=1)

    @torch.no_grad()
    def _quantize_lm_head(self):
        """Quantize lm_head with the same AWQ pipeline as MLP/Attention layers."""
        lm_head = self.model.lm_head
        if not isinstance(lm_head, ParallelLMHead):
            logger.warning("lm_head is not ParallelLMHead, skipping")
            return

        # Get input features after final norm
        hidden_states, _ = self.model.model.norm(
            self.inps, self.layer_kwargs.get("residual")
        )

        # [STEP 1]: Search best scale (same pipeline as MLP/Attention)
        scale_info = self._search_best_scale(
            module=self.model,
            prev_op=self.model.model.norm,
            layers=[lm_head],
            inp=hidden_states,
            module2inspect=lm_head,
        )

        # Apply scale to both final norm and lm_head (preserves math equivalence)
        scale_utils.apply_scale(self.model, [scale_info])

        # [STEP 2]: Search and apply clip
        if self._apply_clip:
            max_val = quantize_utils.compute_best_clip(
                lm_head.weight,
                hidden_states,
                self.quant_config.quant_bits,
                self.quant_config.group_size,
                self.quant_config.has_zero_point,
            )
            quantize_utils.apply_clip(
                self.model, [("lm_head", max_val)], model_utils.get_op_by_name
            )

        # [STEP 3]: Quantize weights
        qweight, scales, zeros = quantize_utils.pseudo_quantize_tensor(
            lm_head.weight.data,
            self.quant_config.quant_bits,
            self.quant_config.group_size,
            self.quant_config.has_zero_point,
        )

        layout = getattr(self.quant_config, "layout", "Wt")
        pack_order = getattr(self.quant_config, "pack_order", "sequential")
        q_cls = WQLinear_Wt if layout == "Wt" else WQLinear_W

        q_linear = q_cls.from_linear(
            linear=lm_head,
            w_bit=self.quant_config.quant_bits,
            group_size=self.quant_config.group_size,
            init_only=False,
            scales=scales,
            zeros=zeros,
            backend=self.backend,
            layout=layout,
            pack_order=pack_order,
        )
        model_utils.set_op_by_name(self.model, "lm_head", q_linear)
        logger.info("lm_head quantized with AWQ scale search")

        # If embeddings were tied, sync embed_tokens weight with dequantized lm_head
        # to preserve the semantic symmetry between input and output projections.
        if getattr(self.model.config, "tie_word_embeddings", False):
            with torch.no_grad():
                dequantized_t = q_linear._dequantize_to_weight_t(torch.float16)
                # dequantized_t shape: (in_features, out_features) = (hidden_dim, vocab_size)
                self.model.model.embed_tokens.weight.data.copy_(dequantized_t.t())
            logger.info("embed_tokens.weight synchronized with dequantized lm_head")

    def run(self):
        self._quantize_and_replace()

        # Quantize lm_head if requested
        if "LM_HEAD" in [t.upper() for t in self.quant_config.quant_targets]:
            self._quantize_lm_head()

        return self.model
