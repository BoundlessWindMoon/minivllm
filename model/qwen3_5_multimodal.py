"""Multimodal wrapper for Qwen3.5 — isolates the transformers vision dependency."""

from itertools import groupby

import torch
from torch import nn

from model.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5Model
from utils.model_config import Qwen3_5Config
from layers.embed_head import ParallelLMHead


class Qwen3_5MultimodalModel(nn.Module):
    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.language_model = Qwen3_5Model(config)
        if getattr(config, "vision_config", None) is not None:
            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                Qwen3_5VisionModel,
            )

            self.visual = Qwen3_5VisionModel(config.vision_config)
        else:
            self.visual = None


class Qwen3_5MultimodalForCausalLM(Qwen3_5ForCausalLM):
    def __init__(self, config: Qwen3_5Config):
        # Bypass Qwen3_5ForCausalLM.__init__ to build our own model graph.
        nn.Module.__init__(self)
        self.config = config
        self.model = Qwen3_5MultimodalModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = (
                self.model.language_model.embed_tokens.weight.data
            )
        self.rope_deltas = None

    def reset(self) -> None:
        super().reset()
        self.rope_deltas = None

    def _get_image_features(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> torch.Tensor:
        if self.model.visual is None:
            raise RuntimeError("Model has no vision encoder (vision_config is None)")
        pixel_values = pixel_values.type(self.model.visual.dtype)
        vision_outputs = self.model.visual(pixel_values, grid_thw=image_grid_thw)
        image_embeds = vision_outputs.pooler_output
        merge_size = self.model.visual.spatial_merge_size**2
        split_sizes = (image_grid_thw.prod(-1) // merge_size).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        image_embeds = torch.cat(image_embeds, dim=0)
        return image_embeds

    def _get_rope_index(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        spatial_merge_size = (
            self.config.vision_config.spatial_merge_size
            if self.config.vision_config
            else 2
        )

        position_ids = torch.zeros(
            3, batch_size, seq_len, dtype=input_ids.dtype, device=device
        )
        mrope_position_deltas = []

        for batch_idx in range(batch_size):
            current_input_ids = input_ids[batch_idx]
            if mm_token_type_ids is not None:
                token_types = mm_token_type_ids[batch_idx]
            else:
                token_types = torch.zeros(seq_len, dtype=torch.int, device=device)
                image_mask = current_input_ids == self.config.image_token_id
                token_types[image_mask] = 1

            input_type_group = []
            for key, group in groupby(enumerate(token_types.tolist()), lambda x: x[1]):
                group = list(group)
                start_index = group[0][0]
                end_index = group[-1][0] + 1
                input_type_group.append((key, start_index, end_index))

            current_pos = 0
            llm_pos_ids_list = []
            image_idx = 0
            for modality_type, start_idx, end_idx in input_type_group:
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=device).view(1, -1).expand(3, -1)
                        + current_pos
                    )
                    current_pos += text_len
                else:
                    if image_grid_thw is not None and image_idx < len(image_grid_thw):
                        grid_thw = image_grid_thw[image_idx]
                        llm_grid_t = grid_thw[0].item()
                        llm_grid_h = grid_thw[1].item() // spatial_merge_size
                        llm_grid_w = grid_thw[2].item() // spatial_merge_size

                        pos_t = (
                            torch.arange(llm_grid_t, device=device).repeat_interleave(
                                llm_grid_h * llm_grid_w
                            )
                            + current_pos
                        )
                        pos_h = (
                            torch.arange(llm_grid_h, device=device)
                            .repeat_interleave(llm_grid_w)
                            .repeat(llm_grid_t)
                            + current_pos
                        )
                        pos_w = (
                            torch.arange(llm_grid_w, device=device).repeat(
                                llm_grid_h * llm_grid_t
                            )
                            + current_pos
                        )
                        vision_pos = torch.stack([pos_t, pos_h, pos_w], dim=0)
                        llm_pos_ids_list.append(vision_pos)
                        current_pos += (
                            max(grid_thw[1].item(), grid_thw[2].item())
                            // spatial_merge_size
                        )
                        image_idx += 1
            if llm_pos_ids_list:
                llm_positions = torch.cat(llm_pos_ids_list, dim=1)
                position_ids[:, batch_idx, :] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(current_pos - seq_len)

        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        inputs_embeds = self.model.language_model.embed_tokens(input_ids)

        if pixel_values is not None and self.model.visual is not None:
            image_embeds = self._get_image_features(pixel_values, image_grid_thw)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask = input_ids == self.config.image_token_id
            if image_mask.any():
                inputs_embeds = inputs_embeds.masked_scatter(
                    image_mask.unsqueeze(-1).expand_as(inputs_embeds), image_embeds
                )

        if pixel_values is not None:
            if self.rope_deltas is None:
                position_ids, rope_deltas = self._get_rope_index(
                    input_ids,
                    image_grid_thw=image_grid_thw,
                    mm_token_type_ids=mm_token_type_ids,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_len = input_ids.shape
                position_ids = (
                    torch.arange(seq_len, device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, batch_size, -1)
                )
                delta = self.rope_deltas[:batch_size].view(1, batch_size, 1)
                position_ids = position_ids + delta
        elif positions is not None and positions.ndim == 3:
            position_ids = positions
        else:
            position_ids = positions

        hidden_states = self.model.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
        )
        logits = self.lm_head(hidden_states)
        return logits
