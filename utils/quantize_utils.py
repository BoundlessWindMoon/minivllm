import torch


@torch.no_grad()
def pseudo_quantize_tensor(
    w: torch.Tensor, quant_bits: int, group_size: int, has_zero_point: bool
):
    org_w_shape = w.shape
    if group_size > 0:
        assert (
            org_w_shape[-1] % group_size == 0
        ), f"org_w_shape ({org_w_shape[-1]}) must be a multiple of group_size ({group_size})!"
        w = w.reshape(-1, group_size)
    assert w.dim() == 2
    assert torch.isnan(w).sum() == 0

    # zero point quantization
    if has_zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**quant_bits - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
        w = (
            torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
        ) * scales
        zeros = zeros.view(org_w_shape[0], -1)
    else:
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (quant_bits - 1) - 1
        min_int = -(2 ** (quant_bits - 1))
        scales = max_val / max_int
        zeros = None
        w = torch.clamp(torch.round(w / scales), min_int, max_int) * scales

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    scales = scales.view(org_w_shape[0], -1)
    w = w.reshape(org_w_shape)

    return w, scales, zeros


@torch.no_grad()
def compute_loss(
    fp16_output: torch.Tensor,
    int_w_output: torch.Tensor,
    device: torch.device,
    max_chunk_memory: int,
):
    loss = 0.0
    fp16_output_flat = fp16_output.view(-1)
    int_w_output_flat = int_w_output.view(-1)
    num_elements = fp16_output_flat.size(0)
    element_size_bytes = fp16_output.element_size()

    # Calculate chunk size dynamically based on max_chunk_memory
    # Divide the max_chunk_memory by twice the element size
    chunk_size = max_chunk_memory // (element_size_bytes * 2)
    chunk_size = min(chunk_size, num_elements)

    # Split the computation into chunks
    fp16_chunks = torch.split(fp16_output_flat, chunk_size)
    int_w_chunks = torch.split(int_w_output_flat, chunk_size)

    # Compute the loss for each chunk
    for fp16_chunk, int_w_chunk in zip(fp16_chunks, int_w_chunks):
        chunk_loss = (
            (fp16_chunk.to(device) - int_w_chunk.to(device)).float().pow(2).sum().item()
        )
        loss += chunk_loss

    # Normalize the loss by the total number of elements
    loss /= num_elements

    return loss


@torch.no_grad()
def compute_best_clip(
    w: torch.Tensor,
    input_feat: torch.Tensor,
    quant_bits: int,
    group_size: int,
    has_zero_point: bool,
    n_grid=20,
    max_shrink=0.5,
    n_sample_token=512,
):
    assert w.dim() == 2
    org_w_shape = w.shape
    # w           [co, ci]      -> [co, 1, n_group, group size]
    # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
    group_size = group_size if group_size > 0 else org_w_shape[1]
    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)

    # Compute input feature step size (minimum 1)
    step_size = max(1, input_feat.shape[1] // n_sample_token)
    input_feat = input_feat[:, ::step_size]

    w = w.reshape(org_w_shape[0], 1, -1, group_size)

    oc_batch_size = 256 if org_w_shape[0] % 256 == 0 else 64  # prevent OOM
    assert org_w_shape[0] % oc_batch_size == 0
    w_all = w
    best_max_val_all = []

    for i_b in range(org_w_shape[0] // oc_batch_size):
        w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

        org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

        best_max_val = org_max_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9
        input_feat = input_feat.to(w.device)
        org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s / n_grid)
            min_val = -max_val
            cur_w = torch.clamp(w, min_val, max_val)
            q_w = pseudo_quantize_tensor(cur_w, quant_bits, group_size, has_zero_point)[
                0
            ]
            cur_out = (input_feat * q_w).sum(dim=-1)

            # co, 1, n_group, 1
            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            del cur_w
            del cur_out
            cur_best_idx = err < min_errs
            min_errs[cur_best_idx] = err[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]
        best_max_val_all.append(best_max_val)

    best_max_val = torch.cat(best_max_val_all, dim=0)

    return best_max_val.squeeze(1)


@torch.no_grad()
def apply_clip(module, clip_list, get_op_by_name_fn):
    for name, max_val in clip_list:
        layer = get_op_by_name_fn(module, name)
        max_val = max_val.to(layer.weight.device)
        org_shape = layer.weight.shape
        layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
        layer.weight.data = torch.clamp(layer.weight.data, -max_val, max_val)
        layer.weight.data = layer.weight.data.reshape(org_shape)
