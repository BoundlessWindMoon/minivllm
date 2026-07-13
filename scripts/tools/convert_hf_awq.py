"""Convert a HuggingFace AutoAWQ GEMM checkpoint into the project's Wt cache.

Source format (HF AutoAWQ v0.2.x, version='gemm'):
    qweight  (in,            out / 8) int32, pack_order = AWQ_ORDER [0,2,4,6,1,3,5,7]
    qzeros   (in/group,      out / 8) int32, same order
    scales   (in/group,      out)     bf16/fp16
    q/k/v_proj and gate/up_proj are stored separately.

Target format (project Wt cache, see quantization/quantized_linear_wt.py):
    qweight, qzeros: same shapes, pack_order = sequential [0..7]
    scales, unpack_zeros, zero_scales: derived
    qkv_proj and gate_up_proj are FUSED along the out_features axis.

The round-trip is bit-exact: HF integers (after reverse_awq_order) are repacked
directly under sequential order — no dense fp dequant + round step, so memory
peaks scale with int8 (in, out_fused) instead of fp32, and we don't accumulate
any rounding error.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from awq.utils.packing_utils import unpack_awq, reverse_awq_order


PROJ_GROUPS = {
    "self_attn.qkv_proj": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    "self_attn.o_proj": ["self_attn.o_proj"],
    "mlp.gate_up_proj": ["mlp.gate_proj", "mlp.up_proj"],
    "mlp.down_proj": ["mlp.down_proj"],
}

PASSTHROUGH_PER_LAYER = [
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
]

PASSTHROUGH_GLOBAL = [
    "model.embed_tokens.weight",
    "model.norm.weight",
    "lm_head.weight",
]


def open_shards(src: Path):
    """Return (index, {tensor_name: shard_handle}) for lazy reads."""
    index_path = src / "model.safetensors.index.json"
    if not index_path.exists():
        single = src / "model.safetensors"
        if not single.exists():
            raise FileNotFoundError(f"Neither index nor single safetensors at {src}")
        f = safe_open(single, framework="pt")
        return {"weight_map": {k: "model.safetensors" for k in f.keys()}}, {
            k: f for k in f.keys()
        }
    with open(index_path) as f:
        idx = json.load(f)
    handles: Dict[str, object] = {}
    shard_handles: Dict[str, object] = {}
    for name, shard in idx["weight_map"].items():
        if shard not in shard_handles:
            shard_handles[shard] = safe_open(src / shard, framework="pt")
        handles[name] = shard_handles[shard]
    return idx, handles


def read_tensor(handles, name: str) -> torch.Tensor:
    return handles[name].get_tensor(name)


def unpack_one(qweight, qzeros, bits=4):
    """HF GEMM -> (intweight (in, out) int8, izeros (num_groups, out) int8). Order reversed."""
    iw, iz = unpack_awq(qweight, qzeros, bits)
    iw, iz = reverse_awq_order(iw, iz, bits)
    iw = torch.bitwise_and(iw, (1 << bits) - 1)
    iz = torch.bitwise_and(iz, (1 << bits) - 1)
    return iw, iz


def _pack_sequential(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack int values (last dim) into int32, sequential order [0..pack_num-1].

    values: int tensor in [0, 2**bits-1], last dim must be a multiple of pack_num.
    Returns int32 tensor with shape values.shape[:-1] + (last_dim / pack_num,).
    """
    pack_num = 32 // bits
    assert values.shape[-1] % pack_num == 0
    shifts = torch.arange(0, 32, bits, dtype=torch.int32, device=values.device)
    view_shape = values.shape[:-1] + (values.shape[-1] // pack_num, pack_num)
    v = values.to(torch.int32).view(view_shape)
    # Non-overlapping bit positions: sum == bitwise OR for ints.
    return (v << shifts).sum(dim=-1).to(torch.int32)


def quantize_dense_w_to_wt(
    slice_obj,
    out_features: int,
    in_features: int,
    group_size: int,
    bits: int,
    device: torch.device,
    chunk_out: int = 8192,
) -> Dict[str, torch.Tensor]:
    """Quantize a dense (out, in) weight to the project's Wt format.

    Per-group min/max symmetric quantization (no AWQ activation-aware search).
    Used for tensors that ship dense in the HF source — e.g. `lm_head` in
    Qwen3-8B-AWQ, where there is no existing integer encoding to repack from.

    `slice_obj` is a safetensors slice handle (`safe_open(...).get_slice(name)`).
    Only one `chunk_out`-column slice is materialised at a time, so the source
    tensor never lives on CPU as a single allocation and GPU peak stays well
    under 500 MB regardless of total size.
    """
    assert in_features % group_size == 0, (in_features, group_size)
    num_groups = in_features // group_size
    qmax = (1 << bits) - 1
    pack_num = 32 // bits
    assert out_features % pack_num == 0, out_features
    assert chunk_out % pack_num == 0, chunk_out

    qw_chunks: List[torch.Tensor] = []
    qz_chunks: List[torch.Tensor] = []
    sc_chunks: List[torch.Tensor] = []
    iz_chunks: List[torch.Tensor] = []

    for c0 in range(0, out_features, chunk_out):
        c1 = min(c0 + chunk_out, out_features)
        # slice_obj[c0:c1] returns (chunk, in) on CPU in source dtype (bf16/fp16).
        W_chunk = slice_obj[c0:c1]
        Wc = W_chunk.to(device=device, dtype=torch.float32).t().contiguous()
        Wc_g = Wc.view(num_groups, group_size, c1 - c0)
        mn = Wc_g.amin(dim=1)
        mx = Wc_g.amax(dim=1)
        scale = (mx - mn).clamp(min=1e-5) / qmax
        zero = (-(mn / scale)).round().clamp(0, qmax)
        intw = (
            (Wc_g / scale.unsqueeze(1) + zero.unsqueeze(1))
            .round()
            .clamp(0, qmax)
            .to(torch.int32)
        )
        intw = intw.view(in_features, c1 - c0)
        qweight_chunk = _pack_sequential(intw, bits)
        qzeros_chunk = _pack_sequential(zero.to(torch.int32), bits)

        qw_chunks.append(qweight_chunk.cpu())
        qz_chunks.append(qzeros_chunk.cpu())
        sc_chunks.append(scale.to(torch.float16).cpu())
        iz_chunks.append(zero.cpu())
        del W_chunk, Wc, Wc_g, mn, mx, scale, zero, intw, qweight_chunk, qzeros_chunk
        if device.type == "cuda":
            torch.cuda.empty_cache()

    qweight = torch.cat(qw_chunks, dim=1).contiguous()
    qw_chunks.clear()
    qzeros = torch.cat(qz_chunks, dim=1).contiguous()
    qz_chunks.clear()
    scales = torch.cat(sc_chunks, dim=1).contiguous()
    sc_chunks.clear()
    izeros = torch.cat(iz_chunks, dim=1)
    iz_chunks.clear()
    unpack_zeros = izeros.to(torch.float16).contiguous()
    zero_scales = (unpack_zeros * scales).to(torch.float16).contiguous()
    del izeros

    return {
        "qweight": qweight,
        "qzeros": qzeros,
        "scales": scales,
        "unpack_zeros": unpack_zeros,
        "zero_scales": zero_scales,
    }


def repack_group(
    handles,
    src_prefix: str,
    parts: List[str],
    group_size: int,
    bits: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Read `parts`, fuse along out_features, repack to Wt-sequential.

    Avoids the dense fp32 dequant + round path used by `WQLinear_Wt.from_linear`
    by re-packing the already-correct AWQ integers directly. Bit-exact by
    construction, and peak memory is proportional to (in, out_fused) int8
    instead of (out_fused, in) fp32 — fits on a 6 GB GPU.
    """
    iws, izs, scs = [], [], []
    for p in parts:
        qweight = read_tensor(handles, f"{src_prefix}.{p}.qweight").to(device)
        qzeros = read_tensor(handles, f"{src_prefix}.{p}.qzeros").to(device)
        scales = read_tensor(handles, f"{src_prefix}.{p}.scales").to(device)
        iw, iz = unpack_one(qweight, qzeros, bits)
        iws.append(iw)
        izs.append(iz)
        scs.append(scales)
        del qweight, qzeros, scales

    intweight = torch.cat(iws, dim=1)   # (in,         out_fused) int8
    izeros = torch.cat(izs, dim=1)      # (num_groups, out_fused) int8
    scales = torch.cat(scs, dim=1)      # (num_groups, out_fused) bf16/fp16
    del iws, izs, scs

    in_features, out_features = intweight.shape
    num_groups = scales.shape[0]
    assert in_features % group_size == 0
    assert num_groups == in_features // group_size
    assert out_features % (32 // bits) == 0

    qweight = _pack_sequential(intweight, bits)              # (in,         out_fused/8) int32
    qzeros = _pack_sequential(izeros, bits)                  # (num_groups, out_fused/8) int32
    scales_fp16 = scales.to(torch.float16).contiguous()      # (num_groups, out_fused)
    unpack_zeros = izeros.to(torch.float16).contiguous()     # (num_groups, out_fused)
    zero_scales = (unpack_zeros * scales_fp16).to(torch.float16)

    out: Dict[str, torch.Tensor] = {
        "qweight": qweight.detach().cpu().contiguous(),
        "qzeros": qzeros.detach().cpu().contiguous(),
        "scales": scales_fp16.detach().cpu().contiguous(),
        "unpack_zeros": unpack_zeros.detach().cpu().contiguous(),
        "zero_scales": zero_scales.detach().cpu().contiguous(),
    }
    del intweight, izeros, scales, qweight, qzeros, scales_fp16, unpack_zeros, zero_scales
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def verify_layer(handles, layer_idx: int, group_size: int, bits: int, device: torch.device):
    """Round-trip integer test on layer_idx's q_proj.

    Compares the integers unpacked from the project's qweight/qzeros (sequential
    order) against the original HF integers (AWQ_ORDER, after reverse). Equality
    here is bit-exact by construction.
    """
    src_prefix = f"model.layers.{layer_idx}"
    p = "self_attn.q_proj"
    qweight = read_tensor(handles, f"{src_prefix}.{p}.qweight").to(device)
    qzeros = read_tensor(handles, f"{src_prefix}.{p}.qzeros").to(device)
    scales = read_tensor(handles, f"{src_prefix}.{p}.scales").to(device)

    iw_hf, iz_hf = unpack_one(qweight, qzeros, bits)
    iw_hf = iw_hf.to(torch.int32)
    iz_hf = iz_hf.to(torch.int32)

    bufs = repack_group(handles, src_prefix, [p], group_size, bits, device)
    qweight_proj = bufs["qweight"].to(device)
    qzeros_proj = bufs["qzeros"].to(device)

    shifts = torch.arange(0, 32, bits, device=device, dtype=torch.int32)
    mask = (1 << bits) - 1
    iw_proj = (
        (qweight_proj.unsqueeze(-1) >> shifts) & mask
    ).reshape(qweight_proj.shape[0], -1).to(torch.int32)
    iz_proj = (
        (qzeros_proj.unsqueeze(-1) >> shifts) & mask
    ).reshape(qzeros_proj.shape[0], -1).to(torch.int32)

    iw_diff = (iw_hf - iw_proj).abs().max().item()
    iz_diff = (iz_hf - iz_proj).abs().max().item()
    print(
        f"[verify] layer {layer_idx} {p}: "
        f"intweight max|diff|={iw_diff} (shape={tuple(iw_hf.shape)}), "
        f"izeros max|diff|={iz_diff} (shape={tuple(iz_hf.shape)})"
    )
    if iw_diff != 0 or iz_diff != 0:
        raise RuntimeError(
            f"Integer round-trip failed (intweight={iw_diff}, izeros={iz_diff})"
        )

    scales_stored = bufs["scales"].to(device)
    scales_expected = scales.to(torch.float16)
    sc_diff = (scales_stored - scales_expected).abs().max().item()
    print(f"[verify] scales (bf16->fp16) max|diff|={sc_diff:.3e}")
    print("[verify] BIT-EXACT integer round-trip")


class ShardWriter:
    """Stream-write tensors into multiple safetensors shards.

    Keeps peak host RAM bounded by `shard_max_bytes` regardless of model size:
    when the current buffer would exceed the limit, it is flushed to disk and
    released. After `finalize()`, the cache layout is either a single
    `model.safetensors` (if everything fit in one shard) or a sharded layout
    with `model-{i:05d}-of-{N:05d}.safetensors` files plus
    `model.safetensors.index.json`.
    """

    def __init__(self, dst_dir: Path, shard_max_bytes: int):
        self.dst_dir = dst_dir
        self.shard_max_bytes = shard_max_bytes
        self.current: Dict[str, torch.Tensor] = {}
        self.current_bytes = 0
        self.shard_idx = 0
        self.shard_files: List[tuple[str, list[str]]] = []
        self.total_tensors = 0

    def add(self, name: str, t: torch.Tensor):
        t = t.detach().cpu().contiguous()
        nbytes = t.numel() * t.element_size()
        if self.current_bytes + nbytes > self.shard_max_bytes and self.current:
            self._flush()
        self.current[name] = t
        self.current_bytes += nbytes
        self.total_tensors += 1

    def _flush(self):
        if not self.current:
            return
        temp_name = f"_tmp_shard_{self.shard_idx:05d}.safetensors"
        save_file(self.current, str(self.dst_dir / temp_name))
        keys = list(self.current.keys())
        self.shard_files.append((temp_name, keys))
        self.current = {}
        self.current_bytes = 0
        self.shard_idx += 1
        print(
            f"[shard] flushed shard {self.shard_idx} "
            f"({len(keys)} tensors, dst={self.dst_dir / temp_name})"
        )

    def finalize(self) -> tuple[int, bool]:
        """Returns (total_size_bytes, is_single_file)."""
        self._flush()
        n = len(self.shard_files)
        if n == 0:
            raise RuntimeError("ShardWriter: no tensors written")

        if n == 1:
            single = self.dst_dir / "model.safetensors"
            (self.dst_dir / self.shard_files[0][0]).rename(single)
            return single.stat().st_size, True

        weight_map: Dict[str, str] = {}
        total_size = 0
        for i, (temp_name, keys) in enumerate(self.shard_files):
            final_name = f"model-{i + 1:05d}-of-{n:05d}.safetensors"
            (self.dst_dir / temp_name).rename(self.dst_dir / final_name)
            for k in keys:
                weight_map[k] = final_name
            total_size += (self.dst_dir / final_name).stat().st_size

        index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        with open(self.dst_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)
        return total_size, False


def convert(
    src: Path,
    dst: Path,
    device: torch.device,
    verify_first: int | None,
    overwrite: bool,
    verify_only: bool = False,
    shard_max_gb: float = 1.5,
    quantize_lm_head: bool = True,
):
    if not verify_only and dst.exists() and any(dst.iterdir()):
        if not overwrite:
            raise RuntimeError(
                f"Destination {dst} exists and is not empty (pass --overwrite)"
            )

    config_path = src / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)

    qcfg = cfg.get("quantization_config", {})
    bits = qcfg.get("bits", 4)
    group_size = qcfg.get("group_size", 128)
    has_zero_point = qcfg.get("zero_point", True)
    version = qcfg.get("version", "gemm")
    if bits != 4 or version != "gemm" or not has_zero_point:
        raise RuntimeError(
            f"Unsupported AWQ config: bits={bits} version={version} zero_point={has_zero_point}"
        )

    num_layers = cfg["num_hidden_layers"]
    tie_emb = cfg.get("tie_word_embeddings", False)

    idx, handles = open_shards(src)

    if verify_first is not None:
        verify_layer(handles, verify_first, group_size, bits, device)

    if verify_only:
        print("[convert] --verify-only set: exiting after verification")
        return

    dst.mkdir(parents=True, exist_ok=True)
    # Clean leftover shard files from a previous failed run.
    for stale in dst.glob("_tmp_shard_*.safetensors"):
        stale.unlink()
    for stale in dst.glob("model-*-of-*.safetensors"):
        stale.unlink()

    writer = ShardWriter(dst, int(shard_max_gb * (1 << 30)))
    quantized_layers: List[str] = []

    for layer_idx in range(num_layers):
        src_prefix = f"model.layers.{layer_idx}"
        for fused_suffix, parts in PROJ_GROUPS.items():
            dst_name = f"{src_prefix}.{fused_suffix}"
            bufs = repack_group(handles, src_prefix, parts, group_size, bits, device)
            for k, v in bufs.items():
                writer.add(f"{dst_name}.{k}", v)
            del bufs
            quantized_layers.append(dst_name)
        for ptname in PASSTHROUGH_PER_LAYER:
            key = f"{src_prefix}.{ptname}"
            if key in idx["weight_map"]:
                t = read_tensor(handles, key).clone().cpu()
                writer.add(key, t)
                del t
        print(f"[convert] layer {layer_idx}/{num_layers - 1} done")

    for key in PASSTHROUGH_GLOBAL:
        if key == "lm_head.weight" and tie_emb and key not in idx["weight_map"]:
            continue
        if key == "lm_head.weight" and quantize_lm_head:
            # handled below — write quantized buffers instead of bf16 passthrough
            continue
        if key not in idx["weight_map"]:
            print(f"[convert] (skip missing) {key}")
            continue
        t = read_tensor(handles, key).clone().cpu()
        writer.add(key, t)
        del t

    if quantize_lm_head and "lm_head.weight" in idx["weight_map"]:
        print("[convert] quantizing lm_head (per-group min/max, no AWQ search) ...")
        slice_obj = handles["lm_head.weight"].get_slice("lm_head.weight")
        shape = slice_obj.get_shape()
        out_f, in_f = int(shape[0]), int(shape[1])
        bufs = quantize_dense_w_to_wt(slice_obj, out_f, in_f, group_size, bits, device)
        for k, v in bufs.items():
            writer.add(f"lm_head.{k}", v)
        del bufs
        quantized_layers.append("lm_head")
        print("[convert] lm_head quantized")

    print(f"[convert] finalizing ({writer.total_tensors} tensors) ...")
    total_bytes, is_single = writer.finalize()

    quant_info = {
        "quant_method": "AWQ",
        "quant_bits": bits,
        "group_size": group_size,
        "has_zero_point": True,
        "layout": "Wt",
        "pack_order": "sequential",
        "quantized_layers": quantized_layers,
        "scaled_activations": {},
        "tie_word_embeddings": tie_emb,
    }
    with open(dst / "quant_config.json", "w") as f:
        json.dump(quant_info, f, indent=2)

    shutil.copy2(config_path, dst / "config.json")

    layout_label = "single model.safetensors" if is_single else "multi-shard"
    print(
        f"[convert] done. cache at {dst} ({layout_label}, "
        f"{total_bytes / 1e9:.2f} GB total, "
        f"{len(quantized_layers)} quantized fused layers)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path, help="HF AutoAWQ checkpoint dir")
    ap.add_argument("--dst", required=True, type=Path, help="Output Wt cache dir")
    ap.add_argument("--device", default="cuda:0", help="Compute device for packing")
    ap.add_argument(
        "--verify-layer",
        type=int,
        default=0,
        help="Run a round-trip self-check on this layer before bulk conversion (negative to skip)",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--verify-only",
        action="store_true",
        help="Run verification only, do not write the cache",
    )
    ap.add_argument(
        "--shard-max-gb",
        type=float,
        default=1.5,
        help="Max size of each safetensors shard in GB (lower = less peak host RAM)",
    )
    ap.add_argument(
        "--quantize-lm-head",
        dest="quantize_lm_head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Quantize lm_head with per-group min/max (no AWQ search). Default on. "
             "Use --no-quantize-lm-head to keep it as bf16 pass-through.",
    )
    args = ap.parse_args()

    src = args.src.expanduser().resolve()
    dst = args.dst.expanduser().resolve()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    verify = args.verify_layer if args.verify_layer >= 0 else None
    convert(
        src,
        dst,
        device,
        verify,
        args.overwrite,
        verify_only=args.verify_only,
        shard_max_gb=args.shard_max_gb,
        quantize_lm_head=args.quantize_lm_head,
    )


if __name__ == "__main__":
    main()
