"""Pluggable KV cache backends for attention layers.

Provides:
  - KVCacheBackend: abstract interface
  - DefaultKVCacheBackend: dense FP16/BF16 cache (current mini-vllm behaviour)
  - KiviKVCacheBackend: 2/4-bit asymmetric KV cache quantization (KIVI)
"""

from abc import ABC, abstractmethod
from typing import Tuple

import torch

# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class KVCacheBackend(ABC):
    """Abstract interface for KV cache storage and retrieval."""

    @abstractmethod
    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        cache_len: int,
        is_prefill: bool,
    ) -> None:
        """Write new k, v into the cache starting at *cache_len*.

        Args:
            k: (batch, num_kv_heads, seq_len, head_dim)
            v: (batch, num_kv_heads, seq_len, head_dim)
            cache_len: cached sequence length before this update
            is_prefill: whether this is the prefill phase
        """
        ...

    @abstractmethod
    def load_kv(self, total_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return full k and v up to *total_len* for attention.

        Returns:
            k, v of shape (batch, num_kv_heads, total_len, head_dim)
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all cached data."""
        ...

    @property
    @abstractmethod
    def max_seq_len(self) -> int:
        """Maximum sequence length this backend can hold."""
        ...


# ---------------------------------------------------------------------------
# Default dense cache
# ---------------------------------------------------------------------------


class DefaultKVCacheBackend(KVCacheBackend):
    """Standard dense FP16/BF16 KV cache."""

    def __init__(
        self,
        batch_size: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self._batch_size = batch_size
        self._num_kv_heads = num_kv_heads
        self._max_seq_len = max_seq_len
        self._head_dim = head_dim
        self._device = device
        self._dtype = dtype

        self.k_cache = torch.zeros(
            batch_size,
            num_kv_heads,
            max_seq_len,
            head_dim,
            dtype=dtype,
            device=device,
        )
        self.v_cache = torch.zeros(
            batch_size,
            num_kv_heads,
            max_seq_len,
            head_dim,
            dtype=dtype,
            device=device,
        )

    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        cache_len: int,
        is_prefill: bool,
    ) -> None:
        seq_len = k.shape[2]
        self.k_cache[:, :, cache_len : cache_len + seq_len, :] = k
        self.v_cache[:, :, cache_len : cache_len + seq_len, :] = v

    def load_kv(self, total_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.k_cache[:, :, :total_len, :],
            self.v_cache[:, :, :total_len, :],
        )

    def reset(self) -> None:
        self.k_cache.zero_()
        self.v_cache.zero_()


# ---------------------------------------------------------------------------
# KIVI quantized cache
# ---------------------------------------------------------------------------


class KiviKVCacheBackend(KVCacheBackend):
    """KIVI-style asymmetric grouped KV cache quantization.

    K is quantized per-channel (along seq_len), V is quantized per-token
    (along head_dim).  The most recent *residual_length* tokens are kept in
    full FP16 precision; older tokens are quantized to *k_bits* / *v_bits*.

    All quantized buffers are pre-allocated to max_seq_len to eliminate
    dynamic torch.cat allocations during decode.
    """

    def __init__(
        self,
        batch_size: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        k_bits: int = 2,
        v_bits: int = 2,
        group_size: int = 32,
        residual_length: int = 32,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        from kernels.kivi.quant_pack import (
            quantize_and_pack_kcache,
            quantize_and_pack_vcache,
        )

        self._pack = quantize_and_pack_kcache
        self._pack_v = quantize_and_pack_vcache

        self._batch_size = batch_size
        self._num_kv_heads = num_kv_heads
        self._max_seq_len = max_seq_len
        self._head_dim = head_dim
        self._device = device
        self._dtype = dtype

        self.k_bits = k_bits
        self.v_bits = v_bits
        self.group_size = group_size
        self.residual_length = residual_length

        k_feat_per_int = 32 // k_bits
        v_feat_per_int = 32 // v_bits

        # ---- K cache state ----
        # _k_quant_cache: packed int32 codes, shape (B, nh, T//k_feat_per_int, D)
        # _k_scale_cache: per-group scale, shape (B, nh, T//group_size, D)
        # _k_mn_cache: per-group min, shape (B, nh, T//group_size, D)
        # _k_full: full-precision residual buffer. Allocated to max_seq_len because
        #          K chunks can be large (multiples of group_size).
        self._k_quant_cache = torch.zeros(
            batch_size,
            num_kv_heads,
            max_seq_len // k_feat_per_int,
            head_dim,
            dtype=torch.int32,
            device=device,
        )
        self._k_scale_cache = torch.zeros(
            batch_size,
            num_kv_heads,
            max_seq_len // group_size,
            head_dim,
            dtype=dtype,
            device=device,
        )
        self._k_mn_cache = torch.zeros(
            batch_size,
            num_kv_heads,
            max_seq_len // group_size,
            head_dim,
            dtype=dtype,
            device=device,
        )
        self._k_full = torch.zeros(
            batch_size,
            num_kv_heads,
            max_seq_len,
            head_dim,
            dtype=dtype,
            device=device,
        )

        self._k_quant_len = 0  # packed seq len written
        self._k_scale_len = 0  # number of groups written
        self._k_full_len = 0  # full-precision tokens in _k_full

        # ---- V cache state ----
        # _v_quant_cache: packed int32 codes, shape (B, nh, T, D//v_feat_per_int)
        # _v_scale_cache: per-group scale, shape (B, nh, T, D//group_size)
        # _v_mn_cache: per-group min, shape (B, nh, T, D//group_size)
        # _v_full: full-precision residual buffer. Allocated to residual_length+1
        #          because V quantizes one token at a time and needs a 1-slot
        #          overflow before sliding.
        self._v_quant_cache = torch.zeros(
            batch_size,
            num_kv_heads,
            max_seq_len,
            head_dim // v_feat_per_int,
            dtype=torch.int32,
            device=device,
        )
        self._v_scale_cache = torch.zeros(
            batch_size,
            num_kv_heads,
            max_seq_len,
            head_dim // group_size,
            dtype=dtype,
            device=device,
        )
        self._v_mn_cache = torch.zeros(
            batch_size,
            num_kv_heads,
            max_seq_len,
            head_dim // group_size,
            dtype=dtype,
            device=device,
        )
        self._v_full = torch.zeros(
            batch_size,
            num_kv_heads,
            residual_length + 1,
            head_dim,
            dtype=dtype,
            device=device,
        )

        self._v_quant_len = 0
        self._v_scale_len = 0
        self._v_full_len = 0

        self._total_len = 0

    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _quantize_k(
        self, k_fp16: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize a K chunk (B, nh, T, D); K is per-channel: group along T, pack along T."""
        return self._pack(k_fp16, self.group_size, self.k_bits)

    def _quantize_v(
        self, v_fp16: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize a V chunk (B, nh, T, D); V is per-token: group along D, pack along D."""
        return self._pack_v(v_fp16, self.group_size, self.v_bits)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        _cache_len: int,
        is_prefill: bool,
    ) -> None:
        """KIVI store logic.

        Prefill:
          - If total tokens <= residual_length: keep all in full.
          - Else: quantize all but the last residual_length tokens.

        Decode (seq_len == 1 typically):
          - Append to full buffer.
          - K: when full exceeds residual_length, quantize oldest group_size chunk.
          - V: when full exceeds residual_length, quantize the oldest token.
        """
        seq_len = k.shape[2]

        if is_prefill:
            self.reset()
            total = seq_len
            if total <= self.residual_length:
                self._k_full[:, :, :total, :] = k
                self._v_full[:, :, :total, :] = v
                self._k_full_len = total
                self._v_full_len = total
            else:
                remainder = total % self.residual_length
                if remainder == 0:
                    n_quant = total
                else:
                    n_quant = total - remainder
                # align to group_size (required by quantize kernel)
                n_quant = (n_quant // self.group_size) * self.group_size

                if n_quant == total:
                    k_quant_part = k.contiguous()
                    self._k_full_len = 0
                else:
                    k_quant_part = k[:, :, :n_quant, :].contiguous()
                    self._k_full[:, :, :total - n_quant, :] = k[:, :, n_quant:, :]
                    self._k_full_len = total - n_quant

                v_quant_part = v[:, :, : -self.residual_length, :].contiguous()
                v_keep = self.residual_length
                self._v_full[:, :, :v_keep, :] = v[:, :, -v_keep:, :]
                self._v_full_len = v_keep

                if n_quant > 0:
                    k_code, k_scale, k_mn = self._quantize_k(k_quant_part)
                    self._k_quant_cache[:, :, : k_code.shape[2], :] = k_code
                    self._k_scale_cache[:, :, : k_scale.shape[2], :] = k_scale
                    self._k_mn_cache[:, :, : k_mn.shape[2], :] = k_mn
                    self._k_quant_len = k_code.shape[2]
                    self._k_scale_len = k_scale.shape[2]
                else:
                    self._k_quant_len = 0
                    self._k_scale_len = 0

                v_code, v_scale, v_mn = self._quantize_v(v_quant_part)
                self._v_quant_cache[:, :, : v_code.shape[2], :] = v_code
                self._v_scale_cache[:, :, : v_scale.shape[2], :] = v_scale
                self._v_mn_cache[:, :, : v_mn.shape[2], :] = v_mn
                self._v_quant_len = v_code.shape[2]
                self._v_scale_len = v_scale.shape[2]
            self._total_len = total
            return

        # ---- Decode ----
        self._k_full[:, :, self._k_full_len : self._k_full_len + seq_len, :] = k
        self._k_full_len += seq_len
        self._v_full[:, :, self._v_full_len : self._v_full_len + seq_len, :] = v
        self._v_full_len += seq_len
        self._total_len += seq_len

        if self._k_full_len == self.residual_length:
            n_quant = self._k_full_len
            n_quant = (n_quant // self.group_size) * self.group_size
            if n_quant > 0:
                k_old = self._k_full[:, :, :n_quant, :].contiguous()
                k_code, k_scale, k_mn = self._quantize_k(k_old)

                qsz = k_code.shape[2]
                self._k_quant_cache[
                    :, :, self._k_quant_len : self._k_quant_len + qsz, :
                ] = k_code
                self._k_quant_len += qsz
                ssz = k_scale.shape[2]
                self._k_scale_cache[
                    :, :, self._k_scale_len : self._k_scale_len + ssz, :
                ] = k_scale
                self._k_mn_cache[
                    :, :, self._k_scale_len : self._k_scale_len + ssz, :
                ] = k_mn
                self._k_scale_len += ssz

                self._k_full_len = 0

        if self._v_full_len > self.residual_length:
            v_old = self._v_full[:, :, :1, :].contiguous()
            v_code, v_scale, v_mn = self._quantize_v(v_old)

            self._v_quant_cache[:, :, self._v_quant_len : self._v_quant_len + 1, :] = (
                v_code
            )
            self._v_quant_len += 1
            self._v_scale_cache[:, :, self._v_scale_len : self._v_scale_len + 1, :] = (
                v_scale
            )
            self._v_mn_cache[:, :, self._v_scale_len : self._v_scale_len + 1, :] = v_mn
            self._v_scale_len += 1

            self._v_full[:, :, : self._v_full_len - 1, :] = self._v_full[
                :, :, 1 : self._v_full_len, :
            ].clone()
            self._v_full_len -= 1

    def _dequant_k(self, code, scale, mn):
        from kernels.kivi.quant_pack import unpack_and_dequant_kcache

        return unpack_and_dequant_kcache(
            code, scale, mn, self.group_size, self.k_bits, out_dtype=self._dtype
        )

    def _dequant_v(self, code, scale, mn):
        from kernels.kivi.quant_pack import unpack_and_dequant_vcache

        return unpack_and_dequant_vcache(
            code, scale, mn, self.group_size, self.v_bits, out_dtype=self._dtype
        )

    def load_kv(self, total_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return dequantized K and V for fallback attention computation.

        Dynamically dequantizes from packed int32 caches on every call.
        """
        k_parts = [self._k_full[:, :, : self._k_full_len, :]]
        if self._k_quant_len > 0:
            k_code = self._k_quant_cache[:, :, : self._k_quant_len, :]
            k_scale = self._k_scale_cache[:, :, : self._k_scale_len, :]
            k_mn = self._k_mn_cache[:, :, : self._k_scale_len, :]
            k_parts.insert(0, self._dequant_k(k_code, k_scale, k_mn))
        k = torch.cat(k_parts, dim=2) if len(k_parts) > 1 else k_parts[0]

        v_parts = [self._v_full[:, :, : self._v_full_len, :]]
        if self._v_quant_len > 0:
            v_code = self._v_quant_cache[:, :, : self._v_quant_len, :]
            v_scale = self._v_scale_cache[:, :, : self._v_scale_len, :]
            v_mn = self._v_mn_cache[:, :, : self._v_scale_len, :]
            v_parts.insert(0, self._dequant_v(v_code, v_scale, v_mn))
        v = torch.cat(v_parts, dim=2) if len(v_parts) > 1 else v_parts[0]

        assert k.shape[2] == total_len, f"K len mismatch: {k.shape[2]} vs {total_len}"
        assert v.shape[2] == total_len, f"V len mismatch: {v.shape[2]} vs {total_len}"
        return k, v

    def get_quantized_state(self):
        """Return quantized K/V tensors in the format expected by fused kernels.

        Returns:
            k_code, k_scale, k_mn, k_full,
            v_code, v_scale, v_mn, v_full
        """
        # Build empty tensors with correct shape dimensions so that downstream
        # code can query .shape[-1] for head_dim even when length is zero.
        if self._k_quant_len > 0:
            k_code = self._k_quant_cache[:, :, : self._k_quant_len, :]
            k_scale = self._k_scale_cache[:, :, : self._k_scale_len, :]
            k_mn = self._k_mn_cache[:, :, : self._k_scale_len, :]
        else:
            k_code = torch.empty(
                self._batch_size,
                self._num_kv_heads,
                0,
                self._head_dim,
                dtype=torch.int32,
                device=self._device,
            )
            k_scale = torch.empty(
                self._batch_size,
                self._num_kv_heads,
                0,
                self._head_dim,
                dtype=self._dtype,
                device=self._device,
            )
            k_mn = torch.empty(
                self._batch_size,
                self._num_kv_heads,
                0,
                self._head_dim,
                dtype=self._dtype,
                device=self._device,
            )

        if self._v_quant_len > 0:
            v_code = self._v_quant_cache[:, :, : self._v_quant_len, :]
            v_scale = self._v_scale_cache[:, :, : self._v_scale_len, :]
            v_mn = self._v_mn_cache[:, :, : self._v_scale_len, :]
        else:
            v_code = torch.empty(
                self._batch_size,
                self._num_kv_heads,
                0,
                self._head_dim // (32 // self.v_bits),
                dtype=torch.int32,
                device=self._device,
            )
            v_scale = torch.empty(
                self._batch_size,
                self._num_kv_heads,
                0,
                self._head_dim // self.group_size,
                dtype=self._dtype,
                device=self._device,
            )
            v_mn = torch.empty(
                self._batch_size,
                self._num_kv_heads,
                0,
                self._head_dim // self.group_size,
                dtype=self._dtype,
                device=self._device,
            )

        k_full = self._k_full[:, :, : self._k_full_len, :]
        v_full = self._v_full[:, :, : self._v_full_len, :]
        return k_code, k_scale, k_mn, k_full, v_code, v_scale, v_mn, v_full

    def reset(self) -> None:
        self._k_quant_len = 0
        self._k_scale_len = 0
        self._k_full_len = 0
        self._v_quant_len = 0
        self._v_scale_len = 0
        self._v_full_len = 0
        self._total_len = 0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_kv_backend(
    backend: str,
    batch_size: int,
    num_kv_heads: int,
    max_seq_len: int,
    head_dim: int,
    device: str,
    dtype: torch.dtype,
    k_bits: int = 2,
    v_bits: int = 2,
    group_size: int = 32,
    residual_length: int = 32,
) -> KVCacheBackend:
    """Create a KV cache backend by name.

    Args:
        backend: "default" for dense FP16/BF16 cache, "kivi" for KIVI quantization.
        batch_size: batch dimension.
        num_kv_heads: number of KV heads.
        max_seq_len: maximum sequence length.
        head_dim: head dimension.
        device: torch device string.
        dtype: torch dtype for full-precision buffers.
        k_bits: K quantization bits (KIVI only).
        v_bits: V quantization bits (KIVI only).
        group_size: quantization group size (KIVI only).
        residual_length: number of recent tokens kept in full precision (KIVI only).

    Returns:
        An instance of KVCacheBackend.
    """
    if backend == "kivi":
        return KiviKVCacheBackend(
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
            head_dim=head_dim,
            k_bits=k_bits,
            v_bits=v_bits,
            group_size=group_size,
            residual_length=residual_length,
            device=device,
            dtype=dtype,
        )
    elif backend == "default":
        return DefaultKVCacheBackend(
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
        )
    else:
        raise ValueError(
            f"Unknown KV cache backend: {backend}. Supported: default, kivi."
        )
