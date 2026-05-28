"""Pluggable KV cache backends for attention layers.

Provides:
  - KVCacheBackend: abstract interface
  - DefaultKVCacheBackend: dense FP16/BF16 cache (current mini-vllm behaviour)
  - KiviKVCacheBackend: 2/4-bit asymmetric KV cache quantization (KIVI)
"""

import math
from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
import torch.nn.functional as F


class KVCacheBackend(ABC):
    """Abstract interface for KV cache storage and retrieval."""

    @abstractmethod
    def update(
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
    def get_kv(self, total_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
            batch_size, num_kv_heads, max_seq_len, head_dim,
            dtype=dtype, device=device,
        )
        self.v_cache = torch.zeros(
            batch_size, num_kv_heads, max_seq_len, head_dim,
            dtype=dtype, device=device,
        )

    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len

    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        cache_len: int,
        is_prefill: bool,
    ) -> None:
        seq_len = k.shape[2]
        self.k_cache[:, :, cache_len : cache_len + seq_len, :] = k
        self.v_cache[:, :, cache_len : cache_len + seq_len, :] = v

    def get_kv(self, total_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.k_cache[:, :, :total_len, :],
            self.v_cache[:, :, :total_len, :],
        )

    def reset(self) -> None:
        self.k_cache.zero_()
        self.v_cache.zero_()


class KiviKVCacheBackend(KVCacheBackend):
    """KIVI-style asymmetric grouped KV cache quantization.

    K is quantized per-channel (along seq_len), V is quantized per-token
    (along head_dim).  The most recent *residual_length* tokens are kept in
    full FP16 precision; older tokens are quantized to *k_bits* / *v_bits*.

    Phase-1 implementation: quantizes storage; dequantizes on read so that
    the attention layer can reuse existing SDPA / manual matmul paths.
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
            unpack_and_dequant_kcache,
            unpack_and_dequant_vcache,
        )

        self._pack = quantize_and_pack_kcache
        self._unpack_k = unpack_and_dequant_kcache
        self._pack_v = quantize_and_pack_vcache
        self._unpack_v = unpack_and_dequant_vcache

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

        # ---- K cache state ----
        # Quantized chunks: each is (B, nh, D, T_i // feat_per_int) int32
        self._k_quant_codes: List[torch.Tensor] = []
        self._k_scales: List[torch.Tensor] = []
        self._k_mns: List[torch.Tensor] = []
        self._k_quant_lens: List[int] = []

        # Full-precision residual: (B, nh, L, D) where L <= residual_length
        self._k_full = torch.zeros(
            batch_size, num_kv_heads, 0, head_dim,
            dtype=dtype, device=device,
        )

        # ---- V cache state ----
        self._v_quant_codes: List[torch.Tensor] = []
        self._v_scales: List[torch.Tensor] = []
        self._v_mns: List[torch.Tensor] = []
        self._v_quant_lens: List[int] = []

        self._v_full = torch.zeros(
            batch_size, num_kv_heads, 0, head_dim,
            dtype=dtype, device=device,
        )

        self._total_len = 0

    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _quantize_k_chunk(self, k_fp16: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize a K chunk (B, nh, T, D) and return (code, scale, mn).

        K is quantized per-channel: group along T, pack along T.
        """
        code, scale, mn = self._pack(k_fp16, self.group_size, self.k_bits)
        return code, scale, mn

    def _dequantize_k_chunks(self) -> torch.Tensor:
        """Dequantize all K chunks and concatenate along T.

        Returns (B, nh, T_quantized, D).
        """
        if not self._k_quant_codes:
            return torch.zeros(
                self._batch_size, self._num_kv_heads, 0, self._head_dim,
                dtype=self._dtype, device=self._device,
            )
        chunks = []
        for code, scale, mn, t_len in zip(
            self._k_quant_codes, self._k_scales, self._k_mns, self._k_quant_lens
        ):
            chunk = self._unpack_k(code, scale, mn, self.group_size, self.k_bits, out_dtype=self._dtype)
            chunks.append(chunk)
        return torch.cat(chunks, dim=2)

    def _quantize_v_chunk(self, v_fp16: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize a V chunk (B, nh, T, D) and return (code, scale, mn).

        V is quantized per-token: group along D, pack along D.
        """
        code, scale, mn = self._pack_v(v_fp16, self.group_size, self.v_bits)
        return code, scale, mn

    def _dequantize_v_chunks(self) -> torch.Tensor:
        """Dequantize all V chunks and concatenate along T.

        Returns (B, nh, T_quantized, D).
        """
        if not self._v_quant_codes:
            return torch.zeros(
                self._batch_size, self._num_kv_heads, 0, self._head_dim,
                dtype=self._dtype, device=self._device,
            )
        chunks = []
        for code, scale, mn, t_len in zip(
            self._v_quant_codes, self._v_scales, self._v_mns, self._v_quant_lens
        ):
            chunk = self._unpack_v(code, scale, mn, self.group_size, self.v_bits, out_dtype=self._dtype)
            chunks.append(chunk)
        return torch.cat(chunks, dim=2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        cache_len: int,
        is_prefill: bool,
    ) -> None:
        """KIVI update logic.

        Prefill:
          - If total tokens <= residual_length: keep all in full.
          - Else: quantize all but the last residual_length tokens.

        Decode (seq_len == 1 typically):
          - Append to full buffer.
          - K: when full reaches residual_length, quantize the whole chunk.
          - V: when full exceeds residual_length, quantize the oldest token.
        """
        seq_len = k.shape[2]

        if is_prefill:
            self.reset()
            total = seq_len
            if total <= self.residual_length:
                self._k_full = k.clone()
                self._v_full = v.clone()
            else:
                remainder = total % self.residual_length
                if remainder == 0:
                    k_quant_part = k.clone()
                    v_quant_part = v.clone()
                    self._k_full = torch.zeros(
                        self._batch_size, self._num_kv_heads, 0, self._head_dim,
                        dtype=self._dtype, device=self._device,
                    )
                    self._v_full = torch.zeros(
                        self._batch_size, self._num_kv_heads, 0, self._head_dim,
                        dtype=self._dtype, device=self._device,
                    )
                else:
                    k_quant_part = k[:, :, : total - remainder, :].contiguous()
                    v_quant_part = v[:, :, : total - remainder, :].contiguous()
                    self._k_full = k[:, :, -remainder:, :].contiguous()
                    self._v_full = v[:, :, -remainder:, :].contiguous()

                k_code, k_scale, k_mn = self._quantize_k_chunk(k_quant_part)
                v_code, v_scale, v_mn = self._quantize_v_chunk(v_quant_part)

                self._k_quant_codes = [k_code]
                self._k_scales = [k_scale]
                self._k_mns = [k_mn]
                self._k_quant_lens = [k_quant_part.shape[2]]

                self._v_quant_codes = [v_code]
                self._v_scales = [v_scale]
                self._v_mns = [v_mn]
                self._v_quant_lens = [v_quant_part.shape[2]]
            self._total_len = total
            return

        # ---- Decode ----
        # Append new token(s) to full buffer
        self._k_full = torch.cat([self._k_full, k], dim=2)
        self._v_full = torch.cat([self._v_full, v], dim=2)
        self._total_len += seq_len

        # --- K: quantize whole chunk when full reaches residual_length ---
        if self._k_full.shape[2] == self.residual_length:
            k_code, k_scale, k_mn = self._quantize_k_chunk(self._k_full)
            self._k_quant_codes.append(k_code)
            self._k_scales.append(k_scale)
            self._k_mns.append(k_mn)
            self._k_quant_lens.append(self.residual_length)
            self._k_full = torch.zeros(
                self._batch_size, self._num_kv_heads, 0, self._head_dim,
                dtype=self._dtype, device=self._device,
            )

        # --- V: quantize oldest token when full exceeds residual_length ---
        if self._v_full.shape[2] > self.residual_length:
            # oldest token is at position 0
            v_old = self._v_full[:, :, :1, :].contiguous()
            v_code, v_scale, v_mn = self._quantize_v_chunk(v_old)
            self._v_quant_codes.append(v_code)
            self._v_scales.append(v_scale)
            self._v_mns.append(v_mn)
            self._v_quant_lens.append(1)
            self._v_full = self._v_full[:, :, 1:, :].contiguous()

    def get_kv(self, total_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return dequantized K and V for attention computation."""
        k_deq = self._dequantize_k_chunks()
        v_deq = self._dequantize_v_chunks()
        k = torch.cat([k_deq, self._k_full], dim=2)
        v = torch.cat([v_deq, self._v_full], dim=2)
        assert k.shape[2] == total_len, f"K len mismatch: {k.shape[2]} vs {total_len}"
        assert v.shape[2] == total_len, f"V len mismatch: {v.shape[2]} vs {total_len}"
        return k, v

    def reset(self) -> None:
        self._k_quant_codes.clear()
        self._k_scales.clear()
        self._k_mns.clear()
        self._k_quant_lens.clear()
        self._v_quant_codes.clear()
        self._v_scales.clear()
        self._v_mns.clear()
        self._v_quant_lens.clear()
        self._k_full = torch.zeros(
            self._batch_size, self._num_kv_heads, 0, self._head_dim,
            dtype=self._dtype, device=self._device,
        )
        self._v_full = torch.zeros(
            self._batch_size, self._num_kv_heads, 0, self._head_dim,
            dtype=self._dtype, device=self._device,
        )
        self._total_len = 0
