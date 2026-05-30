"""Runtime optimization patches applied after model loading.

Examples:
    KIVI KV-cache backend injection
    Megakernel greedy fast-path toggle
"""

from utils.logger import logger


def apply_runtime_patches(model, cfg):
    """Apply runtime optimizations that depend on the fully loaded model.

    Currently handles:
      - Megakernel greedy fast path
      - KIVI KV-cache backend injection
    """
    if hasattr(model, "greedy_fast_path"):
        sampling = cfg.inference.sampling
        if (
            sampling.sample_method == "greedy"
            and sampling.temperature == 1.0
            and sampling.topp == 1.0
        ):
            model.greedy_fast_path = True
            logger.info("Megakernel: enabled greedy fast path (kernel argmax only)")

    # Inject KV-cache backend if requested
    if (
        getattr(cfg.inference, "kv_cache", None)
        and cfg.inference.kv_cache.backend == "kivi"
    ):
        from layers.kv_cache import KiviKVCacheBackend

        kv_cfg = cfg.inference.kv_cache
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is not None:
            for layer in layers:
                attn_module = getattr(getattr(layer, "self_attn", None), "attn", None)
                if attn_module is not None and hasattr(attn_module, "kv_backend"):
                    attn_module.kv_backend = KiviKVCacheBackend(
                        batch_size=1,
                        num_kv_heads=attn_module.num_kv_heads,
                        max_seq_len=attn_module.max_seq_len,
                        head_dim=attn_module.head_dim,
                        k_bits=kv_cfg.k_bits,
                        v_bits=kv_cfg.v_bits,
                        group_size=kv_cfg.group_size,
                        residual_length=kv_cfg.residual_length,
                        device=cfg.env.device,
                        dtype=cfg.env.get_torch_dtype(),
                    )
            logger.info(
                f"[KVCache] Injected KiviKVCacheBackend "
                f"(k_bits={kv_cfg.k_bits}, v_bits={kv_cfg.v_bits}, "
                f"group_size={kv_cfg.group_size}, residual={kv_cfg.residual_length})"
            )

    return model
