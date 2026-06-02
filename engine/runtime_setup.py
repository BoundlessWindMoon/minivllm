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
    kv_cfg = getattr(cfg.inference, "kv_cache", None)
    if kv_cfg and kv_cfg.backend in ("kivi", "default"):
        from layers.kv_cache import create_kv_backend

        def _find_kv_backend_module(module, depth=0):
            """Recursively find the first submodule that has a kv_backend attr."""
            if depth > 3:
                return None
            if hasattr(module, "kv_backend"):
                return module
            for child in module.children():
                found = _find_kv_backend_module(child, depth + 1)
                if found is not None:
                    return found
            return None

        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is not None:
            for layer in layers:
                attn_module = _find_kv_backend_module(layer)
                if attn_module is not None:
                    attn_module.kv_backend = create_kv_backend(
                        backend=kv_cfg.backend,
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
            if kv_cfg.backend == "kivi":
                logger.info(
                    f"[KVCache] Injected KiviKVCacheBackend "
                    f"(k_bits={kv_cfg.k_bits}, v_bits={kv_cfg.v_bits}, "
                    f"group_size={kv_cfg.group_size}, residual={kv_cfg.residual_length})"
                )
            else:
                logger.info("[KVCache] Using default dense FP16/BF16 cache.")

    return model
