"""Multimodal processor loading."""

from transformers import AutoProcessor

from utils.logger import logger


def load_processor(cfg):
    """Load AutoProcessor for multimodal inference if enabled.

    Returns None when multimodal is disabled.
    """
    if getattr(cfg.inference, "multimodal", None) and cfg.inference.multimodal.enabled:
        processor = AutoProcessor.from_pretrained(
            cfg.path.model_path, trust_remote_code=True
        )
        logger.info("Loaded AutoProcessor for multimodal inference.")
        return processor
    return None
