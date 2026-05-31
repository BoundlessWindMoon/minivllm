#!/usr/bin/env python3
"""Multi-turn chat CLI for mini-vllm.

Usage:
    python chat.py --config configs/default.yaml
    python chat.py --config configs/default.yaml --system "You are a helpful assistant."
"""

from __future__ import annotations

import argparse
import time

# Enable GNU readline for better terminal editing (arrow keys, history, etc.)
# when available.  This is a best-effort import; if readline is missing we
# fall back to plain input().
try:
    import readline
except ImportError:
    readline = None

import torch
import torch.distributed as dist

from utils.config import GlobalConfig, print_runtime_config
from engine.model_runner import ModelRunner
from engine.loader import load_model
from engine.runtime_setup import apply_runtime_patches
from chat.dialog_manager import DialogManager
from chat.ui import ChatUI


def setup_distributed(cfg: GlobalConfig) -> None:
    """Initialize process group from config."""
    backend = cfg.env.distributed.backend if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method=cfg.env.distributed.init_method,
        world_size=cfg.env.distributed.world_size,
        rank=cfg.env.distributed.rank,
    )


def reset_kv_cache(model) -> None:
    """Zero out all attention KV caches and reset model decode state."""
    if hasattr(model, "iter_attention_modules"):
        for attn in model.iter_attention_modules():
            if hasattr(attn, "k_cache") and attn.k_cache is not None:
                attn.k_cache.zero_()
            if hasattr(attn, "v_cache") and attn.v_cache is not None:
                attn.v_cache.zero_()
    if hasattr(model, "reset"):
        model.reset()


def main() -> None:
    parser = argparse.ArgumentParser(description="mini-vllm multi-turn chat")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="Optional system prompt",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Env setup
    # ------------------------------------------------------------------ #
    cfg = GlobalConfig.from_yaml(args.config)
    torch.set_default_dtype(cfg.env.get_torch_dtype())
    torch.set_default_device(cfg.env.device)

    setup_distributed(cfg)
    print_runtime_config(cfg)

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    model, tokenizer = load_model(cfg)
    model = apply_runtime_patches(model, cfg)
    runner = ModelRunner(model=model, tokenizer=tokenizer, cfg=cfg)

    model_name = cfg.path.model_path

    # Guard sequence length by the KV cache limit (if configured),
    # otherwise fall back to the model's theoretical maximum.
    max_seq_len = cfg.inference.kv_cache_max_len or getattr(
        model.config, "max_position_embeddings", 4096
    )

    # ------------------------------------------------------------------ #
    # Dialog + UI
    # ------------------------------------------------------------------ #
    ui = ChatUI()
    dialog = DialogManager(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        device=cfg.env.device,
        system_prompt=args.system,
        use_thinking=cfg.inference.use_thinking,
    )

    ui.print_header(model_name=model_name)

    turn = 1
    while True:
        try:
            user_text = ui.input_prompt()
        except (EOFError, KeyboardInterrupt):
            break

        user_text = user_text.strip()
        if not user_text:
            continue
        if user_text in ("/exit", "/quit"):
            break
        if user_text == "/clear":
            dialog.clear()
            reset_kv_cache(model)
            ui.print_info("对话历史已清空，KV cache 已重置")
            turn = 1
            continue
        if user_text == "/think on":
            dialog.use_thinking = True
            ui.print_info("思考模式已开启（下一轮将自动回退到全量 prefill）")
            continue
        if user_text == "/think off":
            dialog.use_thinking = False
            ui.print_info("思考模式已关闭（下一轮将自动回退到全量 prefill）")
            continue

        dialog.add_user(user_text)
        ui.print_user(user_text)

        # Build input and validate length
        try:
            input_ids, cached_len = dialog.get_generation_input()
        except ValueError as exc:
            ui.print_error(str(exc))
            continue

        # Generate
        t0 = time.perf_counter()
        try:
            output_ids = runner.generate(
                input_ids,
                cached_len=cached_len,
                max_new_tokens=cfg.inference.max_new_tokens,
            )
        except Exception as exc:
            ui.print_error(f"生成失败: {exc}")
            continue
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0

        assistant_text = tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        )

        dialog.add_assistant(assistant_text)
        dialog.update_cache(output_ids)

        ui.print_assistant(
            assistant_text,
            turn,
            elapsed_ms=elapsed_ms,
            show_cot=dialog.use_thinking,
        )
        turn += 1

    dist.destroy_process_group()
    ui.print_info("再见！")


if __name__ == "__main__":
    main()
