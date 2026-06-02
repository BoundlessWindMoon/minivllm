#!/usr/bin/env python3
"""Generate figures for multi_turn_prefill_v6.md.

Run:
    python scripts/draw_figures.py
"""

from __future__ import annotations

import matplotlib
from matplotlib import font_manager

# Register user-local CJK font so Chinese labels render correctly.
_FONT_PATH = "/home/sakuya/.local/share/fonts/NotoSansMonoCJKsc-Regular.otf"
font_manager.fontManager.addfont(_FONT_PATH)
matplotlib.rcParams["font.family"] = ["Noto Sans Mono CJK SC", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np


def save(fig: plt.Figure, name: str) -> None:
    """Save figure to assets/images."""
    path = f"/home/sakuya/userspace/mini-vllm/assets/images/{name}"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def _card(ax, x, y, w, h, text, style, fs=10):
    """Draw a rounded card with subtle shadow."""
    sh = FancyBboxPatch(
        (x + 0.04, y - 0.04), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor="#00000008", edgecolor="none", zorder=1
    )
    ax.add_patch(sh)
    body = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=style["bg"], edgecolor=style["fg"],
        linewidth=2.2, alpha=0.96, zorder=2
    )
    ax.add_patch(body)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fs,
        color=style["fg"], fontweight="bold", linespacing=1.25, zorder=3
    )
    return x + w


def _arrow(ax, x1, x2, y, label="", color="#4B5563"):
    ax.annotate(
        "", xy=(x2, y), xytext=(x1, y),
        arrowprops=dict(arrowstyle="->", color=color, lw=2.0),
        zorder=4
    )
    if label:
        ax.text(
            (x1 + x2) / 2, y + 0.38, label,
            ha="center", va="bottom", fontsize=9,
            color=color, fontweight="medium", style="italic", zorder=5
        )


# ------------------------------------------------------------------ #
# Figure 1: Pseudo multi-turn (problem) – for section 1
# ------------------------------------------------------------------ #

def draw_pseudo_timeline() -> None:
    """Figure 1 – 伪多轮：Turn 1 单轮 → Turn 2 仍做单轮（重复计算）."""
    fig, ax = plt.subplots(figsize=(15, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    styles = {
        "prompt": {"bg": "#EFF6FF", "fg": "#1E40AF"},
        "kv": {"bg": "#FFFBEB", "fg": "#92400E"},
        "assistant": {"bg": "#ECFDF5", "fg": "#065F46"},
        "warn": {"bg": "#FEF2F2", "fg": "#991B1B"},
    }

    # ---- Turn 1（单轮） ----
    y1 = 3.55
    ax.add_patch(Rectangle((0.4, 2.75), 14.2, 1.55, facecolor="#EFF6FF", alpha=0.22, zorder=0, edgecolor="none"))
    ax.text(0.65, 4.1, "Turn 1（单轮）", fontsize=14, fontweight="bold", color="#1E40AF")

    x = 1.0
    x = _card(ax, x, y1 - 0.38, 2.0, 0.76, "初始输入\n70 tokens", styles["prompt"])
    x += 0.25
    _arrow(ax, x - 0.25, x + 0.55, y1, "prefill", "#1E40AF")
    x += 0.55
    x = _card(ax, x, y1 - 0.38, 1.5, 0.76, "KV Cache\n(70)", styles["kv"])
    x += 0.25
    _arrow(ax, x - 0.25, x + 0.55, y1, "decode")
    x += 0.55
    x = _card(ax, x, y1 - 0.38, 1.7, 0.76, "Assistant 回复\n30 tokens", styles["assistant"])
    x += 0.25
    _arrow(ax, x - 0.25, x + 0.55, y1, "释放", "#991B1B")
    x += 0.55
    x = _card(ax, x, y1 - 0.33, 1.3, 0.66, "释放 KV", styles["warn"])

    # ---- Turn 2（伪多轮 – 重复计算） ----
    y2 = 1.35
    ax.add_patch(Rectangle((0.4, 0.55), 14.2, 1.55, facecolor="#FEF2F2", alpha=0.28, zorder=0, edgecolor="none"))
    ax.text(0.65, 1.9, "Turn 2（伪多轮）", fontsize=14, fontweight="bold", color="#991B1B")

    x = 1.0
    x = _card(ax, x, y2 - 0.38, 2.0, 0.76, "历史对话\n100 tokens", styles["prompt"])
    # 重复计算标注（放在历史框下方，不和任何框重叠）
    ax.text(
        x - 1.0, y2 - 0.72, "⚠️  重复计算",
        fontsize=10, ha="center", color="#DC2626", fontweight="bold", zorder=6,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF2F2", edgecolor="#DC2626", alpha=0.9)
    )
    ax.text(x + 0.12, y2, "+", ha="center", va="center", fontsize=14,
            fontweight="bold", color="#6B7280", zorder=5)
    x += 0.3
    x = _card(ax, x, y2 - 0.38, 1.3, 0.76, "新输入\n20 tokens", styles["prompt"])
    x += 0.25
    _arrow(ax, x - 0.25, x + 0.55, y2, "prefill", "#991B1B")
    x += 0.55
    x = _card(ax, x, y2 - 0.38, 1.5, 0.76, "KV Cache\n(120)", styles["kv"])
    x += 0.25
    _arrow(ax, x - 0.25, x + 0.55, y2, "decode")
    x += 0.55
    x = _card(ax, x, y2 - 0.38, 1.7, 0.76, "Assistant 回复\n30 tokens", styles["assistant"])
    x += 0.25
    _arrow(ax, x - 0.25, x + 0.55, y2, "释放", "#991B1B")
    x += 0.55
    x = _card(ax, x, y2 - 0.33, 1.3, 0.66, "释放 KV", styles["warn"])

    fig.tight_layout()
    save(fig, "multi_turn_timeline.png")


# ------------------------------------------------------------------ #
# Figure 2: Real multi-turn incremental – for section 3
# ------------------------------------------------------------------ #

def draw_incremental_timeline() -> None:
    """Figure 2 – 真多轮：Turn 1 保留 KV → Turn 2 增量 prefill."""
    fig, ax = plt.subplots(figsize=(15, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    styles = {
        "prompt": {"bg": "#EFF6FF", "fg": "#1E40AF"},
        "kv": {"bg": "#FFFBEB", "fg": "#92400E"},
        "assistant": {"bg": "#ECFDF5", "fg": "#065F46"},
        "retain": {"bg": "#F5F3FF", "fg": "#5B21B6"},
    }

    # ---- Turn 1（单轮，但保留 KV） ----
    y1 = 3.55
    ax.add_patch(Rectangle((0.4, 2.75), 14.2, 1.55, facecolor="#EFF6FF", alpha=0.22, zorder=0, edgecolor="none"))
    ax.text(0.65, 4.1, "Turn 1（单轮）", fontsize=14, fontweight="bold", color="#1E40AF")

    x = 1.0
    x = _card(ax, x, y1 - 0.38, 2.0, 0.76, "初始输入\n70 tokens", styles["prompt"])
    x += 0.25
    _arrow(ax, x - 0.25, x + 0.55, y1, "prefill", "#1E40AF")
    x += 0.55
    x = _card(ax, x, y1 - 0.38, 1.5, 0.76, "KV Cache\n(70)", styles["kv"])
    x += 0.25
    _arrow(ax, x - 0.25, x + 0.55, y1, "decode")
    x += 0.55
    x = _card(ax, x, y1 - 0.38, 1.7, 0.76, "Assistant 回复\n30 tokens", styles["assistant"])
    x += 0.25
    _arrow(ax, x - 0.25, x + 0.55, y1, "保留", "#5B21B6")
    x += 0.55
    x = _card(ax, x, y1 - 0.33, 1.4, 0.66, "保留 KV\n(100)", styles["retain"])

    # ---- Turn 2（增量 prefill） ----
    y2 = 1.35
    ax.add_patch(Rectangle((0.4, 0.55), 14.2, 1.55, facecolor="#ECFDF5", alpha=0.28, zorder=0, edgecolor="none"))
    ax.text(0.65, 1.9, "Turn 2（增量 prefill）", fontsize=14, fontweight="bold", color="#065F46")

    x = 1.0
    x = _card(ax, x, y2 - 0.38, 2.0, 0.76, "历史对话\n100 tokens\n(已缓存)", styles["prompt"])
    ax.text(x - 1.0, y2 - 0.72, "✓ 复用", fontsize=10, ha="center",
            color="#059669", fontweight="bold", zorder=6)

    ax.text(x + 0.12, y2, "+", ha="center", va="center", fontsize=14,
            fontweight="bold", color="#6B7280", zorder=5)
    x += 0.3

    x = _card(ax, x, y2 - 0.38, 1.3, 0.76, "新输入\n20 tokens", styles["prompt"])
    x += 0.25
    _arrow(ax, x - 0.25, x + 0.55, y2, "增量 prefill", "#059669")
    x += 0.55
    x = _card(ax, x, y2 - 0.38, 1.4, 0.76, "追加 KV\n(20)", styles["kv"])
    x += 0.25
    _arrow(ax, x - 0.25, x + 0.55, y2, "decode")
    x += 0.55
    x = _card(ax, x, y2 - 0.38, 1.7, 0.76, "Assistant 回复\n30 tokens", styles["assistant"])
    x += 0.25
    _arrow(ax, x - 0.25, x + 0.55, y2, "保留", "#059669")
    x += 0.55
    x = _card(ax, x, y2 - 0.33, 1.3, 0.66, "保留 KV\n(150)", styles["retain"])

    fig.tight_layout()
    save(fig, "multi_turn_incremental.png")


# ------------------------------------------------------------------ #
# Figure 3: Attention Mask matrix
# ------------------------------------------------------------------ #

def draw_mask_matrix() -> None:
    """Figure 3 – 增量 prefill 的 causal mask 矩阵（网格状）."""
    cache_len = 58
    delta = 42
    total = cache_len + delta

    fig, ax = plt.subplots(figsize=(10, 7.5))
    fig.patch.set_facecolor("white")

    mask = np.ones((delta, total))
    for i in range(delta):
        mask[i, : cache_len + i + 1] = 0

    cmap = plt.matplotlib.colors.ListedColormap(["#4ADE80", "#FB7185"])
    ax.imshow(mask, cmap=cmap, aspect="auto", interpolation="nearest")

    ax.set_xticks(np.arange(-0.5, total, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, delta, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.45, alpha=0.7)
    ax.tick_params(which="minor", size=0)

    ax.set_xlabel("KV Position (j)", fontsize=12, color="#374151")
    ax.set_ylabel("Query Position (i)", fontsize=12, color="#374151")
    ax.set_title(
        "增量 Prefill Causal Mask  (cache_len=58, Δ=42, S=100)",
        fontsize=14, fontweight="bold", pad=20, color="#1F2937",
    )

    ax.set_xticks([0, cache_len - 1, total - 1])
    ax.set_xticklabels(["0", f"{cache_len}", f"{total}"], fontsize=10)
    ax.set_yticks([0, delta - 1])
    ax.set_yticklabels(["0", f"{delta - 1}"], fontsize=10)
    ax.tick_params(axis="both", colors="#4B5563")

    ax.axvline(x=cache_len - 0.5, color="#1F2937", linewidth=2)
    ax.text(
        cache_len / 2, -4.5, "← Cached (58) →",
        ha="center", va="top", fontsize=11, fontweight="bold", color="#1F2937",
    )
    ax.text(
        cache_len + delta / 2, -4.5, "← New (42) →",
        ha="center", va="top", fontsize=11, fontweight="bold", color="#1F2937",
    )

    visible_patch = mpatches.Patch(color="#4ADE80", label="可见 (0)")
    masked_patch = mpatches.Patch(color="#FB7185", label="Masked (−∞)")
    ax.legend(
        handles=[visible_patch, masked_patch],
        loc="lower right", fontsize=10, framealpha=0.95, edgecolor="#E5E7EB",
    )

    fig.tight_layout()
    save(fig, "incremental_mask.png")


if __name__ == "__main__":
    draw_pseudo_timeline()
    draw_incremental_timeline()
    draw_mask_matrix()
    print("All figures generated.")
