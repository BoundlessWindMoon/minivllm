#!/usr/bin/env bash
# Download common lm-eval datasets from hf-mirror.com for offline use.
#
# Note: hf-mirror.com 对部分数据集（hellaswag, piqa 等）未同步预生成 parquet，
# 这些数据集需通过其他渠道获取（如官方 HF Hub、ModelScope 等）。
#
# Usage: bash scripts/download_eval_datasets.sh

set -e

MIRROR="https://hf-mirror.com"
DATASETS_DIR="${DATASETS_DIR:-/tmp/datasets}"

mkdir -p "$DATASETS_DIR"

download_url() {
    local url="$1"
    local out="$2"
    if curl -fsL -o "$out" "$url" 2>/dev/null; then
        echo "  OK: $(basename "$out")"
        return 0
    else
        return 1
    fi
}

# ---- ARC-Easy ----
mkdir -p "$DATASETS_DIR/arc_easy"
echo "[Download] allenai/ai2_arc / ARC-Easy"
for split in train validation test; do
    download_url "$MIRROR/datasets/allenai/ai2_arc/resolve/main/ARC-Easy/${split}-00000-of-00001.parquet" \
        "$DATASETS_DIR/arc_easy/${split}.parquet" || echo "  SKIP: ${split}.parquet"
done

# ---- ARC-Challenge ----
mkdir -p "$DATASETS_DIR/arc_challenge"
echo "[Download] allenai/ai2_arc / ARC-Challenge"
for split in train validation test; do
    download_url "$MIRROR/datasets/allenai/ai2_arc/resolve/main/ARC-Challenge/${split}-00000-of-00001.parquet" \
        "$DATASETS_DIR/arc_challenge/${split}.parquet" || echo "  SKIP: ${split}.parquet"
done

# ---- GSM8K ----
mkdir -p "$DATASETS_DIR/gsm8k"
echo "[Download] openai/gsm8k"
for split in train test; do
    download_url "$MIRROR/datasets/openai/gsm8k/resolve/main/main/${split}-00000-of-00001.parquet" \
        "$DATASETS_DIR/gsm8k/${split}.parquet" || echo "  SKIP: ${split}.parquet"
done

# ---- MMLU (all subset) ----
mkdir -p "$DATASETS_DIR/mmlu"
echo "[Download] cais/mmlu / all"
for split in dev test validation; do
    download_url "$MIRROR/datasets/cais/mmlu/resolve/main/all/${split}-00000-of-00001.parquet" \
        "$DATASETS_DIR/mmlu/${split}.parquet" || echo "  SKIP: ${split}.parquet"
done
# lm-eval expects 'dev' as fewshot_split; datasets loads it as 'validation'
# We keep both names available via symlink if dev.parquet exists
if [ -f "$DATASETS_DIR/mmlu/dev.parquet" ] && [ ! -f "$DATASETS_DIR/mmlu/validation.parquet" ]; then
    ln -sf dev.parquet "$DATASETS_DIR/mmlu/validation.parquet"
fi

# ---- WinoGrande ----
mkdir -p "$DATASETS_DIR/winogrande"
echo "[Download] allenai/winogrande"
for split in train validation test; do
    download_url "$MIRROR/datasets/allenai/winogrande/resolve/main/${split}-00000-of-00001.parquet" \
        "$DATASETS_DIR/winogrande/${split}.parquet" || echo "  SKIP: ${split}.parquet"
done

echo ""
echo "========================================"
echo "Datasets downloaded to $DATASETS_DIR"
echo ""
echo "Supported (parquet available on mirror):"
echo "  arc_easy, arc_challenge, gsm8k, mmlu, winogrande"
echo ""
echo "NOT available on hf-mirror.com (need alternative source):"
echo "  hellaswag, piqa, openbookqa"
echo "========================================"
echo ""
echo "Run benchmark with local tasks:"
echo "  python -m eval.run --config configs/qwen3_5.yaml \\"
echo "      --tasks arc_easy_local,arc_challenge_local,gsm8k_local,mmlu_local \\"
echo "      --include_path eval/tasks_local"
