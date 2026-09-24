#!/bin/bash
#SBATCH --job-name=TAU_remaining
#SBATCH --partition=k2-gpu-v100,k2-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=3-00:00:00
#SBATCH --output=/users/gmadaan/GIT/Neural-Codecs/TAU_remaining-%j.out
#SBATCH --error=/users/gmadaan/GIT/Neural-Codecs/TAU_remaining-%j.err

REPO=/users/gmadaan/GIT/Neural-Codecs
INPUT=/mnt/scratch2/users/gmadaan/TAU-urban-acoustic-scenes-2019-openset/TAU-urban-acoustic-scenes-2019-openset-development/audio
OUTPUT=/mnt/scratch2/users/gmadaan/TAU_CodecFake
NEURAL_CODEC=/users/gmadaan/.local/bin/neural-codec

module load libs/nvidia-cuda/12.4.0/bin 2>/dev/null || true

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "==========="

mkdir -p $OUTPUT

# Already done: F01 SoundStream, F02 SpeechTokenizer, F03 FunCodec nq32ds640,
#               F04 EnCodec 24k, F07 DAC 44k
# Queued (10008918): F05 AudioDec, B01/B02/B03 AcademiCodec
# This job: remaining FunCodec variants + EnCodec 48k + SNAC + DAC 16k/24k

# ── FunCodec variants (paper F02–F06) ─────────────────────────
echo ""
echo "===== FunCodec gr1  (paper F02) ====="
$NEURAL_CODEC decode \
    --codec funcodec_en_libritts_16k_gr1nq32ds320 \
    --input  $INPUT \
    --output $OUTPUT/F02_FunCodec_gr1_16k \
    --device cuda \
|| echo "[WARN] FunCodec gr1 failed"

echo ""
echo "===== FunCodec gr8  (paper F03) ====="
$NEURAL_CODEC decode \
    --codec funcodec_en_libritts_16k_gr8nq32ds320 \
    --input  $INPUT \
    --output $OUTPUT/F03_FunCodec_gr8_16k \
    --device cuda \
|| echo "[WARN] FunCodec gr8 failed"

echo ""
echo "===== FunCodec nq32ds320  (paper F04) ====="
$NEURAL_CODEC decode \
    --codec funcodec_en_libritts_16k_nq32ds320 \
    --input  $INPUT \
    --output $OUTPUT/F04_FunCodec_nq32ds320_16k \
    --device cuda \
|| echo "[WARN] FunCodec nq32ds320 failed"

echo ""
echo "===== FunCodec zh_en nq32ds320  (paper F05) ====="
$NEURAL_CODEC decode \
    --codec funcodec_zh_en_16k_nq32ds320 \
    --input  $INPUT \
    --output $OUTPUT/F05_FunCodec_zh_en_nq32ds320_16k \
    --device cuda \
|| echo "[WARN] FunCodec zh_en nq32ds320 failed"

echo ""
echo "===== FunCodec zh_en nq32ds640  (paper F06) ====="
$NEURAL_CODEC decode \
    --codec funcodec_zh_en_16k_nq32ds640 \
    --input  $INPUT \
    --output $OUTPUT/F06_FunCodec_zh_en_nq32ds640_16k \
    --device cuda \
|| echo "[WARN] FunCodec zh_en nq32ds640 failed"

# ── EnCodec 48k ───────────────────────────────────────────────
echo ""
echo "===== EnCodec 48k ====="
$NEURAL_CODEC decode \
    --codec encodec_48khz \
    --input  $INPUT \
    --output $OUTPUT/EnCodec_48k \
    --device cuda \
|| echo "[WARN] EnCodec 48k failed"

# ── SNAC (24 / 32 / 44 kHz) ───────────────────────────────────
echo ""
echo "===== SNAC 24k ====="
$NEURAL_CODEC decode \
    --codec snac_24khz \
    --input  $INPUT \
    --output $OUTPUT/SNAC_24k \
    --device cuda \
|| echo "[WARN] SNAC 24k failed"

echo ""
echo "===== SNAC 32k ====="
$NEURAL_CODEC decode \
    --codec snac_32khz \
    --input  $INPUT \
    --output $OUTPUT/SNAC_32k \
    --device cuda \
|| echo "[WARN] SNAC 32k failed"

echo ""
echo "===== SNAC 44k ====="
$NEURAL_CODEC decode \
    --codec snac_44khz \
    --input  $INPUT \
    --output $OUTPUT/SNAC_44k \
    --device cuda \
|| echo "[WARN] SNAC 44k failed"

# ── DAC 16k / 24k ─────────────────────────────────────────────
echo ""
echo "===== DAC 16k ====="
$NEURAL_CODEC decode \
    --codec dac_16khz \
    --input  $INPUT \
    --output $OUTPUT/DAC_16k \
    --device cuda \
|| echo "[WARN] DAC 16k failed"

echo ""
echo "===== DAC 24k ====="
$NEURAL_CODEC decode \
    --codec dac_24khz \
    --input  $INPUT \
    --output $OUTPUT/DAC_24k \
    --device cuda \
|| echo "[WARN] DAC 24k failed"

# ── Done ──────────────────────────────────────────────────────
echo ""
echo "===== All remaining runs complete ====="
echo "Output: $OUTPUT"
du -sh $OUTPUT/* 2>/dev/null

sleep infinity
