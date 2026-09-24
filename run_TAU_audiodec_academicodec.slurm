#!/bin/bash
#SBATCH --job-name=TAU_audiodec_academicodec
#SBATCH --partition=k2-gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=3-00:00:00
#SBATCH --output=/users/gmadaan/GIT/Neural-Codecs/TAU_audiodec_academicodec-%j.out
#SBATCH --error=/users/gmadaan/GIT/Neural-Codecs/TAU_audiodec_academicodec-%j.err

REPO=/users/gmadaan/GIT/Neural-Codecs
INPUT=/mnt/scratch2/users/gmadaan/TAU-urban-acoustic-scenes-2019-openset/TAU-urban-acoustic-scenes-2019-openset-development/audio
OUTPUT=/mnt/scratch2/users/gmadaan/TAU_CodecFake

NEURAL_CODEC=/users/gmadaan/.local/bin/neural-codec

module load libs/nvidia-cuda/12.4.0/bin 2>/dev/null || true

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "==========="

mkdir -p $OUTPUT

# ──────────────────────────────────────────────────────────────────────────
#  F05 — AudioDec 24k  (dedicated venv; MUST cd into AudioDec/ first)
# ──────────────────────────────────────────────────────────────────────────
echo ""
echo "===== F05  AudioDec 24k ====="
cd $REPO/AudioDec || { echo "[ERROR] cd $REPO/AudioDec failed"; exit 1; }
pwd
source audiodec/bin/activate
mkdir -p $OUTPUT/F05_AudioDec_24k
python AudioDec.py \
    --model libritts_v1 \
    --cuda 0 \
    -i $INPUT \
    -o $OUTPUT/F05_AudioDec_24k \
|| echo "[WARN] F05 AudioDec failed — continuing."
deactivate
cd $REPO

# ──────────────────────────────────────────────────────────────────────────
#  B01 — HiFi-Codec 16k-320d  (AcademiCodec)
# ──────────────────────────────────────────────────────────────────────────
echo ""
echo "===== B01  HiFi-Codec 16k-320d ====="
$NEURAL_CODEC decode \
    --codec academicodec_hifi_16k_320d \
    --input  $INPUT \
    --output $OUTPUT/B01_AcademiCodec_hifi_16k_320d \
    --device cuda \
|| echo "[WARN] B01 AcademiCodec 16k-320d failed — continuing."

# ──────────────────────────────────────────────────────────────────────────
#  B02 — HiFi-Codec 16k-320d-large-universal  (AcademiCodec)
# ──────────────────────────────────────────────────────────────────────────
echo ""
echo "===== B02  HiFi-Codec 16k-320d-large-universal ====="
$NEURAL_CODEC decode \
    --codec academicodec_hifi_16k_320d_large_uni \
    --input  $INPUT \
    --output $OUTPUT/B02_AcademiCodec_hifi_16k_320d_large_uni \
    --device cuda \
|| echo "[WARN] B02 AcademiCodec 16k-320d-large-uni failed — continuing."

# ──────────────────────────────────────────────────────────────────────────
#  B03 — HiFi-Codec 24k-320d  (AcademiCodec, paper F06)
# ──────────────────────────────────────────────────────────────────────────
echo ""
echo "===== B03  HiFi-Codec 24k-320d ====="
$NEURAL_CODEC decode \
    --codec academicodec_hifi_24k_320d \
    --input  $INPUT \
    --output $OUTPUT/B03_AcademiCodec_hifi_24k_320d \
    --device cuda \
|| echo "[WARN] B03 AcademiCodec 24k-320d failed — continuing."

# ──────────────────────────────────────────────────────────────────────────
#  Done
# ──────────────────────────────────────────────────────────────────────────
echo ""
echo "===== All runs complete ====="
echo "Output: $OUTPUT"
du -sh $OUTPUT/* 2>/dev/null

# Keep GPU allocated until 3-day time limit
sleep infinity
