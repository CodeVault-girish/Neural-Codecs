#!/bin/bash
#SBATCH --job-name=neural_codecs_TAU
#SBATCH --partition=k2-gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=3-00:00:00
#SBATCH --output=/users/gmadaan/GIT/Neural-Codecs/TAU_codec_run-%j.out
#SBATCH --error=/users/gmadaan/GIT/Neural-Codecs/TAU_codec_run-%j.err

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
#  F04 — EnCodec 24k
# ──────────────────────────────────────────────────────────────────────────
echo ""
echo "===== F04  EnCodec 24k ====="
$NEURAL_CODEC decode \
    --codec encodec_24khz \
    --input  $INPUT \
    --output $OUTPUT/F04_EnCodec_24k \
    --device cuda \
|| echo "[WARN] F04 EnCodec failed — continuing."

# ──────────────────────────────────────────────────────────────────────────
#  F07 — DAC 44k
# ──────────────────────────────────────────────────────────────────────────
echo ""
echo "===== F07  DAC 44k ====="
$NEURAL_CODEC decode \
    --codec dac_44khz \
    --input  $INPUT \
    --output $OUTPUT/F07_DAC_44k \
    --device cuda \
|| echo "[WARN] F07 DAC failed — continuing."

# ──────────────────────────────────────────────────────────────────────────
#  F02 — SpeechTokenizer 16k  (must run from repo root — config path is relative)
# ──────────────────────────────────────────────────────────────────────────
echo ""
echo "===== F02  SpeechTokenizer 16k ====="
cd $REPO
$NEURAL_CODEC decode \
    --codec speechtokenizer \
    --input  $INPUT \
    --output $OUTPUT/F02_SpeechTokenizer_16k \
    --device cuda \
|| echo "[WARN] F02 SpeechTokenizer failed — continuing."

# ──────────────────────────────────────────────────────────────────────────
#  F03 — FunCodec 16k
# ──────────────────────────────────────────────────────────────────────────
echo ""
echo "===== F03  FunCodec 16k ====="
$NEURAL_CODEC decode \
    --codec funcodec_16khz \
    --input  $INPUT \
    --output $OUTPUT/F03_FunCodec_16k \
    --device cuda \
|| echo "[WARN] F03 FunCodec failed — continuing."

# ──────────────────────────────────────────────────────────────────────────
#  F01 — SoundStream 16k  (dedicated venv)
# ──────────────────────────────────────────────────────────────────────────
echo ""
echo "===== F01  SoundStream 16k ====="
source $REPO/soundstream_venv/bin/activate
neural-codec decode \
    --codec soundstream_16khz \
    --input  $INPUT \
    --output $OUTPUT/F01_SoundStream_16k \
    --device cuda \
|| echo "[WARN] F01 SoundStream failed — continuing."
deactivate

# ──────────────────────────────────────────────────────────────────────────
#  F05 — AudioDec 24k  (dedicated venv)
# ──────────────────────────────────────────────────────────────────────────
echo ""
echo "===== F05  AudioDec 24k ====="
cd $REPO/AudioDec
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
#  F06 — AcademiCodec 24k  — not in this repo
# ──────────────────────────────────────────────────────────────────────────
echo ""
echo "[SKIP] F06 AcademiCodec — not in Neural-Codecs registry."

# ──────────────────────────────────────────────────────────────────────────
#  Done
# ──────────────────────────────────────────────────────────────────────────
echo ""
echo "===== All codec runs complete ====="
echo "Output: $OUTPUT"
du -sh $OUTPUT/* 2>/dev/null

# Keep GPU allocated until 3-day time limit
sleep infinity
