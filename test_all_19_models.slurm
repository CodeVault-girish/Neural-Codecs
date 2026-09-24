#!/bin/bash
#SBATCH --job-name=test_all_codecs
#SBATCH --partition=k2-gpu-v100,k2-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/users/gmadaan/GIT/Neural-Codecs/test_all19-%j.out
#SBATCH --error=/users/gmadaan/GIT/Neural-Codecs/test_all19-%j.err

REPO=/users/gmadaan/GIT/Neural-Codecs
SAMPLES=$REPO/audio_sample
OUT=$REPO/test_out_all19
NEURAL_CODEC=/users/gmadaan/.local/bin/neural-codec

module load libs/nvidia-cuda/12.4.0/bin 2>/dev/null || true

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "==========="
echo ""

mkdir -p $OUT
PASS=0; FAIL=0
declare -A RESULTS

run_test() {
    local id="$1"
    local label="$2"
    local outdir="$OUT/${id}_${label}"
    mkdir -p "$outdir"
    $NEURAL_CODEC decode --codec "$id" --input "$SAMPLES" --output "$outdir" --device cuda \
        > "$OUT/${id}_${label}.log" 2>&1
    local count=$(find "$outdir" -name "*.wav" | wc -l)
    if [ "$count" -gt 0 ]; then
        echo "  [PASS] $id $label  ($count files)"
        RESULTS["$label"]="PASS ($count)"
        PASS=$((PASS+1))
    else
        echo "  [FAIL] $id $label — 0 files, see ${id}_${label}.log"
        RESULTS["$label"]="FAIL"
        FAIL=$((FAIL+1))
    fi
}

# ── SNAC ────────────────────────────────────────────────────────
echo "--- SNAC ---"
run_test 1  "snac_24khz"
run_test 2  "snac_32khz"
run_test 3  "snac_44khz"

# ── DAC ─────────────────────────────────────────────────────────
echo "--- DAC ---"
run_test 4  "dac_16khz"
run_test 5  "dac_24khz"
run_test 6  "dac_44khz"

# ── EnCodec ─────────────────────────────────────────────────────
echo "--- EnCodec ---"
run_test 7  "encodec_24khz"
run_test 8  "encodec_48khz"

# ── SoundStream ─────────────────────────────────────────────────
echo "--- SoundStream ---"
source $REPO/soundstream_venv/bin/activate
run_test 9  "soundstream_16khz"
deactivate

# ── SpeechTokenizer ─────────────────────────────────────────────
echo "--- SpeechTokenizer ---"
cd $REPO
run_test 10 "speechtokenizer"

# ── FunCodec (all 6 variants) ───────────────────────────────────
echo "--- FunCodec ---"
run_test 11 "funcodec_en_libritts_16k_nq32ds640"
run_test 12 "funcodec_en_libritts_16k_gr1nq32ds320"
run_test 13 "funcodec_en_libritts_16k_gr8nq32ds320"
run_test 14 "funcodec_en_libritts_16k_nq32ds320"
run_test 15 "funcodec_zh_en_16k_nq32ds320"
run_test 16 "funcodec_zh_en_16k_nq32ds640"

# ── AcademiCodec / HiFi-Codec ───────────────────────────────────
echo "--- AcademiCodec ---"
run_test 17 "academicodec_hifi_16k_320d"
run_test 18 "academicodec_hifi_16k_320d_large_uni"
run_test 19 "academicodec_hifi_24k_320d"

# ── AudioDec (external venv) ─────────────────────────────────────
echo "--- AudioDec ---"
cd $REPO/AudioDec || { echo "  [FAIL] AudioDec — cd failed"; FAIL=$((FAIL+1)); RESULTS["AudioDec"]="FAIL (cd)"; }
if [ -d "$REPO/AudioDec" ]; then
    adout="$OUT/F05_AudioDec"
    mkdir -p "$adout"
    source audiodec/bin/activate
    python AudioDec.py --model libritts_v1 --cuda 0 -i "$SAMPLES" -o "$adout" \
        > "$OUT/F05_AudioDec.log" 2>&1
    adcount=$(find "$adout" -name "*.wav" | wc -l)
    if [ "$adcount" -gt 0 ]; then
        echo "  [PASS] AudioDec libritts_v1  ($adcount files)"
        RESULTS["AudioDec"]="PASS ($adcount)"
        PASS=$((PASS+1))
    else
        echo "  [FAIL] AudioDec — 0 files, see F05_AudioDec.log"
        RESULTS["AudioDec"]="FAIL"
        FAIL=$((FAIL+1))
    fi
    deactivate
fi
cd $REPO

# ── Summary ──────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "  RESULTS — PASS: $PASS   FAIL: $FAIL"
echo "=================================================="
for label in "${!RESULTS[@]}"; do
    printf "  %-50s %s\n" "$label" "${RESULTS[$label]}"
done | sort
echo "=================================================="
