#!/bin/bash
# Rename TAU_CodecFake folders to CodecFake paper (Table 1) naming.
# Run this AFTER all SLURM jobs finish.
# Paper: Wu et al., Interspeech 2024, "CodecFake"

BASE=/mnt/scratch2/users/gmadaan/TAU_CodecFake

rename_if_exists() {
    local src="$BASE/$1"
    local dst="$BASE/$2"
    if [ -d "$src" ]; then
        mv "$src" "$dst" && echo "  $2  ← $1"
    else
        echo "  [SKIP] $1 not found"
    fi
}

echo "=== Renaming TAU_CodecFake to paper (Table 1) names ==="

# Already done (completed before this script)
# A  ← F02_SpeechTokenizer_16k
# B1 ← B01_AcademiCodec_hifi_16k_320d
# C  ← F05_AudioDec_24k
# D3 ← F07_DAC_44k
# E  ← F04_EnCodec_24k
# F4 ← F03_FunCodec_16k

# Running folders (rename when their job finishes)
rename_if_exists "B02_AcademiCodec_hifi_16k_320d_large_uni"  "B2"
rename_if_exists "F02_FunCodec_gr1_16k"                      "F1"

# Future folders created by job 10008918
rename_if_exists "B03_AcademiCodec_hifi_24k_320d"            "B3"

# Future folders created by job 10008928
rename_if_exists "F03_FunCodec_gr8_16k"                      "F2"
rename_if_exists "F04_FunCodec_nq32ds320_16k"                "F3"
rename_if_exists "F05_FunCodec_zh_en_nq32ds320_16k"          "F5"
rename_if_exists "F06_FunCodec_zh_en_nq32ds640_16k"          "F6"
rename_if_exists "DAC_16k"                                   "D1"
rename_if_exists "DAC_24k"                                   "D2"

# Delete extras not in the 15 paper models
for extra in EnCodec_48k SNAC_24k SNAC_32k SNAC_44k; do
    if [ -d "$BASE/$extra" ]; then
        rm -rf "$BASE/$extra" && echo "  [DELETED] $extra"
    fi
done

echo ""
echo "=== Final folder list ==="
ls "$BASE"/
echo ""
echo "=== Expected: A B1 B2 B3 C D1 D2 D3 E F1 F2 F3 F4 F5 F6 ==="
