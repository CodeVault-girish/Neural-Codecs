# audio_codec/registry.py
#
# pip_packages     — packages passed to `pip install`
# import_checks    — module names verified via importlib.util.find_spec()
#                    (pip dist name != import name for some packages, e.g. dac/audiotools)
# pip_no_deps      — packages that must be installed with --no-deps first
#                    (used by funcodec to avoid editdistance build failure on Python 3.14+)

CODEC_REGISTRY = {
    # ─── SNAC variants ───
    "1": {
        "name":          "snac_24khz",
        "module":        "audio_codec.codecs.snac",
        "class":         "SNACDecoder",
        "hub_name":      "hubertsiuzdak/snac_24khz",
        "sample_rate":   24000,
        "pip_packages":  ["snac"],
        "import_checks": ["snac"],
    },
    "2": {
        "name":          "snac_32khz",
        "module":        "audio_codec.codecs.snac",
        "class":         "SNACDecoder",
        "hub_name":      "hubertsiuzdak/snac_32khz",
        "sample_rate":   32000,
        "pip_packages":  ["snac"],
        "import_checks": ["snac"],
    },
    "3": {
        "name":          "snac_44khz",
        "module":        "audio_codec.codecs.snac",
        "class":         "SNACDecoder",
        "hub_name":      "hubertsiuzdak/snac_44khz",
        "sample_rate":   44100,
        "pip_packages":  ["snac"],
        "import_checks": ["snac"],
    },

    # ─── DAC variants ───
    # audiotools is bundled inside descript-audio-codec — no separate install needed
    "4": {
        "name":          "dac_16khz",
        "module":        "audio_codec.codecs.dac",
        "class":         "DACDecoder",
        "hub_name":      "descript/dac_16khz",
        "sample_rate":   16000,
        "pip_packages":  ["descript-audio-codec"],
        "import_checks": ["dac", "audiotools"],
    },
    "5": {
        "name":          "dac_24khz",
        "module":        "audio_codec.codecs.dac",
        "class":         "DACDecoder",
        "hub_name":      "descript/dac_24khz",
        "sample_rate":   24000,
        "pip_packages":  ["descript-audio-codec"],
        "import_checks": ["dac", "audiotools"],
    },
    "6": {
        "name":          "dac_44khz",
        "module":        "audio_codec.codecs.dac",
        "class":         "DACDecoder",
        "hub_name":      "descript/dac_44khz",
        "sample_rate":   44100,
        "pip_packages":  ["descript-audio-codec"],
        "import_checks": ["dac", "audiotools"],
    },

    # ─── EnCodec variants ───
    "7": {
        "name":          "encodec_24khz",
        "module":        "audio_codec.codecs.encodec24",
        "class":         "Encodec24Decoder",
        "hub_name":      "facebook/encodec_24khz",
        "sample_rate":   24000,
        "pip_packages":  ["transformers", "encodec"],
        "import_checks": ["transformers", "encodec"],
    },
    "8": {
        "name":          "encodec_48khz",
        "module":        "audio_codec.codecs.encodec48",
        "class":         "Encodec48Decoder",
        "hub_name":      "facebook/encodec_48khz",
        "sample_rate":   48000,
        "pip_packages":  ["transformers", "encodec"],
        "import_checks": ["transformers", "encodec"],
    },

    # ─── SoundStream ───
    # WARNING: soundstream pins numpy<2.0 and huggingface-hub<0.16.
    # Use a separate virtual environment if you need newer versions for other codecs.
    "9": {
        "name":          "soundstream_16khz",
        "module":        "audio_codec.codecs.soundstream",
        "class":         "SoundStreamDecoder",
        "hub_name":      "SoundStream/soundstream_16khz",
        "sample_rate":   16000,
        "pip_packages":  ["soundstream"],
        "import_checks": ["soundstream"],
        "install_notes": [
            "soundstream pins numpy<2.0 and huggingface-hub<0.16.",
            "Use a dedicated venv if other codecs need newer versions.",
        ],
    },

    # ─── SpeechTokenizer ───
    "10": {
        "name":          "speechtokenizer",
        "module":        "audio_codec.codecs.speechtokenizer",
        "class":         "SpeechTokenizerDecoder",
        "config_path":   "config/config.json",
        "ckpt_path":     "checkpoints/SpeechTokenizer.pt",
        "sample_rate":   None,
        "pip_packages":  ["speechtokenizer", "beartype"],
        "import_checks": ["speechtokenizer", "beartype"],
        "install_notes": [
            "After pip install, download the checkpoint manually:",
            "  URL : https://huggingface.co/fnlp/SpeechTokenizer/tree/main/speechtokenizer_hubert_avg",
            "  Save: checkpoints/SpeechTokenizer.pt",
            "  Save: config/config.json",
        ],
    },

    # ─── FunCodec ── paper F4: en-libritts nq32ds640 (original entry) ───
    # funcodec installed with --no-deps (skips editdistance, not needed for inference).
    # Weights (~150 MB) download automatically from HuggingFace on first use.
    "11": {
        "name":          "funcodec_en_libritts_16k_nq32ds640",
        "module":        "audio_codec.codecs.funcodec_decoder",
        "class":         "FunCodecDecoder",
        "hub_name":      "alibaba-damo/audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch",
        "sample_rate":   16000,
        "pip_no_deps":   ["funcodec"],
        "pip_packages":  [
            "huggingface_hub", "librosa", "kaldiio", "einops",
            "thop", "six", "pytorch-wpe", "torch-complex",
            "humanfriendly", "h5py", "typeguard==2.13.3",
        ],
        "import_checks": ["funcodec", "huggingface_hub", "librosa", "kaldiio",
                          "einops", "thop", "humanfriendly", "h5py"],
        "install_notes": [
            "source funcodec/bin/activate",
            "neural-codec decode --codec funcodec_en_libritts_16k_nq32ds640 --input ./wavs --output ./out",
        ],
    },

    # ─── FunCodec (FreqCodec) ── paper F1: gr1nq32ds320 ───
    "12": {
        "name":          "funcodec_en_libritts_16k_gr1nq32ds320",
        "module":        "audio_codec.codecs.funcodec_decoder",
        "class":         "FunCodecDecoder",
        "hub_name":      "alibaba-damo/audio_codec-freqcodec_magphase-en-libritts-16k-gr1nq32ds320-pytorch",
        "sample_rate":   16000,
        "pip_no_deps":   ["funcodec"],
        "pip_packages":  [
            "huggingface_hub", "librosa", "kaldiio", "einops",
            "thop", "six", "pytorch-wpe", "torch-complex",
            "humanfriendly", "h5py", "typeguard==2.13.3",
        ],
        "import_checks": ["funcodec", "huggingface_hub", "librosa", "kaldiio",
                          "einops", "thop", "humanfriendly", "h5py"],
    },

    # ─── FunCodec (FreqCodec) ── paper F2: gr8nq32ds320 ───
    "13": {
        "name":          "funcodec_en_libritts_16k_gr8nq32ds320",
        "module":        "audio_codec.codecs.funcodec_decoder",
        "class":         "FunCodecDecoder",
        "hub_name":      "alibaba-damo/audio_codec-freqcodec_magphase-en-libritts-16k-gr8nq32ds320-pytorch",
        "sample_rate":   16000,
        "pip_no_deps":   ["funcodec"],
        "pip_packages":  [
            "huggingface_hub", "librosa", "kaldiio", "einops",
            "thop", "six", "pytorch-wpe", "torch-complex",
            "humanfriendly", "h5py", "typeguard==2.13.3",
        ],
        "import_checks": ["funcodec", "huggingface_hub", "librosa", "kaldiio",
                          "einops", "thop", "humanfriendly", "h5py"],
    },

    # ─── FunCodec (EnCodec) ── paper F3: en-libritts nq32ds320 ───
    "14": {
        "name":          "funcodec_en_libritts_16k_nq32ds320",
        "module":        "audio_codec.codecs.funcodec_decoder",
        "class":         "FunCodecDecoder",
        "hub_name":      "alibaba-damo/audio_codec-encodec-en-libritts-16k-nq32ds320-pytorch",
        "sample_rate":   16000,
        "pip_no_deps":   ["funcodec"],
        "pip_packages":  [
            "huggingface_hub", "librosa", "kaldiio", "einops",
            "thop", "six", "pytorch-wpe", "torch-complex",
            "humanfriendly", "h5py", "typeguard==2.13.3",
        ],
        "import_checks": ["funcodec", "huggingface_hub", "librosa", "kaldiio",
                          "einops", "thop", "humanfriendly", "h5py"],
    },

    # ─── FunCodec (EnCodec) ── paper F5: zh_en general nq32ds320 ───
    "15": {
        "name":          "funcodec_zh_en_16k_nq32ds320",
        "module":        "audio_codec.codecs.funcodec_decoder",
        "class":         "FunCodecDecoder",
        "hub_name":      "alibaba-damo/audio_codec-encodec-zh_en-general-16k-nq32ds320-pytorch",
        "sample_rate":   16000,
        "pip_no_deps":   ["funcodec"],
        "pip_packages":  [
            "huggingface_hub", "librosa", "kaldiio", "einops",
            "thop", "six", "pytorch-wpe", "torch-complex",
            "humanfriendly", "h5py", "typeguard==2.13.3",
        ],
        "import_checks": ["funcodec", "huggingface_hub", "librosa", "kaldiio",
                          "einops", "thop", "humanfriendly", "h5py"],
    },

    # ─── FunCodec (EnCodec) ── paper F6: zh_en general nq32ds640 ───
    "16": {
        "name":          "funcodec_zh_en_16k_nq32ds640",
        "module":        "audio_codec.codecs.funcodec_decoder",
        "class":         "FunCodecDecoder",
        "hub_name":      "alibaba-damo/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch",
        "sample_rate":   16000,
        "pip_no_deps":   ["funcodec"],
        "pip_packages":  [
            "huggingface_hub", "librosa", "kaldiio", "einops",
            "thop", "six", "pytorch-wpe", "torch-complex",
            "humanfriendly", "h5py", "typeguard==2.13.3",
        ],
        "import_checks": ["funcodec", "huggingface_hub", "librosa", "kaldiio",
                          "einops", "thop", "humanfriendly", "h5py"],
    },

    # ─── AcademiCodec / HiFi-Codec ── paper B1: hifi_16k_320d ───
    # Checkpoint: Dongchao/AcademiCodec  file "HiFi-Codec-16k-320d"  (root, no ext)
    # Config    : AcademiCodec/egs/HiFi-Codec-16k-320d/config_16k_320d.json (cloned repo)
    "17": {
        "name":           "academicodec_hifi_16k_320d",
        "module":         "audio_codec.codecs.academicodec",
        "class":          "AcademiCodecDecoder",
        "hub_name":       "Dongchao/AcademiCodec",
        "hub_ckpt_file":  "HiFi-Codec-16k-320d",
        "local_egs_dir":  "HiFi-Codec-16k-320d",
        "sample_rate":    16000,
        "pip_packages":   ["huggingface_hub", "librosa"],
        "import_checks":  ["huggingface_hub", "librosa"],
        "install_notes":  [
            "Requires AcademiCodec cloned at <repo_root>/../AcademiCodec.",
            "Checkpoint (~242 MB) downloads from Dongchao/AcademiCodec on HuggingFace.",
        ],
    },

    # ─── AcademiCodec / HiFi-Codec ── paper B2: hifi_16k_320d_large_uni ───
    # Checkpoint: Dongchao/AcademiCodec  file "HiFi-Codec-16k-320d-large-universal"
    # Config    : same 320d architecture — uses HiFi-Codec-16k-320d egs config
    "18": {
        "name":           "academicodec_hifi_16k_320d_large_uni",
        "module":         "audio_codec.codecs.academicodec",
        "class":          "AcademiCodecDecoder",
        "hub_name":       "Dongchao/AcademiCodec",
        "hub_ckpt_file":  "HiFi-Codec-16k-320d-large-universal",
        "local_egs_dir":  "HiFi-Codec-16k-320d",
        "sample_rate":    16000,
        "pip_packages":   ["huggingface_hub", "librosa"],
        "import_checks":  ["huggingface_hub", "librosa"],
        "install_notes":  [
            "Large-universal checkpoint shares 320d architecture config.",
        ],
    },

    # ─── AcademiCodec / HiFi-Codec ── paper B3: hifi_24k_320d ───
    # Checkpoint: Dongchao/AcademiCodec  file "HiFi-Codec-24k-320d"
    # Config    : AcademiCodec/egs/HiFi-Codec-24k-320d/config_24k_320d.json
    "19": {
        "name":           "academicodec_hifi_24k_320d",
        "module":         "audio_codec.codecs.academicodec",
        "class":          "AcademiCodecDecoder",
        "hub_name":       "Dongchao/AcademiCodec",
        "hub_ckpt_file":  "HiFi-Codec-24k-320d",
        "local_egs_dir":  "HiFi-Codec-24k-320d",
        "sample_rate":    24000,
        "pip_packages":   ["huggingface_hub", "librosa"],
        "import_checks":  ["huggingface_hub", "librosa"],
        "install_notes":  [
            "24 kHz HiFi-Codec. Checkpoint (~255 MB) auto-downloads from HuggingFace.",
        ],
    },
}

# Codecs that require a separate working directory — cannot be run via the neural-codec CLI
EXTERNAL_CODECS = {
    "audiodec": {
        "name": "AudioDec",
        "steps": [
            "The AudioDec/ folder, checkpoints, and audiodec/ venv are already included in this repo.",
            "Windows : cd AudioDec && audiodec\\Scripts\\python AudioDec.py --model libritts_v1 --cuda -1 -i ..\\audio_sample\\ -o ..\\out\\",
            "Linux   : cd AudioDec && source audiodec/bin/activate && python AudioDec.py --model libritts_v1 --cuda -1 -i ../audio_sample/ -o ../out/",
            "--- If AudioDec/ is missing, do a fresh setup: ---",
            "git clone https://github.com/facebookresearch/AudioDec.git",
            "cd AudioDec && python -m venv audiodec && audiodec\\Scripts\\activate  (Windows)",
            "pip install -r requirements.txt",
            "Download exp.zip from: https://github.com/facebookresearch/AudioDec/releases/download/pretrain_models_v02/exp.zip",
            "Extract exp.zip into AudioDec/  (gives AudioDec/exp/autoencoder/... and AudioDec/exp/vocoder/...)",
            "Copy AudioDec.py from this repo's root into AudioDec/",
        ],
        "encode_usage": "cd AudioDec && audiodec\\Scripts\\python AudioDec.py --model libritts_v1 --cuda -1 -i input/ -o output/",
        "decode_usage": "cd AudioDec && audiodec\\Scripts\\python AudioDec.py --model vctk_v1    --cuda -1 -i input/ -o output/",
        "scp_note": "Models: libritts_v1 (24 kHz), vctk_v1 (48 kHz). Output: <stem>_AudioDec_<model>_<khz>khz.wav",
    },
}
