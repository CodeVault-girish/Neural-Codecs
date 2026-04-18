# audio_codec/registry.py
#
# pip_packages  — packages passed to `pip install`
# import_checks — module names verified via importlib.util.find_spec()
#                 (pip dist name != import name for some packages, e.g. dac/audiotools)

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
}

# Codecs that require cloning external repos — cannot be pip-installed
EXTERNAL_CODECS = {
    "funcodec": {
        "name": "FunCodec",
        "steps": [
            "python -m venv funcodec",
            "funcodec\\Scripts\\activate   (Windows)  OR  source funcodec/bin/activate  (Linux/Mac)",
            "git clone https://github.com/alibaba-damo-academy/FunCodec.git",
            "cd FunCodec && pip install -e .",
            "pip install torch torchaudio numpy soundfile",
            "cd egs/LibriTTS/codec && mkdir -p exp",
            "git lfs install",
            "git clone https://huggingface.co/alibaba-damo/audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch exp/audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch",
        ],
        "encode_usage": (
            "bash encoding_decoding.sh --stage 1 --batch_size 1 --num_workers 1 "
            "--gpu_devices 0 --model_dir exp/audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch "
            "--bit_width 16000 --file_sampling_rate 16000 --wav_scp input.scp --out_dir outputs/codecs"
        ),
        "decode_usage": (
            "bash encoding_decoding.sh --stage 2 --batch_size 1 --num_workers 1 "
            "--gpu_devices 0 --model_dir exp/audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch "
            "--bit_width 16000 --file_sampling_rate 16000 --wav_scp outputs/codecs/codecs.txt --out_dir outputs/recon_wavs"
        ),
        "scp_note": (
            "Build input.scp with:\n"
            '  find /path/to/wavs -name "*.wav" | awk -F/ \'{printf "%s %s\\n", $(NF-1)"_"$NF, $0}\' > input.scp'
        ),
    },
    "audiodec": {
        "name": "AudioDec",
        "steps": [
            "git clone https://github.com/facebookresearch/AudioDec.git",
            "cd AudioDec && pip install -r requirements.txt",
            "Download exp.zip: https://github.com/facebookresearch/AudioDec/releases/download/pretrain_models_v02/exp.zip",
            "Extract exp.zip into AudioDec/  (you should have AudioDec/exp/...)",
            "Copy AudioDec.py from this repo into AudioDec/",
        ],
        "encode_usage": "python AudioDec.py --model libritts_v1 -i input/ -o output/",
        "decode_usage": "python AudioDec.py --model vctk_v1    -i input/ -o output/",
    },
}
