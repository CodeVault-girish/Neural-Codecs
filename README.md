# Neural Codecs

A pipeline library for batch **encode → decode** round-trips through neural audio codec models.
Feed it a folder of WAV files, pick a codec, get reconstructed WAVs — useful for building
codec-distorted datasets, evaluating codec quality, or preprocessing audio for TTS/ASR training.

---

## Codec Support

| ID | Name | Sample Rate | Install | Output |
|----|------|-------------|---------|--------|
| 1 | `snac_24khz` | 24 kHz | `pip install snac` | mono |
| 2 | `snac_32khz` | 32 kHz | `pip install snac` | mono |
| 3 | `snac_44khz` | 44 kHz | `pip install snac` | mono |
| 4 | `dac_16khz` | 16 kHz | `pip install descript-audio-codec` | mono |
| 5 | `dac_24khz` | 24 kHz | `pip install descript-audio-codec` | mono |
| 6 | `dac_44khz` | 44 kHz | `pip install descript-audio-codec` | mono |
| 7 | `encodec_24khz` | 24 kHz | `pip install transformers encodec` | mono |
| 8 | `encodec_48khz` | 48 kHz | `pip install transformers encodec` | stereo |
| 9 | `soundstream_16khz` | 16 kHz | `pip install soundstream` ⚠️ | mono |
| 10 | `speechtokenizer` | 16 kHz | pip + manual checkpoint | mono |
| 11 | `funcodec_16khz` | 16 kHz | dedicated venv ⚠️ | mono |
| — | `AudioDec` | 24 / 48 kHz | included venv ⚠️ | mono/stereo |

> ⚠️ **SoundStream** (`soundstream==0.0.1`) pins `numpy<2.0` and `huggingface-hub<0.16`.
> After installing it, run `pip install --upgrade huggingface-hub` to keep EnCodec working.
> For a fully clean setup, use a dedicated virtual environment for SoundStream.

> ⚠️ **FunCodec** requires a dedicated virtual environment — its dependencies conflict with
> other codecs. A pre-configured `funcodec/` venv is included in this repo. See [below](#11--funcodec).

> ⚠️ **AudioDec** requires its own venv (`AudioDec/audiodec/`) and must be run from the `AudioDec/`
> directory. The repo, checkpoints, and venv are already included. See [below](#audiodec-external).

All model weights download automatically from HuggingFace on first use (except SpeechTokenizer — see [below](#10--speechtokenizer)). AudioDec checkpoints are already bundled in `AudioDec/exp/`.

---

## Project Structure

```
Neural-Codecs/
├── audio_codec/
│   ├── config.py          ← AUTO_INSTALL_DEPS flag lives here
│   ├── registry.py        ← codec metadata (packages, import checks, hub names)
│   ├── installer.py       ← dep-check, auto-install, setup commands
│   ├── cli.py             ← neural-codec CLI entry point
│   └── codecs/
│       ├── snac.py
│       ├── dac.py
│       ├── encodec24.py
│       ├── encodec48.py
│       ├── soundstream.py
│       ├── speechtokenizer.py
│       └── funcodec_decoder.py
├── requirements/
│   ├── base.txt           ← torch, torchaudio, soundfile, numpy, tqdm
│   ├── snac.txt           ← IDs 1–3
│   ├── dac.txt            ← IDs 4–6
│   ├── encodec.txt        ← IDs 7–8
│   ├── soundstream.txt    ← ID 9
│   ├── speechtokenizer.txt← ID 10
│   └── funcodec.txt       ← ID 11  (use inside funcodec/ venv)
├── config/
│   └── config.json        ← SpeechTokenizer model config
├── checkpoints/           ← place SpeechTokenizer.pt here
├── audio_sample/          ← put your input WAV files here
└── pyproject.toml
```

---

## Installation

```bash
git clone https://github.com/CodeVault-girish/NeuralCodecDecoder.git
cd NeuralCodecDecoder
pip install -e .
```

This registers the `neural-codec` CLI command. Base dependencies (`torch`, `torchaudio`,
`soundfile`, `tqdm`) are installed automatically. Per-codec packages are installed on demand.

---

## Auto-Install

Missing codec dependencies are **installed automatically** the first time you run a codec.

Controlled by one flag in `audio_codec/config.py`:

```python
# audio_codec/config.py

AUTO_INSTALL_DEPS = True   # auto-install missing packages before decoding (default)
AUTO_INSTALL_DEPS = False  # print the install command and exit instead
```

| Value | Behaviour |
|-------|-----------|
| `True` | First `decode_folder()` call installs any missing packages, then runs. No manual setup needed. |
| `False` | Prints the missing packages and exact `pip install` / `neural-codec setup` command, then exits cleanly. |

---

## Quick Start

```bash
# 1. See all codecs with live install status
neural-codec list

# 2. Decode a folder — auto-installs deps on first run (AUTO_INSTALL_DEPS=True)
neural-codec decode --codec snac_24khz --input ./audio_sample --output ./out

# 3. Use a different codec
neural-codec decode --codec dac_16khz    --input ./audio_sample --output ./out
neural-codec decode --codec encodec_24khz --input ./audio_sample --output ./out

# 4. Use GPU
neural-codec decode --codec snac_24khz --input ./audio_sample --output ./out --device cuda

# 5. Use codec by numeric ID instead of name
neural-codec decode --codec 7 --input ./audio_sample --output ./out

# 6. Pre-install deps without decoding
neural-codec setup --codec snac_24khz
neural-codec setup --all
```

---

## CLI Reference

### `neural-codec list`

Shows every codec — ID, name, sample rate, install status, and required packages.
Also shows whether `AUTO_INSTALL_DEPS` is currently enabled.

```
neural-codec list
```

---

### `neural-codec setup`

Install dependencies for a codec without running it.

```bash
neural-codec setup --codec snac_24khz       # install by name
neural-codec setup --codec 1                # install by ID
neural-codec setup --all                    # install all pip-installable codecs

# External codecs — prints manual setup steps
neural-codec setup --codec funcodec
neural-codec setup --codec audiodec
```

---

### `neural-codec decode`

Batch encode/decode all WAV files in a folder (recursive).

```bash
neural-codec decode --codec <NAME_OR_ID> --input <DIR> --output <DIR> [--device cpu|cuda]
```

| Flag | Required | Description |
|------|----------|-------------|
| `--codec` | yes | Codec name (`snac_24khz`) or numeric ID (`1`) |
| `--input` | yes | Folder with `.wav` files (searched recursively) |
| `--output` | yes | Folder where decoded files are written |
| `--device` | no | `cpu` (default) or `cuda` |

Output files are named `<original_stem>_<codec_name>.wav`.

---

## Python API

```python
from audio_codec import decode_folder, decoder_list, setup_codec, setup_all

# show all codecs and their install status
decoder_list()

# decode a folder (auto-installs deps if AUTO_INSTALL_DEPS=True)
decode_folder("1",           "audio_sample/", "out/", "cpu")   # by ID
decode_folder("snac_24khz",  "audio_sample/", "out/", "cuda")  # by name

# explicitly install before decoding
setup_codec("snac_24khz")
setup_all()

# check if a codec's deps are satisfied (no side effects)
from audio_codec import deps_satisfied
from audio_codec.registry import CODEC_REGISTRY
print(deps_satisfied(CODEC_REGISTRY["7"]))  # True / False
```

---

## Per-Codec Requirements

### 1 · 2 · 3 — SNAC

**Models:** `snac_24khz` · `snac_32khz` · `snac_44khz`
**Weights:** auto-download from HuggingFace (~30–80 MB each)

```bash
# requirements file
pip install -r requirements/snac.txt

# or via CLI
neural-codec setup --codec snac_24khz
```

| Package | Notes |
|---------|-------|
| `snac` | SNAC model |
| `torchaudio` | resampling |
| `soundfile` | WAV I/O |

```bash
neural-codec decode --codec snac_24khz --input ./wavs --output ./out
neural-codec decode --codec snac_32khz --input ./wavs --output ./out
neural-codec decode --codec snac_44khz --input ./wavs --output ./out
```

---

### 4 · 5 · 6 — DAC (Descript Audio Codec)

**Models:** `dac_16khz` · `dac_24khz` · `dac_44khz`
**Weights:** auto-download via `dac.utils.download()` (~75 MB each)

```bash
pip install -r requirements/dac.txt

neural-codec setup --codec dac_16khz
```

| Package | Notes |
|---------|-------|
| `descript-audio-codec` | DAC model + bundles `audiotools` |
| `soundfile` | WAV I/O |

```bash
neural-codec decode --codec dac_16khz --input ./wavs --output ./out
neural-codec decode --codec dac_24khz --input ./wavs --output ./out
neural-codec decode --codec dac_44khz --input ./wavs --output ./out
```

---

### 7 · 8 — EnCodec (Facebook)

**Models:** `encodec_24khz` (mono) · `encodec_48khz` (stereo)
**Weights:** auto-download from HuggingFace

```bash
pip install -r requirements/encodec.txt

neural-codec setup --codec encodec_24khz
```

| Package | Notes |
|---------|-------|
| `transformers` | HuggingFace model loader |
| `encodec` | EnCodec core |
| `soundfile` | WAV I/O |

```bash
neural-codec decode --codec encodec_24khz --input ./wavs --output ./out  # mono, 24 kHz
neural-codec decode --codec encodec_48khz --input ./wavs --output ./out  # stereo, 48 kHz
```

---

### 9 — SoundStream

**Model:** `soundstream_16khz`
**Weights:** auto-download from HuggingFace (`naturalspeech2.pt`, ~143 MB)

```bash
pip install -r requirements/soundstream.txt

# After installing, restore a newer huggingface-hub so EnCodec still works:
pip install --upgrade huggingface-hub

neural-codec setup --codec soundstream_16khz
```

| Package | Version | Notes |
|---------|---------|-------|
| `soundstream` | 0.0.1 | pins `numpy<2.0` and `huggingface-hub<0.16` |
| `soundfile` | latest | WAV I/O |

> **Dependency conflict with EnCodec:** `soundstream==0.0.1` forces `huggingface-hub<0.16`,
> which breaks `transformers`. Fix: after installing soundstream, run
> `pip install --upgrade huggingface-hub`. Both codecs then work in the same environment.
> For a fully isolated setup, use a dedicated virtual environment:

```bash
python -m venv venv_soundstream
venv_soundstream\Scripts\activate         # Windows
source venv_soundstream/bin/activate      # Linux / Mac

pip install -r requirements/soundstream.txt
neural-codec decode --codec soundstream_16khz --input ./wavs --output ./out
```

---

### 10 — SpeechTokenizer

**Model:** `speechtokenizer` (16 kHz)
**Weights:** manual download required

```bash
pip install -r requirements/speechtokenizer.txt
```

| Package | Notes |
|---------|-------|
| `speechtokenizer` | model loader |
| `beartype` | required runtime dependency of speechtokenizer |
| `soundfile` | WAV I/O |

**The pip packages alone are not enough — you must also download the checkpoint manually.**

**Step 1 — Install packages:**
```bash
neural-codec setup --codec speechtokenizer
```

**Step 2 — Download both files** from HuggingFace:

> [fnlp/SpeechTokenizer → speechtokenizer_hubert_avg](https://huggingface.co/fnlp/SpeechTokenizer/tree/main/speechtokenizer_hubert_avg)

Place them at these exact paths:

```
Neural-Codecs/
  checkpoints/
    SpeechTokenizer.pt     ← download from HuggingFace
  config/
    config.json            ← download from HuggingFace
```

**Step 3 — Decode:**
```bash
neural-codec decode --codec speechtokenizer --input ./wavs --output ./out
```

> If the checkpoint is missing the CLI prints the exact download URL and expected path — no silent failures.

---

### 11 — FunCodec

**Model:** `funcodec_16khz` (16 kHz, mono)
**Weights:** auto-download from HuggingFace (`alibaba-damo/audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch`, ~150 MB)

FunCodec must run in its own virtual environment. A pre-configured `funcodec/` venv is
already included in this repo with all dependencies installed.

#### Why a dedicated venv?

FunCodec 0.2.0 requires `typeguard==2.13.3` (conflicts with newer versions used by other codecs)
and its `editdistance` dependency fails to build on Python 3.14+. FunCodec is therefore
installed with `--no-deps` — `editdistance` is only needed for WER/CER scoring, not inference.

#### Quick Start (using the included venv)

```bash
# Windows
funcodec\Scripts\activate
neural-codec decode --codec funcodec_16khz --input ./audio_sample --output ./out

# Linux / Mac
source funcodec/bin/activate
neural-codec decode --codec funcodec_16khz --input ./audio_sample --output ./out
```

Model weights are downloaded automatically on first run.

#### Fresh venv setup (if the funcodec/ venv is missing)

```bash
python -m venv funcodec
funcodec\Scripts\activate          # Windows
source funcodec/bin/activate       # Linux / Mac

pip install -r requirements/base.txt

# Install funcodec without its broken build dep (editdistance fails on Python 3.14+)
pip install funcodec --no-deps

# Install inference-only runtime deps (verified complete list)
pip install huggingface_hub librosa kaldiio einops thop six \ pytorch-wpe torch-complex humanfriendly h5py "typeguard==2.13.3"
```

Or via the CLI (handles `--no-deps` automatically):
```bash
neural-codec setup --codec funcodec_16khz
```

#### Python API (inside the funcodec venv)

```python
from audio_codec import decode_folder
decode_folder("funcodec_16khz", "audio_sample/", "out/", "cpu")
decode_folder("11",             "audio_sample/", "out/", "cuda")
```

#### Known bugs fixed

**Bug 1 — `Audio2Mel` hardcoded CUDA** (upstream bug in FunCodec 0.2.0)

`funcodec/models/codec_basic.py` ships `Audio2Mel.__init__` with `device='cuda'` as a
default and a hardcoded `.cuda()` call — crashes immediately on CPU-only machines:

```
AssertionError: Torch not compiled with CUDA enabled
```

`FunCodecDecoder` patches `Audio2Mel.__init__` before model load, making it device-agnostic.

**Bug 2 — incomplete dependency list** (`humanfriendly`, `h5py` missing from install)

FunCodec's import chain requires `humanfriendly` (model summary logging) and `h5py`
(dataset loading) but these are not pulled in by a `--no-deps` install. Both are now
listed explicitly in `requirements/funcodec.txt` and `CODEC_REGISTRY["11"]["pip_packages"]`.

**Bug 3 — wrong input tensor shape**

`FunCodec._encode` expects shape `[B, C, T]` (3D). `FunCodecDecoder.decode_file` correctly
reshapes audio to `[1, T]` before passing to `Speech2Token`, which adds the channel dim.

**Bug 4 — `ValueError` crash on second `decode_folder` call** (Windows)

On Windows, `importlib.util.find_spec('humanfriendly')` raises `ValueError` because the
package's `__spec__` attribute is `None`. This caused the dep-check in `installer.py` to
crash on every call after the first (when the package was already installed):

```
ValueError: humanfriendly.__spec__ is None
```

Fixed in `_import_ok()` by catching `ValueError` and treating it as importable — the
`__spec__ = None` case means the package exists but is partially initialised at spec-scan
time; the actual `import` succeeds fine.

**Bug 5 — output filename used full HuggingFace repo path**

Output files were named `<stem>_audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch.wav`
because `self.name` was derived from the HuggingFace hub slug. Fixed in
`FunCodecDecoder.__init__` to use a clean codec name:

```python
self.name = f"funcodec_{sample_rate // 1000}khz"   # -> funcodec_16khz
```

Output files are now correctly named `<stem>_funcodec_16khz.wav`.

---

### AudioDec (external)

**Status:** The `AudioDec/` repo and checkpoints are already included in this repo.
A dedicated `AudioDec/audiodec/` venv is pre-configured with all required packages.

#### Quick Start (using the included setup)

```bash
# Windows — run from inside the AudioDec/ folder
cd AudioDec
audiodec\Scripts\python AudioDec.py --model libritts_v1 --cuda -1 -i ..\audio_sample\ -o ..\out\

# Linux / Mac
cd AudioDec
source audiodec/bin/activate
python AudioDec.py --model libritts_v1 --cuda -1 -i ../audio_sample/ -o ../out/
```

#### CLI flags

| Flag | Description |
|------|-------------|
| `--model` | Model name (see table below) |
| `-i` / `--input` | Input WAV file or directory |
| `-o` / `--output` | Output WAV file or directory |
| `--cuda` | CUDA device index, or `-1` for CPU (default: 0) |

#### Models

| Model | Sample Rate | Notes |
|-------|-------------|-------|
| `libritts_v1` | 24 kHz | English speech |
| `vctk_v1` | 48 kHz | Multi-speaker speech |
| `vctk_v0` | 48 kHz | Earlier checkpoint |
| `vctk_v2` | 48 kHz | Later checkpoint |

Output files are named `<stem>_AudioDec_<model>_<khz>khz.wav`.

#### Fresh setup (if the AudioDec/ folder is missing)

```bash
git clone https://github.com/facebookresearch/AudioDec.git
cd AudioDec

python -m venv audiodec
audiodec\Scripts\activate          # Windows
source audiodec/bin/activate       # Linux / Mac

pip install -r requirements.txt
```

Download [exp.zip](https://github.com/facebookresearch/AudioDec/releases/download/pretrain_models_v02/exp.zip),
extract into `AudioDec/` so you have `AudioDec/exp/autoencoder/...` and `AudioDec/exp/vocoder/...`.

Copy `AudioDec.py` from this repo's root into `AudioDec/` (it fixes two Windows bugs — see below).

#### Known bugs fixed

**Bug 1 — Unicode characters crash on Windows**

The original `AudioDec.py` uses `→` and `…` in print statements. On Windows these trigger a
`UnicodeEncodeError` on cp1252 terminals. Fixed by replacing them with plain ASCII (`->`, `...`).

**Bug 2 — Output directory not created for single-file input**

When passing a single input file with a non-existent output directory, `os.path.isdir` returns
False and the code tries to write to the directory path as if it were a `.wav` filename:

```
TypeError: No format specified and unable to get format from file extension: '../out/'
```

Fixed: if the output path does not end with `.wav`, it is created as a directory automatically
before the `os.path.isdir` check.
