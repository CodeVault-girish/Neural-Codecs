import os
import sys
import glob
import torch
import soundfile as sf
import librosa
import numpy as np
from ..utils import TempWavContext

# AcademiCodec has no setup.py — add cloned repo root to path so
# `import academicodec` resolves to <GIT_ROOT>/AcademiCodec/academicodec/
_ACADEMICODEC_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "AcademiCodec")
)
if os.path.isdir(_ACADEMICODEC_SRC) and _ACADEMICODEC_SRC not in sys.path:
    sys.path.insert(0, _ACADEMICODEC_SRC)


class AcademiCodecDecoder:
    """
    HiFi-Codec / AcademiCodec decoder.

    Checkpoint : downloaded from Dongchao/AcademiCodec on HuggingFace
                 (root-level file, no extension)
    Config     : JSON file from the cloned AcademiCodec repo at
                 <GIT_ROOT>/AcademiCodec/egs/<egs_dir>/config_*.json

    hub_ckpt_file  — filename in HuggingFace repo (e.g. "HiFi-Codec-16k-320d")
    local_egs_dir  — subdirectory name under egs/ in the cloned repo
                     (e.g. "HiFi-Codec-16k-320d").  For large-universal, point
                     to the 320d egs dir since they share the same architecture.
    """

    def __init__(self, hub_name: str, hub_ckpt_file: str, local_egs_dir: str,
                 sample_rate: int = 16000, device: str = "cpu"):
        from academicodec.models.hificodec.vqvae import VQVAE
        from huggingface_hub import hf_hub_download

        self.device      = device
        self.sample_rate = sample_rate
        self.name        = (
            hub_ckpt_file
            .lower()
            .replace("hifi-codec-", "hifi_codec_")
            .replace("-", "_")
        )

        # ── config: from cloned egs/ directory ───────────────────
        egs_path      = os.path.join(_ACADEMICODEC_SRC, "egs", local_egs_dir)
        config_files  = glob.glob(os.path.join(egs_path, "config_*.json"))
        if not config_files:
            raise FileNotFoundError(
                f"No config_*.json found in {egs_path}. "
                f"Ensure AcademiCodec is cloned at {_ACADEMICODEC_SRC}."
            )
        config_path = config_files[0]

        # ── checkpoint: from HuggingFace (root-level, no extension) ──
        print(f"  -> Downloading {hub_ckpt_file} from {hub_name} ...")
        ckpt_path = hf_hub_download(hub_name, hub_ckpt_file)

        print(f"  -> Config     : {os.path.basename(config_path)}")
        print(f"  -> Checkpoint : {os.path.basename(ckpt_path)}")
        print(f"  -> Loading on {device} ...")

        self.model = VQVAE(config_path, ckpt_path, with_encoder=True)
        self.model = self.model.to(device).eval()

    def decode_file(self, src_path: str, out_dir: str) -> str:
        base     = os.path.splitext(os.path.basename(src_path))[0]
        out_name = f"{base}_{self.name}.wav"
        out_path = os.path.join(out_dir, out_name)
        os.makedirs(out_dir, exist_ok=True)

        try:
            with TempWavContext(src_path) as wav_path:
                wav, sr = sf.read(wav_path)
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                if sr != self.sample_rate:
                    wav = librosa.resample(wav, orig_sr=sr,
                                           target_sr=self.sample_rate)

                # encode() expects [B, T]
                x = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    codes = self.model.encode(x)   # [1, T', Nq=4]
                    recon = self.model(codes)       # [1, 1, T']

                recon_np = recon.squeeze().cpu().numpy()
                sf.write(out_path, recon_np, self.sample_rate)
                return out_name

        except Exception as e:
            msg = f"[SKIPPED] {src_path} | {type(e).__name__}: {e}"
            print("\n" + "=" * 80)
            print(msg)
            print("=" * 80 + "\n")
            return None
