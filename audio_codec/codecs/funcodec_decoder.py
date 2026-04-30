import os
import torch
import soundfile as sf
import librosa
import numpy as np


def _patch_audio2mel(device: str):
    """
    Fix a hard-coded CUDA reference inside FunCodec's Audio2Mel class.

    FunCodec 0.2.0 ships Audio2Mel with:
      - default argument  device='cuda'
      - hardcoded         mel_basis = torch.from_numpy(...).cuda()

    Both lines crash on CPU-only machines.  This one-time patch makes
    Audio2Mel honour whatever device is passed in.  It is idempotent —
    calling it multiple times is safe.
    """
    import funcodec.models.codec_basic as cb
    import torch.nn as nn
    from librosa.filters import mel as librosa_mel_fn

    if getattr(cb.Audio2Mel, "_patched", False):
        return

    def _patched_init(self, n_fft=1024, hop_length=256, win_length=1024,
                      sampling_rate=22050, n_mel_channels=80,
                      mel_fmin=0.0, mel_fmax=None, device=device):
        nn.Module.__init__(self)
        window    = torch.hann_window(win_length, device=device).float()
        mel_basis = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft,
                                   n_mels=n_mel_channels,
                                   fmin=mel_fmin, fmax=mel_fmax)
        self.register_buffer("mel_basis",
                             torch.from_numpy(mel_basis).to(device).float())
        self.register_buffer("window", window)
        self.n_fft         = n_fft
        self.hop_length    = hop_length
        self.win_length    = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    cb.Audio2Mel.__init__ = _patched_init
    cb.Audio2Mel._patched  = True


class FunCodecDecoder:
    def __init__(self, hub_name: str, sample_rate: int, device: str = "cpu"):
        self.device      = device
        self.sample_rate = sample_rate
        self.name        = f"funcodec_{sample_rate // 1000}khz"

        # must patch before any FunCodec model import
        _patch_audio2mel(device)

        from huggingface_hub import hf_hub_download
        from funcodec.bin.codec_inference import Speech2Token

        import warnings
        warnings.filterwarnings("ignore")

        print(f"  -> Downloading FunCodec config/weights from {hub_name}...")
        cfg_path   = hf_hub_download(hub_name, "config.yaml")
        model_path = hf_hub_download(hub_name, "model.pth")

        print(f"  -> Loading FunCodec model on {device}...")
        self.model = Speech2Token(
            config_file  = cfg_path,
            model_file   = model_path,
            device       = device,
            sampling_rate = sample_rate,
            bit_width     = sample_rate,
        )

    def decode_file(self, src_path: str, out_dir: str) -> str:
        base     = os.path.splitext(os.path.basename(src_path))[0]
        out_name = f"{base}_{self.name}.wav"
        out_path = os.path.join(out_dir, out_name)
        os.makedirs(out_dir, exist_ok=True)

        # load + mono + resample
        wav, sr = sf.read(src_path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != self.sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)

        # shape: [1, T]  (Speech2Token expects batch dim)
        x = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(self.device)

        result = self.model(x, need_recon=True, bit_width=self.sample_rate)
        # result[2] is the reconstructed waveform: [1, 1, T]
        recon = result[2].squeeze().cpu().numpy()

        sf.write(out_path, recon, self.sample_rate)

        del x, result, recon
        return out_name
