import os
import torch
import numpy as np
import soundfile as sf
from torchaudio.transforms import Resample

try:
    from soundstream import from_pretrained as _ss_from_pretrained
except ImportError as _e:
    raise ImportError(
        "SoundStream package not installed. Run:\n"
        "  neural-codec setup --codec soundstream_16khz\n"
        f"Original error: {_e}"
    ) from _e

def _load_wav(path: str, target_sr: int) -> tuple[torch.Tensor, int]:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)              # (C, T)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)     # mono (1, T)
    if sr != target_sr:
        wav = Resample(sr, target_sr)(wav)
    return wav, target_sr

def _save_wav(path: str, wav: torch.Tensor, sr: int):
    arr = wav.squeeze().cpu().numpy()           # (T,)
    sf.write(path, arr, sr, subtype="PCM_16")

class SoundStreamDecoder:
    def __init__(self, hub_name: str, sample_rate: int, device: str = "cpu"):
        self.device      = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.model_sr    = 16000
        self.sample_rate = sample_rate
        self.name        = f"soundstream_{self.model_sr // 1000}khz"
        print(f"  -> Loading SoundStream model onto {self.device}")
        self.codec = _ss_from_pretrained().eval().to(self.device)

    def decode_file(self, src_path: str, out_dir: str):
        base     = os.path.splitext(os.path.basename(src_path))[0]
        out_name = f"{base}_{self.name}.wav"
        out_path = os.path.join(out_dir, out_name)
        os.makedirs(out_dir, exist_ok=True)

        waveform, _ = _load_wav(src_path, self.model_sr)
        waveform    = waveform.unsqueeze(0).to(self.device)     # (1, 1, T) — batch x chan x time

        with torch.inference_mode():
            recovered = self.codec(waveform, mode="end-to-end") # (1, 1, T')

        recovered = recovered.detach().cpu().squeeze(0)         # (1, T')
        if self.sample_rate != self.model_sr:
            recovered = Resample(self.model_sr, self.sample_rate)(recovered)

        _save_wav(out_path, recovered, self.sample_rate)
        del waveform, recovered
        return out_name
