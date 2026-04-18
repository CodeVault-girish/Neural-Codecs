import os
import torch
import numpy as np
import soundfile as sf
from snac import SNAC

class SNACDecoder:
    def __init__(self, hub_name: str, sample_rate: int, device: str = "cpu"):
        self.sample_rate = sample_rate
        self.device      = torch.device(device)
        print(f"  -> Loading SNAC model {hub_name} onto {self.device}")
        self.model = SNAC.from_pretrained(hub_name).eval().to(self.device)
        self.name  = hub_name.replace("/", "_")

    def _resample(self, wav: torch.Tensor, orig_sr: int) -> torch.Tensor:
        if orig_sr == self.sample_rate:
            return wav
        # simple linear interpolation resample
        orig_len = wav.shape[-1]
        new_len  = int(orig_len * self.sample_rate / orig_sr)
        return torch.nn.functional.interpolate(
            wav.unsqueeze(0), size=new_len, mode="linear", align_corners=False
        ).squeeze(0)

    def _load(self, path: str):
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T)          # (channels, samples)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = self._resample(wav, sr)
        peak = wav.abs().max()
        if peak > 1.0:
            wav = wav / peak
        return wav.unsqueeze(0).to(self.device)

    def _save(self, wav, out_path: str):
        arr = wav.squeeze().cpu().numpy()
        sf.write(out_path, arr, self.sample_rate, subtype="PCM_16")

    def decode_file(self, src_path: str, out_dir: str):
        """Encode/decode one file and immediately delete intermediates."""
        base = os.path.splitext(os.path.basename(src_path))[0]
        out_name = f"{base}_{self.name}.wav"
        out_path = os.path.join(out_dir, out_name)

        wav = self._load(src_path)
        with torch.inference_mode():
            codes   = self.model.encode(wav)
            wav_hat = self.model.decode(codes)
        self._save(wav_hat, out_path)

        # drop big tensors
        del wav, codes, wav_hat
        return out_name
