import os
import torch
import soundfile as sf
from torchaudio.functional import resample
from snac import SNAC


class SNACDecoder:
    def __init__(self, hub_name: str, sample_rate: int, device: str = "cpu"):
        self.sample_rate = sample_rate
        self.device      = torch.device(device)
        self.name        = hub_name.split("/")[-1]   # e.g. snac_24khz
        print(f"  -> Loading SNAC model {hub_name} onto {self.device}")
        self.model = SNAC.from_pretrained(hub_name).eval().to(self.device)

    def _load(self, path: str) -> torch.Tensor:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T)               # (C, T)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)      # mono (1, T)
        if sr != self.sample_rate:
            wav = resample(wav, orig_freq=sr, new_freq=self.sample_rate)
        peak = wav.abs().max()
        if peak > 1.0:
            wav = wav / peak
        return wav.unsqueeze(0).to(self.device)      # (1, 1, T)

    def _save(self, wav: torch.Tensor, path: str):
        arr = wav.squeeze().cpu().numpy()
        sf.write(path, arr, self.sample_rate, subtype="PCM_16")

    def decode_file(self, src_path: str, out_dir: str) -> str | None:
        base     = os.path.splitext(os.path.basename(src_path))[0]
        out_name = f"{base}_{self.name}.wav"
        out_path = os.path.join(out_dir, out_name)

        try:
            wav = self._load(src_path)
            with torch.inference_mode():
                codes   = self.model.encode(wav)
                wav_hat = self.model.decode(codes)
            self._save(wav_hat, out_path)
            del wav, codes, wav_hat
            return out_name
        except Exception as e:
            print(f"\n[SKIPPED] {os.path.basename(src_path)} | {type(e).__name__}: {e}")
            return None
