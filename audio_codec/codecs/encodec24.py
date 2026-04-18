import os
import torch
import numpy as np
import soundfile as sf
from torchaudio.transforms import Resample
from transformers import EncodecModel, AutoProcessor

class Encodec24Decoder:
    def __init__(self, hub_name: str, sample_rate: int, device: str = "cpu"):
        self.device    = torch.device(device)
        print(f"  -> Loading Encodec model {hub_name} onto {self.device}")
        self.model     = EncodecModel.from_pretrained(hub_name).eval().to(self.device)
        self.processor = AutoProcessor.from_pretrained(hub_name)
        self.sample_rate = self.processor.sampling_rate
        self.name      = hub_name.replace("/", "_")

    def _load(self, path: str) -> torch.Tensor:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T)          # (C, T)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True) # mono (1, T)
        if sr != self.sample_rate:
            wav = Resample(sr, self.sample_rate)(wav)
        return wav                              # (1, T)

    def _save(self, wav: torch.Tensor, path: str):
        arr = wav.squeeze().cpu().numpy()       # (T,) mono
        sf.write(path, arr, self.sample_rate, subtype="PCM_16")

    def decode_file(self, src_path: str, out_dir: str):
        base     = os.path.splitext(os.path.basename(src_path))[0]
        out_name = f"{base}_{self.name}.wav"
        out_path = os.path.join(out_dir, out_name)
        os.makedirs(out_dir, exist_ok=True)

        wav      = self._load(src_path)
        audio_np = wav.squeeze(0).cpu().numpy()
        inputs   = self.processor(raw_audio=audio_np, sampling_rate=self.sample_rate, return_tensors="pt")
        input_vals = inputs["input_values"].to(self.device)
        padding    = inputs.get("padding_mask")
        if padding is not None:
            padding = padding.to(self.device)

        with torch.inference_mode():
            enc     = self.model.encode(input_vals, padding_mask=padding)
            decoded = self.model.decode(enc.audio_codes, enc.audio_scales, padding_mask=padding)[0]

        if decoded.ndim == 3:
            decoded = decoded.squeeze(0)
        elif decoded.ndim == 1:
            decoded = decoded.unsqueeze(0)

        self._save(decoded.to(torch.float32), out_path)
        del wav, inputs, input_vals, padding, enc, decoded
        return out_name
