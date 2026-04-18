# audio_codec/codecs/dac.py

import os
import torch
import numpy as np
import soundfile as sf
import dac
from audiotools import AudioSignal

class DACDecoder:
    def __init__(self, hub_name: str, sample_rate: int, device: str = "cpu"):
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        model_type = hub_name.split("/")[-1].replace("dac_", "")
        print(f"  -> Downloading + loading DAC model ({model_type}) onto {self.device}")
        model_path = dac.utils.download(model_type=model_type)
        self.model = dac.DAC.load(model_path).to(self.device).eval()
        self.name = f"dac_{model_type}"

    def decode_file(self, src_path: str, out_dir: str) -> str:
        base     = os.path.splitext(os.path.basename(src_path))[0]
        out_name = f"{base}_{self.name}.wav"
        out_path = os.path.join(out_dir, out_name)
        os.makedirs(out_dir, exist_ok=True)

        signal     = AudioSignal(src_path).to(self.model.device)
        compressed = self.model.compress(signal)
        recon      = self.model.decompress(compressed).to("cpu")
        recon.write(out_path)

        del signal, compressed, recon
        return out_name
