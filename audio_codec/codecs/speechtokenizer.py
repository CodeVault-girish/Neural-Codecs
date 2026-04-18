import os
import torch
import numpy as np
import soundfile as sf
from torchaudio.functional import resample
from speechtokenizer import SpeechTokenizer

class SpeechTokenizerDecoder:
    def __init__(
        self,
        hub_name: str = None,
        sample_rate: int = None,
        device: str = "cpu",
        config_path: str = None,
        ckpt_path: str = None,
    ):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

        self.config_path = config_path or os.path.join(repo_root, "config", "config.json")
        self.ckpt_path   = ckpt_path   or os.path.join(repo_root, "checkpoints", "SpeechTokenizer.pt")

        if not os.path.isfile(self.ckpt_path):
            raise FileNotFoundError(
                f"SpeechTokenizer checkpoint not found at: {self.ckpt_path}\n"
                "Download it from: https://huggingface.co/fnlp/SpeechTokenizer/tree/main/speechtokenizer_hubert_avg\n"
                f"Then place it at: {self.ckpt_path}"
            )
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(
                f"SpeechTokenizer config not found at: {self.config_path}\n"
                "Download config.json from: https://huggingface.co/fnlp/SpeechTokenizer/tree/main/speechtokenizer_hubert_avg\n"
                f"Then place it at: {self.config_path}"
            )

        self.device = torch.device(device)
        print(f"  -> Loading SpeechTokenizer from {self.ckpt_path} onto {self.device}")
        self.model = (
            SpeechTokenizer.load_from_checkpoint(
                config_path=self.config_path,
                ckpt_path=self.ckpt_path,
            )
            .to(self.device)
            .eval()
        )
        self.sample_rate = getattr(self.model, "sample_rate", sample_rate)
        if self.sample_rate is None:
            raise ValueError("Sample rate not set and not found on model")

    def _load(self, path: str) -> torch.Tensor:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T)          # (C, T)
        if wav.size(0) > 1:
            wav = wav[:1]                       # mono
        if sr != self.sample_rate:
            wav = resample(wav, orig_freq=sr, new_freq=self.sample_rate)
        return wav.unsqueeze(0).to(self.device) # (1, 1, T)

    def _save(self, wav: torch.Tensor, path: str):
        arr = wav.detach().cpu()[0].squeeze().numpy()   # (T,)
        sf.write(path, arr, self.sample_rate, subtype="PCM_16")

    def decode_file(self, src_path: str, out_dir: str) -> str:
        base     = os.path.splitext(os.path.basename(src_path))[0]
        out_name = f"{base}_speechtokenizer.wav"
        out_path = os.path.join(out_dir, out_name)
        os.makedirs(out_dir, exist_ok=True)

        wav = self._load(src_path)
        with torch.no_grad():
            codes = self.model.encode(wav)
            recon = self.model.decode(codes)

        self._save(recon, out_path)
        del wav, codes, recon
        return out_name
