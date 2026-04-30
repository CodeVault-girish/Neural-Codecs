#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

### usage ###
# (file):  python demoFile.py --model libritts_v1 -i in.wav        -o out.wav
# (dir):   python demoFile.py --model vctk_v1   -i in_folder    -o out_folder

import os
import torch
import argparse
import numpy as np
import librosa
import soundfile as sf
from utils.audiodec import AudioDec, assign_model

def process_file(audiodec, sample_rate, model_name, suffix, tx_device, filepath, out_dir):
    data, fs = sf.read(filepath, always_2d=True)

    if fs != sample_rate:
        print(f"[{os.path.basename(filepath)}] Resampling {fs}->{sample_rate} Hz...")
        data = np.stack([
            librosa.resample(data[:, ch], orig_sr=fs, target_sr=sample_rate)
            for ch in range(data.shape[1])
        ], axis=1)
        fs = sample_rate

    x = np.expand_dims(data.transpose(1, 0), axis=1)
    x = torch.tensor(x, dtype=torch.float).to(tx_device)

    with torch.no_grad():
        z   = audiodec.tx_encoder.encode(x)
        idx = audiodec.tx_encoder.quantize(z)
        zq  = audiodec.rx_encoder.lookup(idx)
        y   = audiodec.decoder.decode(zq)[:, :, :x.size(-1)]
    y = y.squeeze(1).transpose(1, 0).cpu().numpy()

    base = os.path.splitext(os.path.basename(filepath))[0]
    out_filename = f"{base}_AudioDec_{model_name}_{suffix}.wav"
    out_path = os.path.join(out_dir, out_filename)
    sf.write(out_path, y, fs, "PCM_16")
    print(f"[{os.path.basename(filepath)}] -> {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str, default="libritts_v1",
                        help="Choose model: vctk_v1 (48 kHz) or libritts_v1 (24 kHz)")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Input WAV file or directory")
    parser.add_argument("-o", "--output",type=str, required=True,
                        help="Output WAV file or directory")
    parser.add_argument("--cuda",       type=int, default=0,
                        help="CUDA device index, or -1 for CPU")
    parser.add_argument("--num_threads",type=int, default=4)
    args = parser.parse_args()

    # device
    if args.cuda < 0:
        tx_device = rx_device = "cpu"
    else:
        tx_device = rx_device = f"cuda:{args.cuda}"
    torch.set_num_threads(args.num_threads)

    # model assignment
    sample_rate, enc_ckpt, dec_ckpt = assign_model(args.model)
    khz = sample_rate // 1000
    suffix = f"{khz}khz"
    model_name = args.model

    print("AudioDec initializing...")
    audiodec = AudioDec(tx_device=tx_device, rx_device=rx_device)
    audiodec.load_transmitter(enc_ckpt)
    audiodec.load_receiver(enc_ckpt, dec_ckpt)

    inp, outp = args.input, args.output

    # directory mode
    if os.path.isdir(inp):
        os.makedirs(outp, exist_ok=True)
        wavs = sorted(f for f in os.listdir(inp) if f.lower().endswith(".wav"))
        if not wavs:
            raise ValueError(f"No .wav files found in {inp}")
        for fn in wavs:
            process_file(
                audiodec, sample_rate,
                model_name, suffix,
                tx_device,
                os.path.join(inp, fn),
                outp
            )

    # single-file mode
    else:
        if not os.path.exists(inp):
            raise ValueError(f"Input {inp} does not exist")

        # if output is (or should be) a directory, drop into it
        if not outp.lower().endswith(".wav"):
            os.makedirs(outp, exist_ok=True)

        if os.path.isdir(outp):
            process_file(
                audiodec, sample_rate,
                model_name, suffix,
                tx_device,
                inp,
                outp
            )

        # explicit output filename
        else:
            data, fs = sf.read(inp, always_2d=True)
            if fs != sample_rate:
                print(f"Resampling {fs}->{sample_rate} Hz...")
                data = np.stack([
                    librosa.resample(data[:, ch], orig_sr=fs, target_sr=sample_rate)
                    for ch in range(data.shape[1])
                ], axis=1)
                fs = sample_rate

            x = torch.tensor(
                np.expand_dims(data.transpose(1,0), axis=1),
                dtype=torch.float
            ).to(tx_device)

            with torch.no_grad():
                z   = audiodec.tx_encoder.encode(x)
                idx = audiodec.tx_encoder.quantize(z)
                zq  = audiodec.rx_encoder.lookup(idx)
                y   = audiodec.decoder.decode(zq)[:, :, :x.size(-1)]
            y = y.squeeze(1).transpose(1,0).cpu().numpy()

            sf.write(outp, y, fs, "PCM_16")
            print(f"Output written to {outp}")

if __name__ == "__main__":
    main()
