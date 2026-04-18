#!/usr/bin/env python3
import argparse
import importlib
import os
import time
import gc
import torch
from tqdm import tqdm
from .registry import CODEC_REGISTRY, EXTERNAL_CODECS
from .config import AUTO_INSTALL_DEPS
from .installer import ensure_deps

_LINE  = "-" * 60
_DLINE = "=" * 60


def _resolve_id(name_or_id: str) -> str | None:
    if name_or_id in CODEC_REGISTRY:
        return name_or_id
    for key, info in CODEC_REGISTRY.items():
        if info["name"] == name_or_id:
            return key
    return None


def decoder_list():
    print(f"\n{_DLINE}")
    print(" Neural Codecs - Available Decoders")
    print(_DLINE)
    print(f"\n  {'ID':<5} {'Name':<24} {'Sample Rate':<12} {'Status':<12} {'Packages'}")
    print(f"  {'-'*5} {'-'*24} {'-'*12} {'-'*12} {'-'*30}")
    for key, info in sorted(CODEC_REGISTRY.items(), key=lambda x: int(x[0])):
        sr     = f"{info['sample_rate'] // 1000} kHz" if info["sample_rate"] else "variable"
        pkgs   = ", ".join(info.get("pip_packages", []))
        status = "installed" if ensure_deps(info, auto_install=False) else "not installed"
        # suppress the "missing" print during list — just capture return value silently
        print(f"  {key:<5} {info['name']:<24} {sr:<12} {status:<12} {pkgs}")

    print(f"\n  {'-'*5} {'-'*24}")
    print(f"\n  External codecs (manual setup required):")
    for key, ext in EXTERNAL_CODECS.items():
        print(f"  {'--':<5} {ext['name']:<24} neural-codec setup --codec {key}")

    print(f"\nAuto-install:  {'ENABLED' if AUTO_INSTALL_DEPS else 'DISABLED'}"
          f"  (audio_codec/config.py  AUTO_INSTALL_DEPS)")
    print(f"\nUsage:")
    print(f"  neural-codec setup  --all                             # install all deps")
    print(f"  neural-codec setup  --codec snac_24khz               # install one codec")
    print(f"  neural-codec decode --codec snac_24khz --input ./in --output ./out")
    print(f"  neural-codec decode --codec 1          --input ./in --output ./out --device cuda")
    print()


def _gen_wav_paths(in_dir: str):
    for root, _, files in os.walk(in_dir):
        for fn in sorted(files):
            if fn.lower().endswith(".wav"):
                yield os.path.join(root, fn)


def decode_folder(decoder_id: str, in_dir: str, out_dir: str, device_str: str):
    if decoder_id not in CODEC_REGISTRY:
        print(f"Unknown decoder ID {decoder_id!r}. Run `neural-codec list`.")
        return

    info = CODEC_REGISTRY[decoder_id]

    # ── dependency check / auto-install ──────────────────────
    if not ensure_deps(info, auto_install=AUTO_INSTALL_DEPS):
        return

    # ── load codec module ─────────────────────────────────────
    try:
        module  = importlib.import_module(info["module"])
        Decoder = getattr(module, info["class"])
    except ImportError as e:
        print(f"\nERROR: Import failed for '{info['name']}' even after dependency check.")
        print(f"  {e}")
        print("  Try restarting Python or running in a fresh process.")
        return

    # ── device selection ──────────────────────────────────────
    can_cuda = torch.cuda.is_available()
    if device_str == "cuda" and not can_cuda:
        print("Warning: CUDA not available, falling back to CPU.")
    device = "cuda" if (device_str == "cuda" and can_cuda) else "cpu"

    # ── constructor kwargs ────────────────────────────────────
    ctor_kwargs: dict = {"device": device}
    if "hub_name" in info:
        ctor_kwargs["hub_name"]    = info["hub_name"]
    if info.get("sample_rate") is not None:
        ctor_kwargs["sample_rate"] = info["sample_rate"]
    if "config_path" in info and "ckpt_path" in info:
        ctor_kwargs["config_path"] = info["config_path"]
        ctor_kwargs["ckpt_path"]   = info["ckpt_path"]

    print(f"\n{_LINE}")
    print(f" Codec  : {info['name']}")
    print(f" Device : {device}")
    print(f" Input  : {in_dir}")
    print(f" Output : {out_dir}")
    print(_LINE)
    print(f"\nLoading model...")
    try:
        decoder = Decoder(**ctor_kwargs)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return

    try:
        os.makedirs(out_dir, exist_ok=True)
        total = sum(1 for _ in _gen_wav_paths(in_dir))
        if total == 0:
            print("No WAV files found in input directory.")
            return

        print(f"\nProcessing {total} file(s)...\n")
        pbar = tqdm(total=total, unit="file", ncols=72)
        for src in _gen_wav_paths(in_dir):
            t0 = time.perf_counter()
            decoder.decode_file(src, out_dir)
            elapsed = time.perf_counter() - t0
            torch.cuda.empty_cache()
            gc.collect()
            pbar.update(1)
            pbar.set_postfix_str(f"{elapsed:.2f}s")
        pbar.close()

        print(f"\n{_LINE}")
        print(f" Done. {total} file(s) written to: {out_dir}")
        print(_LINE + "\n")

    finally:
        del decoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def main():
    parser = argparse.ArgumentParser(
        prog="neural-codec",
        description="Neural audio codec pipeline — encode/decode WAV files for dataset creation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  neural-codec list\n"
            "  neural-codec setup --all\n"
            "  neural-codec setup --codec snac_24khz\n"
            "  neural-codec setup --codec funcodec\n"
            "  neural-codec decode --codec snac_24khz --input ./wavs --output ./out\n"
            "  neural-codec decode --codec 7 --input ./wavs --output ./out --device cuda\n"
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="Show all available codecs with IDs and install status")

    sp  = sub.add_parser("setup", help="Install codec dependencies")
    grp = sp.add_mutually_exclusive_group(required=True)
    grp.add_argument("--codec", metavar="NAME_OR_ID",
                     help="Codec name or numeric ID  (e.g. snac_24khz, 1, funcodec, audiodec)")
    grp.add_argument("--all", action="store_true",
                     help="Install dependencies for all pip-installable codecs")

    dp = sub.add_parser("decode", help="Batch decode a folder of WAV files")
    dp.add_argument("--codec", required=True, metavar="NAME_OR_ID",
                    help="Codec name or numeric ID  (e.g. snac_24khz or 1)")
    dp.add_argument("--input",  required=True, metavar="DIR",
                    help="Input folder containing WAV files (searched recursively)")
    dp.add_argument("--output", required=True, metavar="DIR",
                    help="Output folder for decoded WAV files")
    dp.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                    help="Device to run on (default: cpu)")

    args = parser.parse_args()

    if args.cmd == "list":
        decoder_list()

    elif args.cmd == "setup":
        from .installer import setup_codec, setup_all
        if args.all:
            setup_all()
        else:
            setup_codec(args.codec)

    elif args.cmd == "decode":
        codec_id = _resolve_id(args.codec)
        if codec_id is None:
            print(f"Unknown codec: {args.codec!r}")
            print("Run `neural-codec list` to see all available codecs.")
            return
        decode_folder(codec_id, args.input, args.output, args.device)


if __name__ == "__main__":
    main()
