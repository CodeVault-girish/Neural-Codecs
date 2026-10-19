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
from .installer import ensure_deps, deps_satisfied

_LINE  = "-" * 60
_DLINE = "=" * 60


def _resolve_id(name_or_id: str) -> str | None:
    if name_or_id in CODEC_REGISTRY:
        return name_or_id
    for key, info in CODEC_REGISTRY.items():
        if info["name"] == name_or_id:
            return key
    return None


def _pick_device(device_str: str) -> str:
    """Resolve 'auto' → 'cuda' if available else 'cpu', warn on unavailable cuda."""
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_str == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU.")
        return "cpu"
    return device_str


def _load_decoder(info: dict, device: str):
    """Import the decoder class and instantiate it with the right kwargs."""
    try:
        module  = importlib.import_module(info["module"])
        Decoder = getattr(module, info["class"])
    except ImportError as e:
        print(f"\nERROR: Import failed for '{info['name']}' even after dependency check.")
        print(f"  {e}")
        print("  Try restarting Python or running in a fresh process.")
        return None

    ctor_kwargs: dict = {"device": device}
    for key in ("hub_name", "hub_subfolder", "hub_ckpt_file", "local_egs_dir"):
        if key in info:
            ctor_kwargs[key] = info[key]
    if info.get("sample_rate") is not None:
        ctor_kwargs["sample_rate"] = info["sample_rate"]
    if "config_path" in info and "ckpt_path" in info:
        ctor_kwargs["config_path"] = info["config_path"]
        ctor_kwargs["ckpt_path"]   = info["ckpt_path"]

    try:
        return Decoder(**ctor_kwargs)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return None


def decoder_list():
    print(f"\n{_DLINE}")
    print(" Neural Codecs — Available Decoders")
    print(_DLINE)
    print(f"\n  {'ID':<5} {'Name':<42} {'SR':<8} {'Status':<14} {'Packages'}")
    print(f"  {'-'*5} {'-'*42} {'-'*8} {'-'*14} {'-'*30}")
    for key, info in sorted(CODEC_REGISTRY.items(), key=lambda x: int(x[0])):
        sr     = f"{info['sample_rate'] // 1000} kHz" if info["sample_rate"] else "variable"
        pkgs   = ", ".join(info.get("pip_packages", [])[:3])
        status = "installed" if deps_satisfied(info) else "not installed"
        print(f"  {key:<5} {info['name']:<42} {sr:<8} {status:<14} {pkgs}")

    print(f"\n  External codecs (manual setup required):")
    for key, ext in EXTERNAL_CODECS.items():
        print(f"  {'--':<5} {ext['name']:<42} neural-codec setup --codec {key}")

    gpu_info = f"CUDA {torch.version.cuda}" if torch.cuda.is_available() else "not available"
    print(f"\nDevice:       GPU ({gpu_info})  |  CPU always available")
    print(f"Auto-install: {'ENABLED' if AUTO_INSTALL_DEPS else 'DISABLED'}"
          f"  (audio_codec/config.py  AUTO_INSTALL_DEPS)")
    print(f"\nUsage:")
    print(f"  neural-codec test   --codec snac_24khz  --input file.wav")
    print(f"  neural-codec setup  --all")
    print(f"  neural-codec decode --codec snac_24khz --input ./in --output ./out --device auto")
    print()


def _gen_wav_paths(in_dir: str):
    for root, _, files in os.walk(in_dir):
        for fn in sorted(files):
            if fn.lower().endswith(".wav"):
                yield os.path.join(root, fn)


def decode_folder(name_or_id: str, in_dir: str, out_dir: str, device_str: str):
    decoder_id = _resolve_id(name_or_id)
    if decoder_id is None:
        print(f"Unknown codec: {name_or_id!r}. Run `neural-codec list`.")
        return

    info   = CODEC_REGISTRY[decoder_id]
    device = _pick_device(device_str)

    if not ensure_deps(info, auto_install=AUTO_INSTALL_DEPS):
        return

    print(f"\n{_LINE}")
    print(f" Codec  : {info['name']}")
    print(f" Device : {device}")
    print(f" Input  : {in_dir}")
    print(f" Output : {out_dir}")
    print(_LINE)
    print("\nLoading model...")

    decoder = _load_decoder(info, device)
    if decoder is None:
        return

    try:
        os.makedirs(out_dir, exist_ok=True)
        total = sum(1 for _ in _gen_wav_paths(in_dir))
        if total == 0:
            print("No WAV files found in input directory.")
            return

        print(f"\nProcessing {total} file(s)...\n")
        pbar          = tqdm(total=total, unit="file", ncols=72)
        success_count = 0
        for src in _gen_wav_paths(in_dir):
            t0  = time.perf_counter()
            res = decoder.decode_file(src, out_dir)
            if res is not None:
                success_count += 1
            elapsed = time.perf_counter() - t0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            pbar.update(1)
            pbar.set_postfix_str(f"{elapsed:.2f}s")
        pbar.close()

        print(f"\n{_LINE}")
        print(f" Done. {success_count}/{total} file(s) written to: {out_dir}")
        if success_count < total:
            print(f" {total - success_count} file(s) skipped/failed.")
        print(_LINE + "\n")

    finally:
        del decoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def test_codec(name_or_id: str, input_file: str, out_dir: str, device_str: str):
    """Run one file through one (or all) codecs and report pass/fail + timing."""
    if not os.path.isfile(input_file):
        print(f"ERROR: input file not found: {input_file}")
        return

    ids_to_test = (
        list(CODEC_REGISTRY.keys())
        if name_or_id.lower() == "all"
        else [_resolve_id(name_or_id)]
    )
    if ids_to_test == [None]:
        print(f"Unknown codec: {name_or_id!r}. Run `neural-codec list`.")
        return

    device = _pick_device(device_str)
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for cid in ids_to_test:
        info = CODEC_REGISTRY[cid]
        print(f"\n{_LINE}")
        print(f" Testing: {info['name']}  ({device})")
        print(_LINE)

        if not deps_satisfied(info):
            print(f"  [SKIP] dependencies not installed — run: neural-codec setup --codec {info['name']}")
            results.append((info["name"], "SKIP", 0.0))
            continue

        decoder = _load_decoder(info, device)
        if decoder is None:
            results.append((info["name"], "FAIL", 0.0))
            continue

        t0  = time.perf_counter()
        res = decoder.decode_file(input_file, out_dir)
        elapsed = time.perf_counter() - t0

        del decoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if res:
            print(f"  [PASS] {res}  ({elapsed:.2f}s)")
            results.append((info["name"], "PASS", elapsed))
        else:
            results.append((info["name"], "FAIL", elapsed))

    print(f"\n{_DLINE}")
    print(f" Test Results  ({device})")
    print(_DLINE)
    print(f"  {'Codec':<42} {'Result':<8} {'Time'}")
    print(f"  {'-'*42} {'-'*8} {'-'*8}")
    for name, status, t in results:
        print(f"  {name:<42} {status:<8} {t:.2f}s" if t else f"  {name:<42} {status}")
    passed = sum(1 for _, s, _ in results if s == "PASS")
    print(f"\n  {passed}/{len(results)} passed")
    print(_DLINE + "\n")


def main():
    parser = argparse.ArgumentParser(
        prog="neural-codec",
        description="Neural audio codec pipeline — encode/decode WAV files for dataset creation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  neural-codec list\n"
            "  neural-codec test   --codec snac_24khz --input ./audio_sample/sample.wav\n"
            "  neural-codec test   --codec all        --input ./audio_sample/sample.wav\n"
            "  neural-codec setup  --all\n"
            "  neural-codec setup  --codec snac_24khz\n"
            "  neural-codec decode --codec snac_24khz --input ./wavs --output ./out\n"
            "  neural-codec decode --codec 7          --input ./wavs --output ./out --device cuda\n"
            "  neural-codec decode --codec dac_16khz  --input ./wavs --output ./out --device auto\n"
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="Show all codecs with IDs and install status")

    tp = sub.add_parser("test", help="Run one WAV file through a codec and report pass/fail")
    tp.add_argument("--codec",  required=True, metavar="NAME_OR_ID",
                    help="Codec name, numeric ID, or 'all' to test every installed codec")
    tp.add_argument("--input",  required=True, metavar="FILE", help="Input WAV file")
    tp.add_argument("--output", default="./test_out", metavar="DIR",
                    help="Output folder (default: ./test_out)")
    tp.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto",
                    help="Device (default: auto — uses GPU if available)")

    sp  = sub.add_parser("setup", help="Install codec dependencies")
    grp = sp.add_mutually_exclusive_group(required=True)
    grp.add_argument("--codec", metavar="NAME_OR_ID",
                     help="Codec name or numeric ID  (e.g. snac_24khz, 1, funcodec, audiodec)")
    grp.add_argument("--all", action="store_true",
                     help="Install dependencies for all pip-installable codecs")

    dp = sub.add_parser("decode", help="Batch decode a folder of WAV files")
    dp.add_argument("--codec",  required=True, metavar="NAME_OR_ID",
                    help="Codec name or numeric ID  (e.g. snac_24khz or 1)")
    dp.add_argument("--input",  required=True, metavar="DIR",
                    help="Input folder with WAV files (searched recursively)")
    dp.add_argument("--output", required=True, metavar="DIR",
                    help="Output folder for decoded WAV files")
    dp.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto",
                    help="Device (default: auto — uses GPU if available)")

    args = parser.parse_args()

    if args.cmd == "list":
        decoder_list()

    elif args.cmd == "test":
        test_codec(args.codec, args.input, args.output, args.device)

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
