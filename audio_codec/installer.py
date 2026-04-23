# audio_codec/installer.py
import importlib.util
import subprocess
import sys
from .registry import CODEC_REGISTRY, EXTERNAL_CODECS

_LINE  = "-" * 60
_DLINE = "=" * 60


# ── low-level helpers ────────────────────────────────────────

def _pip_install(packages: list):
    cmd = [sys.executable, "-m", "pip", "install"] + packages
    print(f"  $ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def _import_ok(module_name: str) -> bool:
    """Return True if `module_name` can be imported (uses find_spec, no side effects)."""
    return importlib.util.find_spec(module_name) is not None


def _missing_imports(info: dict) -> list[str]:
    """Return the import names from import_checks that are not yet importable."""
    return [m for m in info.get("import_checks", []) if not _import_ok(m)]


def _install_entry(info: dict):
    """pip-install all packages listed in info['pip_packages']."""
    name     = info["name"]
    packages = info.get("pip_packages", [])
    notes    = info.get("install_notes", [])

    print(f"\n[{name}] Installing packages...")
    git_pkgs = [p for p in packages if p.startswith("git+")]
    reg_pkgs = [p for p in packages if not p.startswith("git+")]

    if reg_pkgs:
        _pip_install(reg_pkgs)
    for pkg in git_pkgs:
        _pip_install([pkg])

    print(f"[{name}] Install complete.")
    if notes:
        print(f"\n[{name}] Manual steps still required:")
        for note in notes:
            print(f"  {note}")


# ── public dep-check API (used by cli.py) ───────────────────

def deps_satisfied(info: dict) -> bool:
    """Return True if every import_check for this codec is importable."""
    return len(_missing_imports(info)) == 0


def ensure_deps(info: dict, auto_install: bool) -> bool:
    """
    Ensure all dependencies for `info` are present.

    auto_install=True  → install missing packages automatically, then return True.
    auto_install=False → print what is missing and how to fix it, return False.
    """
    missing = _missing_imports(info)
    if not missing:
        return True

    name = info["name"]
    if auto_install:
        print(f"\n[{name}] Missing imports detected: {', '.join(missing)}")
        _install_entry(info)
        # importlib caches find_spec results; invalidate and re-check
        importlib.invalidate_caches()
        still_missing = _missing_imports(info)
        if still_missing:
            print(f"[{name}] WARNING: still missing after install: {', '.join(still_missing)}")
            print("  The package may need a fresh Python process to be importable.")
            return False
        return True
    else:
        pkgs = " ".join(p for p in info.get("pip_packages", []) if not p.startswith("git+"))
        print(f"\n[{name}] Missing dependencies: {', '.join(missing)}")
        print(f"  To install:        neural-codec setup --codec {name}")
        if pkgs:
            print(f"  Or manually:       pip install {pkgs}")
        print(f"  To auto-install:   set AUTO_INSTALL_DEPS = True in audio_codec/config.py")
        return False


# ── public setup commands (used by `neural-codec setup`) ────

def _resolve(name_or_id: str) -> dict | None:
    if name_or_id in CODEC_REGISTRY:
        return CODEC_REGISTRY[name_or_id]
    for info in CODEC_REGISTRY.values():
        if info["name"] == name_or_id:
            return info
    return None


def _print_external(ext: dict):
    name = ext["name"]
    print(f"\n{_LINE}")
    print(f"[{name}] External repo — manual setup required")
    print(_LINE)
    print("\nSetup steps:")
    for i, step in enumerate(ext["steps"], 1):
        print(f"  {i}. {step}")
    if "encode_usage" in ext:
        print("\nEncode usage:")
        print(f"  {ext['encode_usage']}")
    if "decode_usage" in ext:
        print("\nDecode usage:")
        print(f"  {ext['decode_usage']}")
    if "scp_note" in ext:
        print(f"\nNote: {ext['scp_note']}")


def setup_codec(name_or_id: str):
    info = _resolve(name_or_id)
    if info is not None:
        _install_entry(info)
        return

    key = name_or_id.lower()
    if key in EXTERNAL_CODECS:
        _print_external(EXTERNAL_CODECS[key])
        return

    print(f"Unknown codec: {name_or_id!r}")
    print("Run `neural-codec list` to see all available codecs.")


def setup_all():
    print(_DLINE)
    print(" Neural Codecs - Full Setup")
    print(_DLINE)

    seen: set  = set()
    reg_pkgs: list = []
    git_pkgs: list = []
    for info in CODEC_REGISTRY.values():
        for pkg in info.get("pip_packages", []):
            if pkg not in seen:
                seen.add(pkg)
                (git_pkgs if pkg.startswith("git+") else reg_pkgs).append(pkg)

    if reg_pkgs:
        print(f"\nInstalling: {', '.join(reg_pkgs)}")
        _pip_install(reg_pkgs)
    for pkg in git_pkgs:
        print(f"\nInstalling: {pkg}")
        _pip_install([pkg])

    print(f"\n{_LINE}")
    print("Pip installations complete.")

    for info in CODEC_REGISTRY.values():
        if "install_notes" in info:
            print(f"\n{_LINE}")
            print(f"[{info['name']}] Manual step required:")
            for note in info["install_notes"]:
                print(f"  {note}")

    print(f"\n{_LINE}")
    print("External codecs (require manual setup):")
    for key, ext in EXTERNAL_CODECS.items():
        print(f"\n  [{ext['name']}]")
        print(f"  Full instructions:  neural-codec setup --codec {key}")

    print(f"\n{_DLINE}")
    print("Setup complete. Run `neural-codec list` to see all codecs.")
    print(_DLINE)
