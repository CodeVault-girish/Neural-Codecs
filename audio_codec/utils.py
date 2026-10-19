import os
import sys
import subprocess
import shutil
import tempfile
import soundfile as sf


def ensure_ffmpeg() -> bool:
    """Locate ffmpeg and add it to PATH if needed. Returns True if found."""
    if shutil.which("ffmpeg"):
        return True

    search_dirs = [
        os.path.expanduser("~/.local/bin"),
        os.path.expanduser("~/bin"),
        "/usr/local/bin",
        "/usr/bin",
    ]

    # Search any active conda/venv environment bin dirs
    for prefix_var in ("CONDA_PREFIX", "VIRTUAL_ENV"):
        prefix = os.environ.get(prefix_var)
        if prefix:
            bin_dir = os.path.join(prefix, "Scripts" if sys.platform == "win32" else "bin")
            if bin_dir not in search_dirs:
                search_dirs.append(bin_dir)

    # Scan all conda environments if CONDA_ENVS_PATH is set or discoverable
    for envs_root in filter(None, [
        os.environ.get("CONDA_ENVS_PATH"),
        os.path.expanduser("~/miniconda3/envs"),
        os.path.expanduser("~/anaconda3/envs"),
        os.path.expanduser("~/.conda/envs"),
    ]):
        if os.path.isdir(envs_root):
            for env_name in os.listdir(envs_root):
                bin_dir = os.path.join(
                    envs_root, env_name,
                    "Scripts" if sys.platform == "win32" else "bin"
                )
                if os.path.isdir(bin_dir) and bin_dir not in search_dirs:
                    search_dirs.append(bin_dir)

    ffmpeg_bin = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    for d in search_dirs:
        ff = os.path.join(d, ffmpeg_bin)
        if os.path.isfile(ff) and os.access(ff, os.X_OK):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            return True

    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
        return True
    except ImportError:
        pass

    return False


class TempWavContext:
    """
    Context manager that ensures an audio file is readable by soundfile.
    If the file is not directly readable (e.g. non-standard WAV encoding),
    it converts it to standard PCM 16-bit WAV via ffmpeg and cleans up on exit.
    """

    def __init__(self, src_path: str):
        self.src_path  = src_path
        self.temp_path = None

    def __enter__(self) -> str:
        try:
            sf.info(self.src_path)
            return self.src_path
        except Exception:
            pass

        if not ensure_ffmpeg():
            raise RuntimeError(
                f"File '{self.src_path}' cannot be read by libsndfile "
                "and ffmpeg could not be found. Install ffmpeg or run: "
                "pip install static-ffmpeg"
            )

        fd, self.temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        cmd = [
            "ffmpeg", "-y", "-i", self.src_path,
            "-vn", "-acodec", "pcm_s16le", self.temp_path,
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return self.temp_path
        except subprocess.CalledProcessError as e:
            if os.path.exists(self.temp_path):
                os.remove(self.temp_path)
            err = e.stderr.decode("utf-8", errors="ignore") if e.stderr else str(e)
            raise RuntimeError(f"ffmpeg failed to convert '{self.src_path}': {err}")

    def __exit__(self, *_):
        if self.temp_path and os.path.exists(self.temp_path):
            try:
                os.remove(self.temp_path)
            except OSError:
                pass
