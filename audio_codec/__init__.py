# audio_codec package
from .registry import CODEC_REGISTRY, EXTERNAL_CODECS
from .cli import decoder_list, decode_folder
from .installer import setup_codec, setup_all, ensure_deps, deps_satisfied
from .config import AUTO_INSTALL_DEPS
