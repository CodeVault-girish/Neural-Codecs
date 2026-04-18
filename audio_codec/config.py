# ============================================================
#  Neural Codecs — User Configuration
# ============================================================
#
#  AUTO_INSTALL_DEPS
#  -----------------
#  True  → automatically pip-install any missing codec
#          dependencies the moment you call decode_folder().
#          No manual setup needed; the first run may be slower.
#
#  False → print the missing packages and the install command,
#          then exit without running anything.
#          Use this in controlled environments (venvs, CI, etc.)
#          where you want explicit control over what gets installed.
#
AUTO_INSTALL_DEPS = True
