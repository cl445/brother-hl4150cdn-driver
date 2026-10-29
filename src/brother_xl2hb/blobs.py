"""Directories of the tables extracted from Brother's driver package.

`scripts/extract_blobs.sh` fills both; they are not part of the
repository. Installed, they sit inside the package as well.
"""

from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent

# BRCD dither tables (0600-*/capt-* `_cache09.bin`).
LUT_DIR = _PACKAGE_DIR / "lut"
# Colour grids, interpolation tables and the precomputed inverse LUTs.
COLOR_DATA_DIR = _PACKAGE_DIR / "color_data"
