"""Directories of the tables extracted from Brother's driver packages.

`scripts/extract_blobs.sh <model>` fills them, one subdirectory per
printer model; they are not part of the repository. Installed, they sit
inside the package as well.
"""

from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent

# BRCD dither tables (0600-*/capt-* `_<cache>.bin`), in lut/<model>/.
LUT_ROOT = _PACKAGE_DIR / "lut"
# Colour grids, interpolation tables and the precomputed inverse LUTs, in color_data/<model>/.
COLOR_DATA_ROOT = _PACKAGE_DIR / "color_data"
