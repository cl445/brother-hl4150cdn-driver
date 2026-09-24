#!/usr/bin/env python3
"""brfilter.py - PPM → PJL + XL2HB filter for the Brother HL-4150CDN.

Convenience facade over the pipeline, settings and PPM modules, used by
the tests. Run as a script to invoke the CLI (`python src/brfilter.py`).
"""

from cli import main
from color_lut import rgb_to_cmyk_lut
from pipeline import filter_page, filter_pages
from ppm import read_ppm
from settings import (
    ColorMatching,
    DuplexMode,
    ImproveOutput,
    InputSlot,
    MediaType,
    MonoColor,
    PageSize,
    PrintSettings,
    Resolution,
)

__all__ = [
    "ColorMatching",
    "DuplexMode",
    "ImproveOutput",
    "InputSlot",
    "MediaType",
    "MonoColor",
    "PageSize",
    "PrintSettings",
    "Resolution",
    "filter_page",
    "filter_pages",
    "main",
    "read_ppm",
    "rgb_to_cmyk_lut",
]


if __name__ == "__main__":
    main()
