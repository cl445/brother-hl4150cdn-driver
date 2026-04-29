#!/usr/bin/env python3
"""brfilter.py - PPM → PJL + XL2HB filter for the Brother HL-4150CDN.

Thin facade re-exporting the public surface of the split modules so
existing callers (`from brfilter import …`) keep working. Run as a
script to invoke the CLI.
"""

from cli import main
from pipeline import filter_duplex_pages, filter_page
from ppm import read_ppm
from settings import (
    DUPLEX_MAP,
    ColorMatching,
    DuplexMode,
    ImproveOutput,
    InputSlot,
    MediaType,
    MonoColor,
    PageSize,
    PrintSettings,
    Resolution,
    input_slot_to_tray,
)
from transforms import (
    apply_input_remap_rgb,
    apply_vivid,
    build_input_remap_lut,
    rgb_line_to_cmyk_intensities,
)

__all__ = [
    "DUPLEX_MAP",
    "ColorMatching",
    "DuplexMode",
    "ImproveOutput",
    "InputSlot",
    "MediaType",
    "MonoColor",
    "PageSize",
    "PrintSettings",
    "Resolution",
    "apply_input_remap_rgb",
    "apply_vivid",
    "build_input_remap_lut",
    "filter_duplex_pages",
    "filter_page",
    "input_slot_to_tray",
    "main",
    "read_ppm",
    "rgb_line_to_cmyk_intensities",
]


if __name__ == "__main__":
    main()
