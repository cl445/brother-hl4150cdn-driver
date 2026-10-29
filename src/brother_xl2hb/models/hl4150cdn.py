"""Brother HL-4150CDN: the `HL4150CDN` model and its tables.

What brhl4150cdnfilter writes for this model, taken from its paperinfij2
and ImagingArea files and verified against its output. The protocol
itself is in `brother_xl2hb.protocol`.
"""

from brother_xl2hb.color.transforms import color_table
from brother_xl2hb.models.base import PrinterModel
from brother_xl2hb.protocol.pjl import generate_pjl_footer, generate_pjl_header
from brother_xl2hb.settings import (
    DuplexMode,
    ImproveOutput,
    InputSlot,
    MonoColor,
    PrintSettings,
    Resolution,
)

# ---------------------------------------------------------------------------
# MediaSize enum
# ---------------------------------------------------------------------------
# BeginPage MediaSize as brhl4150cdnfilter writes it: a PCL XL media-size
# enum, or for sizes without one a name string.
MEDIA_SIZE: dict[str, int | bytes] = {
    "Letter": 0,
    "Legal": 1,
    "A4": 2,
    "Executive": 3,
    "Env10": 6,
    "EnvMonarch": 7,
    "EnvC5": 8,
    "EnvDL": 9,
    "JISB5": 11,
    "ISOB5": 12,
    "Postcard": 14,
    "A5": 16,
    "A6": 17,
    "PRA5Rotated": b"A5L",
    "ISOB6": b"B6",
    "JISB6": b"JISB6",
    "Br3x5": b"3x5",
    "FanFoldGermanLegal": b"Folio",
    "EnvPRC5Rotated": b"DL Long Edge",
    "EnvYou4": b"Envelope #4",
    "EnvChou3": b"Envelope MAX",
}

# ---------------------------------------------------------------------------
# Paper sizes: pixels at 600 DPI from paperinfij2, and the printable height
# in points (ury - lly) from Brother's ImagingArea file.
# ---------------------------------------------------------------------------
PAPER_SIZES = {
    "A4": (4760, 6812),
    "Letter": (4900, 6400),
    "Legal": (4900, 8200),
    "Executive": (4148, 6100),
    "A5": (3296, 4760),
    "PRA5Rotated": (4756, 3300),
    "A6": (2272, 3300),
    "ISOB5": (3956, 5708),
    "ISOB6": (2748, 3956),
    "JISB5": (4100, 5872),
    "JISB6": (2824, 4100),
    "EnvDL": (2400, 4996),
    "EnvC5": (3624, 5208),
    "Env10": (2272, 5500),
    "EnvMonarch": (2124, 4300),
    "Br3x5": (1600, 2800),
    "FanFoldGermanLegal": (4900, 7600),
    "EnvPRC5Rotated": (5000, 2400),
    "Postcard": (2164, 3288),
    "EnvYou4": (2280, 5348),
    "EnvChou3": (2632, 5348),
}

IMAGING_HEIGHT_PT = {
    "A4": 818,
    "Letter": 768,
    "Legal": 984,
    "Executive": 732,
    "A5": 571,
    "PRA5Rotated": 396,
    "A6": 396,
    "ISOB5": 685,
    "ISOB6": 475,
    "JISB5": 705,
    "JISB6": 492,
    "EnvDL": 600,
    "EnvC5": 625,
    "Env10": 660,
    "EnvMonarch": 516,
    "Br3x5": 336,
    "FanFoldGermanLegal": 912,
    "EnvPRC5Rotated": 288,
    "Postcard": 395,
    "EnvYou4": 642,
    "EnvChou3": 642,
}

# MediaType strings (each name is prefixed with 'd' on the wire), as
# brhl4150cdnfilter writes them for the rc values Brother's CUPS wrapper
# passes (MediaType=BOND, Env, ...). It sends Bond paper as Transparency.
MEDIA_TYPE_STRINGS = {
    "Plain": b"dRegular",
    "Thin": b"dThin",
    "Thick": b"dThick",
    "Thicker": b"dThick2",
    "Bond": b"dTransparency",
    "Envelope": b"dEnvelopes",
    "EnvThin": b"dEnvthin",
    "EnvThick": b"dEnvthick",
    "Recycled": b"dRecycled",
    "Postcard": b"dPostcard",
    "Label": b"dLabel",
    "Glossy": b"dGlossy",
}

# Fine mode uses the same paper sizes as Normal — only the dithering
# depth, band config, and PJL APTMODE differ.

# Bytes per line of an A4 Normal-mode plane (4768 pixels / 8): the
# reference width of the tests. The renderer passes the width of the
# actual paper size to PlaneBuffer.
BPL = 596

# XL2HB DuplexPageMode values as written by Brother's filter; None = simplex.
DUPLEX_MAP: dict[DuplexMode, int | None] = {
    DuplexMode.NONE: None,
    DuplexMode.NO_TUMBLE: 0x00,
    DuplexMode.TUMBLE: 0x81,
}

# XL2HB MediaSource values as brhl4150cdnfilter writes them. The tray goes
# only into BeginPage; the original never emits @PJL SET SOURCETRAY for
# this model.
MEDIA_SOURCE: dict[InputSlot, int] = {
    InputSlot.AUTO: 1,
    InputSlot.TRAY1: 1001,
    InputSlot.TRAY2: 1005,
    InputSlot.MP_TRAY: 1004,
    InputSlot.MANUAL: 2,
}


# The printable area starts 100 px (at 600 dpi) from the physical page edge;
# SetPageOrigin(100, 100).
PAGE_ORIGIN = (100, 100)

# Paper dimensions in PostScript points (the PPD's PaperDimension).
PAPER_POINTS = {
    "A4": (595, 842),
    "Letter": (612, 792),
    "Legal": (612, 1008),
    "Executive": (522, 756),
    "A5": (420, 595),
    "JISB5": (516, 729),
    "Postcard": (284, 419),
    "EnvDL": (312, 624),
    "EnvC5": (459, 649),
    "Env10": (297, 684),
    "EnvMonarch": (279, 540),
    "PRA5Rotated": (595, 420),
    "A6": (297, 420),
    "ISOB5": (499, 709),
    "ISOB6": (354, 499),
    "JISB6": (363, 516),
    "Br3x5": (216, 360),
    "FanFoldGermanLegal": (612, 936),
    "EnvPRC5Rotated": (624, 312),
    "EnvYou4": (298, 666),
    "EnvChou3": (340, 666),
}

# BRCD dither files: 0600-k_cache09.bin, capt-k-TS_cache09.bin, ...
DITHER_CACHE = "cache09"


def pjl_header(settings: PrintSettings) -> bytes:
    """The PJL header brhl4150cdnfilter writes for `settings`.

    Returns:
        Encoded PJL header bytes.
    """
    # PJL header always reports 600 dpi; Fine mode differs only in dithering.
    color = settings.mono_color != MonoColor.MONO
    return generate_pjl_header(
        resolution=600,
        color=color,
        # Only BRMonoColor=Auto sets the colour-adapt flag (printer_config_init).
        color_adapt=settings.mono_color == MonoColor.AUTO,
        # ECONOMODE is always OFF; toner-save is implemented through the
        # -TS_cache09.bin dither tables instead.
        economode=False,
        less_paper_curl=settings.improve_output == ImproveOutput.LESS_PAPER_CURL,
        fix_intensity=settings.improve_output == ImproveOutput.FIX_INTENSITY,
        apt_mode=(settings.resolution == Resolution.FINE),
        improve_gray=settings.improve_gray,
        ucrgcr=settings.enhance_black,
    )


def page_is_flipped(settings: PrintSettings, back_side: bool) -> bool:  # noqa: FBT001 - PrinterModel hook signature
    """Whether a page is sent bottom-up with the back-side marker.

    Long-edge back pages are sent that way (rows mirrored, columns not).
    BRReverse makes Brother's filter send every page that way, so on a
    long-edge back page the two cancel out (verified against captures).

    Returns:
        True if the page is mirrored top to bottom.
    """
    return (back_side and settings.duplex == DuplexMode.NO_TUMBLE) != settings.reverse


HL4150CDN = PrinterModel(
    name="HL-4150CDN",
    paper_sizes=PAPER_SIZES,
    imaging_height_pt=IMAGING_HEIGHT_PT,
    paper_points=PAPER_POINTS,
    page_origin=PAGE_ORIGIN,
    media_size=MEDIA_SIZE,
    media_types=MEDIA_TYPE_STRINGS,
    media_source=MEDIA_SOURCE,
    duplex_page_mode=DUPLEX_MAP,
    dither_cache=DITHER_CACHE,
    color_table=color_table,
    pjl_header=pjl_header,
    pjl_footer=generate_pjl_footer,
    page_is_flipped=page_is_flipped,
)
