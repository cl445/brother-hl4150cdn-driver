"""Brother HL-4150CDN: paper geometry, media codes and tray values.

What brhl4150cdnfilter writes for this model, taken from its paperinfij2
and ImagingArea files and verified against its output. The protocol
itself is in `brother_xl2hb.protocol`.
"""

import logging

from brother_xl2hb.settings import DuplexMode, InputSlot

logger = logging.getLogger(__name__)

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


def image_height(page_size: str) -> int:
    """Rows of the page image, as the original computes its band height.

    The printable height in points, capped at the paper height in whole
    points, converted to 600 dpi: `min(ury - lly, paper_h * 72 // 600) *
    600 // 72`. Verified against brhl4150cdnfilter for every paper size
    (A4 6812 -> 6808, Letter 6400 -> 6400, EnvDL 4996 -> 4991).

    Returns:
        Source height in rows.
    """
    _, ph = PAPER_SIZES[page_size]
    points = min(IMAGING_HEIGHT_PT[page_size], ph * 72 // 600)
    return points * 600 // 72


def get_image_dimensions(page_size: str) -> tuple[int, int]:
    """Return (source_width, source_height) for BeginImage.

    Width = paper_width rounded up to 32-pixel boundary; height see
    `image_height`.
    """
    pw, _ = PAPER_SIZES[page_size]
    return (pw + 31) & ~31, image_height(page_size)


def get_image_dimensions_fine(page_size: str) -> tuple[int, int]:
    """Return (source_width, source_height) for Fine mode BeginImage.

    Fine mode rounds the paper width up to 8 pixels instead of 32
    (raster_encode.c:352): A4 4760 stays 4760 (Normal 4768), Letter 4900
    becomes 4904. Height is the same as Normal. Verified against
    brhl4150cdnfilter for all 21 paper sizes.
    """
    pw, _ = PAPER_SIZES[page_size]
    return (pw + 7) & ~7, image_height(page_size)


def media_size_code(page_size: str) -> int | bytes:
    """BeginPage MediaSize for `page_size`: enum value or name string.

    Returns:
        The `MEDIA_SIZE` entry; A4 (2) with a warning for unknown sizes.
    """
    size = MEDIA_SIZE.get(page_size)
    if size is None:
        logger.warning("Unknown media size %r, defaulting to A4", page_size)
        return 2
    return size


def media_type_code(media_type: str) -> bytes:
    """BeginPage MediaType string for `media_type`.

    Returns:
        The `MEDIA_TYPE_STRINGS` entry; dRegular with a warning for unknown types.
    """
    type_str = MEDIA_TYPE_STRINGS.get(media_type)
    if type_str is None:
        logger.warning("Unknown media type %r, defaulting to Regular", media_type)
        return b"dRegular"
    return type_str
