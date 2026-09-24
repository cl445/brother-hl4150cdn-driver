"""
Full pipeline integration tests (PPM → XL2HB).

Tests end-to-end output against original driver captures.
"""

import io
import struct

import numpy as np
import pytest

from brfilter import (
    ColorMatching,
    DuplexMode,
    InputSlot,
    MediaType,
    MonoColor,
    PageSize,
    PrintSettings,
    Resolution,
    filter_page,
    filter_pages,
    read_ppm,
)
from fixture_utils import assert_matches_fixture, read_fixture
from xl2hb import PAPER_SIZES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ppm_bytes(width: int, height: int, r: int, g: int, b: int) -> bytes:
    """Create a uniform-color PPM in memory."""
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    pixel = bytes([r, g, b])
    return header + pixel * (width * height)


def _make_ppm_band(width: int, height: int, y_start: int, y_end: int, r: int, g: int, b: int) -> bytes:
    """Create an A4 PPM with a colored band at y_start..y_end, white elsewhere."""
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    white = bytes([255, 255, 255])
    color = bytes([r, g, b])
    white_row = white * width
    color_row = color * width
    rows = [color_row if y_start <= y < y_end else white_row for y in range(height)]
    return header + b"".join(rows)


def _run_pipeline(width: int, height: int, pixel_data: bytes, settings=None) -> bytes:
    """Run the filter pipeline and return the XL2HB output."""
    if settings is None:
        settings = PrintSettings()
    out = io.BytesIO()
    filter_page(width, height, pixel_data, settings, out)
    return out.getvalue()


def _read_ppm_strict(ppm_data: bytes) -> tuple[int, int, int, bytes]:
    """Parse a PPM that we know is valid (asserts non-None for type narrowing)."""
    result = read_ppm(io.BytesIO(ppm_data))
    assert result is not None, "test PPM is malformed"
    return result


def _pipeline_from_ppm(ppm_data: bytes, settings=None) -> bytes:
    """Parse PPM then run pipeline."""
    w, h, _, pixels = _read_ppm_strict(ppm_data)
    return _run_pipeline(w, h, pixels, settings)


# ---------------------------------------------------------------------------
# Structural validation (should all pass)
# ---------------------------------------------------------------------------


class TestOutputStructure:
    """Verify the pipeline produces structurally valid XL2HB output."""

    def test_starts_with_uel(self):
        ppm = _make_ppm_bytes(100, 100, 255, 255, 255)
        out = _pipeline_from_ppm(ppm)
        assert out.startswith(b"\x1b%-12345X")

    def test_ends_with_double_uel(self):
        ppm = _make_ppm_bytes(100, 100, 255, 255, 255)
        out = _pipeline_from_ppm(ppm)
        uel = b"\x1b%-12345X"
        assert out.endswith(uel + uel)

    def test_contains_xl2hb_marker(self):
        ppm = _make_ppm_bytes(100, 100, 255, 255, 255)
        out = _pipeline_from_ppm(ppm)
        assert b") BROTHER XL2HB" in out

    def test_contains_begin_session(self):
        ppm = _make_ppm_bytes(100, 100, 255, 255, 255)
        out = _pipeline_from_ppm(ppm)
        assert bytes([0x41]) in out  # BeginSession opcode

    def test_contains_end_session(self):
        ppm = _make_ppm_bytes(100, 100, 255, 255, 255)
        out = _pipeline_from_ppm(ppm)
        assert bytes([0x42]) in out  # EndSession opcode

    def test_white_page_has_no_read_image(self):
        """All-white input should produce no ReadImage opcodes."""
        ppm = _make_ppm_bytes(4760, 100, 255, 255, 255)
        out = _pipeline_from_ppm(ppm)
        # ReadImage opcode = 0xB1
        # Check it's not in the binary payload (between header markers)
        marker = b") BROTHER XL2HB"
        payload_start = out.index(marker)
        uel = b"\x1b%-12345X"
        # Find the footer
        footer_pos = out.rfind(uel + uel)
        payload = out[payload_start:footer_pos]
        assert 0xB1 not in payload

    def test_black_page_has_read_image(self):
        """All-black input should produce at least one ReadImage."""
        ppm = _make_ppm_bytes(4760, 100, 0, 0, 0)
        out = _pipeline_from_ppm(ppm)
        assert bytes([0xB1]) in out


# ---------------------------------------------------------------------------
# White page verification (exact match — no dithering needed)
# ---------------------------------------------------------------------------


class TestWhitePagePipeline:
    def test_white_page_matches_capture(self):
        """All-white PPM → byte-for-byte match with a4_white.xl2hb."""
        w, h = 4760, 6812
        pixels = bytes([255]) * (w * h * 3)
        assert_matches_fixture("a4_white", _run_pipeline(w, h, pixels))


# ---------------------------------------------------------------------------
# Black page verification (exact match — K-only, no dithering)
# ---------------------------------------------------------------------------


class TestBlackPagePipeline:
    def test_black_page_matches_capture(self):
        """All-black PPM → byte-for-byte match with a4_black.xl2hb.

        This requires the compression to match exactly.
        """
        w, h = 4760, 6812
        pixels = bytes(w * h * 3)  # all (0,0,0) = black
        assert_matches_fixture("a4_black", _run_pipeline(w, h, pixels))


# ---------------------------------------------------------------------------
# Capture-paired PPM tests (require matching PPM files)
# ---------------------------------------------------------------------------

_PPM_CAPTURE_PAIRS = [
    ("test_fullwidth_k", "K-only fullwidth"),
    ("test_halfpage_k", "Half-page K"),
    ("test_narrow_k", "Narrow K strip"),
    ("test_1pt_black", "1-point black"),
    ("test_fullwidth_c", "Full-width color (C/M/Y/K)"),
]


class TestPPMCapturePairs:
    """Tests that use the actual PPM files that generated each capture."""

    @pytest.mark.parametrize(("name", "desc"), _PPM_CAPTURE_PAIRS, ids=[n for n, _ in _PPM_CAPTURE_PAIRS])
    def test_ppm_to_xl2hb_matches_capture(self, name, desc):
        """The PPM behind each capture reproduces it byte for byte."""
        w, h, _, pixels = _read_ppm_strict(read_fixture(f"{name}.ppm"))
        assert_matches_fixture(name, _run_pipeline(w, h, pixels))


# ---------------------------------------------------------------------------
# Color page tests (generated PPM band → byte-exact match with captures)
# ---------------------------------------------------------------------------

_A4_W, _A4_H = 4760, 6812

_COLOR_PAIRS = [
    ("cyan_100", 3000, 3100, 0, 255, 255),
    ("gray50_1000", 1000, 2000, 128, 128, 128),
    ("red_100", 3000, 3100, 255, 0, 0),
    ("gray75_1000", 1000, 2000, 64, 64, 64),
]


def _run_color_band_test(name, y0, y1, r, g, b):
    """Run a color-band pipeline test against capture."""
    ppm = _make_ppm_band(_A4_W, _A4_H, y0, y1, r, g, b)
    assert_matches_fixture(name, _pipeline_from_ppm(ppm))


class TestColorPages:
    @pytest.mark.parametrize(
        ("name", "y0", "y1", "r", "g", "b"),
        _COLOR_PAIRS,
        ids=[p[0] for p in _COLOR_PAIRS],
    )
    def test_color_band_matches_capture(self, name, y0, y1, r, g, b):
        """Generated color-band PPM → byte-for-byte match with capture."""
        _run_color_band_test(name, y0, y1, r, g, b)


# ---------------------------------------------------------------------------
# Setting variants on cyan_100 PPM (cyan band y=3000..3100 on white A4)
#
# Captures from akator-ws02 with brhl4150cdnfilter, varying single RC settings.
# Brightness/contrast/RGB-keys and saturation are per-pixel adjustments before
# the 3D LUT; Vivid selects the sRGB LUT instead of the default one.
# ---------------------------------------------------------------------------


_SETTING_VARIANTS = [
    ("baseline", PrintSettings()),
    ("saturation_p20", PrintSettings(saturation=20)),  # no-op on saturated cyan
    ("green_p20", PrintSettings(green=20)),  # no-op on cyan (G already 255)
    ("blue_p20", PrintSettings(blue=20)),  # no-op on cyan (B already 255)
    ("cm_none", PrintSettings(color_matching=ColorMatching.NONE)),
    ("brightness_p20", PrintSettings(brightness=20)),
    ("brightness_n20", PrintSettings(brightness=-20)),
    ("contrast_p20", PrintSettings(contrast=20)),
    ("red_p20", PrintSettings(red=20)),
    ("combined", PrintSettings(brightness=10, contrast=10, saturation=10)),
    ("toner_save", PrintSettings(toner_save=True)),
    ("contrast_n20", PrintSettings(contrast=-20)),
    ("saturation_n20", PrintSettings(saturation=-20)),
    ("vivid", PrintSettings(color_matching=ColorMatching.VIVID)),
]


def _run_settings_variant(name, settings):
    ppm = _make_ppm_band(_A4_W, _A4_H, 3000, 3100, 0, 255, 255)
    assert_matches_fixture(f"cyan_100_{name}", _pipeline_from_ppm(ppm, settings))


class TestSettingVariants:
    """Per-RC-setting byte-exact tests against original-driver captures."""

    @pytest.mark.parametrize(
        ("name", "settings"),
        _SETTING_VARIANTS,
        ids=[v[0] for v in _SETTING_VARIANTS],
    )
    def test_setting_variant_matches(self, name, settings):
        _run_settings_variant(name, settings)


# ---------------------------------------------------------------------------
# Paper sizes other than A4 (captures from brhl4150cdnfilter under i386 emulation)
# ---------------------------------------------------------------------------


def _make_ppm_edges(page_size: str) -> bytes:
    """Full-size page with hash noise at the top, a ramp at the bottom and a bar on the right edge."""
    w, h = PAPER_SIZES[page_size]
    page = np.full((h, w, 3), 255, np.uint8)
    y, x = np.mgrid[0:60, 0:w].astype(np.uint32)
    for ch in range(3):
        page[20:80, :, ch] = ((x * 2654435761 + y * 40503 + ch * 97 + 1) >> 13) & 255
    page[h - 60 :] = ((np.arange(w) * 7 + 1) & 255).astype(np.uint8)[None, :, None]
    page[:, w - 30 :] = (0, 90, 200)
    return f"P6\n{w} {h}\n255\n".encode("ascii") + page.tobytes()


_PAPER_SIZE_CAPTURES = ["Letter", "A5", "EnvDL", "Br3x5", "EnvYou4"]


class TestPaperSizes:
    @pytest.mark.parametrize("name", _PAPER_SIZE_CAPTURES)
    def test_matches_capture(self, name):
        out = _pipeline_from_ppm(_make_ppm_edges(name), PrintSettings(page_size=PageSize(name)))
        assert_matches_fixture(f"size_{name}", out)


# ---------------------------------------------------------------------------
# Grayscale mode: BRMonoColor=Mono
#
# Captures from brhl4150cdnfilter (run under i386 emulation, verified to
# reproduce the akator-ws02 captures byte for byte) with BRMonoColor=Mono.
# ---------------------------------------------------------------------------


def _make_ppm_ramps() -> bytes:
    """A4 PPM: gray ramp rows 500..1500, colour ramp rows 2000..2600, white elsewhere."""
    header = f"P6\n{_A4_W} {_A4_H}\n255\n".encode("ascii")
    x = np.arange(_A4_W) * 256 // _A4_W
    gray = np.repeat(x, 3).astype(np.uint8).tobytes()
    colour = np.stack([x, 255 - x, (np.arange(_A4_W) * 7) & 255], axis=1).astype(np.uint8).tobytes()
    white = b"\xff" * (_A4_W * 3)
    rows = [gray if 500 <= y < 1500 else colour if 2000 <= y < 2600 else white for y in range(_A4_H)]
    return header + b"".join(rows)


_MONO_BANDS = [
    ("gray50", 1000, 2000, 128, 128, 128),
    ("red", 3000, 3100, 255, 0, 0),
    ("mixed", 2000, 2100, 30, 200, 90),
]

_MONO_RAMP_VARIANTS = [
    ("ramp", {}),
    ("ramp_bright", {"brightness": 20}),
    ("ramp_contrast", {"contrast": -20}),
    ("ramp_ts", {"toner_save": True}),
]


# Colour pixels on which the binary's 80-bit trunc((new_range / old_range) * (mid - min))
# in compress_adjust_saturation differs from exact floor division, at saturation +20 or +7.
_X87_SATURATION_PIXELS = [
    (59, 30, 1), (4, 164, 124), (181, 6, 216), (231, 9, 83), (90, 156, 13), (16, 107, 120), (228, 193, 18),
    (22, 250, 136), (118, 28, 208), (237, 37, 187), (95, 238, 51), (73, 127, 241), (242, 110, 99), (135, 245, 223),
    (30, 1, 59), (204, 6, 171), (127, 242, 12), (19, 65, 157), (209, 117, 25), (32, 170, 124), (132, 40, 224),
    (249, 49, 149), (151, 197, 59), (73, 96, 165), (192, 139, 86), (100, 208, 154), (140, 117, 209), (245, 139, 192),
]  # fmt: skip


def _make_ppm_mixed() -> bytes:
    """Deterministic A4 page: hash noise, sparse two-tone pixels, per-row colours, ramps, patches, black.

    Busy content that exercises literal/run/context-skip transitions in all
    four plane encoders and the per-pixel colour adjustments.
    """
    page = np.full((_A4_H, _A4_W, 3), 255, np.uint8)
    y, x = np.mgrid[0:64, 0:_A4_W].astype(np.uint32)
    for ch in range(3):
        page[100:164, :, ch] = ((x * 2654435761 + y * 40503 + ch * 97) >> 13) & 255
    page[200:264] = np.where((((x * 7919 + y * 104729) >> 5) & 1)[..., None] == 1, 230, 30).astype(np.uint8)
    rows = np.arange(64, dtype=np.uint32)
    row_colours = np.stack([(rows * 37) & 255, (rows * 91 + 50) & 255, (rows * 53 + 120) & 255], axis=1)
    page[300:364] = row_colours[:, None, :].astype(np.uint8)
    ramp = (np.arange(_A4_W) * 256 // _A4_W).astype(np.uint8)
    page[400:464, :, 0] = ramp
    page[400:464, :, 1] = 255 - ramp
    page[400:464, :, 2] = 128
    for j, x0 in enumerate(range(0, _A4_W, 170)):
        page[500:564, x0 : x0 + 85] = ((j * 67) & 255, (j * 151 + 30) & 255, (j * 23 + 200) & 255)
        page[600:664, x0 : x0 + 170] = _X87_SATURATION_PIXELS[j]
    page[700:764, : _A4_W // 2] = 0  # pure black: K only, or rich black with BREnhanceBlkPrt
    page[700:764, _A4_W // 2 :] = (0, 0, 1)
    return f"P6\n{_A4_W} {_A4_H}\n255\n".encode("ascii") + page.tobytes()


_MIXED_VARIANTS = [
    ("baseline", PrintSettings()),
    ("saturation_p20", PrintSettings(saturation=20)),
    ("saturation_p7", PrintSettings(saturation=7)),
    ("saturation_n20", PrintSettings(saturation=-20)),
    ("contrast_n20", PrintSettings(contrast=-20)),
    ("brightness_p15", PrintSettings(brightness=15)),
    ("vivid", PrintSettings(color_matching=ColorMatching.VIVID)),
    ("toner_save", PrintSettings(toner_save=True)),
    ("cm_none", PrintSettings(color_matching=ColorMatching.NONE)),
    ("cm_none_adjusted", PrintSettings(color_matching=ColorMatching.NONE, brightness=15, saturation=-20, red=10)),
    ("improve_gray", PrintSettings(improve_gray=True)),
    ("enhance_black", PrintSettings(enhance_black=True)),
    ("glossy", PrintSettings(media_type=MediaType.GLOSSY)),
    ("vivid_toner_save_gray", PrintSettings(color_matching=ColorMatching.VIVID, toner_save=True, improve_gray=True)),
]


class TestMixedPage:
    """Busy page against brhl4150cdnfilter captures (run under i386 emulation)."""

    @pytest.mark.parametrize(("name", "settings"), _MIXED_VARIANTS, ids=[v[0] for v in _MIXED_VARIANTS])
    def test_matches_capture(self, name, settings):
        assert_matches_fixture(f"mixed_{name}", _pipeline_from_ppm(_make_ppm_mixed(), settings))


class TestMonoMode:
    @pytest.mark.parametrize(("name", "y0", "y1", "r", "g", "b"), _MONO_BANDS, ids=[m[0] for m in _MONO_BANDS])
    def test_band_matches_capture(self, name, y0, y1, r, g, b):
        ppm = _make_ppm_band(_A4_W, _A4_H, y0, y1, r, g, b)
        out = _pipeline_from_ppm(ppm, PrintSettings(mono_color=MonoColor.MONO))
        assert_matches_fixture(f"mono_{name}", out)

    @pytest.mark.parametrize(("name", "kwargs"), _MONO_RAMP_VARIANTS, ids=[v[0] for v in _MONO_RAMP_VARIANTS])
    def test_ramp_matches_capture(self, name, kwargs):
        out = _pipeline_from_ppm(_make_ppm_ramps(), PrintSettings(mono_color=MonoColor.MONO, **kwargs))
        assert_matches_fixture(f"mono_{name}", out)

    @pytest.mark.parametrize(
        "kwargs",
        [{"saturation": 20}, {"color_matching": ColorMatching.VIVID}, {"improve_gray": True}],
        ids=["saturation", "vivid", "improve_gray"],
    )
    def test_colour_only_settings_are_ignored(self, kwargs):
        """The original driver's output for these equals plain mono."""
        out = _pipeline_from_ppm(_make_ppm_ramps(), PrintSettings(mono_color=MonoColor.MONO, **kwargs))
        assert_matches_fixture("mono_ramp", out)

    def test_only_k_plane(self):
        ppm = _make_ppm_bytes(4760, 10, 128, 0, 0)
        out = _pipeline_from_ppm(ppm, PrintSettings(mono_color=MonoColor.MONO))
        assert b"RENDERMODE=GRAYSCALE" in out
        # ReadImage colour-treatment attribute (plane id) for C/M/Y never appears.
        for plane_id in (1, 2, 3):
            assert bytes([0xC1, plane_id, 0x00, 0xF8, 0x81]) not in out
        assert bytes([0xC1, 0x00, 0x00, 0xF8, 0x81]) in out


class TestPipelineSettings:
    @pytest.mark.parametrize(
        ("slot", "attr"),
        [
            (InputSlot.AUTO, b"\xc0\x01\xf8\x26"),
            (InputSlot.TRAY1, b"\xc1\xe9\x03\xf8\x26"),
            (InputSlot.TRAY2, b"\xc1\xed\x03\xf8\x26"),
            (InputSlot.MP_TRAY, b"\xc1\xec\x03\xf8\x26"),
            (InputSlot.MANUAL, b"\xc0\x02\xf8\x26"),
        ],
    )
    def test_input_slot_goes_into_media_source(self, slot, attr):
        """BeginPage MediaSource as brhl4150cdnfilter writes it; no PJL SOURCETRAY."""
        out = _pipeline_from_ppm(_make_ppm_bytes(4760, 10, 0, 0, 0), PrintSettings(input_slot=slot))
        assert attr in out
        assert b"SOURCETRAY" not in out

    @pytest.mark.parametrize(
        ("mono_color", "line"),
        [
            (MonoColor.AUTO, b"COLORADAPT=ON"),
            (MonoColor.FULL_COLOR, b"COLORADAPT=OFF"),
            (MonoColor.MONO, b"COLORADAPT=OFF"),
        ],
    )
    def test_color_adapt_only_for_auto(self, mono_color, line):
        out = _pipeline_from_ppm(_make_ppm_bytes(4760, 10, 0, 0, 0), PrintSettings(mono_color=mono_color))
        assert line in out

    def test_toner_save_keeps_economode_off(self):
        """Toner save uses the -TS dither tables; ECONOMODE stays OFF (matches original driver, pjl.c:58)."""
        ppm = _make_ppm_bytes(4760, 10, 0, 0, 0)
        settings = PrintSettings(toner_save=True)
        out = _pipeline_from_ppm(ppm, settings)
        assert b"ECONOMODE=OFF" in out
        assert b"ECONOMODE=ON" not in out


# ---------------------------------------------------------------------------
# Fine mode framing
# ---------------------------------------------------------------------------


class TestFineResolution:
    """Fine mode framing: PJL, session, image dimensions and depth.

    Verified against original Brother driver captures: Fine mode uses
    RESOLUTION=600 in PJL, a 600x600 session, the Normal source height
    but the unrounded width (A4: 4760x6808), COLOR_DEPTH=1 and APTMODE=ON4.
    """

    def _find_ubyte_attr(self, data: bytes, attr_id: int) -> int:
        """Find a ubyte attribute value by attribute ID."""
        marker = bytes([0xF8, attr_id])
        idx = data.index(marker)
        # ubyte format: TAG_UBYTE (0xC0) value TAG_ATTR attr_id
        return data[idx - 1]

    def _find_uint16_attr(self, data: bytes, attr_id: int) -> int:
        """Find a uint16 attribute value by attribute ID."""
        marker = bytes([0xF8, attr_id])
        idx = data.index(marker)
        lo, hi = data[idx - 2], data[idx - 1]
        return struct.unpack("<H", bytes([lo, hi]))[0]

    def _find_uint16_xy_attr(self, data: bytes, attr_id: int) -> tuple[int, int]:
        """Find a uint16_xy attribute (x, y) by attribute ID."""
        marker = bytes([0xF8, attr_id])
        idx = data.index(marker)
        x = struct.unpack("<H", data[idx - 4 : idx - 2])[0]
        y = struct.unpack("<H", data[idx - 2 : idx])[0]
        return x, y

    def test_fine_mode_source_height(self):
        """Fine mode A4 source_height = 6808 (same as Normal, verified from captures)."""
        fine = _run_pipeline(1, 1, b"\xff\xff\xff", PrintSettings(resolution=Resolution.FINE))
        fine_h = self._find_uint16_attr(fine, 0x6B)  # ATTR_SOURCE_HEIGHT
        assert fine_h == 6808

    def test_fine_mode_source_width(self):
        """Fine mode A4 source_width = 4760 (no 32-pixel rounding, verified from captures)."""
        fine = _run_pipeline(1, 1, b"\xff\xff\xff", PrintSettings(resolution=Resolution.FINE))
        fine_w = self._find_uint16_attr(fine, 0x6C)  # ATTR_SOURCE_WIDTH
        assert fine_w == 4760

    def test_fine_mode_color_depth_1(self):
        """Fine mode must use COLOR_DEPTH=1 (verified from captures)."""
        fine = _run_pipeline(1, 1, b"\xff\xff\xff", PrintSettings(resolution=Resolution.FINE))
        fine_cd = self._find_ubyte_attr(fine, 0x62)  # ATTR_COLOR_DEPTH
        assert fine_cd == 1

    def test_normal_mode_color_depth_0(self):
        """Normal mode must use COLOR_DEPTH=0."""
        normal = _run_pipeline(1, 1, b"\xff\xff\xff", PrintSettings(resolution=Resolution.NORMAL))
        normal_cd = self._find_ubyte_attr(normal, 0x62)  # ATTR_COLOR_DEPTH
        assert normal_cd == 0

    def test_fine_mode_same_session_units(self):
        """Fine mode must keep session UnitsPerMeasure at 600x600."""
        normal = _run_pipeline(1, 1, b"\xff\xff\xff", PrintSettings(resolution=Resolution.NORMAL))
        fine = _run_pipeline(1, 1, b"\xff\xff\xff", PrintSettings(resolution=Resolution.FINE))

        nx, ny = self._find_uint16_xy_attr(normal, 0x89)  # ATTR_UNITS_PER_MEASURE
        fx, fy = self._find_uint16_xy_attr(fine, 0x89)
        assert (nx, ny) == (600, 600)
        assert (fx, fy) == (600, 600), (
            f"Fine session must be 600x600 (got {fx}x{fy}). Original driver keeps session resolution unchanged."
        )

    def test_fine_mode_pjl_resolution_600(self):
        """Fine mode must set PJL RESOLUTION=600 (same as Normal)."""
        fine = _run_pipeline(1, 1, b"\xff\xff\xff", PrintSettings(resolution=Resolution.FINE))
        assert b"RESOLUTION=600" in fine
        assert b"RESOLUTION=1200" not in fine

    def test_fine_mode_aptmode_on4(self):
        """Fine mode must set APTMODE=ON4 in PJL."""
        fine = _run_pipeline(1, 1, b"\xff\xff\xff", PrintSettings(resolution=Resolution.FINE))
        assert b"APTMODE=ON4" in fine

    def test_normal_mode_aptmode_off(self):
        """Normal mode must set APTMODE=OFF in PJL."""
        normal = _run_pipeline(1, 1, b"\xff\xff\xff", PrintSettings(resolution=Resolution.NORMAL))
        assert b"APTMODE=OFF" in normal


# ---------------------------------------------------------------------------
# Fine mode E2E: byte-for-byte match against original driver captures
# ---------------------------------------------------------------------------


class TestFineWhitePagePipeline:
    def test_fine_white_matches_capture(self):
        """Fine all-white PPM -> byte-for-byte match with fine_white.xl2hb."""
        w, h = 4760, 6812
        pixels = bytes([255]) * (w * h * 3)
        settings = PrintSettings(resolution=Resolution.FINE)
        assert_matches_fixture("fine_white", _run_pipeline(w, h, pixels, settings))


class TestFineBlackPagePipeline:
    def test_fine_black_matches_capture(self):
        """Fine all-black PPM -> byte-for-byte match with fine_black.xl2hb.

        Verifies the full Fine pipeline: 4bpp dithering, two-stage compression,
        PlaneBuffer headers (bit_depth=12, quant_type=0, comp_size=0),
        COLOR_DEPTH=1, band config, and APTMODE=ON4.
        """
        w, h = 4760, 6812
        pixels = bytes(w * h * 3)  # all (0,0,0) = black
        settings = PrintSettings(resolution=Resolution.FINE)
        assert_matches_fixture("fine_black", _run_pipeline(w, h, pixels, settings))


# ---------------------------------------------------------------------------
# Skip blank pages
# ---------------------------------------------------------------------------


class TestSkipBlank:
    def test_skip_blank_empty_page(self):
        """With skip_blank=True, an all-white page should produce no output."""
        settings = PrintSettings(skip_blank=True)
        out = _run_pipeline(1, 1, b"\xff\xff\xff", settings)
        assert out == b""

    def test_skip_blank_off_produces_output(self):
        """With skip_blank=False (default), a white page still produces output."""
        settings = PrintSettings(skip_blank=False)
        out = _run_pipeline(1, 1, b"\xff\xff\xff", settings)
        assert len(out) > 0


def test_filter_page_accepts_ppm_wider_than_printable_area():
    """cli.py feeds the uncropped GS render (4958 px) into an A4 page (4760 px)."""
    width, height = 4958, 16
    pixel_data = bytes(range(256)) * (width * height * 3 // 256) + bytes(width * height * 3 % 256)
    out = io.BytesIO()
    filter_page(width, height, pixel_data, PrintSettings(), out)
    assert out.getvalue()


# ---------------------------------------------------------------------------
# Duplex: byte-exact against brhl4150cdnfilter (4 asymmetric pages, one session)
# ---------------------------------------------------------------------------


def _duplex_test_pages() -> list[tuple[int, int, bytes]]:
    """Four A4 pages whose content is asymmetric in x and y, so flips show up."""
    colours = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 0, 0)]
    pages = []
    for i in range(4):
        page = np.full((_A4_H, _A4_W, 3), 255, np.uint8)
        page[300 + i * 100 : 420 + i * 100, 200:1400] = (0, 0, 0)
        page[4000:4150, 2600 + i * 200 : 4500] = colours[i]
        pages.append((_A4_W, _A4_H, page.tobytes()))
    return pages


@pytest.mark.parametrize(
    ("fixture", "duplex"),
    [
        ("duplex4_none", "None"),
        ("duplex4_long_edge", "DuplexNoTumble"),
        ("duplex4_short_edge", "DuplexTumble"),
    ],
)
def test_duplex_job_matches_brother_capture(fixture, duplex):
    """Whole 4-page job in one session, incl. the flipped long-edge back pages."""
    out = io.BytesIO()
    filter_pages(_duplex_test_pages(), PrintSettings(duplex=DuplexMode(duplex)), out)
    assert_matches_fixture(fixture, out.getvalue())


@pytest.mark.parametrize("duplex", ["None", "DuplexNoTumble", "DuplexTumble"])
@pytest.mark.parametrize("n_pages", [3, 4])
def test_reverse_matches_reversed_input(duplex, n_pages):
    """Reverse order renders forward and spools, but must equal rendering the reversed pages."""
    pages = _duplex_test_pages()[:n_pages]
    expected = io.BytesIO()
    filter_pages(pages[::-1], PrintSettings(duplex=DuplexMode(duplex)), expected)

    out = io.BytesIO()
    settings = PrintSettings(duplex=DuplexMode(duplex), reverse=True)
    filter_pages(iter(pages), settings, out, page_count=n_pages)
    assert out.getvalue() == expected.getvalue()


def test_reverse_long_edge_requires_page_count():
    settings = PrintSettings(duplex=DuplexMode.NO_TUMBLE, reverse=True)
    with pytest.raises(ValueError, match="page_count"):
        filter_pages(iter(_duplex_test_pages()), settings, io.BytesIO())


def _as_row_blocks(pixel_data: bytes, width: int, height: int, block_rows: int):
    rows = np.frombuffer(pixel_data, dtype=np.uint8).reshape(height, width * 3)
    return (rows[i : i + block_rows] for i in range(0, height, block_rows))


@pytest.mark.parametrize(
    ("duplex", "n_pages"),
    [
        ("None", 2),
        ("DuplexNoTumble", 2),  # page 2 is a mirrored back side
    ],
)
def test_streamed_row_blocks_match_whole_pages(duplex, n_pages):
    """Pages given as row blocks (as page_stream delivers them) render identically."""
    settings = PrintSettings(duplex=DuplexMode(duplex))
    pages = _duplex_test_pages()[:n_pages]
    expected = io.BytesIO()
    filter_pages(pages, settings, expected)

    streamed = [(w, h, _as_row_blocks(data, w, h, 333)) for w, h, data in pages]
    out = io.BytesIO()
    filter_pages(streamed, settings, out)
    assert out.getvalue() == expected.getvalue()
