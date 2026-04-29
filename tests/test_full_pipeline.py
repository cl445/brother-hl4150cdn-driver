"""
Full pipeline integration tests (PPM → XL2HB).

Tests end-to-end output against original driver captures.
"""

import io

import pytest

from brfilter import MonoColor, PageSize, PrintSettings, Resolution, filter_page, read_ppm
from fixture_utils import read_fixture

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
        assert 0xB1 not in payload or payload.count(bytes([0xB1])) == 0

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
        expected = read_fixture("a4_white.xl2hb")
        if expected is None:
            pytest.skip("a4_white.xl2hb not available")

        w, h = 4760, 6812
        pixels = bytes(w * h * 3) + bytes([255] * (w * h * 3))
        # Actually: white = (255,255,255) for all pixels
        pixels = bytes([255]) * (w * h * 3)
        out = _run_pipeline(w, h, pixels)
        assert out == expected, f"White page mismatch: got {len(out)}B, expected {len(expected)}B"


# ---------------------------------------------------------------------------
# Black page verification (exact match — K-only, no dithering)
# ---------------------------------------------------------------------------


class TestBlackPagePipeline:
    def test_black_page_matches_capture(self):
        """All-black PPM → byte-for-byte match with a4_black.xl2hb.

        This requires the compression to match exactly.
        """
        expected = read_fixture("a4_black.xl2hb")
        if expected is None:
            pytest.skip("a4_black.xl2hb not available")

        w, h = 4760, 6812
        pixels = bytes(w * h * 3)  # all (0,0,0) = black
        out = _run_pipeline(w, h, pixels)
        if out != expected:
            # Find first diff
            for i in range(min(len(out), len(expected))):
                if out[i] != expected[i]:
                    pytest.fail(
                        f"Black page mismatch at byte {i}: "
                        f"got 0x{out[i]:02x}, expected 0x{expected[i]:02x}. "
                        f"Output {len(out)}B vs expected {len(expected)}B"
                    )
            pytest.fail(f"Length mismatch: {len(out)}B vs {len(expected)}B")


# ---------------------------------------------------------------------------
# Capture-paired PPM tests (require matching PPM files)
# ---------------------------------------------------------------------------

_PPM_CAPTURE_PAIRS_PASS = [
    ("test_fullwidth_k", "K-only fullwidth"),
    ("test_halfpage_k", "Half-page K"),
    ("test_narrow_k", "Narrow K strip"),
    ("test_1pt_black", "1-point black"),
    ("test_fullwidth_c", "Full-width color (C/M/Y/K)"),
]

_PPM_CAPTURE_PAIRS_XFAIL = []


class TestPPMCapturePairs:
    """Tests that use the actual PPM files that generated each capture."""

    @pytest.mark.parametrize(("name", "desc"), _PPM_CAPTURE_PAIRS_PASS, ids=[n for n, _ in _PPM_CAPTURE_PAIRS_PASS])
    def test_ppm_to_xl2hb_matches_capture_k(self, name, desc):
        """K-only PPMs that already match the driver output."""
        ppm_data = read_fixture(f"{name}.ppm")
        expected = read_fixture(f"{name}.xl2hb")

        if ppm_data is None or expected is None:
            pytest.skip(f"Missing {name}.ppm or {name}.xl2hb")

        w, h, _, pixels = _read_ppm_strict(ppm_data)

        out = _run_pipeline(w, h, pixels)
        assert out == expected, f"{name}: output {len(out)}B vs expected {len(expected)}B"

    @pytest.mark.parametrize(("name", "desc"), _PPM_CAPTURE_PAIRS_XFAIL, ids=[n for n, _ in _PPM_CAPTURE_PAIRS_XFAIL])
    @pytest.mark.xfail(reason="C/M planes use JPEG-LS-like compression; our RLE encoder differs")
    def test_ppm_to_xl2hb_matches_capture_color(self, name, desc):
        """Run PPM through pipeline, compare with driver capture."""
        ppm_data = read_fixture(f"{name}.ppm")
        expected = read_fixture(f"{name}.xl2hb")

        if ppm_data is None or expected is None:
            pytest.skip(f"Missing {name}.ppm or {name}.xl2hb")

        w, h, _, pixels = _read_ppm_strict(ppm_data)

        out = _run_pipeline(w, h, pixels)
        assert out == expected, f"{name}: output {len(out)}B vs expected {len(expected)}B"


# ---------------------------------------------------------------------------
# Color page tests (generated PPM band → byte-exact match with captures)
# ---------------------------------------------------------------------------

_A4_W, _A4_H = 4760, 6812

_COLOR_PAIRS_PASS = [
    ("cyan_100", 3000, 3100, 0, 255, 255),
    ("gray50_1000", 1000, 2000, 128, 128, 128),
    ("red_100", 3000, 3100, 255, 0, 0),
    ("gray75_1000", 1000, 2000, 64, 64, 64),
]


def _run_color_band_test(name, y0, y1, r, g, b):
    """Run a color-band pipeline test against capture."""
    expected = read_fixture(f"{name}.xl2hb")
    if expected is None:
        pytest.skip(f"{name}.xl2hb not available")

    ppm = _make_ppm_band(_A4_W, _A4_H, y0, y1, r, g, b)
    out = _pipeline_from_ppm(ppm)
    if out != expected:
        for i in range(min(len(out), len(expected))):
            if out[i] != expected[i]:
                pytest.fail(
                    f"{name}: first diff at byte {i}: "
                    f"got 0x{out[i]:02x}, expected 0x{expected[i]:02x}. "
                    f"Output {len(out)}B vs expected {len(expected)}B"
                )
        pytest.fail(f"{name}: length mismatch: {len(out)}B vs {len(expected)}B")


class TestColorPages:
    @pytest.mark.parametrize(
        ("name", "y0", "y1", "r", "g", "b"),
        _COLOR_PAIRS_PASS,
        ids=[p[0] for p in _COLOR_PAIRS_PASS],
    )
    def test_color_band_matches_capture(self, name, y0, y1, r, g, b):
        """Generated color-band PPM → byte-for-byte match with capture."""
        _run_color_band_test(name, y0, y1, r, g, b)


# ---------------------------------------------------------------------------
# Setting variants on cyan_100 PPM (cyan band y=3000..3100 on white A4)
#
# Captures from akator-ws02 with brhl4150cdnfilter, varying single RC settings.
# These exercise the color-correction LUT bake (brightness/contrast/RGB-keys
# fold into the 3D RGB cube; saturation+vivid stay per-pixel in the C driver).
# ---------------------------------------------------------------------------


from brfilter import ColorMatching  # noqa: E402

_SETTING_VARIANTS_PASS = [
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
]

_SETTING_VARIANTS_XFAIL = [
    # 88 byte diffs in image payload — Python's color separation/dither for non-pure
    # input colors like (25,230,230) deviates from Brother. Independent of the input remap.
    ("contrast_n20", PrintSettings(contrast=-20)),
    # The major saturation off-by-one (R=51 vs Brother's 50) was fixed by switching
    # to truncate-toward-zero (matches the FPU chop mode the binary sets at
    # 0x0804f88b). Residual 21-byte RLE drift in 2 clusters — same color-pipeline
    # divergence class as contrast_n20.
    ("saturation_n20", PrintSettings(saturation=-20)),
    # Vivid takes a separate code path in compress_separate_dispatch
    # (Brother selects mono_cm vs cmyk_cm differently when ColorMatching=Vivid).
    ("vivid", PrintSettings(color_matching=ColorMatching.VIVID)),
]


def _run_settings_variant(name, settings):
    expected = read_fixture(f"cyan_100_{name}.xl2hb")
    if expected is None:
        pytest.skip(f"cyan_100_{name}.xl2hb not available")
    ppm = _make_ppm_band(_A4_W, _A4_H, 3000, 3100, 0, 255, 255)
    out = _pipeline_from_ppm(ppm, settings)
    if out != expected:
        for i in range(min(len(out), len(expected))):
            if out[i] != expected[i]:
                pytest.fail(
                    f"cyan_100_{name}: first diff at byte {i}: "
                    f"got 0x{out[i]:02x}, expected 0x{expected[i]:02x}. "
                    f"Output {len(out)}B vs expected {len(expected)}B"
                )
        pytest.fail(f"cyan_100_{name}: length mismatch: {len(out)}B vs {len(expected)}B")


class TestSettingVariants:
    """Per-RC-setting byte-exact tests against original-driver captures."""

    @pytest.mark.parametrize(
        ("name", "settings"),
        _SETTING_VARIANTS_PASS,
        ids=[v[0] for v in _SETTING_VARIANTS_PASS],
    )
    def test_setting_variant_matches(self, name, settings):
        _run_settings_variant(name, settings)

    @pytest.mark.xfail(
        reason=(
            "saturation_n20: 21-byte residual after the FPU-truncate fix (deeper "
            "color-pipeline drift). vivid/contrast_n20: separate code paths still diverge."
        ),
    )
    @pytest.mark.parametrize(
        ("name", "settings"),
        _SETTING_VARIANTS_XFAIL,
        ids=[v[0] for v in _SETTING_VARIANTS_XFAIL],
    )
    def test_setting_variant_matches_xfail(self, name, settings):
        _run_settings_variant(name, settings)


# ---------------------------------------------------------------------------
# Pipeline settings tests
# ---------------------------------------------------------------------------


class TestPipelineSettings:
    def test_mono_mode_only_k_plane(self):
        """In mono mode, only K plane should have data."""
        ppm = _make_ppm_bytes(4760, 10, 128, 0, 0)
        settings = PrintSettings(mono_color=MonoColor.MONO)
        out = _pipeline_from_ppm(ppm, settings)
        # Verify it produces valid output with PJL GRAYSCALE
        assert b"\x1b%-12345X" in out
        assert b"RENDERMODE=GRAYSCALE" in out

    def test_toner_save_keeps_economode_off(self):
        """Toner save uses the -TS dither tables; ECONOMODE stays OFF (matches original driver, pjl.c:58)."""
        ppm = _make_ppm_bytes(4760, 10, 0, 0, 0)
        settings = PrintSettings(toner_save=True)
        out = _pipeline_from_ppm(ppm, settings)
        assert b"ECONOMODE=OFF" in out
        assert b"ECONOMODE=ON" not in out

    def test_letter_size_dimensions(self):
        """Letter size should use 4900x6400 pixel dimensions."""
        ppm = _make_ppm_bytes(4900, 100, 255, 255, 255)
        settings = PrintSettings(page_size=PageSize.LETTER)
        out = _pipeline_from_ppm(ppm, settings)
        # Verify image dimensions in BeginImage
        # Source width should be 4928 (4900 rounded to 32)
        # Source height should be 6396 (6400 - 4)
        assert b"\xc1\x40\x13" in out  # uint16 LE for 4928 = 0x1340


# ---------------------------------------------------------------------------
# Target: fine resolution (1200 DPI)
# ---------------------------------------------------------------------------


class TestFineResolution:
    """Fine mode uses same source dimensions as Normal but different dithering/band config.

    Verified against original Brother driver captures: Fine mode uses
    RESOLUTION=600 in PJL, 600x600 session, same source/dest dimensions
    as Normal (A4: 4768x6808), COLOR_DEPTH=1, and APTMODE=ON4.
    """

    def _find_ubyte_attr(self, data: bytes, attr_id: int) -> int:
        """Find a ubyte attribute value by attribute ID."""
        marker = bytes([0xF8, attr_id])
        idx = data.index(marker)
        # ubyte format: TAG_UBYTE (0xC0) value TAG_ATTR attr_id
        return data[idx - 1]

    def _find_uint16_attr(self, data: bytes, attr_id: int) -> int:
        """Find a uint16 attribute value by attribute ID."""
        import struct

        marker = bytes([0xF8, attr_id])
        idx = data.index(marker)
        lo, hi = data[idx - 2], data[idx - 1]
        return struct.unpack("<H", bytes([lo, hi]))[0]

    def _find_uint16_xy_attr(self, data: bytes, attr_id: int) -> tuple[int, int]:
        """Find a uint16_xy attribute (x, y) by attribute ID."""
        import struct

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
        expected = read_fixture("fine_white.xl2hb")
        if expected is None:
            pytest.skip("fine_white.xl2hb not available")

        w, h = 4760, 6812
        pixels = bytes([255]) * (w * h * 3)
        settings = PrintSettings(resolution=Resolution.FINE)
        out = _run_pipeline(w, h, pixels, settings)
        assert out == expected, f"Fine white mismatch: got {len(out)}B, expected {len(expected)}B"


class TestFineBlackPagePipeline:
    def test_fine_black_matches_capture(self):
        """Fine all-black PPM -> byte-for-byte match with fine_black.xl2hb.

        Verifies the full Fine pipeline: 4bpp dithering, two-stage compression,
        PlaneBuffer headers (bit_depth=12, quant_type=0, comp_size=0),
        COLOR_DEPTH=1, band config, and APTMODE=ON4.
        """
        expected = read_fixture("fine_black.xl2hb")
        if expected is None:
            pytest.skip("fine_black.xl2hb not available")

        w, h = 4760, 6812
        pixels = bytes(w * h * 3)  # all (0,0,0) = black
        settings = PrintSettings(resolution=Resolution.FINE)
        out = _run_pipeline(w, h, pixels, settings)
        if out != expected:
            for i in range(min(len(out), len(expected))):
                if out[i] != expected[i]:
                    pytest.fail(
                        f"Fine black mismatch at byte {i}: "
                        f"got 0x{out[i]:02x}, expected 0x{expected[i]:02x}. "
                        f"Output {len(out)}B vs expected {len(expected)}B"
                    )
            pytest.fail(f"Length mismatch: {len(out)}B vs {len(expected)}B")


# ---------------------------------------------------------------------------
# Target: skip blank optimization
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
