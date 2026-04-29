"""
Print settings tests.

Tests PrintSettings defaults, RC file parsing, paper/media configuration,
and PJL settings propagation.
"""

import io
import tempfile

import pytest

from brfilter import (
    DuplexMode,
    PrintSettings,
    filter_duplex_pages,
    filter_page,
)
from xl2hb import (
    MEDIA_SIZE,
    MEDIA_TYPE_STRINGS,
    PAPER_SIZES,
    generate_pjl_header,
    get_image_dimensions,
)

# ---------------------------------------------------------------------------
# Default settings
# ---------------------------------------------------------------------------


class TestDefaultSettings:
    def test_defaults(self):
        s = PrintSettings()
        assert s.media_type == "Plain"
        assert s.page_size == "A4"
        assert s.input_slot == "AutoSelect"
        assert s.resolution == "Normal"
        assert s.copies == 1
        assert s.duplex == "None"
        assert s.mono_color == "Auto"
        assert s.color_matching == "Normal"
        assert s.improve_gray is False
        assert s.enhance_black is False
        assert s.toner_save is False
        assert s.improve_output == "OFF"
        assert s.brightness == 0
        assert s.contrast == 0
        assert s.red == 0
        assert s.green == 0
        assert s.blue == 0
        assert s.saturation == 0
        assert s.skip_blank is False
        assert s.reverse is False


# ---------------------------------------------------------------------------
# RC file parsing
# ---------------------------------------------------------------------------


class TestRCFileParsing:
    def _write_rc(self, content: str) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".rc", delete=False) as f:
            f.write(content)
            return f.name

    def test_parse_standard_rc(self):
        rc = self._write_rc("""\
[hl4150cdn]
MediaType=Thick
PageSize=Letter
InputSlot=Tray1
BRResolution=Fine
Copies=3
Duplex=DuplexNoTumble
BRMonoColor=Mono
BRColorMatching=Vivid
BRGray=ON
BREnhanceBlkPrt=ON
TonerSaveMode=ON
BRImproveOutput=BRLessPaperCurl
Brightness=10
Contrast=-5
RedKey=3
GreenKey=-2
BlueKey=1
Saturation=8
BRSkipBlank=ON
BRReverse=ON
""")
        s = PrintSettings.from_rc_file(rc)
        assert s.media_type == "Thick"
        assert s.page_size == "Letter"
        assert s.input_slot == "Tray1"
        assert s.resolution == "Fine"
        assert s.copies == 3
        assert s.duplex == "DuplexNoTumble"
        assert s.mono_color == "Mono"
        assert s.color_matching == "Vivid"
        assert s.improve_gray is True
        assert s.enhance_black is True
        assert s.toner_save is True
        assert s.improve_output == "BRLessPaperCurl"
        assert s.brightness == 10
        assert s.contrast == -5
        assert s.red == 3
        assert s.green == -2
        assert s.blue == 1
        assert s.saturation == 8
        assert s.skip_blank is True
        assert s.reverse is True

    def test_parse_minimal_rc(self):
        rc = self._write_rc("[hl4150cdn]\nMediaType=Envelope\n")
        s = PrintSettings.from_rc_file(rc)
        assert s.media_type == "Envelope"
        assert s.page_size == "A4"  # default preserved

    def test_parse_original_rc(self):
        """Parse the actual driver RC file."""
        from pathlib import Path

        rc_path = str(
            Path(__file__).resolve().parent.parent
            / "original"
            / "extracted_lpr"
            / "usr"
            / "local"
            / "Brother"
            / "Printer"
            / "hl4150cdn"
            / "inf"
            / "brhl4150cdnrc"
        )
        if not Path(rc_path).exists():
            pytest.skip("Original RC file not available")
        s = PrintSettings.from_rc_file(rc_path)
        assert s.page_size == "Letter"  # US default
        assert s.media_type == "Plain"
        assert s.copies == 1


# ---------------------------------------------------------------------------
# Paper sizes
# ---------------------------------------------------------------------------


class TestPaperSizes:
    @pytest.mark.parametrize(
        ("name", "expected_wh"),
        [
            ("A4", (4760, 6812)),
            ("Letter", (4900, 6400)),
            ("Legal", (4900, 8200)),
            ("Executive", (4148, 6100)),
            ("A5", (3296, 4760)),
            ("JISB5", (4100, 5872)),
            ("Postcard", (2164, 3288)),
            ("EnvDL", (2400, 4996)),
            ("EnvC5", (3624, 5208)),
            ("Env10", (2272, 5500)),
            ("EnvMonarch", (2124, 4300)),
        ],
    )
    def test_paper_dimensions(self, name, expected_wh):
        assert PAPER_SIZES[name] == expected_wh

    @pytest.mark.parametrize("name", list(PAPER_SIZES.keys()))
    def test_all_sizes_have_media_enum(self, name):
        """Every paper size should have a corresponding MediaSize enum."""
        assert name in MEDIA_SIZE, f"No MediaSize enum for {name}"

    @pytest.mark.parametrize("name", list(PAPER_SIZES.keys()))
    def test_image_dimensions_32bit_aligned(self, name):
        """Image width must be rounded up to 32-pixel boundary."""
        sw, _sh = get_image_dimensions(name)
        assert sw % 32 == 0, f"{name}: source_width {sw} not 32-aligned"

    @pytest.mark.parametrize("name", list(PAPER_SIZES.keys()))
    def test_image_height_minus_4(self, name):
        """Image height = paper height - 4."""
        _, ph = PAPER_SIZES[name]
        _, sh = get_image_dimensions(name)
        assert sh == ph - 4

    @pytest.mark.parametrize("name", list(PAPER_SIZES.keys()))
    def test_bpl_calculation(self, name):
        """BPL = (source_width + 7) // 8."""
        sw, _ = get_image_dimensions(name)
        expected_bpl = (sw + 7) // 8
        # Since sw is always 32-aligned, BPL = sw // 8
        assert expected_bpl == sw // 8


# ---------------------------------------------------------------------------
# Media types
# ---------------------------------------------------------------------------


class TestMediaTypes:
    @pytest.mark.parametrize(
        ("name", "expected_prefix"),
        [
            ("Plain", b"dRegular"),
            ("Thin", b"dThin"),
            ("Thick", b"dThick"),
            ("Thicker", b"dThicker"),
            ("Bond", b"dBond"),
            ("Envelope", b"dEnvelope"),
            ("EnvThick", b"dEnvThick"),
            ("Recycled", b"dRecycled"),
            ("Label", b"dLabel"),
            ("Glossy", b"dGlossy"),
        ],
    )
    def test_media_type_strings(self, name, expected_prefix):
        assert MEDIA_TYPE_STRINGS[name] == expected_prefix


# ---------------------------------------------------------------------------
# PJL settings propagation
# ---------------------------------------------------------------------------


class TestPJLSettingsPropagation:
    def test_economode_on(self):
        header = generate_pjl_header(economode=True)
        assert b"ECONOMODE=ON" in header

    def test_economode_off(self):
        header = generate_pjl_header(economode=False)
        assert b"ECONOMODE=OFF" in header

    def test_color_mode(self):
        header = generate_pjl_header(color=True)
        assert b"RENDERMODE=COLOR" in header

    def test_mono_mode(self):
        header = generate_pjl_header(color=False)
        assert b"RENDERMODE=GRAYSCALE" in header

    def test_less_paper_curl(self):
        header = generate_pjl_header(less_paper_curl=True)
        assert b"LESSPAPERCURL=ON" in header

    def test_fix_intensity(self):
        header = generate_pjl_header(fix_intensity=True)
        assert b"FIXINTENSITYUP=ON" in header

    def test_apt_mode(self):
        header = generate_pjl_header(apt_mode=True)
        assert b"APTMODE=ON" in header

    def test_resolution_600(self):
        header = generate_pjl_header(resolution=600)
        assert b"RESOLUTION=600" in header

    def test_resolution_1200(self):
        header = generate_pjl_header(resolution=1200)
        assert b"RESOLUTION=1200" in header


# ---------------------------------------------------------------------------
# Target: duplex support
# ---------------------------------------------------------------------------


class TestDuplex:
    @staticmethod
    def _make_tiny_page(duplex: DuplexMode) -> bytes:
        """Run filter_page with a 1x1 white pixel and return the output."""
        settings = PrintSettings(duplex=duplex)
        pixel_data = b"\xff\xff\xff"  # 1x1 white
        buf = io.BytesIO()
        filter_page(1, 1, pixel_data, settings, buf)
        return buf.getvalue()

    def test_duplex_long_edge_sets_mode(self):
        """DuplexNoTumble should set duplex_mode=1 in BeginPage."""
        data = self._make_tiny_page(DuplexMode.NO_TUMBLE)
        # attr 0x34 (duplex_mode) with ubyte value 1: c0 01 f8 34
        assert b"\xc0\x01\xf8\x34" in data

    def test_duplex_short_edge_sets_mode(self):
        """DuplexTumble should set duplex_mode=2 in BeginPage."""
        data = self._make_tiny_page(DuplexMode.TUMBLE)
        # attr 0x34 (duplex_mode) with ubyte value 2: c0 02 f8 34
        assert b"\xc0\x02\xf8\x34" in data

    def test_duplex_sends_two_pages(self):
        """Duplex mode should produce front and back in one session."""
        settings = PrintSettings(duplex=DuplexMode.NO_TUMBLE)
        pages = [
            (1, 1, b"\xff\xff\xff"),  # front (white)
            (1, 1, b"\xff\xff\xff"),  # back (white)
        ]
        buf = io.BytesIO()
        filter_duplex_pages(pages, settings, buf)
        data = buf.getvalue()

        # Verify exactly one session
        # BeginSession: ...F8 89 41 (after ATTR_UNITS_PER_MEASURE)
        assert data.count(b"\xf8\x89\x41") == 1, "Expected exactly 1 BeginSession"
        # EndSession: standalone 0x42 byte after CloseDataSource (0x49)
        assert b"\x49\x42" in data, "Expected EndSession after CloseDataSource"

        # Verify two BeginPage opcodes (duplex_mode attr + 0x43)
        # Each BeginPage is preceded by the duplex mode attribute
        assert data.count(b"\xf8\x34\x43") >= 2, "Expected 2 BeginPage opcodes"

        # Verify duplex mode attribute is present in both pages (long-edge: value 1)
        assert data.count(b"\xc0\x01\xf8\x34") >= 2, "Expected duplex_mode=1 in both pages"


# ---------------------------------------------------------------------------
# Target: multiple copies
# ---------------------------------------------------------------------------


class TestCopies:
    def test_copies_propagated_to_page(self):
        """copies > 1 should set PageCopies attribute in BeginImage and EndPage."""
        settings = PrintSettings(copies=3)
        buf = io.BytesIO()
        filter_page(1, 1, b"\xff\xff\xff", settings, buf)
        data = buf.getvalue()
        # attr 0x31 (PageCopies) with uint16 value 3: c1 03 00 f8 31
        assert b"\xc1\x03\x00\xf8\x31" in data


# ---------------------------------------------------------------------------
# Target: toner save mode effect
# ---------------------------------------------------------------------------


class TestTonerSave:
    def test_toner_save_reduces_coverage(self):
        """Toner save should reduce ink coverage (lighter output)."""
        import numpy as np

        from dither import dither_channel_1bpp, load_dither_tables

        width = 256
        # Mid-gray row: ink level ~127 (pixel brightness 128 -> ink 127)
        row = bytes([128] * width)

        normal_ch = load_dither_tables()
        ts_ch = load_dither_tables(toner_save=True)

        normal_dots = 0
        ts_dots = 0
        # Dither multiple rows to average out pattern effects
        for y in range(32):
            normal_out = dither_channel_1bpp(row, y, width, normal_ch["K"])
            ts_out = dither_channel_1bpp(row, y, width, ts_ch["K"])
            normal_dots += np.unpackbits(np.frombuffer(normal_out, dtype=np.uint8)).sum()
            ts_dots += np.unpackbits(np.frombuffer(ts_out, dtype=np.uint8)).sum()

        assert ts_dots < normal_dots, f"Toner save dots ({ts_dots}) should be fewer than normal ({normal_dots})"
