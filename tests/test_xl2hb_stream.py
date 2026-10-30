"""
Full stream integration tests.

Reconstructs XL2HB output from parsed capture data and compares
byte-for-byte against the original driver capture files.
"""

import logging
from io import BytesIO

import pytest

from brother_xl2hb.models import HL4150CDN
from brother_xl2hb.protocol.xl2hb import XL2HBWriter
from brother_xl2hb.settings import InputSlot
from fixture_utils import assert_bytes_equal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# BeginPage of the captures: A4, auto-select tray, plain paper.
_A4_PLAIN = {
    "media_size": HL4150CDN.media_size["A4"],
    "media_source": HL4150CDN.media_source[InputSlot.AUTO],
    "media_type": HL4150CDN.media_types["Plain"],
}


def _reconstruct_stream(cap) -> bytes:
    """Rebuild the full XL2HB file from a parsed CaptureFixture."""
    buf = BytesIO()
    buf.write(cap.pjl_header)

    w = XL2HBWriter(buf)
    w.write_stream_header()
    w.write_begin_session()
    w.write_open_data_source()
    w.write_begin_page(**_A4_PLAIN)
    w.write_set_page_origin(100, 100)

    sw, sh = HL4150CDN.image_dimensions("A4")
    w.write_begin_image(sw, sh)

    for block in cap.blocks:
        w.write_read_image(
            block.start_line,
            block.block_height,
            block.plane_id,
            block.expected_blob,
        )

    w.write_end_image()
    w.write_end_page()
    w.write_close_data_source()
    w.write_end_session()

    buf.write(cap.pjl_footer)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Full stream reconstruction tests
# ---------------------------------------------------------------------------


class TestWhitePage:
    def test_white_page(self, all_captures):
        """a4_white: zero blocks, byte-for-byte match."""
        cap = all_captures["a4_white"]
        actual = _reconstruct_stream(cap)
        assert_bytes_equal(actual, cap.raw, "a4_white")


class TestSingleBlockK:
    @pytest.mark.parametrize("name", ["test_fullwidth_k", "allblack_1000", "halfblack_1000"])
    def test_single_block_k(self, all_captures, name):
        """Single K-plane block captures."""
        cap = all_captures[name]
        actual = _reconstruct_stream(cap)
        assert_bytes_equal(actual, cap.raw, name)


class TestMultiBlockFlush:
    def test_a4_black_multi_block(self, all_captures):
        """a4_black: 4 K blocks with buffer flush boundaries."""
        cap = all_captures["a4_black"]
        actual = _reconstruct_stream(cap)
        assert_bytes_equal(actual, cap.raw, "a4_black")


class TestMultiPlane:
    @pytest.mark.parametrize("name", ["test_fullwidth_y", "red_100"])
    def test_multi_plane_my(self, all_captures, name):
        """M+Y two-plane captures."""
        cap = all_captures[name]
        actual = _reconstruct_stream(cap)
        assert_bytes_equal(actual, cap.raw, name)


class TestAllPlanes:
    @pytest.mark.parametrize("name", ["test_fullwidth_c", "gray75_1000"])
    def test_all_planes(self, all_captures, name):
        """All 4 planes (K,C,M,Y) captures."""
        cap = all_captures[name]
        actual = _reconstruct_stream(cap)
        assert_bytes_equal(actual, cap.raw, name)


# ---------------------------------------------------------------------------
# Flush boundary verification
# ---------------------------------------------------------------------------


class TestFlushBoundaries:
    def test_a4_black_flush_boundaries(self, all_captures):
        """Verify a4_black flush start_lines at 0, 2249, 4498, 6747."""
        cap = all_captures["a4_black"]
        assert len(cap.blocks) == 4
        expected_starts = [0, 2249, 4498, 6747]
        actual_starts = [b.start_line for b in cap.blocks]
        assert actual_starts == expected_starts, f"Flush boundaries: expected {expected_starts}, got {actual_starts}"


# ---------------------------------------------------------------------------
# Flush order verification
# ---------------------------------------------------------------------------


class TestFlushOrder:
    def test_m_before_y(self, all_captures):
        """In multi-plane captures, M(2) appears before Y(3)."""
        for name in ["test_fullwidth_y", "red_100"]:
            cap = all_captures[name]
            plane_ids = [b.plane_id for b in cap.blocks]
            m_indices = [i for i, p in enumerate(plane_ids) if p == 2]
            y_indices = [i for i, p in enumerate(plane_ids) if p == 3]
            if m_indices and y_indices:
                assert m_indices[0] < y_indices[0], f"{name}: M should come before Y, got plane order {plane_ids}"

    def test_flush_order_consecutive_pairs(self, all_captures):
        """Consecutive blocks with the same start_line follow CMYK order.

        Blocks at the same start_line from different flush events may not be
        adjacent, so we only check consecutive pairs — these come from the
        same flush event and must respect the C(1),M(2),Y(3),K(0) order.
        """
        flush_rank = {1: 0, 2: 1, 3: 2, 0: 3}
        for name in ["test_fullwidth_c", "gray75_1000"]:
            cap = all_captures[name]
            for i in range(len(cap.blocks) - 1):
                a, b = cap.blocks[i], cap.blocks[i + 1]
                if a.start_line == b.start_line:
                    assert flush_rank[a.plane_id] < flush_rank[b.plane_id], (
                        f"{name} blocks {i},{i + 1}: plane {a.plane_id} should "
                        f"come before {b.plane_id} in CMYK flush order"
                    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestUnknownMediaDefaults:
    """Unknown media size/type should fall back to defaults with a warning."""

    def test_unknown_media_size(self, caplog):
        """Unknown media size defaults to A4 (enum 2) and logs a warning."""
        with caplog.at_level(logging.WARNING, logger="brother_xl2hb.models.hl4150cdn"):
            size = HL4150CDN.media_size_code("Folio")
        assert "Unknown media size" in caplog.text
        assert size == 2
        buf = BytesIO()
        XL2HBWriter(buf).write_begin_page(**{**_A4_PLAIN, "media_size": size})
        assert b"\xc0\x02\xf8\x25" in buf.getvalue()  # TAG_UBYTE 0x02 TAG_ATTR ATTR_MEDIA_SIZE

    def test_unknown_media_type(self, caplog):
        """Unknown media type defaults to 'dRegular' and logs a warning."""
        with caplog.at_level(logging.WARNING, logger="brother_xl2hb.models.hl4150cdn"):
            media_type = HL4150CDN.media_type_code("Cardstock")
        assert "Unknown media type" in caplog.text
        buf = BytesIO()
        XL2HBWriter(buf).write_begin_page(**{**_A4_PLAIN, "media_type": media_type})
        assert b"dRegular" in buf.getvalue()
