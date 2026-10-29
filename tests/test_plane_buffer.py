"""
PlaneBuffer unit tests.

Parametrized tests verify blob output against every block in every capture.
Hardcoded edge-case tests cover empty buffers, resets, thresholds, etc.
"""

import struct

import pytest

from brother_xl2hb.protocol.xl2hb import PLANE_BUF_FLUSH_THRESH, PLANE_BUF_INIT_FREE, PlaneBuffer
from fixture_utils import BPL, capture_block_params

BLOCK_PARAMS = capture_block_params()

# ---------------------------------------------------------------------------
# Parametrized from captures
# ---------------------------------------------------------------------------


class TestBlockBlobMatchesCapture:
    """Feed extracted entries into PlaneBuffer, compare blob byte-for-byte."""

    @pytest.mark.parametrize(("capture_name", "block_idx"), BLOCK_PARAMS)
    def test_block_blob_matches_capture(self, all_captures, capture_name, block_idx):
        cap = all_captures[capture_name]
        block = cap.blocks[block_idx]

        pb = PlaneBuffer(plane_id=block.plane_id, bpl=BPL)
        pb.start_line = block.start_line
        for entry in block.entries:
            pb.append_scanline(entry)

        result = pb.flush()
        assert result is not None, "flush() returned None but block has entries"
        _start, _count, blob = result
        assert blob == block.expected_blob, (
            f"Blob mismatch for {capture_name} block {block_idx}: "
            f"got {len(blob)}B, expected {len(block.expected_blob)}B"
        )


# ---------------------------------------------------------------------------
# Hardcoded edge cases
# ---------------------------------------------------------------------------


class TestPlaneBufferEdgeCases:
    def test_empty_buffer_returns_none(self):
        pb = PlaneBuffer(plane_id=0, bpl=BPL)
        assert pb.flush() is None

    def test_empty_compressed_skipped(self):
        pb = PlaneBuffer(plane_id=0, bpl=BPL)
        pb.append_scanline(b"")  # empty → skipped
        assert pb.flush() is None
        assert pb.line_count == 0

    def test_single_entry(self):
        pb = PlaneBuffer(plane_id=0, bpl=BPL)
        entry = b"\x01\x02\x03"
        pb.append_scanline(entry, line_idx=42)
        result = pb.flush()
        assert result is not None
        start, count, blob = result
        assert start == 42
        assert count == 1
        assert sum(blob) % 256 == 0

        # Verify entry is embedded in blob body
        # Header = 14B, then leading_size(2B) + data(3B) + pad(1B) + trailing_size(2B) = 8B
        # Then terminator 00 00, then checksum
        assert len(blob) == 14 + 8 + 2 + 1  # 25

        # Verify leading/trailing sizes
        leading = struct.unpack_from(">H", blob, 14)[0]
        trailing = struct.unpack_from(">H", blob, 14 + 2 + 4)[0]  # after padded data
        assert leading == 3
        assert trailing == 3

    def test_reset_clears_state(self):
        pb = PlaneBuffer(plane_id=0, bpl=BPL)
        pb.append_scanline(b"\x01\x02", line_idx=10)
        assert pb.line_count == 1

        pb.reset(start_line=100)
        assert pb.line_count == 0
        assert pb.start_line == 100
        assert pb.flush() is None

    def test_is_nearly_full_threshold(self):
        pb = PlaneBuffer(plane_id=0, bpl=BPL)
        assert not pb.is_nearly_full()
        assert pb.free_space == PLANE_BUF_INIT_FREE

        # Fill up to near the threshold
        # Each entry with 4 bytes of data = 2 + 4 + 2 = 8 bytes in entries buffer
        entry = b"\x01\x02\x03\x04"  # 4 bytes, already 4-aligned
        target_fill = PLANE_BUF_INIT_FREE - PLANE_BUF_FLUSH_THRESH + 1
        entries_needed = target_fill // 8 + 1
        for _i in range(entries_needed):
            pb.append_scanline(entry)
        assert pb.is_nearly_full()

    @pytest.mark.parametrize(
        ("plane_id", "expected_qt", "expected_cs"),
        [
            (0, 2, 12),  # K
            (1, 2, 20),  # C
            (2, 4, 10),  # M
            (3, 2, 12),  # Y
        ],
        ids=["K", "C", "M", "Y"],
    )
    def test_plane_params_in_header(self, plane_id, expected_qt, expected_cs):
        """Verify quant_type and comp_size in header for each plane."""
        pb = PlaneBuffer(plane_id=plane_id, bpl=BPL)
        pb.append_scanline(b"\xaa\xbb")
        result = pb.flush()
        assert result is not None
        _, _, blob = result
        qt = blob[2]
        cs = blob[3]
        assert qt == expected_qt
        assert cs == expected_cs
