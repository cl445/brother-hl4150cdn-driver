"""
Buffer management tests.

Tests the fill-and-flush cycle, multi-plane coordination, flush ordering,
and the complete driver buffer loop that produces correct block boundaries.
"""

import struct

import pytest

from xl2hb import (
    BPL,
    FLUSH_ORDER,
    PLANE_BUF_FLUSH_THRESH,
    PLANE_BUF_INIT_FREE,
    PLANE_PARAMS,
    PlaneBuffer,
)

# ---------------------------------------------------------------------------
# Buffer lifecycle
# ---------------------------------------------------------------------------


class TestBufferLifecycle:
    def test_fresh_buffer_empty(self):
        pb = PlaneBuffer(plane_id=0)
        assert pb.line_count == 0
        assert pb.flush() is None

    def test_append_increments_line_count(self):
        pb = PlaneBuffer(plane_id=0)
        pb.append_scanline(b"\x01\x02\x03")
        assert pb.line_count == 1
        pb.append_scanline(b"\x04\x05")
        assert pb.line_count == 2

    def test_empty_scanline_skipped(self):
        pb = PlaneBuffer(plane_id=0)
        pb.append_scanline(b"")
        assert pb.line_count == 0

    def test_start_line_from_first_entry(self):
        pb = PlaneBuffer(plane_id=0)
        pb.append_scanline(b"\x01", line_idx=42)
        pb.append_scanline(b"\x02", line_idx=43)
        result = pb.flush()
        assert result is not None
        assert result[0] == 42  # start_line

    def test_start_line_skips_empty_lines(self):
        """start_line should be the index of the first NON-EMPTY line."""
        pb = PlaneBuffer(plane_id=0)
        pb.append_scanline(b"", line_idx=10)  # empty → skipped
        pb.append_scanline(b"", line_idx=11)  # empty → skipped
        pb.append_scanline(b"\x01", line_idx=12)
        result = pb.flush()
        assert result is not None
        assert result[0] == 12


# ---------------------------------------------------------------------------
# Free space tracking
# ---------------------------------------------------------------------------


class TestFreeSpace:
    def test_initial_free_space(self):
        pb = PlaneBuffer(plane_id=0)
        assert pb.free_space == PLANE_BUF_INIT_FREE

    def test_free_decreases_with_entries(self):
        pb = PlaneBuffer(plane_id=0)
        initial = pb.free_space
        # 3 bytes data → padded to 4 → entry = 2+4+2 = 8 bytes
        pb.append_scanline(b"\x01\x02\x03")
        assert pb.free_space == initial - 8

    def test_4byte_aligned_entries(self):
        pb = PlaneBuffer(plane_id=0)
        initial = pb.free_space
        # 4 bytes data → already aligned → entry = 2+4+2 = 8 bytes
        pb.append_scanline(b"\x01\x02\x03\x04")
        assert pb.free_space == initial - 8

    def test_5byte_entry_padded(self):
        pb = PlaneBuffer(plane_id=0)
        initial = pb.free_space
        # 5 bytes data → padded to 8 → entry = 2+8+2 = 12 bytes
        pb.append_scanline(b"\x01\x02\x03\x04\x05")
        assert pb.free_space == initial - 12


# ---------------------------------------------------------------------------
# Nearly-full detection and flush threshold
# ---------------------------------------------------------------------------


class TestNearlyFull:
    def test_not_nearly_full_initially(self):
        pb = PlaneBuffer(plane_id=0)
        assert not pb.is_nearly_full()

    def test_becomes_nearly_full(self):
        pb = PlaneBuffer(plane_id=0)
        # Fill with large entries to cross threshold
        big_entry = bytes(100)
        while not pb.is_nearly_full():
            pb.append_scanline(big_entry)
        assert pb.is_nearly_full()

    def test_threshold_value(self):
        """Buffer becomes nearly full when free < FLUSH_THRESH."""
        assert PLANE_BUF_FLUSH_THRESH == 0x1688  # 5768
        assert PLANE_BUF_INIT_FREE == 0x7FF2  # 32754


# ---------------------------------------------------------------------------
# Flush output structure
# ---------------------------------------------------------------------------


class TestFlushOutput:
    def test_flush_returns_correct_tuple(self):
        pb = PlaneBuffer(plane_id=0)
        pb.append_scanline(b"\xaa\xbb", line_idx=100)
        result = pb.flush()
        assert result is not None
        start, count, blob = result
        assert start == 100
        assert count == 1
        assert isinstance(blob, bytes)

    def test_blob_has_valid_header(self):
        pb = PlaneBuffer(plane_id=1)  # C plane
        pb.append_scanline(b"\x01\x02\x03")
        result = pb.flush()
        assert result is not None
        _, _, blob = result
        # Parse 14-byte header
        assert blob[0] == 0  # comp_type
        assert blob[1] == 4  # bit_depth
        assert blob[2] == 2  # quant_type for C
        assert blob[3] == 20  # comp_size for C

    def test_blob_checksum_valid(self):
        pb = PlaneBuffer(plane_id=0)
        for i in range(10):
            pb.append_scanline(bytes([i] * 20))
        result = pb.flush()
        assert result is not None
        _, _, blob = result
        assert sum(blob) % 256 == 0

    def test_blob_data_size_matches(self):
        """data_size field in header == total blob length."""
        pb = PlaneBuffer(plane_id=0)
        pb.append_scanline(b"\x01\x02\x03\x04")
        result = pb.flush()
        assert result is not None
        _, _, blob = result
        data_size = struct.unpack_from(">I", blob, 8)[0]
        assert data_size == len(blob)

    def test_blob_terminates_with_zero_size(self):
        """Entry list ends with 0x0000 terminator."""
        pb = PlaneBuffer(plane_id=0)
        pb.append_scanline(b"\x01\x02")
        result = pb.flush()
        assert result is not None
        _, _, blob = result
        # After header(14) + entry(2+4+2=8), terminator at offset 22
        assert blob[22:24] == b"\x00\x00"


# ---------------------------------------------------------------------------
# Reset behavior
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_all(self):
        pb = PlaneBuffer(plane_id=0)
        pb.append_scanline(b"\x01")
        pb.append_scanline(b"\x02")
        pb.reset(start_line=500)
        assert pb.line_count == 0
        assert pb.start_line == 500
        assert pb.free_space == PLANE_BUF_INIT_FREE
        assert pb.flush() is None

    def test_reset_allows_reuse(self):
        pb = PlaneBuffer(plane_id=0)
        pb.append_scanline(b"\x01", line_idx=0)
        first = pb.flush()
        pb.reset(start_line=100)
        pb.append_scanline(b"\x02", line_idx=100)
        second = pb.flush()
        assert first is not None
        assert second is not None
        assert first[0] == 0  # first start_line
        assert second[0] == 100  # second start_line


# ---------------------------------------------------------------------------
# Multi-plane flush coordination
# ---------------------------------------------------------------------------


class TestMultiPlaneFlush:
    def test_flush_order_constant(self):
        """Flush order is C(1), M(2), Y(3), K(0)."""
        assert FLUSH_ORDER == (1, 2, 3, 0)

    def test_independent_plane_buffers(self):
        """Each plane maintains independent state."""
        buffers = {i: PlaneBuffer(plane_id=i) for i in range(4)}
        buffers[0].append_scanline(b"\x01")  # K only
        assert buffers[0].line_count == 1
        assert buffers[1].line_count == 0
        assert buffers[2].line_count == 0
        assert buffers[3].line_count == 0

    def test_all_planes_flushed_together(self):
        """When any plane is nearly full, ALL should be flushed."""
        buffers = {i: PlaneBuffer(plane_id=i) for i in range(4)}

        # Fill K plane to nearly full
        big_entry = bytes(100)
        while not buffers[0].is_nearly_full():
            for pid in range(4):
                buffers[pid].append_scanline(big_entry)

        # All planes should have data to flush
        for pid in range(4):
            result = buffers[pid].flush()
            assert result is not None, f"Plane {pid} should have data"


# ---------------------------------------------------------------------------
# Driver fill-and-flush loop simulation
# ---------------------------------------------------------------------------


class TestFillFlushLoop:
    def test_single_plane_flush_cycle(self):
        """Simulate the K-only fill-flush loop for a few lines."""
        pb = PlaneBuffer(plane_id=0)
        entry = bytes(20)  # compressed scanline
        flushes = []

        for line_idx in range(100):
            pb.append_scanline(entry, line_idx)
            if pb.is_nearly_full():
                result = pb.flush()
                if result:
                    flushes.append(result)
                pb.reset(line_idx + 1)

        # Final flush
        result = pb.flush()
        if result:
            flushes.append(result)

        # Should have at least one flush
        assert len(flushes) >= 1
        # All lines accounted for
        total_lines = sum(f[1] for f in flushes)
        assert total_lines == 100

    def test_multi_plane_flush_cycle(self):
        """Simulate the full CMYK fill-flush loop."""
        buffers = {i: PlaneBuffer(plane_id=i) for i in range(4)}
        entry = bytes(20)
        flushes = []  # list of (plane_id, start, count)

        for line_idx in range(200):
            for pid in range(4):
                buffers[pid].append_scanline(entry, line_idx)

            if any(buffers[pid].is_nearly_full() for pid in range(4)):
                for pid in FLUSH_ORDER:
                    result = buffers[pid].flush()
                    if result:
                        flushes.append((pid, result[0], result[1]))
                for pid in range(4):
                    buffers[pid].reset(line_idx + 1)

        # Final flush
        for pid in FLUSH_ORDER:
            result = buffers[pid].flush()
            if result:
                flushes.append((pid, result[0], result[1]))

        # Verify flush order within each group
        # Group flushes by start_line
        groups: dict[int, list[int]] = {}
        for pid, start, _count in flushes:
            groups.setdefault(start, []).append(pid)

        flush_rank = {1: 0, 2: 1, 3: 2, 0: 3}
        for start, planes in groups.items():
            ranks = [flush_rank[p] for p in planes]
            assert ranks == sorted(ranks), f"Flush order violated at start={start}: got planes {planes}"

    def test_a4_black_line_count(self):
        """A4 at 600dpi has 6812 lines — verify total lines match."""
        pb = PlaneBuffer(plane_id=0)
        # Simulate all-black K-plane with small compressed entries
        entry = b"\xff\xff"  # tiny entry
        total_lines = 0
        paper_h = 6812

        for line_idx in range(paper_h):
            pb.append_scanline(entry, line_idx)
            if pb.is_nearly_full():
                result = pb.flush()
                if result:
                    total_lines += result[1]
                pb.reset(line_idx + 1)

        result = pb.flush()
        if result:
            total_lines += result[1]

        assert total_lines == paper_h


# ---------------------------------------------------------------------------
# Plane-specific configuration
# ---------------------------------------------------------------------------


class TestPlaneConfig:
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
    def test_plane_params(self, plane_id, expected_qt, expected_cs):
        qt, cs = PLANE_PARAMS[plane_id]
        assert qt == expected_qt
        assert cs == expected_cs

    @pytest.mark.parametrize("plane_id", [0, 1, 2, 3])
    def test_bpl_default(self, plane_id):
        pb = PlaneBuffer(plane_id=plane_id)
        assert pb.bpl == BPL
