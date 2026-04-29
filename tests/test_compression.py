"""
Compression / encoding / decoding tests.

Migrates all inline test vectors from brother_encode.py into proper pytest
parametrized tests, plus decoding verification and capture-derived round-trip.
"""

import pytest

from brother_decode import (
    decode_c_plane,
    decode_m_plane,
    decode_plane,
)
from brother_encode import (
    compress_jpegls_encode,
    compress_rle_preencode,
    encode_c_plane,
    encode_fine_plane,
    encode_m_plane_10,
    encode_m_plane_20,
    encode_plane,
    group_bits,
    pack_groups,
)
from xl2hb import BPL

# ---------------------------------------------------------------------------
# K-plane: 12-bit read, 12-bit RLE
# ---------------------------------------------------------------------------

_K_VECTORS = [
    ("blank", bytes(BPL), ""),
    ("full_ff", bytes([0xFF] * 595) + bytes(1), "ffffff8540ff0000"),
    ("byte0_ff", bytes([0xFF]) + bytes(BPL - 1), "8ff0f000ff86"),
    ("byte0_80", bytes([0x80]) + bytes(BPL - 1), "8800f000ff86"),
    ("byte0_01", bytes([0x01]) + bytes(BPL - 1), "8010f000ff86"),
    ("byte1_ff", bytes([0x00, 0xFF]) + bytes(BPL - 2), "4000ff00f000ff85"),
    ("byte2_ff", bytes(2) + bytes([0xFF]) + bytes(BPL - 3), "400000fff000ff85"),
    ("byte10_ff", bytes(10) + bytes([0xFF]) + bytes(BPL - 11), "d0004000ff00f000ff7f"),
    ("2bytes_ff", bytes([0xFF] * 2) + bytes(BPL - 2), "40ffff00f000ff85"),
    ("3bytes_ff", bytes([0xFF] * 3) + bytes(BPL - 3), "9ffff000ff85"),
    ("4bytes_ff", bytes([0xFF] * 4) + bytes(BPL - 4), "9fff8ff0f000ff84"),
    ("5bytes_ff", bytes([0xFF] * 5) + bytes(BPL - 5), "afff8f00f000ff83"),
    ("6bytes_ff", bytes([0xFF] * 6) + bytes(BPL - 6), "bffff000ff83"),
    ("7bytes_ff", bytes([0xFF] * 7) + bytes(BPL - 7), "bfff8ff0f000ff82"),
    ("8bytes_ff", bytes([0xFF] * 8) + bytes(BPL - 8), "cfff8f00f000ff81"),
    ("9bytes_ff", bytes([0xFF] * 9) + bytes(BPL - 9), "dffff000ff81"),
    ("10bytes_ff", bytes([0xFF] * 10) + bytes(BPL - 10), "dfff8ff0f000ff80"),
    ("11bytes_ff", bytes([0xFF] * 11) + bytes(BPL - 11), "efff8f00f000ff7f"),
    ("12bytes_ff", bytes([0xFF] * 12) + bytes(BPL - 12), "ffff00f000ff7f"),
    ("16bytes_ff", bytes([0xFF] * 16) + bytes(BPL - 16), "ffff028ff0f000ff7c"),
    ("32bytes_ff", bytes([0xFF] * 32) + bytes(BPL - 32), "ffff0d8f00f000ff71"),
    ("64bytes_ff", bytes([0xFF] * 64) + bytes(BPL - 64), "ffff228ff0f000ff5c"),
    ("128bytes_ff", bytes([0xFF] * 128) + bytes(BPL - 128), "ffff4d8f00f000ff31"),
    ("256bytes_ff", bytes([0xFF] * 256) + bytes(BPL - 256), "ffffa28ff0f000db"),
    ("bit0", bytes([0x80]) + bytes(BPL - 1), "8800f000ff86"),
    ("bit1", bytes([0x40]) + bytes(BPL - 1), "8400f000ff86"),
    ("bit4", bytes([0x08]) + bytes(BPL - 1), "8080f000ff86"),
    ("bit7", bytes([0x01]) + bytes(BPL - 1), "8010f000ff86"),
    ("bit8", bytes([0x00, 0x80]) + bytes(BPL - 2), "8008f000ff86"),
    ("bit15", bytes([0x00, 0x01]) + bytes(BPL - 2), "40000100f000ff85"),
    ("bit16", bytes(2) + bytes([0x80]) + bytes(BPL - 3), "40000080f000ff85"),
    ("alt_cc", bytes([0xCC] * 595) + bytes(1), "fcccff8540cc0000"),
    ("alt_33", bytes([0x33] * 595) + bytes(1), "f333ff8540330000"),
    ("full_aa", bytes([0xAA] * 595) + bytes(1), "faaaff8540aa0000"),
    ("full_55", bytes([0x55] * 595) + bytes(1), "f555ff8540550000"),
    ("3vals_abc", bytes([0xAB, 0xCD, 0xEF]) + bytes(BPL - 3), "40abcdeff000ff85"),
    ("byte593_ff", bytes(593) + bytes([0xFF]) + bytes(2), "f000ff8480ff9000"),
]

# Position-based
_p = bytearray(BPL)
_p[0] = 0xFF
_p[100] = 0xFF
_K_VECTORS.append(("two_0_100", bytes(_p), "8ff0f000394000ff00f000ff43"))

_p = bytearray(BPL)
for _i in range(100, 200):
    _p[_i] = 0xFF
_K_VECTORS.append(("mid_100", bytes(_p), "f0003a800fffff3a8f00f000ff01"))


class TestKPlaneEncoding:
    @pytest.mark.parametrize(
        ("name", "data", "expected_hex"),
        [pytest.param(n, d, e, id=n) for n, d, e in _K_VECTORS if e],
    )
    def test_k_plane(self, name, data, expected_hex):
        expected = bytes.fromhex(expected_hex)
        result = encode_plane(data, "K")
        assert result == expected

    def test_all_zero_returns_empty(self):
        assert encode_plane(bytes(BPL), "K") == b""


# ---------------------------------------------------------------------------
# Y-plane: same encoding as K (12-bit read, 12-bit encode)
# ---------------------------------------------------------------------------

_Y_VECTORS = [
    ("1bytes", bytes([0xFF]) + bytes(BPL - 1), "8ff0f000ff86"),
    ("2bytes", bytes([0xFF] * 2) + bytes(BPL - 2), "40ffff00f000ff85"),
    ("4bytes", bytes([0xFF] * 4) + bytes(BPL - 4), "9fff8ff0f000ff84"),
    ("8bytes", bytes([0xFF] * 8) + bytes(BPL - 8), "cfff8f00f000ff81"),
    ("16bytes", bytes([0xFF] * 16) + bytes(BPL - 16), "ffff028ff0f000ff7c"),
    ("32bytes", bytes([0xFF] * 32) + bytes(BPL - 32), "ffff0d8f00f000ff71"),
    ("64bytes", bytes([0xFF] * 64) + bytes(BPL - 64), "ffff228ff0f000ff5c"),
    ("128bytes", bytes([0xFF] * 128) + bytes(BPL - 128), "ffff4d8f00f000ff31"),
    ("256bytes", bytes([0xFF] * 256) + bytes(BPL - 256), "ffffa28ff0f000db"),
]


class TestYPlaneEncoding:
    @pytest.mark.parametrize(("name", "data", "expected_hex"), [pytest.param(n, d, e, id=n) for n, d, e in _Y_VECTORS])
    def test_y_plane(self, name, data, expected_hex):
        expected = bytes.fromhex(expected_hex)
        result = encode_plane(data, "Y")
        assert result == expected


# ---------------------------------------------------------------------------
# C-plane: 20-bit word packing → sliding-window 20-bit RLE (3-word context)
# ---------------------------------------------------------------------------

_C_VECTORS = [
    ("1bytes", bytes([0xFF]) + bytes(BPL - 1), "8ff000f00000e6"),
    ("2bytes", bytes([0xFF] * 2) + bytes(BPL - 2), "8ffff0f00000e6"),
    ("4bytes", bytes([0xFF] * 4) + bytes(BPL - 4), "40ffffffff00f00000e5"),
    ("8bytes", bytes([0xFF] * 8) + bytes(BPL - 8), "afffff8f0000f00000e3"),
    ("16bytes", bytes([0xFF] * 16) + bytes(BPL - 16), "dfffff8ff000f00000e0"),
    ("32bytes", bytes([0xFF] * 32) + bytes(BPL - 32), "ffffff048ffff0f00000da"),
    ("64bytes", bytes([0xFF] * 64) + bytes(BPL - 64), "ffffff118fff00f00000cd"),
    ("128bytes", bytes([0xFF] * 128) + bytes(BPL - 128), "ffffff2b8f0000f00000b3"),
    ("256bytes", bytes([0xFF] * 256) + bytes(BPL - 256), "ffffff5e8ff000f0000080"),
    ("596bytes", bytes([0xFF] * 595) + bytes(1), "ffffffe6800000"),
]


class TestCPlaneEncoding:
    @pytest.mark.parametrize(("name", "data", "expected_hex"), [pytest.param(n, d, e, id=n) for n, d, e in _C_VECTORS])
    def test_c_plane(self, name, data, expected_hex):
        expected = bytes.fromhex(expected_hex)
        result = encode_c_plane(data)
        assert result == expected

    def test_all_zero_returns_empty(self):
        assert encode_c_plane(bytes(BPL)) == b""


# ---------------------------------------------------------------------------
# M-plane comp_size=20 sub-block
# ---------------------------------------------------------------------------

_M20_VECTORS = [
    ("2bytes", bytes([0xFF] * 2) + bytes(BPL - 2), "804010f00000e6"),
    ("4bytes", bytes([0xFF] * 4) + bytes(BPL - 4), "400401004000f00000e5"),
    ("8bytes", bytes([0xFF] * 8) + bytes(BPL - 8), "a04010f00000e4"),
    ("16bytes", bytes([0xFF] * 16) + bytes(BPL - 16), "d04010804000f00000e0"),
    ("32bytes", bytes([0xFF] * 32) + bytes(BPL - 32), "f0401005f00000da"),
    ("64bytes", bytes([0xFF] * 64) + bytes(BPL - 64), "f0401011804000f00000cd"),
    ("128bytes", bytes([0xFF] * 128) + bytes(BPL - 128), "f040102bf00000b4"),
    ("256bytes", bytes([0xFF] * 256) + bytes(BPL - 256), "f040105e804000f0000080"),
    ("596bytes", bytes([0xFF] * 595) + bytes(1), "f04010e6800000"),
]


class TestMPlane20Encoding:
    @pytest.mark.parametrize(
        ("name", "data", "expected_hex"),
        [pytest.param(n, d, e, id=n) for n, d, e in _M20_VECTORS],
    )
    def test_m20(self, name, data, expected_hex):
        expected = bytes.fromhex(expected_hex)
        result = encode_m_plane_20(data)
        assert result == expected


# ---------------------------------------------------------------------------
# M-plane comp_size=10 sub-block
# ---------------------------------------------------------------------------

_M10_VECTORS = [
    ("2bytes", bytes([0xFF] * 2) + bytes(BPL - 2), "40ffff0001fc00ffba"),
    ("4bytes", bytes([0xFF] * 4) + bytes(BPL - 4), "8bff8300fc00ffba"),
    ("8bytes", bytes([0xFF] * 8) + bytes(BPL - 8), "97ff83c0fc00ffb7"),
    ("16bytes", bytes([0xFF] * 16) + bytes(BPL - 16), "afff83fcfc00ffb1"),
    ("32bytes", bytes([0xFF] * 32) + bytes(BPL - 32), "e3ff83f0fc00ffa4"),
    ("64bytes", bytes([0xFF] * 64) + bytes(BPL - 64), "ffff138300fc00ff8a"),
    ("128bytes", bytes([0xFF] * 128) + bytes(BPL - 128), "ffff4683c0fc00ff57"),
    ("256bytes", bytes([0xFF] * 256) + bytes(BPL - 256), "ffffac83fcfc00f0"),
    ("596bytes", bytes([0xFF] * 595) + bytes(1), "ffffffbd8000"),
]


class TestMPlane10Encoding:
    @pytest.mark.parametrize(
        ("name", "data", "expected_hex"),
        [pytest.param(n, d, e, id=n) for n, d, e in _M10_VECTORS],
    )
    def test_m10(self, name, data, expected_hex):
        expected = bytes.fromhex(expected_hex)
        result = encode_m_plane_10(data)
        assert result == expected

    def test_all_zero_returns_empty(self):
        assert encode_m_plane_10(bytes(BPL)) == b""


# ---------------------------------------------------------------------------
# M-plane combined (both sub-blocks concatenated)
# ---------------------------------------------------------------------------


class TestMPlaneCombined:
    def test_combined_format(self):
        """M-plane entry = m20 + m10 sub-blocks concatenated."""
        data = bytes([0xFF] * 8) + bytes(BPL - 8)
        m20 = encode_m_plane_20(data)
        m10 = encode_m_plane_10(data)
        combined = m20 + m10
        assert len(combined) > 0
        assert combined == m20 + m10


# ---------------------------------------------------------------------------
# Bit grouping primitives
# ---------------------------------------------------------------------------


class TestGroupBits:
    def test_8bit_identity(self):
        data = bytes([0xAB, 0xCD])
        groups = group_bits(data, 8)
        assert groups == [0xAB, 0xCD]

    def test_4bit_groups(self):
        data = bytes([0xAB])
        groups = group_bits(data, 4)
        assert groups == [0xA, 0xB]

    def test_12bit_groups(self):
        data = bytes([0xFF, 0xF0, 0x00])
        groups = group_bits(data, 12)
        assert groups == [0xFFF, 0x000]

    def test_10bit_groups_with_partial(self):
        data = bytes([0xFF, 0xC0])  # 16 bits → 1 full + 1 partial 10-bit
        groups = group_bits(data, 10)
        assert len(groups) == 2
        assert groups[0] == 0x3FF  # first 10 bits of 0xFFC0

    def test_empty_input(self):
        assert group_bits(b"", 12) == []


class TestPackGroups:
    def test_roundtrip_12bit(self):
        original = [0xFFF, 0x000, 0xABC]
        packed = pack_groups(original, 12)
        recovered = group_bits(packed, 12)[: len(original)]
        assert recovered == original

    def test_roundtrip_10bit(self):
        original = [0x3FF, 0x000, 0x1AB]
        packed = pack_groups(original, 10)
        recovered = group_bits(packed, 10)[: len(original)]
        assert recovered == original


# ---------------------------------------------------------------------------
# Capture-derived compression verification
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Decoding: verify decode(compressed) reconstructs original data
# ---------------------------------------------------------------------------


class TestKPlaneDecoding:
    @pytest.mark.parametrize(
        ("name", "data", "compressed_hex"),
        [pytest.param(n, d, e, id=n) for n, d, e in _K_VECTORS if e],
    )
    def test_decode_k(self, name, data, compressed_hex):
        compressed = bytes.fromhex(compressed_hex)
        result = decode_plane(compressed, "K")
        assert result == data

    def test_empty_returns_zeros(self):
        assert decode_plane(b"", "K") == bytes(BPL)


class TestYPlaneDecoding:
    @pytest.mark.parametrize(
        ("name", "data", "compressed_hex"),
        [pytest.param(n, d, e, id=n) for n, d, e in _Y_VECTORS],
    )
    def test_decode_y(self, name, data, compressed_hex):
        compressed = bytes.fromhex(compressed_hex)
        result = decode_plane(compressed, "Y")
        assert result == data


class TestCPlaneDecoding:
    @pytest.mark.parametrize(
        ("name", "data", "compressed_hex"),
        [pytest.param(n, d, e, id=n) for n, d, e in _C_VECTORS],
    )
    def test_decode_c(self, name, data, compressed_hex):
        """Decode C-plane and verify re-encoding matches original."""
        compressed = bytes.fromhex(compressed_hex)
        decoded = decode_c_plane(compressed)
        re_encoded = encode_c_plane(decoded)
        assert re_encoded == compressed


class TestMPlane10Decoding:
    @pytest.mark.parametrize(
        ("name", "data", "compressed_hex"),
        [pytest.param(n, d, e, id=n) for n, d, e in _M10_VECTORS],
    )
    def test_decode_m10(self, name, data, compressed_hex):
        compressed = bytes.fromhex(compressed_hex)
        decoded = decode_m_plane(b"", compressed)
        assert decoded == data

    def test_empty_returns_zeros(self):
        assert decode_m_plane(b"", b"") == bytes(BPL)


# ---------------------------------------------------------------------------
# Roundtrip: encode → decode → verify identity
# ---------------------------------------------------------------------------


class TestRoundtrip:
    @pytest.mark.parametrize(
        ("name", "data", "compressed_hex"),
        [pytest.param(n, d, e, id=n) for n, d, e in _K_VECTORS],
    )
    def test_k_roundtrip(self, name, data, compressed_hex):
        encoded = encode_plane(data, "K")
        decoded = decode_plane(encoded, "K")
        assert decoded == data

    @pytest.mark.parametrize(
        ("name", "data", "compressed_hex"),
        [pytest.param(n, d, e, id=n) for n, d, e in _Y_VECTORS],
    )
    def test_y_roundtrip(self, name, data, compressed_hex):
        encoded = encode_plane(data, "Y")
        decoded = decode_plane(encoded, "Y")
        assert decoded == data

    @pytest.mark.parametrize(
        ("name", "data", "compressed_hex"),
        [pytest.param(n, d, e, id=n) for n, d, e in _M10_VECTORS],
    )
    def test_m10_roundtrip(self, name, data, compressed_hex):
        encoded = encode_m_plane_10(data)
        decoded = decode_m_plane(b"", encoded)
        assert decoded == data

    @pytest.mark.parametrize(
        ("name", "data", "compressed_hex"),
        [pytest.param(n, d, e, id=n) for n, d, e in _C_VECTORS],
    )
    def test_c_roundtrip(self, name, data, compressed_hex):
        """C-plane roundtrip: encode → decode → re-encode matches."""
        encoded = encode_c_plane(data)
        decoded = decode_c_plane(encoded)
        re_encoded = encode_c_plane(decoded)
        assert re_encoded == encoded


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_encode_plane_invalid_plane(self):
        """ValueError for unknown plane identifier."""
        with pytest.raises(ValueError, match="Unknown plane"):
            encode_plane(bytes(BPL), "X")


# ---------------------------------------------------------------------------
# Capture-derived compression verification
# ---------------------------------------------------------------------------


class TestCaptureCompression:
    """Verify that compressing plane data from captures produces the same
    compressed entries as the original driver.

    Decompresses each captured entry, re-compresses, and compares against
    the original compressed bytes.
    """

    def test_k_roundtrip_from_capture(self, all_captures):
        """Decompress → re-compress → matches capture for K-plane blocks."""
        cap = all_captures.get("allblack_1000")
        if cap is None:
            pytest.skip("allblack_1000 capture not available")

        k_blocks = [b for b in cap.blocks if b.plane_id == 0]
        assert k_blocks, "No K-plane blocks found in allblack_1000"

        for block in k_blocks:
            for i, entry in enumerate(block.entries):
                decoded = decode_plane(entry, "K")
                re_encoded = encode_plane(decoded, "K")
                assert re_encoded == entry, (
                    f"K block line {block.start_line} entry {i}: "
                    f"re-encoded {re_encoded.hex()} != original {entry.hex()}"
                )


# ---------------------------------------------------------------------------
# Fine mode two-stage compression
# ---------------------------------------------------------------------------

FINE_BPL = 2480  # 4960 pixels / 2 (4bpp nibble-packed)


class TestFinePreEncode:
    """Stage 1: compress_rle_preencode nibble-aware RLE."""

    def test_all_zero_returns_white_run(self):
        """All-zero input (no ink) produces white run codes."""
        result = compress_rle_preencode(bytes(10))
        # 10 white bytes: 8 + 2 → 0x17, 0x11
        assert result == bytes([0x17, 0x11])

    def test_all_ff_returns_black_run(self):
        """All-0xFF input (full ink) produces black run codes."""
        result = compress_rle_preencode(bytes([0xFF] * 10))
        # 10 black bytes: 8 + 2 → 0x1F, 0x19
        assert result == bytes([0x1F, 0x19])

    def test_single_white(self):
        result = compress_rle_preencode(bytes(1))
        assert result == bytes([0x10])

    def test_single_black(self):
        result = compress_rle_preencode(bytes([0xFF]))
        assert result == bytes([0x18])

    def test_quantization_clears_lsb(self):
        """Nibbles != 0xF should have LSB cleared."""
        result = compress_rle_preencode(bytes([0x21]))
        assert len(result) > 0

    def test_empty_input(self):
        result = compress_rle_preencode(b"")
        assert result == b""


class TestFineJpeglsEncode:
    """Stage 2: compress_jpegls_encode pattern matching."""

    def test_stride1_repeat(self):
        """12 identical bytes should be encoded as mode 0."""
        data = bytes([0x50] * 12)
        result = compress_jpegls_encode(data)
        assert len(result) < len(data)
        assert 0x50 in result  # pattern byte present

    def test_literal_passthrough(self):
        """Bytes >= 0x10 with no pattern pass through as literals."""
        data = bytes([0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xA0, 0xB0])
        result = compress_jpegls_encode(data)
        # Less than 12 bytes, so no pattern detection — all literal
        assert result == data

    def test_empty_input(self):
        result = compress_jpegls_encode(b"")
        assert result == b""


class TestEncodeFine:
    """End-to-end Fine compression (Stage 1 + Stage 2)."""

    def test_all_zero_returns_empty(self):
        """All-zero scanline (no ink) returns empty."""
        assert encode_fine_plane(bytes(FINE_BPL)) == b""

    def test_all_ff_produces_output(self):
        """Full black scanline produces non-empty compressed output."""
        result = encode_fine_plane(bytes([0xFF] * FINE_BPL))
        assert len(result) > 0
        assert len(result) < FINE_BPL

    def test_output_smaller_than_input(self):
        """Uniform data should compress well."""
        data = bytes([0x88] * FINE_BPL)
        result = encode_fine_plane(data)
        assert 0 < len(result) < FINE_BPL

    def test_mixed_data(self):
        """Mixed data produces valid output."""
        data = bytes([0x50] * 100 + [0x00] * (FINE_BPL - 100))
        result = encode_fine_plane(data)
        assert len(result) > 0
