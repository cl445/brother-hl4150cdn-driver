"""
Brother HL-4150CDN XL2HB raster decoder.

Inverse of brother_encode.py — decodes RLE-compressed plane data back to
raw 1bpp scanline bytes.

See brother_encode.py for the full encoding format documentation.
"""

from math import ceil

from brother_encode import BPL, group_bits, pack_groups


def _read_count_ext(data: bytes, pos: int) -> tuple[int, int]:
    """Read additive count extension bytes.

    Reads 0xFF bytes (each adds 255) until a byte < 0xFF is found
    (adds that final byte).

    Returns:
        (count, new_pos)
    """
    count = 0
    while pos < len(data):
        b = data[pos]
        pos += 1
        if b < 0xFF:
            count += b
            break
        count += 255
    return count, pos


def decode_rle(compressed: bytes, value_bits: int = 12) -> list[int]:
    """Decode N-bit RLE (12-bit or 20-bit format).

    Args:
        compressed: RLE-compressed data bytes.
        value_bits: Bit width of each value (12 or 20).

    Returns:
        List of N-bit group values.
    """
    if not compressed:
        return []

    v_rest_bits = value_bits - 4
    v_rest_bytes = ceil(v_rest_bits / 8)

    groups: list[int] = []
    pos = 0

    while pos < len(compressed):
        b = compressed[pos]
        pos += 1

        if b >= 0x80:
            # Run header
            v_hi = b & 0x0F
            run_info = (b >> 4) & 0x07

            # Read remaining value bytes (big-endian)
            v_rest = 0
            for _i in range(v_rest_bytes):
                v_rest = (v_rest << 8) | compressed[pos]
                pos += 1
            value = (v_hi << (v_rest_bytes * 8)) | v_rest

            if run_info < 7:
                count = run_info + 1
            else:
                ext, pos = _read_count_ext(compressed, pos)
                count = 8 + ext

            groups.extend([value] * count)

        elif b >= 0x40:
            # Literal header
            count_field = b & 0x3F

            if count_field < 0x3F:
                count = count_field + 2
            else:
                ext, pos = _read_count_ext(compressed, pos)
                count = 0x41 + ext

            # Read packed N-bit values
            bits_needed = count * value_bits
            bytes_needed = ceil(bits_needed / 8)
            packed_data = compressed[pos : pos + bytes_needed]
            pos += bytes_needed

            decoded = group_bits(packed_data, value_bits)[:count]
            groups.extend(decoded)

        elif b == 0x3F:
            # Extended zero-skip
            ext, pos = _read_count_ext(compressed, pos)
            count = 64 + ext
            groups.extend([0] * count)

        else:
            # 0x00-0x3E: Zero-skip (short)
            count = b + 1
            groups.extend([0] * count)

    return groups


def decode_rle_10bit(compressed: bytes) -> list[int]:
    """Decode 10-bit RLE (M-plane sub-block 2).

    Args:
        compressed: RLE-compressed data bytes.

    Returns:
        List of 10-bit group values.
    """
    if not compressed:
        return []

    groups: list[int] = []
    pos = 0

    while pos < len(compressed):
        b = compressed[pos]
        pos += 1

        if b >= 0x80:
            # Run header
            v_hi = b & 0x03
            count_field = (b >> 2) & 0x1F

            v_lo = compressed[pos]
            pos += 1
            value = (v_hi << 8) | v_lo

            if count_field < 31:
                count = count_field + 1
            else:
                ext, pos = _read_count_ext(compressed, pos)
                count = 32 + ext

            groups.extend([value] * count)

        elif b >= 0x40:
            # Literal header
            count_field = b & 0x3F

            if count_field < 0x3F:
                count = count_field + 2
            else:
                ext, pos = _read_count_ext(compressed, pos)
                count = 65 + ext

            # Read packed 10-bit values
            bits_needed = count * 10
            bytes_needed = ceil(bits_needed / 8)
            packed_data = compressed[pos : pos + bytes_needed]
            pos += bytes_needed

            decoded = group_bits(packed_data, 10)[:count]
            groups.extend(decoded)

        elif b == 0x3F:
            # Extended zero-skip
            ext, pos = _read_count_ext(compressed, pos)
            count = 64 + ext
            groups.extend([0] * count)

        else:
            # 0x00-0x3E: Zero-skip (short)
            count = b + 1
            groups.extend([0] * count)

    return groups


def decode_plane(compressed: bytes, plane: str = "K", n_bytes: int = BPL) -> bytes:
    """Decode K or Y plane entry to raw bytes.

    Args:
        compressed: RLE-compressed plane data.
        plane: Plane identifier ('K' or 'Y').
        n_bytes: Expected output length in bytes.

    Returns:
        Raw 1bpp plane data (n_bytes bytes).
    """
    if not compressed:
        return bytes(n_bytes)

    groups = decode_rle(compressed, value_bits=12)
    packed = pack_groups(groups, 12)
    # Truncate or pad to exact byte count
    if len(packed) >= n_bytes:
        return packed[:n_bytes]
    return packed + bytes(n_bytes - len(packed))


def decode_c_plane(compressed: bytes, n_bytes: int = BPL) -> bytes:
    """Decode C-plane entry to raw bytes.

    C-plane packs raw bytes into 20-bit words MSB-first, then applies
    sliding-window 20-bit RLE compression. This decoder reverses that:
    decode 20-bit RLE, then unpack 20-bit words back to bytes.

    Args:
        compressed: RLE-compressed C-plane data (20-bit format).
        n_bytes: Expected output length in bytes.

    Returns:
        Raw 1bpp plane data (n_bytes bytes).
    """
    if not compressed:
        return bytes(n_bytes)

    groups_20 = decode_rle(compressed, value_bits=20)

    # Unpack 20-bit words back to raw bytes
    packed = pack_groups(groups_20, 20)
    if len(packed) >= n_bytes:
        return packed[:n_bytes]
    return packed + bytes(n_bytes - len(packed))


def decode_m_plane(compressed_20: bytes, compressed_10: bytes, n_bytes: int = BPL) -> bytes:
    """Decode M-plane from both sub-blocks.

    The 10-bit sub-block contains full 10-bit group values (bit 5 NOT
    zeroed), so it alone suffices for byte reconstruction. The 20-bit
    sub-block is redundant for reconstruction.

    Args:
        compressed_20: RLE-compressed 20-bit sub-block (ignored for
            reconstruction, but accepted for API completeness).
        compressed_10: RLE-compressed 10-bit sub-block.
        n_bytes: Expected output length in bytes.

    Returns:
        Raw 1bpp plane data (n_bytes bytes).
    """
    if not compressed_10:
        return bytes(n_bytes)

    groups = decode_rle_10bit(compressed_10)
    packed = pack_groups(groups, 10)
    if len(packed) >= n_bytes:
        return packed[:n_bytes]
    return packed + bytes(n_bytes - len(packed))
