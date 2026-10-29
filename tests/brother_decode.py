"""
Brother HL-4150CDN XL2HB raster decoder.

Inverse of plane_encoders.py and fine_encoder.py — decodes RLE-compressed
plane data back to raw 1bpp scanline bytes.

See rle.py and plane_encoders.py for the encoding format documentation.
"""

from math import ceil

from brother_xl2hb.codec.rle import group_bits, pack_groups
from brother_xl2hb.protocol.xl2hb import BPL


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


# ---------------------------------------------------------------------------
# Fine mode (APT): inverse of fine_encoder
# ---------------------------------------------------------------------------

# Stage-2 headers 0x00..0x0F: (repeat unit, base count, count is extended).
# Units: "b" one byte, "bb" a byte pair, "b+x" a fixed byte followed by one
# fresh byte per repeat, "b+xy" a fixed byte and two fresh bytes, "bb+x" a
# fixed pair and one fresh byte, "bbb" a byte triple.
_FINE_HEADERS = {
    0: ("b", 3, False), 1: ("b", 4, False), 2: ("b", 5, False), 3: ("b", 6, True),
    4: ("bb", 2, False), 5: ("bb", 3, False), 6: ("bb", 4, False), 7: ("bb", 5, True),
    8: ("b+x", 3, False), 9: ("b+x", 4, True),
    10: ("b+xy", 3, False), 11: ("b+xy", 4, True),
    12: ("bb+x", 2, False), 13: ("bb+x", 3, True),
    14: ("bbb", 2, False), 15: ("bbb", 3, True),
}  # fmt: skip


def decode_fine_stage2(data: bytes) -> bytes:
    """Undo `compress_pattern_encode`: expand pattern headers into Stage-1 codes.

    Bytes 0x10..0xFF are literal Stage-1 codes; 0x00..0x0F start a pattern
    (`_FINE_HEADERS`), whose count may be extended by `_read_count_ext`.

    Returns:
        The Stage-1 byte stream.

    Raises:
        ValueError: If the stream ends inside a pattern.
    """
    out = bytearray()
    pos = 0
    while pos < len(data):
        head = data[pos]
        pos += 1
        if head >= 0x10:
            out.append(head)
            continue
        unit, count, extended = _FINE_HEADERS[head]
        if extended:
            extra, pos = _read_count_ext(data, pos)
            count += extra
        fixed = 2 if unit.startswith("bb") else 1
        if unit == "bbb":
            fixed = 3
        fresh = {"b+x": 1, "b+xy": 2, "bb+x": 1}.get(unit, 0)
        need = fixed + fresh * count
        if pos + need > len(data):
            msg = f"pattern 0x{head:02x} x{count} at {pos - 1} needs {need} bytes, {len(data) - pos} left"
            raise ValueError(msg)
        lead = data[pos : pos + fixed]
        pos += fixed
        if not fresh:
            out += lead * count
            continue
        for _ in range(count):
            out += lead + data[pos : pos + fresh]
            pos += fresh
    return bytes(out)


def _fine_nibble(code: int) -> int:
    """Nibble of a 3-bit Stage-1 field: 7 is 0xF, otherwise (code + 1) * 2."""
    return 0xF if code == 7 else (code + 1) * 2


def decode_fine_stage1(data: bytes) -> bytes:
    """Undo `compress_rle_preencode`: expand Stage-1 codes into 4bpp bytes.

    The result is the encoder input after its quantisation (each nibble
    other than 0xF has its LSB cleared).

    Returns:
        The 4bpp scanline bytes.
    """
    out = bytearray()
    for code in data:
        low = code & 7
        mid = (code >> 3) & 3
        if code < 0x10:
            msg = f"0x{code:02x} is a Stage-2 header, not a Stage-1 code"
            raise ValueError(msg)
        if code < 0x18:  # white run
            out += bytes(low + 1)
        elif code < 0x20:  # black run
            out += b"\xff" * (low + 1)
        elif code < 0x40:  # 1-2 white bytes, then 0xhF (bit 4 clear) or 0x0l (bit 4 set)
            out += bytes(((code >> 3) & 1) + 1)
            out.append(_fine_nibble(low) if code & 0x10 else (_fine_nibble(low) << 4) | 0xF)
        elif code < 0x80:  # one byte: high nibble from bits 5..3, low nibble from bits 2..0
            out.append((((code >> 3) & 7) * 2) << 4 | _fine_nibble(low))
        elif code < 0xA0:  # 1-4 black bytes, then 0xh0 with h = 2 * low
            out += b"\xff" * (mid + 1)
            out.append((low * 2) << 4)
        elif code < 0xC0:  # 0-3 white bytes, then 0xh0
            out += bytes(mid)
            out.append(_fine_nibble(low) << 4)
        elif code < 0xE0:  # 0-3 black bytes, then 0xFl with l = 2 * low
            out += b"\xff" * mid
            out.append(0xF0 | low * 2)
        else:  # 1-4 black bytes, then 0xhF with h = 2 * low
            out += b"\xff" * (mid + 1)
            out.append((low * 2) << 4 | 0xF)
    return bytes(out)


def decode_fine_plane(compressed: bytes) -> bytes:
    """Decode one Fine-mode scanline entry (Stage 2, then Stage 1).

    Returns:
        The quantised 4bpp scanline bytes.
    """
    return decode_fine_stage1(decode_fine_stage2(compressed))
