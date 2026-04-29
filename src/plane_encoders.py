"""Per-plane RLE encoders for the XL2HB Normal-mode raster blocks.

* :func:`encode_plane` for K and Y (12-bit sliding-window RLE).
* :func:`encode_c_plane` for C (20-bit sliding-window RLE).
* :func:`encode_m_plane_10` for the M plane's 10-bit sub-block.
* :func:`encode_m_plane_20` for the M plane's 20-bit sub-block.

All four return an empty bytestring when the input scanline is all-zero
(no ink), and fall back to a raw literal dump when their compressed
output would otherwise exceed the input length.
"""

import numpy as np

from rle import (
    CONFIG_10BIT,
    CONFIG_12BIT,
    CONFIG_20BIT,
    _SwRleConfig,
    data_to_encode_groups,
    finalize_compressed,
    group_bits,
    rle_encode,
    sw_rle_encode,
)

# Per-plane parameters: (read_group_size, encode_group_size).
_PLANE_GROUP_SIZES = {
    "K": (12, 12),
    "C": (20, 12),
    "M": (12, 12),  # M plane has additional sub-blocks handled separately.
    "Y": (12, 12),
}


def _encode_via_sw_rle(
    words: list[int],
    data: bytes,
    config: _SwRleConfig,
    encode_group: int,
) -> bytes:
    """Run the sliding-window RLE encoder + raw-fallback finalize step.

    All-zero word sequences short-circuit to an empty bytestring; if the
    encoded output exceeds the raw input by more than 0x14 bytes,
    :func:`finalize_compressed` swaps it for a literal dump.

    Returns:
        Compressed bytes, or empty if the input is all-zero.
    """
    if not words or not any(words):
        return b""
    output = sw_rle_encode(words, config)
    return finalize_compressed(output, data, len(data), encode_group)


def encode_plane(data: bytes, plane: str = "K") -> bytes:
    """Encode a single scanline of K or Y plane data using 12-bit sliding-window RLE.

    Args:
        data: Raw 1bpp plane data (BPL bytes per line).
        plane: Color plane identifier ('K' or 'Y').

    Returns:
        Compressed data bytes, empty if the line is all-zero.

    Raises:
        ValueError: If `plane` is not one of 'K', 'C', 'M', 'Y'.
    """
    if plane not in _PLANE_GROUP_SIZES:
        msg = f"Unknown plane {plane!r}, expected one of {set(_PLANE_GROUP_SIZES)}"
        raise ValueError(msg)
    read_group, encode_group = _PLANE_GROUP_SIZES[plane]
    groups = data_to_encode_groups(data, read_group, encode_group)
    return _encode_via_sw_rle(groups, data, CONFIG_12BIT, encode_group)


def encode_c_plane(data: bytes) -> bytes:
    """Encode C-plane data (comp_size=20 block).

    Packs input bytes into 20-bit words MSB-first, then runs the
    sliding-window RLE encoder with a 3-word context.

    Returns:
        Compressed bytes, empty if the line is all-zero.
    """
    return _encode_via_sw_rle(group_bits(data, 20), data, CONFIG_20BIT, 20)


def encode_m_plane_10(data: bytes) -> bytes:
    """Encode M-plane data using 10-bit sliding-window RLE.

    Uses a 5-word context window for context-skip prediction. Total
    groups = ceil(total_bits / 10) = 477 for the standard BPL=596.

    Returns:
        Compressed bytes, empty if the line is all-zero.
    """
    return _encode_via_sw_rle(group_bits(data, 10), data, CONFIG_10BIT, 10)


def encode_m_plane_20(data: bytes) -> bytes:
    """Encode the M-plane comp_size=20 sub-block.

    Reads 10-bit groups, extracts bit 5 of each, places it at bit 4 of
    a 5-bit field, then pairs two 5-bit fields into 20-bit values.

    Returns:
        Compressed bytes for the 20-bit sub-block.
    """
    total_bits = len(data) * 8
    n_10bit = total_bits // 10
    total_20bit = -(-total_bits // 20)

    ten_bit = np.array(group_bits(data, 10)[:n_10bit], dtype=np.uint32)
    bit5 = ((ten_bit >> 5) & 1) << 4

    if len(bit5) % 2 != 0:
        bit5 = np.append(bit5, 0)

    pairs = bit5.reshape(-1, 2)
    groups_20 = ((pairs[:, 0] << 10) | pairs[:, 1]).tolist()

    while len(groups_20) < total_20bit:
        groups_20.append(0)

    return rle_encode(groups_20, value_bits=20)
