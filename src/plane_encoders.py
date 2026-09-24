"""Per-plane RLE encoders for the XL2HB Normal-mode raster blocks.

* :func:`encode_plane` for K and Y (12-bit sliding-window RLE).
* :func:`encode_c_plane` for C (20-bit sliding-window RLE).
* :func:`encode_m_plane_10` for M (10-bit sliding-window RLE).

All three return an empty bytestring when the input scanline is all-zero
(no ink), and fall back to a raw literal dump when their compressed
output would otherwise exceed the input length.
"""

from rle import (
    CONFIG_10BIT,
    CONFIG_12BIT,
    CONFIG_20BIT,
    SwRleConfig,
    finalize_compressed,
    group_bits,
    sw_rle_encode,
)

try:
    from _rle_fast import encode_sw_rle  # type: ignore[import-not-found]

    _HAS_CYTHON_SW_RLE = True
except ImportError:
    _HAS_CYTHON_SW_RLE = False

# Planes encoded with the 12-bit model. Every bit of the line is read: the
# last 12-bit word keeps the trailing bits and is zero-padded, as
# read_word_16 does in the original. This only shows when the line width is
# not a multiple of 12 bits and its last pixels carry ink (sizes without
# right padding, e.g. A5).
_12BIT_PLANES = ("K", "Y")


def _encode_via_sw_rle(
    words: list[int],
    data: bytes,
    config: SwRleConfig,
    word_bits: int,
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
    return finalize_compressed(output, data, word_bits)


def encode_plane(data: bytes, plane: str = "K") -> bytes:
    """Encode a single scanline of K or Y plane data using 12-bit sliding-window RLE.

    Args:
        data: Raw 1bpp plane data (BPL bytes per line).
        plane: Color plane identifier ('K' or 'Y').

    Returns:
        Compressed data bytes, empty if the line is all-zero.

    Raises:
        ValueError: If `plane` is not 'K' or 'Y'.
    """
    if plane not in _12BIT_PLANES:
        msg = f"Unknown plane {plane!r}, expected one of {_12BIT_PLANES}"
        raise ValueError(msg)
    if _HAS_CYTHON_SW_RLE:
        return encode_sw_rle(data, 12)
    return _encode_via_sw_rle(group_bits(data, 12), data, CONFIG_12BIT, 12)


def encode_c_plane(data: bytes) -> bytes:
    """Encode C-plane data (comp_size=20 block).

    Packs input bytes into 20-bit words MSB-first, then runs the
    sliding-window RLE encoder with a 3-word context.

    Returns:
        Compressed bytes, empty if the line is all-zero.
    """
    if _HAS_CYTHON_SW_RLE:
        return encode_sw_rle(data, 20)
    return _encode_via_sw_rle(group_bits(data, 20), data, CONFIG_20BIT, 20)


def encode_m_plane_10(data: bytes) -> bytes:
    """Encode M-plane data using 10-bit sliding-window RLE.

    Uses a 5-word context window for context-skip prediction. Total
    groups = ceil(total_bits / 10) = 477 for the standard BPL=596.

    Returns:
        Compressed bytes, empty if the line is all-zero.
    """
    if _HAS_CYTHON_SW_RLE:
        return encode_sw_rle(data, 10)
    return _encode_via_sw_rle(group_bits(data, 10), data, CONFIG_10BIT, 10)
