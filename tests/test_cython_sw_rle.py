"""Native sliding-window RLE encoder: byte-identity vs. the pure-Python encoder."""

import numpy as np
import pytest

from rle import CONFIG_10BIT, CONFIG_12BIT, CONFIG_20BIT, finalize_compressed, group_bits_py, sw_rle_encode

_rle_fast = pytest.importorskip("_rle_fast")
if not hasattr(_rle_fast, "encode_sw_rle"):
    pytest.skip("_rle_fast built without encode_sw_rle; rebuild the extension", allow_module_level=True)
encode_sw_rle = _rle_fast.encode_sw_rle

# (bits, config) for every plane encoder in plane_encoders.
_VARIANTS = {
    "K/Y 12-bit": (12, CONFIG_12BIT),
    "C 20-bit": (20, CONFIG_20BIT),
    "M 10-bit": (10, CONFIG_10BIT),
}


def _python_encode(data: bytes, bits: int, config) -> bytes:
    """Reference: plane_encoders._encode_via_sw_rle over the Python word split."""
    words = group_bits_py(data, bits)
    if not words or not any(words):
        return b""
    return finalize_compressed(sw_rle_encode(words, config), data, bits)


def _rows() -> list[bytes]:
    """Scanlines covering runs, literals, context skips, count extensions and raw fallback."""
    rng = np.random.default_rng(4150)
    bpl = 596  # A4 at 600 dpi
    rows = [
        b"",
        bytes(bpl),
        b"\xff" * bpl,
        b"\x01",
        b"\x80\x00\x01",
        bytes(rng.integers(0, 256, bpl, dtype=np.uint8)),  # dense noise → raw fallback
        bytes([0xAA, 0x55] * (bpl // 2)),
        bytes([0x12, 0x34, 0x56] * 200),  # period matches the 3-word context
        bytes([0x00, 0x00, 0x0F, 0xF0, 0x00]) * 120,
        b"\x00" * 300 + b"\xff" * 296,
        b"\xff" * 5000,  # long runs → count-extension bytes
        bytes(rng.integers(0, 256, 6000, dtype=np.uint8)),  # literal overflow (> 0xFFF words)
    ]
    for density in (0.01, 0.05, 0.2, 0.5):
        for length in (bpl, 597, 1193):
            bits = rng.random(length * 8) < density
            rows.append(np.packbits(bits).tobytes())
    # Dithered-looking rows: a period-32 ordered pattern with sparse breaks.
    tile = rng.integers(0, 256, 4, dtype=np.uint8)
    base = np.tile(tile, bpl // 4 + 1)[:bpl]
    for flips in (1, 5, 40):
        row = base.copy()
        row[rng.integers(0, bpl, flips)] ^= 0xFF
        rows.append(row.tobytes())
    rows.extend(
        bytes(rng.integers(0, 256, n, dtype=np.uint8) & rng.integers(0, 256, n, dtype=np.uint8)) for n in range(1, 40)
    )
    return rows


@pytest.mark.parametrize("variant", _VARIANTS)
def test_matches_python_encoder(variant):
    bits, config = _VARIANTS[variant]
    for data in _rows():
        expected = _python_encode(data, bits, config)
        assert encode_sw_rle(data, bits) == expected, (variant, len(data), data[:16])


@pytest.mark.parametrize("variant", _VARIANTS)
def test_matches_python_encoder_random_sparse(variant):
    """Many short sparse rows: exercises every state transition cheaply."""
    bits, config = _VARIANTS[variant]
    rng = np.random.default_rng(hash(variant) & 0xFFFF)
    for _ in range(2000):
        n = int(rng.integers(1, 64))
        vals = rng.choice(np.array([0, 0, 0, 0xFF, 0x0F, 0xF0, 0x81], dtype=np.uint8), n)
        data = vals.tobytes()
        expected = _python_encode(data, bits, config)
        assert encode_sw_rle(data, bits) == expected, (variant, data.hex())


def test_rejects_unknown_word_size():
    with pytest.raises(ValueError, match="unsupported"):
        encode_sw_rle(b"\x01", 16)
