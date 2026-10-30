"""Native sliding-window RLE encoder: byte-identity vs. the pure-Python fallback in plane_encoders."""

import numpy as np
import pytest

from brother_xl2hb.codec import plane_encoders
from brother_xl2hb.codec.rle import group_bits_py

_rle_fast = pytest.importorskip("brother_xl2hb._native._rle_fast")
if not hasattr(_rle_fast, "encode_sw_rle"):
    pytest.skip("_rle_fast built without encode_sw_rle; rebuild the extension", allow_module_level=True)
encode_sw_rle = _rle_fast.encode_sw_rle

# Word size per plane encoder of plane_encoders.
_VARIANTS = {
    "K/Y 12-bit": (12, lambda data: plane_encoders.encode_plane(data, "K")),
    "C 20-bit": (20, plane_encoders.encode_c_plane),
    "M 10-bit": (10, plane_encoders.encode_m_plane_10),
}


@pytest.fixture
def python_encoder(monkeypatch):
    """The plane encoders as they run without _rle_fast: Python word split and sw_rle_encode."""
    monkeypatch.setattr(plane_encoders, "_rle_fast", None)
    monkeypatch.setattr(plane_encoders, "group_bits", group_bits_py)

    def encode(variant: str, data: bytes) -> bytes:
        return _VARIANTS[variant][1](data)

    return encode


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
def test_matches_python_encoder(variant, python_encoder):
    bits, _ = _VARIANTS[variant]
    for data in _rows():
        expected = python_encoder(variant, data)
        assert encode_sw_rle(data, bits) == expected, (variant, len(data), data[:16])


@pytest.mark.parametrize("variant", _VARIANTS)
def test_matches_python_encoder_random_sparse(variant, python_encoder):
    """Many short sparse rows: exercises every state transition cheaply."""
    bits, _ = _VARIANTS[variant]
    rng = np.random.default_rng(hash(variant) & 0xFFFF)
    for _ in range(2000):
        n = int(rng.integers(1, 64))
        vals = rng.choice(np.array([0, 0, 0, 0xFF, 0x0F, 0xF0, 0x81], dtype=np.uint8), n)
        data = vals.tobytes()
        expected = python_encoder(variant, data)
        assert encode_sw_rle(data, bits) == expected, (variant, data.hex())


@pytest.mark.parametrize("variant", _VARIANTS)
@pytest.mark.parametrize("n_literals", [2047, 2048, 4095, 4096, 9000])
def test_matches_python_encoder_literal_overflow(variant, n_literals, python_encoder):
    """Literal stretches past the overflow limit, with a zero tail so no raw fallback hides them."""
    bits, _ = _VARIANTS[variant]
    words = [(i % ((1 << bits) - 2)) + 1 for i in range(n_literals)] + [0] * 6000
    value = 0
    for w in words:
        value = (value << bits) | w
    n_bits = len(words) * bits
    data = (value << (-n_bits % 8)).to_bytes((n_bits + 7) // 8, "big")
    assert encode_sw_rle(data, bits) == python_encoder(variant, data)


def test_rejects_unknown_word_size():
    with pytest.raises(ValueError, match="unsupported"):
        encode_sw_rle(b"\x01", 16)
