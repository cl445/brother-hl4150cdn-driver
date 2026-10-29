"""Native Fine codec, 4bpp dither and Fine band kernel: byte-identity vs. the pure-Python reference."""

import io
import random

import numpy as np
import pytest

from brother_xl2hb.codec import fine_encoder
from brother_xl2hb.codec.fine_encoder import compress_pattern_encode, compress_rle_preencode, encode_fine_plane_py
from brother_xl2hb.color import color_lut, dither
from brother_xl2hb.job import filter_pages
from brother_xl2hb.models import HL4150CDN
from brother_xl2hb.render import bands
from brother_xl2hb.render.bands import band_kernel_colour
from brother_xl2hb.render.page import init_channels
from brother_xl2hb.settings import ColorMatching, MediaType, PageSize, PrintSettings, Resolution

_fine_fast = pytest.importorskip("brother_xl2hb._native._fine_fast")

_LENGTHS = [0, 1, 2, 11, 12, 13, 1023, 1024, 1025, 2380, 4000]
_PALETTE = [0x00, 0xFF, 0x0F, 0xF0, 0x0E, 0xE0, 0xF6, 0x6F, 0x44, 0x20, 0x02, 0x11, 0xEE, 0x1F, 0xF1]


def _random_line(rng: random.Random, n: int) -> bytes:
    """4bpp scanline of white/black runs, repeated short units (strides 1..6), saturated nibbles and noise."""
    out = bytearray()
    while len(out) < n:
        kind = rng.random()
        if kind < 0.25:
            out += bytes([rng.choice((0x00, 0xFF))]) * rng.randint(1, 600 if rng.random() < 0.1 else 40)
        elif kind < 0.55:
            unit = bytes(rng.choice(_PALETTE) for _ in range(rng.randint(1, 6)))
            out += unit * rng.randint(1, 400 if rng.random() < 0.1 else 12)
        elif kind < 0.7:
            # Stride-2/3 patterns with a varying slot: modes 1, 3 and 4.
            stride = rng.choice((2, 3))
            fixed = bytes(rng.choice(_PALETTE) for _ in range(stride - 1))
            for _ in range(rng.randint(2, 300)):
                out += fixed[: rng.randint(1, stride - 1)] if rng.random() < 0.02 else fixed
                out.append(rng.randrange(256))
        else:
            out += bytes(rng.choice(_PALETTE) if rng.random() < 0.7 else rng.randrange(256) for _ in range(8))
    return bytes(out[:n])


def _edge_lines() -> list[bytes]:
    lines = [b"", b"\x00", b"\xff", b"\x01", b"\x10", bytes(2380), b"\xff" * 2380, b"\x88" * 2380]
    for n in (5, 6, 7, 11, 12, 13, 24, 1023, 1024, 1025, 1100, 4000):
        lines += [b"\xab" * n, b"\x12\x34" * (n // 2), b"\x12\x34\x56" * (n // 3), b"\x00\xff" * (n // 2)]
        lines += [bytes(b for i in range(n) for b in ((0x40, i & 0xFF)))]  # mode 1, long
        lines += [bytes(b for i in range(n) for b in ((0x40, 0x47, i & 0xFF)))]  # mode 4, long
        lines += [bytes(b for i in range(n) for b in ((0x40, i & 0xFF, (i * 7) & 0xFF)))]  # mode 3, long
    # Every byte value alone, after white and after black runs of every length up to 12.
    for v in range(256):
        for run in range(13):
            lines += [bytes(run) + bytes([v]), b"\xff" * run + bytes([v]), bytes([v]) + bytes(run)]
    return lines


def _cases() -> list[bytes]:
    rng = random.Random(4150)
    return [_random_line(rng, n) for n in _LENGTHS for _ in range(60)] + _edge_lines()


_CASES = _cases()


def test_stage1_matches_python():
    for data in _CASES:
        assert _fine_fast.rle_preencode(data) == compress_rle_preencode(data), data[:24].hex()


def test_stage2_matches_python():
    for data in _CASES:
        assert _fine_fast.pattern_encode(compress_rle_preencode(data)) == compress_pattern_encode(
            compress_rle_preencode(data)
        ), data[:24].hex()


def test_stage2_matches_python_on_raw_bytes():
    """Stage 2 on arbitrary input, header-range bytes 0x00..0x0F included."""
    for data in _CASES:
        assert _fine_fast.pattern_encode(data) == compress_pattern_encode(data), data[:24].hex()


def test_encode_fine_plane_matches_python():
    for data in _CASES:
        assert _fine_fast.encode_fine_plane(data) == encode_fine_plane_py(data), data[:24].hex()


def test_encode_fine_plane_accepts_buffers():
    data = _random_line(random.Random(7), 2380)
    expected = encode_fine_plane_py(data)
    assert _fine_fast.encode_fine_plane(bytearray(data)) == expected
    assert _fine_fast.encode_fine_plane(memoryview(data)) == expected
    assert _fine_fast.encode_fine_plane(np.frombuffer(data, np.uint8)) == expected


def test_encode_fine_plane_dispatch(monkeypatch):
    data = _random_line(random.Random(8), 2380)
    assert fine_encoder.HAS_CYTHON_FINE
    native = fine_encoder.encode_fine_plane(data)
    monkeypatch.setattr(fine_encoder, "HAS_CYTHON_FINE", False)
    assert fine_encoder.encode_fine_plane(data) == native == encode_fine_plane_py(data)


@pytest.mark.parametrize("seed", range(3))
def test_dither_row_4bpp_matches_numpy(seed, monkeypatch):
    """Synthetic nibble tables (values above 15 too), odd widths, strided rows."""
    rng = np.random.default_rng(seed)
    h, w = (31, 37) if seed else (1, 1)
    levels = rng.integers(0, 256 if seed == 2 else 16, (256, h, w), dtype=np.uint8)
    channel = dither.DitherChannel(width=w, height=h, threshold_matrix=None, nibble_levels=levels)
    for width in (0, 1, 2, 3, 37, 4760, 4761):
        row = rng.integers(0, 256, 2 * width + 10, dtype=np.uint8)[::2]
        for y in (0, 5, 30, 31, 6811):
            native = dither.dither_channel_4bpp_arr(row, y, width, channel)
            monkeypatch.setattr(dither, "HAS_CYTHON_DITHER_4BPP", False)
            reference = dither.dither_channel_4bpp_arr(row, y, width, channel)
            monkeypatch.setattr(dither, "HAS_CYTHON_DITHER_4BPP", True)
            assert native == reference, (width, y)


def test_dither_row_4bpp_rejects_bad_shapes():
    with pytest.raises(ValueError, match="row shorter"):
        _fine_fast.dither_row_4bpp(np.zeros(3, np.uint8), np.zeros((256, 2, 2), np.uint8), 0, 4)
    with pytest.raises(ValueError, match="levels not of shape"):
        _fine_fast.dither_row_4bpp(np.zeros(4, np.uint8), np.zeros((255, 2, 2), np.uint8), 0, 4)


# ---------------------------------------------------------------------------
# Fine band kernel (render_band_fine) vs. the per-line path
# ---------------------------------------------------------------------------

_A4_W = 4760


@pytest.fixture(scope="module")
def fine_channels():
    channels = init_channels(PrintSettings(resolution=Resolution.FINE), HL4150CDN)
    if any(channel.nibble_levels is None for channel in channels.values()):
        pytest.skip("Fine BRCD tables not available (run scripts/extract_blobs.sh)")
    return channels


@pytest.fixture(params=["interp", "random_lut"])
def colour_source(request, monkeypatch):
    """Grid interpolation (white maps to no ink, so the overrun quirk fires) or a random inverse LUT."""
    if request.param == "interp":
        monkeypatch.setattr(color_lut, "inverse_lut", lambda table=None: None)
    else:
        lut = np.random.default_rng(4150).integers(0, 256, color_lut._INVERSE_LUT_SHAPE, dtype=np.uint8)
        monkeypatch.setattr(color_lut, "inverse_lut", lambda table=None: lut)
    return request.param


def _fine_page(width: int, height: int, seed: int) -> bytes:
    """Noise, ramps ending in white, solid bars and rows that end in two equal or near-equal pixels."""
    rng = np.random.default_rng(seed)
    page = np.full((height, width, 3), 255, np.uint8)
    page[10:120] = rng.integers(0, 256, (110, width, 3), dtype=np.uint8)
    page[130:200] = (np.arange(width) * 256 // width).astype(np.uint8)[None, :, None]
    page[200:260, : width // 2] = np.linspace(0, 255, width // 2, dtype=np.uint8)[None, :, None]
    page[270:290] = (0, 200, 90)
    page[300:310, width - 7 :] = (254, 255, 255)  # ink only at the right edge
    page[320:340, 100 : width - 3] = (0, 0, 0)
    page[350:370, : width - 2] = (128, 128, 128)
    page[350:360, width - 2 :] = (250, 255, 255)  # last two equal, near white
    page[360:370, width - 2] = (250, 255, 255)  # last two differ, both white after brightening
    page[360:370, width - 1] = (251, 255, 255)
    page[380:390, : width // 3] = (0, 0, 255)  # coloured run, then white to the end
    return page.tobytes()


def _render(pages, settings, *, kernel: bool, threads: int, monkeypatch) -> bytes:
    monkeypatch.setattr(bands, "HAS_FINE_BAND_KERNEL", kernel)
    monkeypatch.setenv("BRHL4150CDN_RENDER_THREADS", str(threads))
    out = io.BytesIO()
    filter_pages(pages, settings, out)
    return out.getvalue()


_FINE = Resolution.FINE
_SETTINGS = {
    "baseline": PrintSettings(resolution=_FINE),
    "cm_none": PrintSettings(resolution=_FINE, color_matching=ColorMatching.NONE),
    "vivid": PrintSettings(resolution=_FINE, color_matching=ColorMatching.VIVID),
    "saturation_p10": PrintSettings(resolution=_FINE, saturation=10),
    "brightness_p20": PrintSettings(resolution=_FINE, brightness=20),
    "input_remap": PrintSettings(resolution=_FINE, brightness=-20, contrast=10, red=5, blue=-7),
    "toner_save": PrintSettings(resolution=_FINE, toner_save=True),
    "improve_gray": PrintSettings(resolution=_FINE, improve_gray=True),
    "glossy": PrintSettings(resolution=_FINE, media_type=MediaType.GLOSSY),
}


@pytest.mark.parametrize("name", list(_SETTINGS))
def test_fine_kernel_matches_per_line_path(name, fine_channels, colour_source, monkeypatch):
    settings = _SETTINGS[name]
    pages = [(_A4_W, 400, _fine_page(_A4_W, 400, seed=1))]
    expected = _render(pages, settings, kernel=False, threads=0, monkeypatch=monkeypatch)
    assert _render(pages, settings, kernel=True, threads=0, monkeypatch=monkeypatch) == expected
    assert _render(pages, settings, kernel=True, threads=3, monkeypatch=monkeypatch) == expected


@pytest.mark.parametrize(
    ("width", "page_size"),
    [
        (3001, PageSize.A4),  # narrower than the image: white padding on the right
        (5000, PageSize.A4),  # wider than the image: cut at sw
        (4900, PageSize.LETTER),  # image width rounded up to 4904
        (4904, PageSize.LETTER),  # overrun geometry at the padded width
    ],
)
def test_fine_kernel_matches_per_line_path_for_widths(width, page_size, fine_channels, colour_source, monkeypatch):
    settings = PrintSettings(resolution=_FINE, page_size=page_size)
    pages = [(width, 400, _fine_page(width, 400, seed=2))]
    expected = _render(pages, settings, kernel=False, threads=0, monkeypatch=monkeypatch)
    assert _render(pages, settings, kernel=True, threads=2, monkeypatch=monkeypatch) == expected


def test_fine_kernel_selection(fine_channels, monkeypatch):
    table = color_lut.DEFAULT_TABLE
    assert band_kernel_colour(table, fine_channels, is_fine=True) is not None
    monkeypatch.setattr(bands, "HAS_FINE_BAND_KERNEL", False)
    assert band_kernel_colour(table, fine_channels, is_fine=True) is None
    monkeypatch.setattr(bands, "HAS_FINE_BAND_KERNEL", True)
    # Bayer fallback channels (no BRCD tables) have no nibble tables: per-line path.
    assert band_kernel_colour(table, dither._build_bayer_tables(toner_save=False), is_fine=True) is None


def test_fine_kernel_rejects_bad_shapes(fine_channels):
    levels = [np.ascontiguousarray(fine_channels[c].nibble_levels) for c in "KCMY"]
    grid, weights, black = color_lut.interp_arrays(color_lut.DEFAULT_TABLE)
    rows = np.full((2, 30), 255, np.uint8)
    flags = np.zeros(2, np.uint8)
    lengths = np.empty((2, 4), np.int32)
    empty = np.empty(0, np.uint8)
    args = (rows, flags, flags, 10, 0, 10, empty, grid, weights, black)
    _fine_fast.render_band_fine(*args, *levels, lengths)
    assert lengths.tolist() == [[0, 0, 0, 0]] * 2
    with pytest.raises(ValueError, match="one entry per row"):
        _fine_fast.render_band_fine(rows, flags[:1], flags, *args[3:], *levels, lengths)
    with pytest.raises(ValueError, match="shorter than width"):
        _fine_fast.render_band_fine(rows, flags, flags, 11, *args[4:], *levels, lengths)
    with pytest.raises(ValueError, match=r"\(256, h, w\)"):
        _fine_fast.render_band_fine(*args, levels[0][:255], *levels[1:], lengths)
