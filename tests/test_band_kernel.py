"""Native band kernel (+ render threads): byte-identity vs. the per-line path."""

import io

import numpy as np
import pytest

import color_lut
import pipeline
from fixture_utils import read_fixture
from pipeline import filter_duplex_pages
from settings import ColorMatching, DuplexMode, PrintSettings
from test_full_pipeline import _SETTING_VARIANTS_PASS, _duplex_test_pages, _run_settings_variant

if not pipeline.HAS_BAND_KERNEL:
    pytest.skip("_band_fast extension not built", allow_module_level=True)

_A4_W, _A4_H = 4760, 6812


@pytest.fixture(scope="module")
def random_lut() -> np.ndarray:
    """A random inverse LUT: every colour, white included, maps to some ink."""
    rng = np.random.default_rng(4150)
    return rng.integers(0, 256, color_lut._INVERSE_LUT_SHAPE, dtype=np.uint8)


@pytest.fixture(autouse=True)
def _use_random_lut(request, monkeypatch, random_lut) -> None:
    if "real_lut" not in request.fixturenames:
        monkeypatch.setattr(color_lut, "_load_inverse_lut", lambda: random_lut)


@pytest.fixture(scope="module")
def real_lut_path(tmp_path_factory):
    return color_lut.write_inverse_lut(tmp_path_factory.mktemp("lut") / "inverse_lut.npy")


@pytest.fixture
def real_lut(monkeypatch, real_lut_path):
    """The real inverse LUT, as installed by install.sh."""
    monkeypatch.setattr(color_lut, "INVERSE_LUT_PATH", real_lut_path)
    color_lut._load_inverse_lut.cache_clear()
    yield
    color_lut._load_inverse_lut.cache_clear()


def _page(width: int, height: int, seed: int) -> bytes:
    """Noise, gradients, solid bars, near-white and pure-white rows."""
    rng = np.random.default_rng(seed)
    page = np.full((height, width, 3), 255, np.uint8)
    page[40:400] = rng.integers(0, 256, (360, width, 3), dtype=np.uint8)
    page[500:900, : width // 2] = np.linspace(0, 255, width // 2, dtype=np.uint8)[None, :, None]
    page[1000:1100] = (0, 200, 90)
    page[1200:1210, width - 7 :] = (254, 255, 255)  # ink only at the right edge
    page[1300:1400, 100:3000] = (0, 0, 0)
    page[1500:1600] = (128, 128, 128)
    return page.tobytes()


def _render(pages, settings, *, kernel: bool, threads: int, monkeypatch) -> bytes:
    monkeypatch.setattr(pipeline, "HAS_BAND_KERNEL", kernel)
    monkeypatch.setenv("BRHL4150CDN_RENDER_THREADS", str(threads))
    out = io.BytesIO()
    filter_duplex_pages(pages, settings, out)
    return out.getvalue()


_SETTINGS = {
    "baseline": PrintSettings(),
    "cm_none": PrintSettings(color_matching=ColorMatching.NONE),
    "vivid": PrintSettings(color_matching=ColorMatching.VIVID),
    "saturation_p10": PrintSettings(saturation=10),
    "saturation_n15": PrintSettings(saturation=-15),
    "input_remap": PrintSettings(brightness=-20, contrast=10, red=5, blue=-7),
    "saturation_remap": PrintSettings(saturation=5, brightness=12),
    "gamma": PrintSettings(gamma_select=1),  # tone curve, no white-row short cut
    "gamma_brightness": PrintSettings(gamma_select=0, brightness=10, contrast=-5),
    "toner_save": PrintSettings(toner_save=True),
}


@pytest.mark.parametrize("name", list(_SETTINGS))
def test_kernel_matches_per_line_path(name, monkeypatch):
    settings = _SETTINGS[name]
    pages = [(_A4_W, 1700, _page(_A4_W, 1700, seed=1))]
    expected = _render(pages, settings, kernel=False, threads=0, monkeypatch=monkeypatch)
    assert _render(pages, settings, kernel=True, threads=0, monkeypatch=monkeypatch) == expected
    assert _render(pages, settings, kernel=True, threads=3, monkeypatch=monkeypatch) == expected


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (3001, 1700),  # narrower than the image: white padding on the right
        (5000, 1650),  # wider than the image: cut at sw
        (_A4_W, _A4_H + 40),  # taller than the paper
    ],
)
def test_kernel_matches_per_line_path_for_page_sizes(width, height, monkeypatch):
    pages = [(width, height, _page(width, height, seed=2))]
    expected = _render(pages, PrintSettings(), kernel=False, threads=0, monkeypatch=monkeypatch)
    assert _render(pages, PrintSettings(), kernel=True, threads=2, monkeypatch=monkeypatch) == expected


def test_kernel_matches_for_long_edge_back_pages_and_streamed_blocks(monkeypatch):
    """Back pages are reversed views (negative row stride); streamed blocks are cropped views."""
    settings = PrintSettings(duplex=DuplexMode.NO_TUMBLE)
    raw = [_page(_A4_W, _A4_H, seed=s) for s in (3, 4)]
    pages = [(_A4_W, _A4_H, data) for data in raw]
    expected = _render(pages, settings, kernel=False, threads=0, monkeypatch=monkeypatch)

    def cropped_blocks(data: bytes):
        # Printable window of a wider render, as page_stream yields it.
        wide = np.full((_A4_H, _A4_W + 200, 3), 7, np.uint8)
        wide[:, 100 : 100 + _A4_W] = np.frombuffer(data, np.uint8).reshape(_A4_H, _A4_W, 3)
        rows = wide.reshape(_A4_H, -1)[:, 300 : 300 + _A4_W * 3]
        return (rows[i : i + 97] for i in range(0, _A4_H, 97))

    streamed = [(_A4_W, _A4_H, cropped_blocks(data)) for data in raw]
    assert _render(streamed, settings, kernel=True, threads=3, monkeypatch=monkeypatch) == expected


def test_kernel_reports_short_pages(monkeypatch):
    monkeypatch.setenv("BRHL4150CDN_RENDER_THREADS", "2")
    rows = np.zeros((300, _A4_W * 3), np.uint8)
    with pytest.raises(ValueError, match="page ended after 300 rows"):
        filter_duplex_pages([(_A4_W, 500, iter([rows]))], PrintSettings(), io.BytesIO())


def test_fine_mode_keeps_per_line_path(monkeypatch):
    from settings import Resolution

    channels = pipeline._init_channels(PrintSettings(resolution=Resolution.FINE))
    assert pipeline._band_kernel_lut(PrintSettings(), channels, is_fine=True) is None
    assert pipeline._band_kernel_lut(PrintSettings(), channels, is_fine=False) is not None


def test_render_threads_from_environment(monkeypatch):
    monkeypatch.setenv("BRHL4150CDN_RENDER_THREADS", "0")
    assert pipeline._render_threads() == 0
    monkeypatch.delenv("BRHL4150CDN_RENDER_THREADS")
    assert 1 <= pipeline._render_threads() <= 3


@pytest.mark.parametrize(("name", "settings"), _SETTING_VARIANTS_PASS)
def test_kernel_matches_brother_captures(name, settings, real_lut, monkeypatch):
    """With the real inverse LUT the banded render is byte-exact against brhl4150cdnfilter."""
    assert pipeline._band_kernel_lut(settings, pipeline._init_channels(settings), is_fine=False) is not None
    monkeypatch.setenv("BRHL4150CDN_RENDER_THREADS", "3")
    _run_settings_variant(name, settings)


def test_kernel_matches_brother_duplex_capture(real_lut, monkeypatch):
    expected = read_fixture("duplex4_long_edge.xl2hb")
    if expected is None:
        pytest.skip("duplex4_long_edge.xl2hb not available")
    settings = PrintSettings(duplex=DuplexMode.NO_TUMBLE)
    assert _render(_duplex_test_pages(), settings, kernel=True, threads=3, monkeypatch=monkeypatch) == expected
