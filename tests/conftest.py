"""Session-wide pytest fixtures."""

import pytest

from extract_fixtures import CaptureFixture
from fixture_utils import CAPTURE_NAMES, parse_capture


@pytest.fixture(scope="session")
def all_captures() -> dict[str, CaptureFixture]:
    """The parsed framing captures, keyed by name."""
    return {name: parse_capture(name) for name in CAPTURE_NAMES}


@pytest.fixture(scope="session")
def color_data_root(tmp_path_factory):
    """A colour data root with the HL-4150CDN's grids, for inverse LUTs written by the tests.

    Holds `hl4150cdn/` with symlinks to the extracted grids and
    interpolation tables (no inverse LUTs); see `use_color_data_root`.
    """
    from brother_xl2hb.models import HL4150CDN

    root = tmp_path_factory.mktemp("color_data")
    model_dir = root / HL4150CDN.key
    model_dir.mkdir()
    for blob in HL4150CDN.color_data_dir.glob("*.bin"):
        (model_dir / blob.name).symlink_to(blob)
    return root


@pytest.fixture
def use_color_data_root(monkeypatch):
    """Point every model's colour data at `root` for this test: `use_color_data_root(root)`.

    Tables built afterwards (`PrinterModel.grid`, `color_table`) carry the
    new directory, so the caches keyed by them never mix it up with the
    real one.
    """
    from brother_xl2hb.models import base

    def use(root):
        monkeypatch.setattr(base, "COLOR_DATA_ROOT", root)
        return root

    return use
