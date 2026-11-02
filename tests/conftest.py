"""Session-wide pytest fixtures, and a check that the built native kernels are current."""

import contextlib
import importlib
import re
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

import pytest
from _pytest.mark import ParameterSet

from brother_xl2hb import _native
from extract_fixtures import CaptureFixture
from fixture_utils import CAPTURE_NAMES, parse_capture

_NATIVE_DIR = Path(__file__).parent.parent / "src" / "brother_xl2hb" / "_native"
_RUST_CRATE = Path(__file__).parent.parent / "rust"


_CIMPORT = re.compile(r"^from brother_xl2hb\._native\.(\w+) cimport", re.MULTILINE)


def stale_extensions(native_dir: Path = _NATIVE_DIR) -> list[str]:
    """Extensions built for this Python that are older than their sources.

    The sources of `_x` are `_x.pyx`, `_x.pxd` and the `.pxd` of every
    module it cimports from `_native`. The tests would otherwise check an
    old build against the pure-Python reference and miss what the `.pyx`
    now does. Only the file Python imports counts: the first one in
    `EXTENSION_SUFFIXES` order.

    Returns:
        File names of the stale extensions.
    """
    stale = []
    for pyx in sorted(native_dir.glob("*.pyx")):
        names = {pyx.stem, *_CIMPORT.findall(pyx.read_text())}
        sources = [pyx, *(native_dir / f"{name}.pxd" for name in names)]
        source_mtime = max(path.stat().st_mtime for path in sources if path.exists())
        built = next(
            (path for suffix in EXTENSION_SUFFIXES if (path := pyx.with_name(pyx.stem + suffix)).exists()), None
        )
        if built is not None and built.stat().st_mtime < source_mtime:
            stale.append(built.name)
    return stale


def rust_build_is_stale(native_dir: Path = _NATIVE_DIR, crate: Path = _RUST_CRATE) -> bool:
    """Whether the Rust build exists and is older than a file of its crate (Cargo.toml, Cargo.lock, build.rs, src/).

    Returns:
        True if `_rust.abi3.so` needs a rebuild.
    """
    built = native_dir / "_rust.abi3.so"
    if not built.exists():
        return False
    sources = [crate / "Cargo.toml", crate / "Cargo.lock", crate / "build.rs", *(crate / "src").rglob("*.rs")]
    return any(path.stat().st_mtime > built.stat().st_mtime for path in sources if path.exists())


def native_variants(name: str) -> list[ParameterSet]:
    """Every built implementation of the native kernel `name`, for a parametrized fixture.

    Empty when BRXL2HB_BACKEND or BRXL2HB_PURE_PYTHON turn the native kernels off.

    Returns:
        pytest.param(module, id="rust" | "cython") for the Rust and the Cython build, as far as they exist.
    """
    if not _native.backend_chain():
        return []
    variants = []
    rust = _native.rust_build()
    if not isinstance(rust, str) and hasattr(rust, name):
        variants.append(pytest.param(getattr(rust, name), id="rust"))
    with contextlib.suppress(ImportError):
        variants.append(pytest.param(importlib.import_module(f"brother_xl2hb._native.{name}"), id="cython"))
    return variants


def pytest_sessionstart(session: pytest.Session) -> None:
    """Stop before testing a native kernel that was not rebuilt after its source changed."""
    if stale := stale_extensions():
        msg = (
            f"native extensions older than their .pyx/.pxd: {', '.join(stale)}; rebuild with "
            "`uv run --with cython,setuptools python setup_cython.py build_ext --inplace`"
        )
        raise pytest.UsageError(msg)
    if rust_build_is_stale():
        msg = "_rust.abi3.so is older than the crate in rust/; rebuild with `./scripts/build_rust.sh`"
        raise pytest.UsageError(msg)


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


@pytest.fixture(scope="session")
def installed_inverse_luts(color_data_root):
    """`color_data_root` with all the HL-4150CDN's inverse LUTs, as install.sh writes them.

    Building them takes about 20 s, so the session does it once for every
    test that needs the real LUTs.
    """
    from brother_xl2hb.color.color_lut import write_inverse_luts
    from brother_xl2hb.models import HL4150CDN

    write_inverse_luts(HL4150CDN.inverse_lut_tables, color_data_root / HL4150CDN.key)
    return color_data_root


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
