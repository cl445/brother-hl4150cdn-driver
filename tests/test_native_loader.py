"""The native-kernel loader: availability, BRXL2HB_PURE_PYTHON, the per-job log line."""

import logging
import os
import subprocess
import sys

import pytest

from brother_xl2hb import _native


@pytest.fixture
def fresh_loader(monkeypatch):
    """Let `_native.load` import again in this test; the cache is restored afterwards."""
    monkeypatch.setattr(_native, "_unavailable", {})
    _native.load.cache_clear()
    yield
    _native.load.cache_clear()


def test_unknown_module_is_rejected() -> None:
    with pytest.raises(ValueError, match="_nope_fast"):
        _native.load("_nope_fast")


@pytest.mark.parametrize(("value", "forced"), [("", False), ("0", False), ("1", True), ("yes", True)])
def test_pure_python_var(monkeypatch, value, forced) -> None:
    monkeypatch.setenv(_native.PURE_PYTHON_VAR, value)
    assert _native.pure_python_forced() is forced


def test_disabled_kernels_are_not_missing(fresh_loader, monkeypatch, caplog) -> None:
    monkeypatch.setenv(_native.PURE_PYTHON_VAR, "1")
    assert all(_native.load(name) is None for name in _native.MODULES)
    assert _native.missing() == []
    with caplog.at_level(logging.INFO, logger=_native.__name__):
        _native.log_status()
    [record] = caplog.records
    assert record.levelno == logging.INFO
    assert f"_fine_fast: disabled by {_native.PURE_PYTHON_VAR}" in record.getMessage()


def test_failed_import_is_a_warning(fresh_loader, monkeypatch, caplog) -> None:
    monkeypatch.delenv(_native.PURE_PYTHON_VAR, raising=False)
    real_import = _native.importlib.import_module

    def import_module(name: str):
        if name.endswith("._fine_fast"):
            msg = "built for another Python"
            raise ImportError(msg)
        return real_import(name)

    monkeypatch.setattr(_native.importlib, "import_module", import_module)
    assert _native.load("_fine_fast") is None
    assert "_fine_fast" in _native.missing()
    with caplog.at_level(logging.INFO, logger=_native.__name__):
        _native.log_status()
    [record] = caplog.records
    assert record.levelno == logging.WARNING
    message = record.getMessage()
    assert "_fine_fast: built for another Python" in message
    assert "setup_cython.py build_ext --inplace" in message
    assert "cups/install.sh" in message


def test_pure_python_var_switches_every_caller() -> None:
    """With the variable set at import time, no caller holds a native module."""
    code = (
        "from brother_xl2hb.codec import fine_encoder, plane_encoders, rle\n"
        "from brother_xl2hb.color import color_lut, dither, saturation\n"
        "from brother_xl2hb.render import bands\n"
        "held = [rle._rle_fast, plane_encoders._rle_fast, fine_encoder._fine_fast, color_lut._color_fast,\n"
        "        dither._dither_fast, dither._fine_fast, bands._band_fast, bands._fine_fast,\n"
        "        saturation._color_fast]\n"
        "assert held == [None] * len(held), held\n"
    )
    env = {**os.environ, _native.PURE_PYTHON_VAR: "1"}
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


def test_stale_extensions(tmp_path) -> None:
    from importlib.machinery import EXTENSION_SUFFIXES

    from conftest import stale_extensions

    def at(name: str, mtime: int) -> None:
        path = tmp_path / name
        path.touch()
        os.utime(path, (mtime, mtime))

    rle, band = f"_rle_fast{EXTENSION_SUFFIXES[0]}", f"_band_fast{EXTENSION_SUFFIXES[0]}"
    for name in ("_rle_fast.pyx", "_rle_fast.pxd", rle, "_band_fast.pyx", band, "_dither_fast.pyx"):
        at(name, 1000)
    (tmp_path / "_band_fast.pyx").write_text("from brother_xl2hb._native._rle_fast cimport OutBuf\n")
    os.utime(tmp_path / "_band_fast.pyx", (1000, 1000))
    assert stale_extensions(tmp_path) == []  # _dither_fast is not built: nothing to compare
    at("_rle_fast.pxd", 2000)  # cimported by _band_fast
    assert stale_extensions(tmp_path) == [band, rle]
    at(rle, 3000)
    at(band, 3000)
    assert stale_extensions(tmp_path) == []
    at("_band_fast.pyx", 4000)
    assert stale_extensions(tmp_path) == [band]
