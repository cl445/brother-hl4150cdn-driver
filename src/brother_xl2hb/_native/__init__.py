"""Optional Cython kernels; every caller falls back to pure Python without them.

Callers get an extension through :func:`load`, keep it in a module
attribute named after it (``_rle_fast = _native.load("_rle_fast")``) and
check that attribute for None when they are called, so setting it to None
(e.g. with ``monkeypatch.setattr``) switches that caller to its fallback.

Setting ``BRXL2HB_PURE_PYTHON=1`` makes :func:`load` return None for every
extension. It is read when the callers are imported.
"""

import functools
import importlib
import logging
import os
from types import ModuleType

# Every extension built by setup_cython.py.
MODULES = ("_rle_fast", "_dither_fast", "_color_fast", "_band_fast", "_fine_fast")

PURE_PYTHON_VAR = "BRXL2HB_PURE_PYTHON"

REBUILD_HINT = (
    "rerun cups/install.sh, or in a checkout "
    "`uv run --with cython,setuptools python setup_cython.py build_ext --inplace`"
)

logger = logging.getLogger(__name__)

# Why an extension is not in use, by name; filled by `load`.
_unavailable: dict[str, str] = {}
_DISABLED = f"disabled by {PURE_PYTHON_VAR}"


def pure_python_forced() -> bool:
    """Whether `BRXL2HB_PURE_PYTHON` asks for the pure-Python fallbacks.

    Returns:
        True unless the variable is unset, empty or "0".
    """
    return os.environ.get(PURE_PYTHON_VAR, "").strip() not in ("", "0")


@functools.cache
def load(name: str) -> ModuleType | None:
    """Import the extension `name` once.

    Returns:
        The module, or None if it does not import or `BRXL2HB_PURE_PYTHON` is set.

    Raises:
        ValueError: If `name` is not one of `MODULES`.
    """
    if name not in MODULES:
        msg = f"Unknown native module {name!r}, expected one of {MODULES}"
        raise ValueError(msg)
    if pure_python_forced():
        _unavailable[name] = _DISABLED
        return None
    try:
        return importlib.import_module(f"{__name__}.{name}")
    except ImportError as exc:
        _unavailable[name] = str(exc)
        return None


def missing() -> list[str]:
    """Extensions that failed to import.

    Returns:
        Their names in `MODULES` order; those disabled by `BRXL2HB_PURE_PYTHON` are not missing.
    """
    return [name for name in MODULES if load(name) is None and _unavailable[name] != _DISABLED]


def summary() -> str:
    """Say which kernels are in use, and why the others are not.

    Returns:
        One line for the log.
    """
    absent = [name for name in MODULES if load(name) is None]
    if not absent:
        return f"native kernels: all {len(MODULES)} loaded"
    reasons = "; ".join(f"{name}: {_unavailable[name]}" for name in absent)
    return f"native kernels not in use, pure-Python fallbacks instead ({reasons})"


def log_status() -> None:
    """Log :func:`summary` once per job: a WARNING if an extension failed to import, INFO otherwise."""
    if missing():
        logger.warning("%s. Pages render much slower; to rebuild, %s.", summary(), REBUILD_HINT)
    else:
        logger.info("%s", summary())
