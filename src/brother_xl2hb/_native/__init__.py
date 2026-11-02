"""Optional native kernels (Rust or Cython); every caller falls back to pure Python without them.

Callers get an extension through :func:`load`, keep it in a module
attribute named after it (``_rle_fast = _native.load("_rle_fast")``) and
check that attribute for None when they are called, so setting it to None
(e.g. with ``monkeypatch.setattr``) switches that caller to its fallback.

The Rust build (``_rust``, see ``rust/``) holds the kernels ported so far
as submodules named like the Cython extensions. :func:`load` takes a
kernel from it first, then from Cython. `BRXL2HB_BACKEND` narrows that:
``cython`` skips Rust, ``python`` skips both (as does
``BRXL2HB_PURE_PYTHON=1``). Both are read when the callers are imported.
"""

import functools
import importlib
import logging
import os
from types import ModuleType

# Every extension built by setup_cython.py.
MODULES = ("_rle_fast", "_dither_fast", "_color_fast", "_band_fast", "_fine_fast")

PURE_PYTHON_VAR = "BRXL2HB_PURE_PYTHON"
BACKEND_VAR = "BRXL2HB_BACKEND"

# The native backends `load` tries, by BRXL2HB_BACKEND value, in order.
_BACKEND_CHAIN = {"auto": ("rust", "cython"), "cython": ("cython",), "python": ()}

REBUILD_HINT = (
    "rerun cups/install.sh, or in a checkout "
    "`uv run --with cython,setuptools python setup_cython.py build_ext --inplace`"
)

logger = logging.getLogger(__name__)

# Why an extension is not in use, by name; filled by `load`.
_unavailable: dict[str, str] = {}
# Which backend an extension in use came from, by name; filled by `load`.
_backend: dict[str, str] = {}
_DISABLED = f"disabled by {PURE_PYTHON_VAR}"
_DISABLED_BACKEND = f"disabled by {BACKEND_VAR}=python"


def pure_python_forced() -> bool:
    """Whether `BRXL2HB_PURE_PYTHON` asks for the pure-Python fallbacks.

    Returns:
        True unless the variable is unset, empty or "0".
    """
    return os.environ.get(PURE_PYTHON_VAR, "").strip() not in ("", "0")


def backend_chain() -> tuple[str, ...]:
    """The native backends `BRXL2HB_BACKEND` allows, in the order :func:`load` tries them.

    Returns:
        A subset of ("rust", "cython"); empty for pure Python.

    Raises:
        ValueError: If `BRXL2HB_BACKEND` is not auto, cython or python.
    """
    if pure_python_forced():
        return ()
    value = os.environ.get(BACKEND_VAR, "").strip().lower() or "auto"
    if value not in _BACKEND_CHAIN:
        msg = f"{BACKEND_VAR}={value!r}, expected one of {', '.join(_BACKEND_CHAIN)}"
        raise ValueError(msg)
    return _BACKEND_CHAIN[value]


@functools.cache
def rust_build() -> ModuleType | str:
    """Import the Rust build once.

    Returns:
        The module, or why it does not import.
    """
    name = f"{__name__}._rust"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        return "not built" if exc.name == name else str(exc)
    except ImportError as exc:
        return str(exc)


@functools.cache
def load(name: str) -> ModuleType | None:
    """Import the extension `name` once, from the first backend that has it.

    Returns:
        The module, or None if no allowed backend provides it.

    Raises:
        ValueError: If `name` is not one of `MODULES`.
    """
    if name not in MODULES:
        msg = f"Unknown native module {name!r}, expected one of {MODULES}"
        raise ValueError(msg)
    chain = backend_chain()
    if not chain:
        _unavailable[name] = _DISABLED if pure_python_forced() else _DISABLED_BACKEND
        return None
    reasons = []
    for backend in chain:
        if backend == "rust":
            rust = rust_build()
            module = getattr(rust, name, None) if isinstance(rust, ModuleType) else None
            if module is None:
                reasons.append(f"rust: {rust if isinstance(rust, str) else 'not ported'}")
                continue
        else:
            try:
                module = importlib.import_module(f"{__name__}.{name}")
            except ImportError as exc:
                reasons.append(f"cython: {exc}")
                continue
        _backend[name] = backend
        return module
    _unavailable[name] = "; ".join(reasons)
    return None


def backend(name: str) -> str:
    """Which backend runs the kernel `name`.

    Returns:
        "rust", "cython" or "python".
    """
    return _backend.get(name, "python") if load(name) is not None else "python"


def missing() -> list[str]:
    """Extensions that failed to import.

    Returns:
        Their names in `MODULES` order; those disabled by `BRXL2HB_PURE_PYTHON` are not missing.
    """
    return [name for name in MODULES if load(name) is None and _unavailable[name] not in (_DISABLED, _DISABLED_BACKEND)]


def summary() -> str:
    """Say which kernels are in use, from which backend, and why the others are not.

    Returns:
        One line for the log.
    """
    absent = [name for name in MODULES if load(name) is None]
    rust = [name for name in MODULES if backend(name) == "rust"]
    if not absent:
        source = f" (Rust: {', '.join(rust)}; Cython: the rest)" if rust else ""
        return f"native kernels: all {len(MODULES)} loaded{source}"
    reasons = "; ".join(f"{name}: {_unavailable[name]}" for name in absent)
    if len(absent) == len(MODULES):
        return f"native kernels not in use, pure-Python fallbacks instead ({reasons})"
    cython = [name for name in MODULES if backend(name) == "cython"]
    by_backend = (("Rust", rust), ("Cython", cython))
    sources = "; ".join(f"{label}: {', '.join(names)}" for label, names in by_backend if names)
    return f"native kernels partly in use ({sources}), pure-Python fallbacks for the rest ({reasons})"


def log_status() -> None:
    """Log :func:`summary` once per job: a WARNING if an extension failed to import, INFO otherwise."""
    if missing():
        logger.warning("%s. Pages render much slower; to rebuild, %s.", summary(), REBUILD_HINT)
    else:
        logger.info("%s", summary())
