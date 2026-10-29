"""Every registered model is complete: its PPD, its CUPS filter shim and the scripts know it.

A model is more than its `PrinterModel`: CUPS needs `cups/br<key>.ppd` and
the shim `cups/br<key>-filter`, and `extract_blobs.sh`, `install.sh` and
`uninstall.sh` each have a block for it. These tests fail as soon as one
of them is missing or disagrees with the model's tables.
"""

import logging
import os
import re
from pathlib import Path

import pytest

from brother_xl2hb.models import MODELS
from brother_xl2hb.models.base import PrinterModel

_ROOT = Path(__file__).resolve().parent.parent
_MODELS = pytest.mark.parametrize("model", MODELS.values(), ids=MODELS.keys())

# PPD options whose every choice the model must accept as it is.
_CHECKED_OPTIONS = ("PageSize", "Duplex", "BRInputSlot", "BRMediaType", "BRMonoColor", "BRSkipBlank")


def _ppd_text(model: PrinterModel) -> str:
    return (_ROOT / "cups" / f"br{model.key}.ppd").read_text(encoding="latin-1")


def _choices(ppd: str, option: str) -> list[str]:
    """The choice keywords of `option` in the PPD, e.g. `A4` from `*PageSize A4/A4: "..."`."""
    return re.findall(rf"^\*{option} ([^/:\s]+)", ppd, re.MULTILINE)


def _paper_dimensions(ppd: str) -> dict[str, tuple[int, int]]:
    """PaperDimension of every PageSize in the PPD, in points."""
    dims = re.findall(r'^\*PaperDimension ([^/:\s]+)[^:]*:\s*"([\d.]+) ([\d.]+)"', ppd, re.MULTILINE)
    return {name: (round(float(w)), round(float(h))) for name, w, h in dims}


@pytest.mark.parametrize("key", MODELS)
def test_registry_key_is_the_models_key(key):
    assert MODELS[key].key == key


@_MODELS
def test_ppd_page_sizes_match_the_model(model):
    """The PPD offers exactly the model's page sizes, at the size Ghostscript renders them."""
    ppd = _ppd_text(model)
    assert set(_choices(ppd, "PageSize")) == set(model.paper_sizes)
    assert _paper_dimensions(ppd) == dict(model.paper_points)


@_MODELS
@pytest.mark.parametrize("option", _CHECKED_OPTIONS)
def test_every_ppd_choice_is_supported(model, option, caplog):
    """Each choice the PPD offers parses without a warning and stays as chosen."""
    choices = _choices(_ppd_text(model), option)
    assert choices, f"{option} missing from the PPD"
    for choice in choices:
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            model.settings_from_cups_options(f"{option}={choice}")
        assert not caplog.records, f"{option}={choice}: {caplog.text}"


@_MODELS
def test_cups_filter_shim_runs_the_model(model):
    shim = _ROOT / "cups" / f"br{model.key}-filter"
    assert shim.is_file()
    assert os.access(shim, os.X_OK)
    assert f'main(MODELS["{model.key}"])' in shim.read_text()


@_MODELS
@pytest.mark.parametrize("script", ["scripts/extract_blobs.sh", "cups/install.sh", "cups/uninstall.sh"])
def test_scripts_know_the_model(model, script):
    """Each script has a `case` block for the model."""
    assert re.search(rf"^\s*{model.key}\)", (_ROOT / script).read_text(), re.MULTILINE)
