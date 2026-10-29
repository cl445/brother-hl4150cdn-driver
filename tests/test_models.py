"""PrinterModel: the generic geometry and lookups, and the registered models' tables."""

import dataclasses
import io
import logging

import numpy as np
import pytest

from brother_xl2hb import cli
from brother_xl2hb.job import filter_pages
from brother_xl2hb.models import DEFAULT_MODEL, HL4150CDN, MODELS
from brother_xl2hb.models.base import PageSend
from brother_xl2hb.protocol.xl2hb import XL2HBWriter
from brother_xl2hb.render.page import JobContext, render_page
from brother_xl2hb.settings import DuplexMode, InputSlot, MediaType, PageSize, PrintSettings


@pytest.mark.parametrize("model", MODELS.values(), ids=MODELS.keys())
def test_tables_are_consistent(model):
    """Every supported page size has its geometry and code; every table key is a valid setting."""
    sizes = set(model.paper_sizes)
    assert sizes <= {size.value for size in PageSize}
    assert set(model.paper_points) == sizes
    assert set(model.media_size) == sizes
    if model.imaging_height_pt is not None:
        assert set(model.imaging_height_pt) == sizes
    assert set(model.media_types) <= {media.value for media in MediaType}
    assert set(model.media_source) <= set(InputSlot)
    assert DuplexMode.NONE in model.duplex_page_mode
    for table in model.aliases.values():
        assert all(isinstance(value, str) for value in table.values())


def test_hl4150cdn_supports_every_setting():
    assert set(HL4150CDN.paper_sizes) == {size.value for size in PageSize}
    assert set(HL4150CDN.media_types) == {media.value for media in MediaType}
    assert set(HL4150CDN.media_source) == set(InputSlot)
    assert set(HL4150CDN.duplex_page_mode) == set(DuplexMode)
    assert HL4150CDN.skip_blank
    settings = PrintSettings(page_size=PageSize.ENV_YOU4, duplex=DuplexMode.TUMBLE, skip_blank=True)
    assert HL4150CDN.restrict(settings) is settings


def test_restrict_replaces_unsupported_choices(caplog):
    simplex = dataclasses.replace(
        HL4150CDN,
        name="Simplex",
        duplex_page_mode={DuplexMode.NONE: None},
        media_source={InputSlot.AUTO: 1, InputSlot.TRAY1: 1001},
        skip_blank=False,
    )
    settings = PrintSettings(duplex=DuplexMode.NO_TUMBLE, input_slot=InputSlot.TRAY2, skip_blank=True, copies=3)
    with caplog.at_level(logging.WARNING, logger="brother_xl2hb.models.base"):
        restricted = simplex.restrict(settings)
    assert restricted.duplex == DuplexMode.NONE
    assert restricted.input_slot == InputSlot.AUTO
    assert restricted.skip_blank is False
    assert restricted.copies == 3
    assert settings.duplex == DuplexMode.NO_TUMBLE  # the caller's settings stay as they were
    assert "Simplex does not support duplex=DuplexNoTumble" in caplog.text


@pytest.mark.parametrize(
    ("option", "attr", "expected"),
    [
        ("BRMediaType=Env", "media_type", MediaType.ENVELOPE),
        ("BRMediaType=env", "media_type", MediaType.ENVELOPE),
        ("PageSize=A5Rotated", "page_size", PageSize.A5_ROTATED),
        ("PageSize=B5", "page_size", PageSize.ISOB5),
        ("PageSize=B6", "page_size", PageSize.ISOB6),
    ],
)
def test_hl4150cdn_spellings(option, attr, expected):
    assert getattr(HL4150CDN.settings_from_cups_options(option), attr) == expected


def test_hl4150cdn_spellings_in_the_rc_file(tmp_path):
    rc = tmp_path / "brhl4150cdnrc"
    rc.write_text("[hl4150cdn]\nMediaType=Env\nPageSize=B5\n")
    settings = HL4150CDN.settings_from_rc_file(str(rc))
    assert settings.media_type == MediaType.ENVELOPE
    assert settings.page_size == PageSize.ISOB5


def test_without_the_model_brothers_spellings_are_unknown():
    assert PrintSettings.from_cups_options("PageSize=B5").page_size == PageSize.A4


def test_default_model_is_the_hl4150cdn():
    assert MODELS["hl4150cdn"] is HL4150CDN
    assert DEFAULT_MODEL is HL4150CDN


@pytest.mark.parametrize(
    ("page_size", "fine", "expected"),
    [
        ("A4", False, (4768, 6808)),  # width to 32 px, height capped by the ImagingArea (818 pt)
        ("A4", True, (4760, 6808)),  # Fine: width to 8 px
        ("Letter", True, (4904, 6400)),
        ("EnvDL", False, (2400, 4991)),  # paper height in whole points (599) below the imaging height
    ],
)
def test_hl4150cdn_image_dimensions(page_size, fine, expected):
    assert HL4150CDN.image_dimensions(page_size, fine=fine) == expected


def test_image_height_without_imaging_area_is_the_paper_height_in_whole_points():
    model = dataclasses.replace(HL4150CDN, imaging_height_pt=None)
    for page_size, (_, paper_h) in model.paper_sizes.items():
        assert model.image_height(page_size) == paper_h * 72 // 600 * 600 // 72


@pytest.mark.parametrize(
    ("duplex", "reverse", "index", "flipped"),
    [
        (DuplexMode.NONE, False, 1, False),
        (DuplexMode.NONE, True, 0, True),
        (DuplexMode.NO_TUMBLE, False, 0, False),
        (DuplexMode.NO_TUMBLE, False, 1, True),  # long-edge back side
        (DuplexMode.NO_TUMBLE, True, 1, False),  # back side and reverse cancel out
        (DuplexMode.NO_TUMBLE, True, 2, True),
        (DuplexMode.TUMBLE, False, 1, False),
    ],
)
def test_hl4150cdn_sends_a_single_copy_once(duplex, reverse, index, flipped):
    settings = PrintSettings(duplex=duplex, reverse=reverse)
    assert HL4150CDN.page_sends(settings, index) == (PageSend(flip=flipped, copies=1),)


def test_a_page_sent_twice_is_rendered_twice_from_streamed_rows():
    """A model may send a page more than once; streamed row blocks are collected first."""
    sends = (PageSend(flip=False, copies=1), PageSend(flip=True, copies=2))
    twice = dataclasses.replace(HL4150CDN, page_sends=lambda settings, index: sends)
    width, height = 64, HL4150CDN.paper_sizes["Postcard"][1]
    rows = np.random.default_rng(0).integers(0, 256, (height, width * 3), dtype=np.uint8)
    blocks = (rows[i : i + 100] for i in range(0, height, 100))
    settings = PrintSettings(page_size=PageSize.POSTCARD)

    out = io.BytesIO()
    filter_pages([(width, height, blocks)], settings, out, twice)

    job = JobContext.create(settings, HL4150CDN)
    expected = []
    for send in sends:
        page = io.BytesIO()
        render_page(XL2HBWriter(page), (width, height, rows), job, send)
        expected.append(page.getvalue())
    assert expected[0] != expected[1]
    stream = out.getvalue()
    assert expected[0] + expected[1] in stream


def test_unknown_names_fall_back_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="brother_xl2hb.models.base"):
        assert HL4150CDN.media_size_code("Folio") == 2
        assert HL4150CDN.media_type_code("Cardstock") == b"dRegular"
    assert "Unknown media size" in caplog.text
    assert "Unknown media type" in caplog.text


def test_cli_rejects_an_unknown_model(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--model", "hl9999"])
    assert "invalid choice" in capsys.readouterr().err


def test_missing_grids_are_an_error_without_generated_fallback(tmp_path, use_color_data_root):
    """Only a model the generated grids approximate prints without its extracted grids."""
    from brother_xl2hb.color import color_lut

    use_color_data_root(tmp_path)  # no grids for any model
    other = dataclasses.replace(HL4150CDN, generated_grids=False)
    with pytest.raises(FileNotFoundError, match=r"run scripts/extract_blobs\.sh hl4150cdn"):
        color_lut.interp_arrays(other.grid())
    lut, _, _ = color_lut.interp_arrays(HL4150CDN.grid())
    assert lut.size == 4913 * 4
