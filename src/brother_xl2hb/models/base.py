"""`PrinterModel`: what one printer's filter writes, as data plus a few hooks.

The XL2HB protocol, the codecs and the rendering are shared; a model
supplies its paper geometry, the BeginPage codes for media and trays,
its dither table set and the decisions that differ between Brother's
filters (colour grid, PJL, which pages go bottom-up).
"""

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from brother_xl2hb.blobs import COLOR_DATA_ROOT, LUT_ROOT
from brother_xl2hb.color.color_lut import ColorTable, Profile
from brother_xl2hb.settings import DuplexMode, InputSlot, PrintSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class PrinterModel:
    """One printer model, as its manufacturer's filter drives it.

    Attributes:
        key: short name for the CLI, file names and blob directories, e.g.
            `hl4150cdn`.
        name: model name, e.g. `HL-4150CDN`.
        paper_sizes: paper size in pixels at 600 dpi (paperinfij2), keyed
            by PPD PageSize.
        imaging_height_pt: printable height in points (ury - lly of the
            ImagingArea file) where the model has one; it caps the image
            height (see `image_height`).
        paper_points: the PPD's PaperDimension in points, the page size
            Ghostscript renders.
        page_origin: SetPageOrigin in pixels; also where the printable area
            starts in Ghostscript's full-page render.
        media_size: BeginPage MediaSize, a PCL XL enum value or a name string.
        media_types: BeginPage MediaType string per PPD media type.
        media_source: BeginPage MediaSource per input slot.
        duplex_page_mode: DuplexPageMode per duplex setting; None writes
            SimplexPageMode.
        dither_cache: suffix of the BRCD dither files, e.g. `cache09` for
            `0600-k_cache09.bin`.
        inverse_lut_grids: (profile, improve_gray) of the default-variant
            grids that get a precomputed inverse LUT (cups/install.sh).
        color_table: the colour grid the filter selects for the settings;
            built with `grid`.
        pjl_header: PJL before the XL2HB stream.
        pjl_footer: PJL after it.
        page_is_flipped: whether a page is sent bottom-up with the
            back-side marker; `back_side` is the second page of a duplex
            sheet.
    """

    key: str
    name: str
    paper_sizes: Mapping[str, tuple[int, int]]
    imaging_height_pt: Mapping[str, int] | None
    paper_points: Mapping[str, tuple[int, int]]
    page_origin: tuple[int, int]
    media_size: Mapping[str, int | bytes]
    media_types: Mapping[str, bytes]
    media_source: Mapping[InputSlot, int]
    duplex_page_mode: Mapping[DuplexMode, int | None]
    dither_cache: str
    inverse_lut_grids: tuple[tuple[Profile, bool], ...]
    color_table: Callable[[PrintSettings], ColorTable]
    pjl_header: Callable[[PrintSettings], bytes]
    pjl_footer: Callable[[], bytes]
    page_is_flipped: Callable[[PrintSettings, bool], bool]

    @property
    def lut_dir(self) -> Path:
        """The model's BRCD dither tables: `lut/<key>/` in the package."""
        return LUT_ROOT / self.key

    @property
    def color_data_dir(self) -> Path:
        """The model's colour grids and inverse LUTs: `color_data/<key>/` in the package."""
        return COLOR_DATA_ROOT / self.key

    def grid(
        self,
        profile: Profile = "rgb",
        *,
        improve_gray: bool = False,
        variant: str = "default",
        rich_black: bool = False,
    ) -> ColorTable:
        """The model's colour grid with these properties.

        Returns:
            The table, marked `precomputed` if it is one of `inverse_lut_grids`.
        """
        precomputed = variant == "default" and not rich_black and (profile, improve_gray) in self.inverse_lut_grids
        return ColorTable(
            profile,
            improve_gray,
            variant,
            rich_black,
            data_dir=self.color_data_dir,
            precomputed=precomputed,
        )

    @property
    def inverse_lut_tables(self) -> tuple[ColorTable, ...]:
        """The tables `cups/install.sh` precomputes an inverse LUT for."""
        return tuple(self.grid(profile, improve_gray=gray) for profile, gray in self.inverse_lut_grids)

    def image_height(self, page_size: str) -> int:
        """Rows of the page image, as the original computes its band height.

        The paper height in whole points, capped at the printable height
        where the model has an ImagingArea file, converted back to 600 dpi:
        `min(ury - lly, paper_h * 72 // 600) * 600 // 72`.

        Returns:
            Source height in rows.
        """
        _, ph = self.paper_sizes[page_size]
        points = ph * 72 // 600
        if self.imaging_height_pt is not None:
            points = min(self.imaging_height_pt[page_size], points)
        return points * 600 // 72

    def image_dimensions(self, page_size: str, *, fine: bool = False) -> tuple[int, int]:
        """Return (source_width, source_height) for BeginImage.

        The paper width rounded up to 32 pixels, in Fine mode to 8
        (raster_encode.c:352); the height is `image_height`.

        Returns:
            Image width and height in pixels.
        """
        pw, _ = self.paper_sizes[page_size]
        align = 7 if fine else 31
        return (pw + align) & ~align, self.image_height(page_size)

    def media_size_code(self, page_size: str) -> int | bytes:
        """BeginPage MediaSize for `page_size`: enum value or name string.

        Returns:
            The `media_size` entry; A4 (2) with a warning for unknown sizes.
        """
        size = self.media_size.get(page_size)
        if size is None:
            logger.warning("Unknown media size %r, defaulting to A4", page_size)
            return 2
        return size

    def media_type_code(self, media_type: str) -> bytes:
        """BeginPage MediaType string for `media_type`.

        Returns:
            The `media_types` entry; dRegular with a warning for unknown types.
        """
        type_str = self.media_types.get(media_type)
        if type_str is None:
            logger.warning("Unknown media type %r, defaulting to Regular", media_type)
            return b"dRegular"
        return type_str
