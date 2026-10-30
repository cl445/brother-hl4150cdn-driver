"""Command-line entry point: read PPM pages from stdin, write XL2HB to stdout."""

import argparse
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

from brother_xl2hb.color.color_lut import write_inverse_luts
from brother_xl2hb.job import chain_peeked, drop_blank_pages, filter_pages
from brother_xl2hb.models import DEFAULT_MODEL, MODELS
from brother_xl2hb.render.ppm import read_ppm
from brother_xl2hb.settings import DuplexMode, MonoColor, PageSize, PrintSettings, Resolution

if TYPE_CHECKING:
    from brother_xl2hb.render.rows import Page

logger = logging.getLogger(__name__)


def iter_ppm_stream(stream: BinaryIO) -> Iterator[tuple[int, int, bytes]]:
    """Yield (width, height, pixel_data) for each page of a concatenated P6 stream.

    Pages are read one at a time, so only the current page is held in memory.

    Yields:
        Page dimensions and packed RGB pixel data.
    """
    page_num = 0
    while (result := read_ppm(stream)) is not None:
        page_num += 1
        width, height, maxval, pixel_data = result
        logger.debug("PPM page %d: %dx%d, maxval=%d", page_num, width, height, maxval)
        yield width, height, pixel_data


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: read PPM pages from stdin, write XL2HB to stdout.

    All pages of a concatenated PPM stream (as Ghostscript writes it) are
    rendered into one XL2HB session, like the CUPS filter does.
    """
    parser = argparse.ArgumentParser(description="Brother XL2HB printer filter (PPM -> PJL + XL2HB)")
    parser.add_argument(
        "--model",
        choices=sorted(MODELS),
        default=DEFAULT_MODEL.key,
        help=f"Printer model (default: {DEFAULT_MODEL.key})",
    )
    parser.add_argument("--rc", "-r", help="Path to the model's Brother RC file (e.g. brhl4150cdnrc)", default=None)
    parser.add_argument("--paper", "-p", help="Paper size (e.g. A4, Letter)", default=None)
    parser.add_argument("--mono", "-m", action="store_true", help="Print monochrome")
    parser.add_argument("--toner-save", "-t", action="store_true", help="Toner save mode")
    parser.add_argument(
        "--fine",
        "-f",
        action="store_true",
        help="Fine mode (600x2400 dpi), byte-identical to the original filter",
    )
    parser.add_argument("--duplex", choices=["none", "long", "short"], default="none")
    parser.add_argument("--debug", "-d", action="store_true", help="Debug output to stderr")
    parser.add_argument(
        "--precompute-lut",
        nargs="?",
        const="",
        default=None,
        metavar="DIR",
        help="Precompute the model's RGB→KCMY inverse LUTs into DIR and exit. "
        "Default DIR: the model's colour data directory.",
    )

    args = parser.parse_args(argv)

    if args.precompute_lut is not None:
        logging.basicConfig(
            level=logging.DEBUG if args.debug else logging.INFO,
            format="%(levelname)s: %(name)s: %(message)s",
            stream=sys.stderr,
        )
        model = MODELS[args.model]
        write_inverse_luts(model.inverse_lut_tables, Path(args.precompute_lut) if args.precompute_lut else None)
        return

    if args.rc is not None and not Path(args.rc).is_file():
        parser.error(f"RC file not found: {args.rc}")
    model = MODELS[args.model]
    settings = model.settings_from_rc_file(args.rc) if args.rc is not None else PrintSettings()

    if args.paper:
        try:
            settings.page_size = PageSize(model.spelled("page_size", args.paper))
        except ValueError:
            parser.error(f"unknown paper size {args.paper!r}")
    if args.mono:
        settings.mono_color = MonoColor.MONO
    if args.fine:
        settings.resolution = Resolution.FINE
    if args.toner_save:
        settings.toner_save = True
    if args.duplex == "long":
        settings.duplex = DuplexMode.NO_TUMBLE
    elif args.duplex == "short":
        settings.duplex = DuplexMode.TUMBLE

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(levelname)s: %(name)s: %(message)s",
        stream=sys.stderr,
    )

    settings = model.restrict(settings)
    logger.debug("Settings: %s", settings)

    pages: Iterator[Page] = iter_ppm_stream(sys.stdin.buffer)
    first = next(pages, None)
    if first is None:
        logger.error("No PPM data on stdin")
        sys.exit(1)
    pages = chain_peeked(first, pages)
    del first

    if settings.skip_blank:
        pages = drop_blank_pages(pages)

    first = next(pages, None)
    if first is None:
        logger.info("All pages blank, nothing to print")
        return
    pages = chain_peeked(first, pages)
    del first
    output = sys.stdout.buffer
    filter_pages(pages, settings, output, model)
    output.flush()


if __name__ == "__main__":
    main()
