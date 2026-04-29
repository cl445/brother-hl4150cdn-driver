"""Command-line entry point: read PPM from stdin, write XL2HB to stdout."""

import argparse
import logging
import sys
from pathlib import Path

from pipeline import filter_page
from ppm import read_ppm
from settings import DuplexMode, MonoColor, PageSize, PrintSettings, Resolution

logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point: read PPM from stdin, write XL2HB to stdout."""
    parser = argparse.ArgumentParser(description="Brother HL-4150CDN filter (PPM -> XL2HB)")
    parser.add_argument("--rc", "-r", help="Path to RC file (brhl4150cdnrc)", default=None)
    parser.add_argument("--paper", "-p", help="Paper size (e.g. A4, Letter)", default=None)
    parser.add_argument("--mono", "-m", action="store_true", help="Print monochrome")
    parser.add_argument("--toner-save", "-t", action="store_true", help="Toner save mode")
    parser.add_argument("--fine", "-f", action="store_true", help="Fine quality (2400 dpi class dithering)")
    parser.add_argument("--duplex", choices=["none", "long", "short"], default="none")
    parser.add_argument("--debug", "-d", action="store_true", help="Debug output to stderr")

    args = parser.parse_args()

    settings = PrintSettings.from_rc_file(args.rc) if args.rc and Path(args.rc).exists() else PrintSettings()

    if args.paper:
        settings.page_size = PageSize(args.paper)
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

    logger.debug("Settings: %s", settings)

    result = read_ppm(sys.stdin.buffer)
    if result is None:
        logger.error("No PPM data on stdin")
        sys.exit(1)
    width, height, maxval, pixel_data = result
    logger.debug("PPM: %dx%d, maxval=%d", width, height, maxval)

    filter_page(width, height, pixel_data, settings, sys.stdout.buffer)


if __name__ == "__main__":
    main()
