"""CUPS filter for the Brother HL-4150CDN.

Receives PostScript from CUPS, rasterizes it with Ghostscript and streams
the pages through the XL2HB pipeline (job.filter_pages).

Usage (called by CUPS through cups/brhl4150cdn-filter):
    brhl4150cdn-filter job-id user title copies options [filename]
"""

import contextlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import BinaryIO, cast

from brother_xl2hb.job import chain_peeked, drop_blank_pages, filter_pages
from brother_xl2hb.models import DEFAULT_MODEL
from brother_xl2hb.models.base import PrinterModel
from brother_xl2hb.render.page_stream import iter_ppm_pages
from brother_xl2hb.render.rows import PageData
from brother_xl2hb.settings import PrintSettings, read_ppd_defaults

# Seconds to wait for Ghostscript to exit on its own after the job failed
# (it has usually closed its output already) before it is killed.
GS_EXIT_GRACE = 2.0


logger = logging.getLogger("brhl4150cdn-filter")

Page = tuple[int, int, PageData]


def find_gs() -> str:
    """Locate the Ghostscript binary or exit.

    Returns:
        Path to the gs executable.
    """
    gs = shutil.which("gs")
    if gs is None:
        # CUPS runs filters with a minimal PATH; try the usual install prefixes.
        for alt in ["/opt/homebrew/bin/gs", "/usr/local/bin/gs"]:
            if Path(alt).is_file():
                gs = alt
                break
    if gs is None:
        logger.error("Ghostscript (gs) not found in PATH")
        sys.exit(1)
    return gs


def build_gs_command(input_path: str, page_size: str, model: PrinterModel) -> list[str]:
    """Build the Ghostscript command for rasterizing PS to PPM at the model's paper size.

    Returns:
        The argument vector; `input_path` "-" reads the job from stdin.
    """
    cmd = [
        find_gs(),
        "-sDEVICE=ppmraw",
        "-r600",
        "-dBATCH",
        "-dNOPAUSE",
        "-dSAFER",
        "-dQUIET",
        "-sOutputFile=-",
    ]

    w_pt, h_pt = model.paper_points[page_size]
    cmd.extend([f"-dDEVICEWIDTHPOINTS={w_pt}", f"-dDEVICEHEIGHTPOINTS={h_pt}", input_path])
    return cmd


def iter_printable_pages(stream: BinaryIO, page_size: str, model: PrinterModel) -> Iterator[Page]:
    """Yield (width, height, row_blocks) per GS page, cut to the printable area.

    Pages are read on a background thread in blocks of scanlines (see
    page_stream), so Ghostscript keeps rasterising while a page renders
    and no page-sized buffer is allocated.

    Yields:
        Printable width and height, and the page's row blocks.
    """
    target_w, target_h = model.paper_sizes[page_size]
    window = (*model.page_origin, target_w, target_h)
    for page_num, (src_w, src_h, blocks) in enumerate(iter_ppm_pages(stream, window), start=1):
        logger.info("page %d: %dx%d pixels from GS, printable %dx%d", page_num, src_w, src_h, target_w, target_h)
        yield target_w, target_h, blocks


class PageCounter:
    """Counts the pages taken from an iterable, even if iteration fails."""

    def __init__(self, pages: Iterable[Page]) -> None:
        """Wrap `pages`; `count` starts at zero."""
        self._pages = pages
        self.count = 0

    def __iter__(self) -> Iterator[Page]:
        """Yield the wrapped pages, counting each one.

        Yields:
            The wrapped pages.
        """
        for page in self._pages:
            self.count += 1
            yield page


def send_pages(gs_stdout: BinaryIO, settings: PrintSettings, output: BinaryIO, model: PrinterModel) -> int:
    """Render Ghostscript's PPM stream into one XL2HB job on `output`.

    Each page is rendered and sent as soon as it is read, like Brother's
    filter, so only the current page is held in RAM. Nothing is written
    when Ghostscript produces no page. If the stream breaks off, the
    exception propagates with the job's complete pages already written and
    the session closed (see filter_pages).

    Returns:
        Number of pages read from Ghostscript.
    """
    counter = PageCounter(iter_printable_pages(gs_stdout, settings.page_size, model))
    pages: Iterable[Page] = counter
    if settings.skip_blank:
        pages = drop_blank_pages(pages)
    pages = iter(pages)
    try:
        first = next(pages, None)
        if first is not None:
            pages = chain_peeked(first, pages)
            del first
            filter_pages(pages, settings, output, model)
    except Exception as exc:
        exc.add_note(f"while reading page {counter.count} from Ghostscript")
        raise
    finally:
        with contextlib.suppress(OSError):
            output.flush()
    return counter.count


def stop_gs(proc: subprocess.Popen[bytes], timeout: float) -> int | None:
    """Give Ghostscript `timeout` seconds to exit, then kill it.

    Returns:
        Its exit status, or None if it had to be killed.
    """
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return None


def log_gs_stderr(stderr_file: BinaryIO, level: int) -> None:
    """Log what Ghostscript wrote to stderr, one CUPS log line per line."""
    stderr_file.seek(0)
    for line in stderr_file.read().decode("utf-8", errors="replace").splitlines():
        if line.strip():
            logger.log(level, "GS: %s", line.rstrip())


def print_job(input_path: str, settings: PrintSettings, output: BinaryIO, model: PrinterModel = DEFAULT_MODEL) -> int:
    """Run Ghostscript on `input_path` and send its pages to `output`.

    Returns:
        The filter's exit status: 0 on success, 1 on any failure.
    """
    gs_cmd = build_gs_command(input_path, settings.page_size, model)
    logger.info("running: %s", " ".join(gs_cmd))

    # Ghostscript's stderr goes to a file: a pipe nobody reads while the
    # pages render would block it once it filled up.
    with tempfile.TemporaryFile() as gs_stderr:
        # The command is fixed apart from the gs path and the job file CUPS gave us.
        gs_proc = subprocess.Popen(gs_cmd, stdout=subprocess.PIPE, stderr=gs_stderr)  # noqa: S603
        gs_stdout = cast("BinaryIO", gs_proc.stdout)  # a pipe: stdout=PIPE
        try:
            try:
                page_count = send_pages(gs_stdout, settings, output, model)
            except Exception as exc:
                status = stop_gs(gs_proc, GS_EXIT_GRACE)
                if status:
                    # Ghostscript died and its output broke off.
                    logger.error("Ghostscript failed with exit status %d: %s", status, exc)
                    for note in getattr(exc, "__notes__", ()):
                        logger.error("%s", note)
                    log_gs_stderr(gs_stderr, logging.ERROR)
                else:
                    logger.exception("job aborted")
                    log_gs_stderr(gs_stderr, logging.INFO)
                return 1

            status = gs_proc.wait()
            log_gs_stderr(gs_stderr, logging.ERROR if status else logging.INFO)
            if status:
                logger.error("Ghostscript exited with status %d", status)
                return 1
            if page_count == 0:
                logger.error("No pages produced by Ghostscript")
                return 1
            logger.info("total pages: %d", page_count)
            return 0
        finally:
            # Reap Ghostscript on every path, including interrupts.
            if gs_proc.poll() is None:
                gs_proc.kill()
                gs_proc.wait()
            gs_stdout.close()


def main(model: PrinterModel = DEFAULT_MODEL) -> None:
    """Run as CUPS calls a filter for `model`: ``job-id user title copies options [filename]``."""
    # CUPS puts every stderr line into its error_log, keyed by the prefix.
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: brhl4150cdn-filter: %(message)s",
        stream=sys.stderr,
    )
    if len(sys.argv) < 6:
        logger.error("Usage: %s job-id user title copies options [filename]", sys.argv[0])
        sys.exit(1)

    job_id = sys.argv[1]
    user = sys.argv[2]
    title = sys.argv[3]
    copies = int(sys.argv[4])
    options_str = sys.argv[5]
    input_file = sys.argv[6] if len(sys.argv) > 6 else None

    logger.info("job=%s user=%s title=%r copies=%d", job_id, user, title, copies)

    # The queue's PPD defaults first (CUPS passes only the job's own
    # options), then the job options on top.
    ppd_path = os.environ.get("PPD")
    ppd_defaults = read_ppd_defaults(ppd_path) if ppd_path else {}
    # Copies are already made upstream: pstops puts NumCopies into the
    # PostScript and Ghostscript renders every page that many times. A
    # PageCopies of `copies` on top would print copies² sheets of each page,
    # so the job carries 1, as Brother's cupswrapper hands its filter.
    settings = model.settings_from_cups_options(options_str, 1, ppd_defaults)
    logger.info(
        "settings: page_size=%s duplex=%s resolution=%s mono=%s color_matching=%s gray=%s toner_save=%s",
        settings.page_size,
        settings.duplex,
        settings.resolution,
        settings.mono_color,
        settings.color_matching,
        settings.improve_gray,
        settings.toner_save,
    )

    # Ghostscript reads the job straight from stdin, so it starts
    # rasterising while pdftops is still producing PostScript.
    status = print_job(input_file or "-", settings, sys.stdout.buffer, model)
    if status == 0:
        logger.info("done")
    sys.exit(status)


if __name__ == "__main__":
    main()
