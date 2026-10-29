"""CUPS filter for a Brother XL2HB printer.

Receives PostScript from CUPS, rasterizes it with Ghostscript and streams
the pages through the XL2HB pipeline (job.filter_pages). Each model has a
shim, cups/br<model>-filter, that calls `main` with its `PrinterModel`.

Usage (called by CUPS through the shim):
    br<model>-filter job-id user title copies options [filename]
"""

import contextlib
import dataclasses
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import BinaryIO, cast

import numpy as np

from brother_xl2hb.job import chain_peeked, drop_blank_pages, filter_pages
from brother_xl2hb.models import DEFAULT_MODEL
from brother_xl2hb.models.base import PrinterModel
from brother_xl2hb.render.page_stream import iter_ppm_pages
from brother_xl2hb.render.rows import Page
from brother_xl2hb.settings import DuplexMode, PrintSettings, read_ppd_defaults

# Seconds to wait for Ghostscript to exit on its own after the job failed
# (it has usually closed its output already) before it is killed.
GS_EXIT_GRACE = 2.0

# Uncollated copies reach Ghostscript as NumCopies (pstops and pdftops put
# it into the PostScript), and Ghostscript emits each page that many times
# in a row. In duplex that prints a page on both sides of a sheet. For
# duplex copies Ghostscript is therefore kept at one copy per page and runs
# the job again for each further copy instead. Between two runs it emits a
# 8x8 pt copy mark page (BRCopyEnd), and `sheets_per_copy` gives a copy
# that ends on a front side a blank back, so the next one starts on a new
# sheet, as in the copies pstops collates itself. The filter decides that
# after BRSkipBlank has dropped blank pages, so the blank back survives it
# and the copy's page count is the one that is printed.
DUPLEX_COPIES_PROLOGUE = (
    "/BRNumCopies 1 def "
    "/setpagedevice { dup /NumCopies known { "
    "dup /NumCopies get dup type /integertype eq { userdict /BRNumCopies 3 -1 roll put } { pop } ifelse "
    "dup length dict copy dup /NumCopies 1 put "
    "} if //setpagedevice } bind def "
    "/BRCopyEnd { currentpagedevice /PageSize get << /PageSize [8 8] >> setpagedevice showpage "
    "<< /PageSize 3 -1 roll >> setpagedevice } bind def"
)
DUPLEX_COPIES_RUN = "<{job}> run BRNumCopies 1 sub {{ BRCopyEnd <{job}> run }} repeat"
# A Ghostscript page at most this many pixels wide and high is a copy mark
# (BRCopyEnd's 8x8 pt page is 67x67 at 600 dpi); no printable page is that small.
COPY_MARK_MAX_PIXELS = 100


logger = logging.getLogger(__name__)


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


def build_gs_command(
    input_path: str, page_size: str, model: PrinterModel, *, collate_copies: bool = False
) -> list[str]:
    """Build the Ghostscript command for rasterizing PS to PPM at the model's paper size.

    With `collate_copies`, the NumCopies the job asks for are rendered as
    whole runs of the job instead of repeated pages (DUPLEX_COPIES_PROLOGUE);
    the job has to be a file then.

    Returns:
        The argument vector; `input_path` "-" reads the job from stdin.

    Raises:
        ValueError: `collate_copies` with the job on stdin.
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
    cmd.extend([f"-dDEVICEWIDTHPOINTS={w_pt}", f"-dDEVICEHEIGHTPOINTS={h_pt}"])
    if not collate_copies:
        cmd.append(input_path)
        return cmd
    if input_path == "-":
        msg = "collated copies need the job as a file, Ghostscript runs it once per copy"
        raise ValueError(msg)
    # Under -dSAFER PostScript may open only permitted files. The job is
    # run from PostScript every time: Ghostscript drops the permission for
    # a file given on its command line once it has run it.
    run = DUPLEX_COPIES_RUN.format(job=os.fsencode(input_path).hex())
    cmd.extend([f"--permit-file-read={input_path}", "-c", DUPLEX_COPIES_PROLOGUE, run])
    return cmd


def iter_printable_pages(
    stream: BinaryIO, page_size: str, model: PrinterModel, *, copy_marks: bool = False
) -> Iterator[Page | None]:
    """Yield (width, height, row_blocks) per GS page, cut to the printable area.

    Pages are read on a background thread in blocks of scanlines (see
    page_stream), so Ghostscript keeps rasterising while a page renders
    and no page-sized buffer is allocated. With `copy_marks`, Ghostscript's
    copy mark pages (see DUPLEX_COPIES_PROLOGUE) come out as None.

    Yields:
        Printable width and height, and the page's row blocks; None for a copy mark.
    """
    target_w, target_h = model.paper_sizes[page_size]
    window = (*model.page_origin, target_w, target_h)
    page_num = 0
    for src_w, src_h, blocks in iter_ppm_pages(stream, window):
        if copy_marks and src_w <= COPY_MARK_MAX_PIXELS and src_h <= COPY_MARK_MAX_PIXELS:
            logger.info("end of copy")
            yield None
            continue
        page_num += 1
        logger.info("page %d: %dx%d pixels from GS, printable %dx%d", page_num, src_w, src_h, target_w, target_h)
        yield target_w, target_h, blocks


def sheets_per_copy(pages: Iterable[Page | None], *, skip_blank: bool) -> Iterator[Page]:
    """Start each duplex copy on a new sheet; None in `pages` ends a copy.

    A copy that ends on a front side gets a blank back. With `skip_blank`,
    blank pages are dropped first, so a copy's sides are counted as they
    are printed and the blank back is kept.

    Yields:
        The pages to print, the blank backs included.
    """
    sides = 0
    size = (0, 0)
    for item in pages:
        if item is None:
            if sides % 2:
                yield _blank_page(*size)
            sides = 0
            continue
        page = next(drop_blank_pages((item,)), None) if skip_blank else item
        if page is None:
            continue
        size = page[:2]
        sides += 1
        yield page


def _blank_page(width: int, height: int) -> Page:
    """A white page of `width` x `height` pixels, as row blocks of one shared band.

    Returns:
        The page.
    """
    band = np.full((min(height, 128), width * 3), 255, np.uint8)
    blocks = [band] * (height // band.shape[0])
    if rest := height % band.shape[0]:
        blocks.append(band[:rest])
    return width, height, blocks


class PageCounter:
    """Counts the pages taken from an iterable, even if iteration fails; copy marks (None) do not count."""

    def __init__(self, pages: Iterable[Page | None]) -> None:
        """Wrap `pages`; `count` starts at zero."""
        self._pages = pages
        self.count = 0

    def __iter__(self) -> Iterator[Page | None]:
        """Yield the wrapped pages, counting each one.

        Yields:
            The wrapped pages and copy marks.
        """
        for page in self._pages:
            if page is not None:
                self.count += 1
            yield page


def send_pages(
    gs_stdout: BinaryIO, settings: PrintSettings, output: BinaryIO, model: PrinterModel, *, copy_marks: bool = False
) -> int:
    """Render Ghostscript's PPM stream into one XL2HB job on `output`.

    Each page is rendered and sent as soon as it is read, like Brother's
    filter, so only the current page is held in RAM. Nothing is written
    when Ghostscript produces no page. If the stream breaks off, the
    exception propagates with the job's complete pages already written and
    the session closed (see filter_pages).

    Returns:
        Number of pages read from Ghostscript.
    """
    counter = PageCounter(iter_printable_pages(gs_stdout, settings.page_size, model, copy_marks=copy_marks))
    pages: Iterable[Page]
    if copy_marks:
        pages = sheets_per_copy(counter, skip_blank=settings.skip_blank)
        # Blank pages are dropped already; the blank backs must stay.
        settings = dataclasses.replace(settings, skip_blank=False)
    else:
        pages = (page for page in counter if page is not None)
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


def print_job(
    input_path: str, settings: PrintSettings, output: BinaryIO, model: PrinterModel = DEFAULT_MODEL, copies: int = 1
) -> int:
    """Run Ghostscript on `input_path` and send its pages to `output`.

    `copies` is the job's copy count from CUPS; in duplex, copies are
    collated (see DUPLEX_COPIES_PROLOGUE).

    Returns:
        The filter's exit status: 0 on success, 1 on any failure.
    """
    collate_copies = settings.duplex != DuplexMode.NONE and copies > 1
    if collate_copies and input_path == "-":
        # Ghostscript runs the job once per copy, so stdin goes to a file first.
        with tempfile.NamedTemporaryFile(prefix="brjob-", suffix=".ps") as spool:
            shutil.copyfileobj(sys.stdin.buffer, spool)
            spool.flush()
            return print_job(spool.name, settings, output, model, copies)

    gs_cmd = build_gs_command(input_path, settings.page_size, model, collate_copies=collate_copies)
    logger.info("running: %s", " ".join(gs_cmd))

    # Ghostscript's stderr goes to a file: a pipe nobody reads while the
    # pages render would block it once it filled up.
    with tempfile.TemporaryFile() as gs_stderr:
        # The command is fixed apart from the gs path and the job file CUPS gave us.
        gs_proc = subprocess.Popen(gs_cmd, stdout=subprocess.PIPE, stderr=gs_stderr)  # noqa: S603
        gs_stdout = cast("BinaryIO", gs_proc.stdout)  # a pipe: stdout=PIPE
        try:
            try:
                page_count = send_pages(gs_stdout, settings, output, model, copy_marks=collate_copies)
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
        format=f"%(levelname)s: br{model.key}-filter: %(message)s",
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
    # PostScript and Ghostscript renders every page that many times (in
    # duplex, the whole job; see DUPLEX_COPIES_PROLOGUE). A PageCopies of
    # `copies` on top would print copies² sheets of each page, so the job
    # carries 1, as Brother's cupswrapper hands its filter.
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
    status = print_job(input_file or "-", settings, sys.stdout.buffer, model, copies)
    if status == 0:
        logger.info("done")
    sys.exit(status)


if __name__ == "__main__":
    main()
