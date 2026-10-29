"""Trace the original filter's Fine-mode (4bpp) dither calls under gdb.

Builds a test page with the `fine_diff` helpers, runs brhl4150cdnfilter
under qemu-i386 with gdb-multiarch attached (Docker image brhl-gdb:i386,
built on demand from the Dockerfile below) and records, for every call of
dither_ordered_4ch_4bpp (0x08061404) in the traced line range, the
run-length CMYK input the dither receives and the nibble-packed CMYK lines
it writes. The DitherContext tables (per-channel matrix size, intensity
LUT, 4bpp pattern data) are dumped once.

    uv run python scripts/trace_dither.py solid:0,255,255 --lines 16
    uv run python scripts/trace_dither.py hramp --lines 4 --show 64
    uv run python scripts/trace_dither.py solid:128,128,128 --rc Saturation=20 --save build/trace.npz

The traced filter's own XL2HB output is checked against the traced dither
output: every traced line must equal what `decode_fine_plane` decodes from
the capture, after the encoder's nibble quantisation.

Import `trace_dither(...)` for the data as numpy arrays.

Why qemu: Docker Desktop's linux/386 emulation has no ptrace, so gdb inside
an i386 container cannot run the filter. qemu-user's gdbstub works.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import fine_diff  # noqa: E402

from brother_decode import decode_fine_plane  # noqa: E402
from extract_fixtures import PLANE_NAMES, parse_xl2hb_capture  # noqa: E402

GDB_IMAGE = "brhl-gdb:i386"
DOCKERFILE = """\
FROM --platform=linux/386 i386/debian:bookworm-slim AS sysroot
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends qemu-user gdb-multiarch python3 \\
    && rm -rf /var/lib/apt/lists/*
COPY --from=sysroot /usr /sysroot/usr
RUN ln -s usr/lib /sysroot/lib && ln -s usr/bin /sysroot/bin && ln -s usr/sbin /sysroot/sbin
"""
CACHE_DIR = fine_diff.ROOT / "build" / "trace-dither-cache"
CHANNELS = "CMYK"  # order of the four input and output arguments
TRACER_VERSION = "1"


@dataclass
class DitherTable:
    """One DitherContext channel slot as the original holds it in memory."""

    name: str
    width: int  # matrix width in pixels (struct field >> 3)
    height: int
    struct_width: int  # raw matrix_width field
    row_bytes: int  # raw +0x04 field
    lut: npt.NDArray[np.int64]  # intensity -> byte offset into `pattern`
    pattern: npt.NDArray[np.uint8] = field(repr=False)

    def level(self, value: int) -> npt.NDArray[np.uint8]:
        """The (height, width) nibble matrix used for intensity `value`."""
        size = self.width * self.height
        return self.pattern[self.lut[value] : self.lut[value] + size].reshape(self.height, self.width)

    def levels(self) -> npt.NDArray[np.uint8]:
        """All 256 level matrices, shape (256, height, width)."""
        return np.stack([self.level(v) for v in range(256)])


@dataclass
class DitherCall:
    """One traced call of dither_ordered_4ch_4bpp."""

    index: int  # call number among all 4ch calls of the run
    y: int
    x_start: int
    width: int
    runs: npt.NDArray[np.uint16]
    run_values: dict[str, npt.NDArray[np.uint8]]  # per channel, one value per run
    out: dict[str, npt.NDArray[np.uint8]]  # per channel, nibble-packed line
    ret: int | None  # plane mask: bit 0 K, 1 C, 2 M, 3 Y
    args: list[int]

    def pixels(self, channel: str) -> npt.NDArray[np.uint8]:
        """The channel input expanded to one value per pixel (from x_start)."""
        return np.repeat(self.run_values[channel], self.runs)

    def levels(self, channel: str) -> npt.NDArray[np.uint8]:
        """The output as one 4-bit level per pixel."""
        packed = self.out[channel]
        return np.stack([packed >> 4, packed & 0xF], axis=1).reshape(-1)


@dataclass
class DitherTrace:
    """All traced dither calls of one filter run, with the dumped tables and output."""

    calls: list[DitherCall]
    all_y: list[int]  # y of every 4ch call, traced or not
    calls_1ch: int
    tables: dict[str, DitherTable]
    xl2hb: bytes  # the traced filter's output


def ensure_image() -> None:
    """Build the gdb image if it is missing."""
    inspect = ["docker", "image", "inspect", GDB_IMAGE]
    if subprocess.run(inspect, capture_output=True, check=False).returncode == 0:  # noqa: S603 - fixed argv
        return
    print(f"building {GDB_IMAGE} ...", file=sys.stderr)
    build = ["docker", "build", "-t", GDB_IMAGE, "-"]
    subprocess.run(build, input=DOCKERFILE.encode(), check=True)  # noqa: S603 - fixed argv


def run_traced(ppm: bytes, rc: str, y0: int, y1: int, max_calls: int) -> Path:
    """Run the filter under gdb, cached; return the directory with trace.json, trace.bin, out.xl2hb.

    Raises:
        SystemExit: If the traced run does not produce its outputs.
    """
    key = hashlib.sha256(ppm + rc.encode() + f"{y0},{y1},{max_calls},{TRACER_VERSION}".encode()).hexdigest()[:16]
    work = CACHE_DIR / key
    if (work / "trace.json").exists():
        return work
    ensure_image()
    work.mkdir(parents=True, exist_ok=True)
    (work / "page.ppm").write_bytes(ppm)
    (work / "rc").write_text(rc)
    script = (
        f"qemu-i386 -L /sysroot -g 1234 {fine_diff.FILTER} -pi {fine_diff.PAPERINF} -rc /w/rc"
        " < /w/page.ppm > /w/out.xl2hb & "
        'gdb-multiarch -q -batch -ex "set architecture i386" -ex "target remote :1234"'
        f" -x /s/trace_dither_gdb.py -ex continue {fine_diff.FILTER}; wait"
    )
    cmd = [
        "docker", "run", "--rm",
        "-e", f"TRACE_Y0={y0}", "-e", f"TRACE_Y1={y1}", "-e", f"TRACE_MAX={max_calls}",
        "-v", f"{fine_diff.BROTHER_DIR}:/usr/local/Brother:ro",
        "-v", f"{work}:/w", "-v", f"{SCRIPTS}:/s:ro",
        GDB_IMAGE, "sh", "-c", script,
    ]  # fmt: skip
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    (work / "page.ppm").unlink()
    if not (work / "trace.json").exists() or not (work / "out.xl2hb").stat().st_size:
        msg = f"traced run failed ({result.returncode}):\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        raise SystemExit(msg)
    return work


def load_trace(work: Path) -> DitherTrace:
    """Parse a traced run's files into numpy arrays."""
    meta = json.loads((work / "trace.json").read_text())
    blob = np.frombuffer((work / "trace.bin").read_bytes(), np.uint8)

    def get(ref: list[int]) -> npt.NDArray[np.uint8]:
        return blob[ref[0] : ref[0] + ref[1]]

    tables = {}
    for name, ch in meta["ctx"]["channels"].items():
        tables[name.upper()] = DitherTable(
            name=name.upper(),
            width=ch["matrix_width"] >> 3,
            height=ch["matrix_height"],
            struct_width=ch["matrix_width"],
            row_bytes=ch["row_bytes"],
            lut=np.array(ch["lut"], np.int64) - ch["pattern_base"],
            pattern=get(ch["pattern"]),
        )
    calls = []
    for rec in meta["calls"]:
        if rec["kind"] != "4ch":
            continue
        a = rec["args"]
        calls.append(
            DitherCall(
                index=rec["index"],
                y=rec["y"],
                x_start=a[7],
                width=a[9],
                runs=get(rec["runs"]).view(np.uint16),
                run_values={c: get(ref) for c, ref in zip(CHANNELS, rec["inputs"], strict=True)},
                out={c: get(ref) for c, ref in zip(CHANNELS, rec.get("out", []), strict=True)},
                ret=rec.get("ret"),
                args=a,
            )
        )
    return DitherTrace(
        calls=calls,
        all_y=meta["all_y"]["4ch"],
        calls_1ch=len(meta["all_y"]["1ch"]),
        tables=tables,
        xl2hb=(work / "out.xl2hb").read_bytes(),
    )


def trace_dither(
    pattern: str = "solid:0,255,255",
    *,
    top: int = 100,
    lines: int = 16,
    rc: list[str] | None = None,
    y0: int | None = None,
    y1: int | None = None,
    max_calls: int = 64,
    seed: int = 0,
) -> DitherTrace:
    """Trace the original's 4bpp dither on a fine_diff test page.

    By default the traced range is the pattern rows top..top+lines.
    """
    page = fine_diff.make_page(pattern, top, lines, seed)
    y0 = top if y0 is None else y0
    y1 = top + lines if y1 is None else y1
    return load_trace(run_traced(fine_diff.ppm_bytes(page), fine_diff.make_rc(rc or []), y0, y1, max_calls))


# ---------------------------------------------------------------------------
# Validation and report
# ---------------------------------------------------------------------------

PLANE_IDS = {name: pid for pid, name in PLANE_NAMES.items()}


def validate(trace: DitherTrace) -> bool:
    """Compare each traced output line with the decoded XL2HB line of the same plane."""
    capture = parse_xl2hb_capture(trace.xl2hb, "traced")
    planes = {c: fine_diff.by_line(capture, PLANE_IDS[c]) for c in CHANNELS}
    ok = True
    stats = {c: [0, 0, 0] for c in CHANNELS}  # match, mismatch, raw-not-quantised
    for call in trace.calls:
        for c in CHANNELS:
            raw = call.out[c].tobytes()[: fine_diff.FINE_ROW_BYTES]
            entry = planes[c].get(call.y)
            decoded = decode_fine_plane(entry) if entry is not None else bytes(fine_diff.FINE_ROW_BYTES)
            q = fine_diff.quantise(raw)
            if q != raw:
                stats[c][2] += 1
            if q == decoded:
                stats[c][0] += 1
            else:
                stats[c][1] += 1
                ok = False
                at = fine_diff.first_diff(q, decoded)
                print(f"  MISMATCH {c} y={call.y} at byte {at}:")
                print(f"    traced  {fine_diff.hex_window(q, at, 24)}")
                print(f"    decoded {fine_diff.hex_window(decoded, at, 24)}")
    for c, (good, bad, unq) in stats.items():
        print(f"  {c}: {good} lines match, {bad} differ; raw output had odd non-F nibbles on {unq} lines")
    return ok


def simulate(call: DitherCall, table: DitherTable, channel: str) -> npt.NDArray[np.uint8]:
    """Model of the routine: level = table.level(v)[y % h][x % w], x counted from 0 at x_start.

    Returns:
        One 4-bit level per pixel of the call.
    """
    px = call.pixels(channel)
    cols = (call.x_start + np.arange(len(px))) % table.width
    return table.levels()[px, call.y % table.height, cols]


def overflow_zeroed(call: DitherCall) -> set[str]:
    """Planes whose byte 0 the original clobbers at the end of this line.

    The all-zero fast path ends with `*out = 0` one byte past the run. If the
    line ends in such a run (all four inputs 0, length >= 2) and the width is
    even, that byte is one past each output buffer. The caller's buffers are
    contiguous C, M, Y, K with a stride of width/2, so C's overrun zeroes
    byte 0 (pixels 0 and 1) of M, M's of Y and Y's of K.
    """
    last_zero = call.runs[-1] >= 2 and all(call.run_values[c][-1] == 0 for c in CHANNELS)
    end = (call.x_start + int(call.runs.sum())) // 2
    ptrs = call.args[10:14]
    out = set()
    if last_zero and (call.x_start + int(call.runs.sum())) % 2 == 0:
        for prev, nxt, name in zip(ptrs, ptrs[1:], CHANNELS[1:], strict=False):
            if prev + end == nxt:
                out.add(name)
    return out


def check_model(trace: DitherTrace) -> bool:
    """Check the traced output against `simulate` (plus the overrun quirk) on the traced input and tables."""
    ok = True
    for c in CHANNELS:
        bad = []
        for call in trace.calls:
            model = simulate(call, trace.tables[c], c)
            if c in overflow_zeroed(call):
                model[:2] = 0
            if not np.array_equal(model, call.levels(c)[: int(call.runs.sum())]):
                bad.append(call.y)
        print(
            f"  {c}: model matches {len(trace.calls) - len(bad)} of {len(trace.calls)} lines"
            + (f", bad y {bad[:8]}" if bad else "")
        )
        ok &= not bad
    return ok


def compare_python_tables(trace: DitherTrace) -> None:
    """Compare the dumped tables with what brother_xl2hb.color.dither loads for Fine mode."""
    from brother_xl2hb.color.dither import load_brcd_tables  # noqa: PLC0415
    from brother_xl2hb.models.hl4150cdn import DITHER_CACHE  # noqa: PLC0415

    lut_dir = fine_diff.BROTHER_DIR / "Printer" / "hl4150cdn" / "inf" / "lut"
    py = load_brcd_tables(str(lut_dir), fine=True, cache=DITHER_CACHE) or {}
    for c in CHANNELS:
        t, p = trace.tables[c], py.get(c)
        if p is None or getattr(p, "nibble_levels", None) is None:
            print(f"  {c}: python has no nibble_levels (w={getattr(p, 'width', None)}, h={getattr(p, 'height', None)})")
            continue
        same = p.nibble_levels.shape == (256, t.height, t.width) and np.array_equal(p.nibble_levels, t.levels())
        print(f"  {c}: original {t.width}x{t.height}, python {p.width}x{p.height}, levels identical: {same}")


def summary(values: npt.NDArray[np.uint8]) -> str:
    """Return a one-line min/max/mean/distinct summary of `values`."""
    u = np.unique(values)
    head = " ".join(map(str, u[:8])) + (" ..." if len(u) > 8 else "")
    return f"min {values.min()} max {values.max()} mean {values.mean():.1f} distinct {len(u)} [{head}]"


def report(trace: DitherTrace, show: int) -> None:
    """Print the traced calls with their inputs and outputs."""
    ys = trace.all_y
    print(f"4ch calls: {len(ys)} (y {min(ys) if ys else '-'}..{max(ys) if ys else '-'}), 1ch calls: {trace.calls_1ch}")
    print("tables:", ", ".join(f"{t.name} {t.width}x{t.height}" for t in trace.tables.values()))
    for call in trace.calls:
        print(
            f"call #{call.index} y={call.y} x_start={call.x_start} width={call.width}"
            f" runs={len(call.runs)} (sum {int(call.runs.sum())}) ret={call.ret}"
        )
        print(f"  runs[:{show}] {call.runs[:show].tolist()}")
        for c in CHANNELS:
            px = call.pixels(c)
            print(f"  in  {c}: {px[:show].tolist()}  | {summary(px)}")
        for c in CHANNELS:
            lv = call.levels(c)
            print(f"  out {c}: {call.out[c][: show // 2].tobytes().hex(' ')}  | levels {summary(lv)}")


def save_npz(trace: DitherTrace, path: Path) -> None:
    """Save per-call pixel inputs and level outputs plus the tables."""
    data: dict[str, npt.NDArray] = {"y": np.array([c.y for c in trace.calls])}
    for ch in CHANNELS:
        data[f"in_{ch}"] = np.array([c.pixels(ch) for c in trace.calls])
        data[f"out_{ch}"] = np.array([c.out[ch] for c in trace.calls])
    for name, t in trace.tables.items():
        data[f"table_{name}"] = t.levels()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    print(f"saved {path}")


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pattern", help="test page pattern, see fine_diff.py")
    parser.add_argument("--top", type=int, default=100, help="first row of the pattern (default 100)")
    parser.add_argument("--lines", type=int, default=16, help="rows of pattern (default 16)")
    parser.add_argument("--seed", type=int, default=0, help="seed for the noise pattern")
    parser.add_argument("--rc", action="append", default=[], metavar="KEY=VALUE", help="override an rc setting")
    parser.add_argument("--y0", type=int, help="first traced line (default --top)")
    parser.add_argument("--y1", type=int, help="end of the traced lines (default --top + --lines)")
    parser.add_argument("--max-calls", type=int, default=64, help="max traced calls (default 64)")
    parser.add_argument("--show", type=int, default=32, help="pixels to print per line (default 32)")
    parser.add_argument("--save", type=Path, help="save the trace as npz")
    args = parser.parse_args()

    trace = trace_dither(
        args.pattern, top=args.top, lines=args.lines, rc=args.rc,
        y0=args.y0, y1=args.y1, max_calls=args.max_calls, seed=args.seed,
    )  # fmt: skip
    report(trace, args.show)
    untraced = fine_diff.run_original(
        fine_diff.ppm_bytes(fine_diff.make_page(args.pattern, args.top, args.lines, args.seed)),
        fine_diff.make_rc(args.rc),
    )
    print("traced output equals untraced run:", trace.xl2hb == untraced)
    print("validation against the XL2HB planes:")
    ok = validate(trace)
    print("VALIDATION", "PASSED" if ok else "FAILED")
    print("table-lookup model on the traced input:")
    check_model(trace)
    print("dumped tables vs dither.load_brcd_tables(fine=True):")
    compare_python_tables(trace)
    if args.save:
        save_npz(trace, args.save)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
