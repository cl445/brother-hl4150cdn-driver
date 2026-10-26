"""Differential harness for Fine mode: original filter vs. the Python pipeline.

Renders a test page, prints it with Brother's brhl4150cdnfilter (i386, in
Docker) and with `pipeline.filter_page`, then compares the two XL2HB
streams: PJL header, block layout, and every compressed scanline entry per
plane. For each mismatching entry it shows the 4bpp line our encoder got,
our Stage-1 and Stage-2 output and the original's bytes, and can dump them
for offline work on `fine_encoder`.

The original's output is cached by a hash of page and rc file, so reruns
only cost the Python side.

    uv run python scripts/fine_diff.py solid:0,255,255 --lines 8
    uv run python scripts/fine_diff.py hramp --lines 64 --dump build/fine-diff
    uv run python scripts/fine_diff.py noise --rc Saturation=20

Patterns: white, black, solid:R,G,B, hramp (gray ramp across the width),
steps (16 gray steps across the width), bars (C/M/Y/R/G/B/K/gray rows),
noise (seeded random RGB), file:<path.ppm>.

Needs Docker and the blobs from scripts/extract_blobs.sh.
"""

import argparse
import hashlib
import io
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import numpy.typing as npt

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import pipeline  # noqa: E402
from extract_fixtures import PLANE_NAMES, parse_xl2hb_capture  # noqa: E402
from fine_encoder import compress_pattern_encode, compress_rle_preencode  # noqa: E402
from settings import PrintSettings, Resolution  # noqa: E402

BROTHER_DIR = ROOT / ".brother-blobs" / "extracted" / "usr" / "local" / "Brother"
RC_TEMPLATE = BROTHER_DIR / "Printer" / "hl4150cdn" / "inf" / "brhl4150cdnrc"
CACHE_DIR = ROOT / "build" / "fine-diff-cache"
DOCKER_IMAGE = "i386/debian:bookworm-slim"
FILTER = "/usr/local/Brother/Printer/hl4150cdn/lpd/brhl4150cdnfilter"
PAPERINF = "/usr/local/Brother/Printer/hl4150cdn/inf/paperinfij2"

# A4 at 600 dpi as the original reads it; rows missing at EOF print black there.
A4_W, A4_H = 4760, 6812


# ---------------------------------------------------------------------------
# Test pages
# ---------------------------------------------------------------------------


def make_page(pattern: str, top: int, lines: int, seed: int) -> npt.NDArray[np.uint8]:
    """Return an A4 (height, width, 3) page: white, with `pattern` in rows top..top+lines.

    Raises:
        SystemExit: For an unknown pattern or a file of the wrong size.
    """
    if pattern.startswith("file:"):
        return read_ppm(Path(pattern[5:]))
    page = np.full((A4_H, A4_W, 3), 255, np.uint8)
    band = page[top : top + lines]
    n = band.shape[0]
    if pattern == "white":
        pass
    elif pattern == "black":
        band[:] = 0
    elif pattern.startswith("solid:"):
        band[:] = [int(v) for v in pattern[6:].split(",")]
    elif pattern == "hramp":
        band[:] = (np.arange(A4_W) * 256 // A4_W).astype(np.uint8)[None, :, None]
    elif pattern == "steps":
        band[:] = (np.arange(A4_W) * 16 // A4_W * 17).astype(np.uint8)[None, :, None]
    elif pattern == "bars":
        colours = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)]
        colours.append((128, 128, 128))
        for i in range(n):
            band[i] = colours[i * len(colours) // n]
    elif pattern == "noise":
        band[:] = np.random.default_rng(seed).integers(0, 256, band.shape, dtype=np.uint8)
    else:
        msg = f"unknown pattern {pattern!r}"
        raise SystemExit(msg)
    return page


def read_ppm(path: Path) -> npt.NDArray[np.uint8]:
    """Read a binary A4 PPM (P6, maxval 255).

    Raises:
        SystemExit: If the file is not a 4760x6812 P6 image.
    """
    data = path.read_bytes()
    fields: list[bytes] = []
    pos = 0
    while len(fields) < 4:
        while data[pos : pos + 1].isspace():
            pos += 1
        if data[pos : pos + 1] == b"#":
            pos = data.index(b"\n", pos)
            continue
        end = pos
        while not data[end : end + 1].isspace():
            end += 1
        fields.append(data[pos:end])
        pos = end
    if fields[0] != b"P6" or (int(fields[1]), int(fields[2])) != (A4_W, A4_H) or fields[3] != b"255":
        msg = f"{path}: need a {A4_W}x{A4_H} P6 PPM with maxval 255"
        raise SystemExit(msg)
    return np.frombuffer(data, np.uint8, A4_W * A4_H * 3, pos + 1).reshape(A4_H, A4_W, 3)


def ppm_bytes(page: npt.NDArray[np.uint8]) -> bytes:
    """Serialise a page as a binary PPM."""
    return b"P6\n%d %d\n255\n" % (page.shape[1], page.shape[0]) + page.tobytes()


def make_rc(overrides: list[str]) -> str:
    """Return the default rc file with Fine, A4 and the given Key=Value overrides.

    Raises:
        SystemExit: If an override is not Key=Value or names an unknown key.
    """
    values = {"BRResolution": "Fine", "PageSize": "A4"}
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            msg = f"--rc expects Key=Value, got {item!r}"
            raise SystemExit(msg)
        values[key] = value
    lines = []
    for template_line in RC_TEMPLATE.read_text().splitlines():
        key, sep, _ = template_line.partition("=")
        lines.append(f"{key}={values.pop(key)}" if sep and key in values else template_line)
    if values:
        msg = f"unknown rc keys: {', '.join(values)}"
        raise SystemExit(msg)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Running both drivers
# ---------------------------------------------------------------------------


def run_original(ppm: bytes, rc: str) -> bytes:
    """Run brhl4150cdnfilter in Docker, cached by input hash.

    Returns:
        The original's XL2HB output.

    Raises:
        SystemExit: If the blobs are missing or the filter fails.
    """
    if not RC_TEMPLATE.exists():
        msg = "Brother blobs missing, run scripts/extract_blobs.sh first"
        raise SystemExit(msg)
    key = hashlib.sha256(ppm + rc.encode()).hexdigest()[:16]
    cached = CACHE_DIR / f"{key}.xl2hb"
    if cached.exists():
        return cached.read_bytes()
    work = CACHE_DIR / f"{key}.work"
    work.mkdir(parents=True, exist_ok=True)
    (work / "page.ppm").write_bytes(ppm)
    (work / "rc").write_text(rc)
    cmd = [
        "docker", "run", "--rm", "--platform", "linux/386",
        "-v", f"{BROTHER_DIR}:/usr/local/Brother:ro",
        "-v", f"{work}:/w",
        DOCKER_IMAGE,
        "sh", "-c", f"{FILTER} -pi {PAPERINF} -rc /w/rc < /w/page.ppm > /w/out.xl2hb",
    ]  # fmt: skip
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    out = work / "out.xl2hb"
    if result.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        msg = f"original filter failed ({result.returncode}):\n{result.stderr}"
        raise SystemExit(msg)
    out.rename(cached)
    for f in work.iterdir():
        f.unlink()
    work.rmdir()
    return cached.read_bytes()


def run_python(page: npt.NDArray[np.uint8], settings: PrintSettings) -> tuple[bytes, dict[int, list]]:
    """Run the Python pipeline, recording every Fine encoder call.

    Wraps the Fine entries of `pipeline._DITHER_FNS` and `_PLANE_ENCODERS`
    for the duration of the call. The dither call for a plane immediately
    precedes its encoder call, which is how each encoder input gets its
    line index.

    Returns:
        (XL2HB output, {plane: [(line, 4bpp input, Stage-2 output), ...]}) with
        only the non-empty outputs, in emission order.
    """
    calls: dict[int, list] = defaultdict(list)
    current = {"line": -1, "plane": 0}
    dither_fns = pipeline._DITHER_FNS  # noqa: SLF001
    plane_encoders = pipeline._PLANE_ENCODERS[True]  # noqa: SLF001
    dither = dither_fns[True]
    encoders = dict(plane_encoders)

    def recording_dither(row_arr, y, width, channel=None):
        if y != current["line"]:
            current.update(line=y, plane=0)
        return dither(row_arr, y, width, channel)

    def recording_encoder(plane):
        def encode(data: bytes) -> bytes:
            out = encoders[plane](data)
            if out:
                calls[plane].append((current["line"], bytes(data), out))
            current["plane"] += 1
            return out

        return encode

    dither_fns[True] = recording_dither
    for plane in encoders:
        plane_encoders[plane] = recording_encoder(plane)
    try:
        out = io.BytesIO()
        pipeline.filter_page(A4_W, A4_H, page, settings, out)
    finally:
        dither_fns[True] = dither
        plane_encoders.update(encoders)
    return out.getvalue(), calls


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def first_diff(a: bytes, b: bytes) -> int:
    """Index of the first differing byte, or the shorter length if one is a prefix."""
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return min(len(a), len(b))


def hex_window(data: bytes, at: int, width: int = 16) -> str:
    """Hex dump of `width` bytes around offset `at`."""
    start = max(0, at - width // 2)
    return f"[{start}:] {data[start : start + width].hex(' ')}"


def block_layout(capture) -> list[tuple[str, int, int, int]]:
    """(plane, start line, block height, entry count) of every ReadImage block."""
    return [
        (PLANE_NAMES.get(b.plane_id, str(b.plane_id)), b.start_line, b.block_height, len(b.entries))
        for b in capture.blocks
    ]


def compare(expected: bytes, actual: bytes, calls: dict[int, list], show: int, dump: Path | None) -> bool:
    """Print a report of the differences; return True if the streams are identical."""
    print(f"size: original {len(expected)}, python {len(actual)}")
    if expected == actual:
        print("IDENTICAL")
        return True

    exp, act = parse_xl2hb_capture(expected, "original"), parse_xl2hb_capture(actual, "python")
    print("PJL header:", "same" if exp.pjl_header == act.pjl_header else "DIFFERENT")
    if exp.pjl_header != act.pjl_header:
        i = first_diff(exp.pjl_header, act.pjl_header)
        print(f"  original {exp.pjl_header[max(0, i - 20) : i + 40]!r}")
        print(f"  python   {act.pjl_header[max(0, i - 20) : i + 40]!r}")

    exp_layout, act_layout = block_layout(exp), block_layout(act)
    print(f"blocks: original {len(exp_layout)}, python {len(act_layout)}", end="")
    if exp_layout == act_layout:
        print(", same layout")
    else:
        i = next(
            (i for i, (x, y) in enumerate(zip(exp_layout, act_layout, strict=False)) if x != y),
            min(map(len, (exp_layout, act_layout))),
        )
        print(f", first layout difference at block {i}")
        print(f"  original (plane, start, height, entries): {exp_layout[i : i + 4]}")
        print(f"  python                                  : {act_layout[i : i + 4]}")

    headers_differ = sum(
        e.expected_blob[:14] != a.expected_blob[:14] for e, a in zip(exp.blocks, act.blocks, strict=False)
    )
    if headers_differ:
        print(f"plane headers: {headers_differ} blocks differ in the 14-byte header")

    all_ok = True
    for plane in range(4):
        exp_entries = [e for b in exp.blocks if b.plane_id == plane for e in b.entries]
        act_entries = [e for b in act.blocks if b.plane_id == plane for e in b.entries]
        recorded = calls.get(plane, [])
        n = min(len(exp_entries), len(act_entries))
        bad = [i for i in range(n) if exp_entries[i] != act_entries[i]]
        name = PLANE_NAMES[plane]
        status = f"{name}: entries original {len(exp_entries)}, python {len(act_entries)}, mismatching {len(bad)}"
        if not bad and len(exp_entries) == len(act_entries):
            print(status + "  ok")
            continue
        all_ok = False
        if bad:
            exp_bytes = sum(len(exp_entries[i]) for i in bad)
            act_bytes = sum(len(act_entries[i]) for i in bad)
            status += f" (bytes in those: original {exp_bytes}, python {act_bytes})"
        print(status)
        for i in bad[:show]:
            report_entry(plane, i, exp_entries[i], act_entries[i], recorded[i] if i < len(recorded) else None)
        if dump is not None and bad:
            dump_entries(dump, plane, bad, exp_entries, act_entries, recorded)
    return all_ok


def report_entry(plane: int, idx: int, exp: bytes, act: bytes, call: tuple | None) -> None:
    """Print one mismatching entry with our encoder input and Stage-1 output."""
    line = call[0] if call else "?"
    at = first_diff(exp, act)
    print(f"  {PLANE_NAMES[plane]} entry {idx} (page line {line}): ", end="")
    print(f"len original {len(exp)}, python {len(act)}, first diff at {at}")
    print(f"    original  {hex_window(exp, at)}")
    print(f"    python    {hex_window(act, at)}")
    if call:
        stage1 = compress_rle_preencode(call[1])
        assert compress_pattern_encode(stage1) == act, "recorded call does not reproduce the entry"
        nonzero = next((i for i, v in enumerate(call[1]) if v), 0)
        print(f"    4bpp in   len {len(call[1])}, first non-zero at {nonzero}: {hex_window(call[1], nonzero + 8)}")
        print(f"    stage 1   len {len(stage1)}: {hex_window(stage1, at)}")


def dump_entries(dump: Path, plane: int, bad: list[int], exp: list[bytes], act: list[bytes], recorded: list) -> None:
    """Save mismatching entries as npz: 4bpp input, our output, the original's output."""
    dump.mkdir(parents=True, exist_ok=True)
    rows = [i for i in bad if i < len(recorded)]
    path = dump / f"plane_{PLANE_NAMES[plane]}.npz"
    np.savez_compressed(
        path,
        lines=np.array([recorded[i][0] for i in rows]),
        inputs=np.array([np.frombuffer(recorded[i][1], np.uint8) for i in rows], dtype=object),
        python=np.array([np.frombuffer(act[i], np.uint8) for i in rows], dtype=object),
        original=np.array([np.frombuffer(exp[i], np.uint8) for i in rows], dtype=object),
    )
    print(f"    dumped {len(rows)} entries to {path}")


def main() -> int:
    """Run both drivers on one test page and print the comparison."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pattern", help="test page pattern, see above")
    parser.add_argument("--top", type=int, default=100, help="first row of the pattern (default 100)")
    parser.add_argument("--lines", type=int, default=16, help="rows of pattern, the rest is white (default 16)")
    parser.add_argument("--seed", type=int, default=0, help="seed for the noise pattern")
    parser.add_argument("--rc", action="append", default=[], metavar="KEY=VALUE", help="override an rc setting")
    parser.add_argument("--show", type=int, default=3, help="mismatching entries to show per plane (default 3)")
    parser.add_argument("--dump", type=Path, help="directory to save mismatching entries to")
    parser.add_argument("--save", type=Path, help="directory to save both XL2HB streams to")
    args = parser.parse_args()

    page = make_page(args.pattern, args.top, args.lines, args.seed)
    rc = make_rc(args.rc)
    # Both drivers read the same rc file.
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rc_path = CACHE_DIR / "last.rc"
    rc_path.write_text(rc)
    settings = PrintSettings.from_rc_file(str(rc_path))
    if settings.resolution != Resolution.FINE:
        print("note: rc is not set to Fine, comparing Normal mode")

    expected = run_original(ppm_bytes(page), rc)
    actual, calls = run_python(page, settings)
    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
        (args.save / "original.xl2hb").write_bytes(expected)
        (args.save / "python.xl2hb").write_bytes(actual)
    return 0 if compare(expected, actual, calls, args.show, args.dump) else 1


if __name__ == "__main__":
    sys.exit(main())
