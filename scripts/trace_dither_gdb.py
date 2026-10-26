"""gdb-side tracer for the 4bpp dither routines of brhl4150cdnfilter.

Loaded by `trace_dither.py` into gdb-multiarch, which is attached to the
filter running under `qemu-i386 -g` (Docker Desktop's i386 emulation has
no ptrace, so plain gdb inside an i386 container cannot trace it).

Breaks at the entry of dither_ordered_4ch_4bpp (0x08061404) and
dither_ordered_1ch_4bpp (0x080618c8). Every call is counted with its y
argument; for calls whose y lies in [TRACE_Y0, TRACE_Y1) the arguments, the
run-length input arrays and, via a FinishBreakpoint, the output buffers and
the return value are recorded. The DitherContext (g_dither_ctx at
0x08261960) and the pattern data its LUTs point to are dumped once.

Configured through environment variables:
    TRACE_OUT   output directory (default /w)
    TRACE_Y0    first traced y (default 0)
    TRACE_Y1    end of traced y range, exclusive (default 1 << 30)
    TRACE_MAX   max traced calls (default 64)

Output: TRACE_OUT/trace.json (metadata, arguments, all calls) and
TRACE_OUT/trace.bin (raw byte blobs referenced by offset/length).

Argument layout of 0x08061404 (cdecl, [ebp+off] after the prologue,
read here as 14 dwords at [esp+4] on entry), verified in the disassembly:
    +0x08 config (short*, &ps->compression_mode)
    +0x0c C, +0x10 M, +0x14 Y, +0x18 K   one 8-bit value per run
    +0x1c runs (uint16 run lengths)  +0x20 run_count
    +0x24 x_start  +0x28 y  +0x2c width (pixels, unused by the routine)
    +0x30 C out, +0x34 M out, +0x38 Y out, +0x3c K out   (nibble-packed)
Return value: plane mask, bit 0 K, 1 C, 2 M, 3 Y.
DitherContext slots (+0x04 + i*0x408): K, C, M, Y; K in -> slot 0 -> K out.
"""

import json
import os
import pathlib
import struct

import gdb  # type: ignore[import-not-found]

ADDR_4CH = 0x08061404
ADDR_1CH = 0x080618C8
G_DITHER_CTX = 0x08261960
CHANNEL_BASE = 0x04
CHANNEL_STRIDE = 0x408
CHANNEL_ORDER = ("k", "c", "m", "y")  # as the refactored C names the four slots

OUT_DIR = os.environ.get("TRACE_OUT", "/w")
Y0 = int(os.environ.get("TRACE_Y0", "0"))
Y1 = int(os.environ.get("TRACE_Y1", str(1 << 30)))
MAX_CALLS = int(os.environ.get("TRACE_MAX", "64"))

inferior = gdb.selected_inferior()
blob = bytearray()
state = {"calls": [], "all_y": {"4ch": [], "1ch": []}, "ctx": None, "traced": 0}


def read(addr: int, size: int) -> bytes:
    """Read `size` bytes of inferior memory at `addr`."""
    return bytes(inferior.read_memory(addr, size)) if size > 0 else b""


def u32(addr: int) -> int:
    """Read a little-endian uint32 from inferior memory."""
    return struct.unpack("<I", read(addr, 4))[0]


def store(data: bytes) -> list[int]:
    """Append `data` to the blob file, returning [offset, length]."""
    off = len(blob)
    blob.extend(data)
    return [off, len(data)]


def esp() -> int:
    """Return the current stack pointer."""
    return int(gdb.parse_and_eval("$esp")) & 0xFFFFFFFF


def dump_ctx() -> dict:
    """Dump the four DitherChannel structs and the pattern bytes their LUTs point to."""
    base = u32(G_DITHER_CTX)
    head = read(base, 4)
    channels = {}
    for i, name in enumerate(CHANNEL_ORDER):
        ch = base + CHANNEL_BASE + i * CHANNEL_STRIDE
        width, height, row_bytes = struct.unpack("<HHI", read(ch, 8))
        lut = list(struct.unpack("<256I", read(ch + 8, 1024)))
        level_size = (width >> 3) * height
        lo, hi = min(lut), max(lut) + level_size
        channels[name] = {
            "struct_addr": ch,
            "matrix_width": width,
            "matrix_height": height,
            "row_bytes": row_bytes,
            "lut": lut,
            "level_size": level_size,
            "pattern_base": lo,
            "pattern": store(read(lo, hi - lo)),
        }
    return {"base": base, "head": list(head), "channels": channels}


class Finish(gdb.FinishBreakpoint):
    """Records the return value and output buffers when a traced call returns."""

    def __init__(self, record: dict, outs: list[int], nbytes: int) -> None:
        """Set the finish breakpoint on the newest frame for `record`."""
        super().__init__(gdb.newest_frame(), internal=True)
        self.silent = True
        self.record, self.outs, self.nbytes = record, outs, nbytes

    def stop(self) -> bool:
        """Store return value and outputs; never stop."""
        self.record["ret"] = int(gdb.parse_and_eval("$eax")) & 0xFF
        self.record["out"] = [store(read(p, self.nbytes)) for p in self.outs]
        return False

    def out_of_scope(self) -> None:
        """Mark the record when the frame was left without a normal return."""
        self.record["ret"] = "out_of_scope"


class Entry(gdb.Breakpoint):
    """Entry breakpoint of a dither routine that counts and records calls."""

    def __init__(self, addr: int, kind: str) -> None:
        """Break at `addr`; `kind` is "4ch" or "1ch"."""
        super().__init__(f"*{addr:#x}", internal=True)
        self.silent = True
        self.kind = kind

    def stop(self) -> bool:
        """Record the call; never stop."""
        sp = esp()
        args = list(struct.unpack("<14I", read(sp + 4, 56)))
        y = struct.unpack("<i", struct.pack("<I", args[8]))[0] if self.kind == "4ch" else None
        if self.kind == "1ch":
            # Signature not recovered; record the stack words, guess y from the same slot.
            y = struct.unpack("<i", struct.pack("<I", args[8]))[0]
        state["all_y"][self.kind].append(y)
        if state["ctx"] is None:
            try:
                state["ctx"] = dump_ctx()
            except gdb.error as exc:  # pragma: no cover - diagnostic only
                state["ctx"] = {"error": str(exc)}
        if not (Y0 <= y < Y1) or state["traced"] >= MAX_CALLS:
            return False
        state["traced"] += 1
        record = {"kind": self.kind, "index": len(state["all_y"][self.kind]) - 1, "args": args, "y": y}
        if self.kind == "4ch":
            config = args[0]
            run_count, x_start, width = args[6], args[7], args[9]
            record["config_shorts"] = list(struct.unpack("<16h", read(config, 32)))
            record["runs"] = store(read(args[5], 2 * run_count))
            record["inputs"] = [store(read(args[i], run_count)) for i in (1, 2, 3, 4)]
            outs = args[10:14]
            nbytes = (x_start + width + 1) // 2
        else:
            outs = []
            nbytes = 0
        record["out_ptrs"] = outs
        state["calls"].append(record)
        if outs:
            Finish(record, outs, nbytes)
        return False


def write_results() -> None:
    """Write trace.bin and trace.json to OUT_DIR."""
    with (pathlib.Path(OUT_DIR) / "trace.bin").open("wb") as f:
        f.write(blob)
    with (pathlib.Path(OUT_DIR) / "trace.json").open("w") as f:
        json.dump(state, f)


Entry(ADDR_4CH, "4ch")
Entry(ADDR_1CH, "1ch")
gdb.events.exited.connect(lambda _event: write_results())
