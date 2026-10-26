"""Run leaf functions of brhl4150cdnfilter in a CPU emulator.

Loads the i386 ELF's PT_LOAD segments into Unicorn and calls a function
with the cdecl convention, so self-contained routines of the original
(no calls, no syscalls) can serve as a reference for their Python ports.

    uv run --with unicorn python scripts/native_codec.py [N] [--cython]

compares `fine_encoder` against the binary on N random scanlines
(default 2000) and prints the first mismatch per stage. With --cython it
checks the native `_fine_fast` codec instead (build it first, see
setup_cython.py).
"""

import argparse
import importlib
import random
import struct
import sys
from pathlib import Path

from unicorn import UC_ARCH_X86, UC_MODE_32, UC_PROT_ALL, Uc
from unicorn.x86_const import UC_X86_REG_EAX, UC_X86_REG_EIP, UC_X86_REG_ESP

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from fine_encoder import compress_pattern_encode, compress_rle_preencode  # noqa: E402

BINARY = ROOT / ".brother-blobs/extracted/usr/local/Brother/Printer/hl4150cdn/lpd/brhl4150cdnfilter"

RLE_PREENCODE = 0x08050F6C  # (src, dst, len) -> len in eax
PATTERN_ENCODE = 0x0805139C  # (src, dst, len, char *checksum) -> len in eax

_STACK_TOP = 0x7FF00000
_STACK_SIZE = 0x100000
_SCRATCH = 0x60000000  # input, output and checksum buffers
_SCRATCH_SIZE = 0x40000
_RETURN = 0x5FFFF000  # return address; emulation stops when it gets there


def _page(n: int, *, up: bool = False) -> int:
    return (n + 0xFFF) & ~0xFFF if up else n & ~0xFFF


class NativeFilter:
    """The filter binary mapped into a fresh 32-bit x86 emulator."""

    def __init__(self, path: Path = BINARY) -> None:
        """Map the binary's loadable segments plus stack and scratch memory."""
        elf = path.read_bytes()
        self.uc = Uc(UC_ARCH_X86, UC_MODE_32)
        e_phoff, e_phentsize, e_phnum = struct.unpack_from("<I", elf, 0x1C)[0], *struct.unpack_from("<HH", elf, 0x2A)
        for i in range(e_phnum):
            p_type, p_offset, p_vaddr, _, p_filesz, p_memsz = struct.unpack_from("<6I", elf, e_phoff + i * e_phentsize)
            if p_type != 1:  # PT_LOAD
                continue
            start, end = _page(p_vaddr), _page(p_vaddr + p_memsz, up=True)
            self.uc.mem_map(start, end - start, UC_PROT_ALL)
            self.uc.mem_write(p_vaddr, elf[p_offset : p_offset + p_filesz])
        self.uc.mem_map(_STACK_TOP - _STACK_SIZE, _STACK_SIZE, UC_PROT_ALL)
        self.uc.mem_map(_SCRATCH, _SCRATCH_SIZE, UC_PROT_ALL)
        self.uc.mem_map(_RETURN, 0x1000, UC_PROT_ALL)

    def call(self, addr: int, *args: int) -> int:
        """Call `addr` with 32-bit cdecl arguments and return eax."""
        esp = _STACK_TOP - 0x1000
        frame = struct.pack(f"<{len(args) + 1}I", _RETURN, *args)
        esp -= len(frame)
        self.uc.mem_write(esp, frame)
        self.uc.reg_write(UC_X86_REG_ESP, esp)
        self.uc.reg_write(UC_X86_REG_EIP, addr)
        self.uc.emu_start(addr, _RETURN)
        return self.uc.reg_read(UC_X86_REG_EAX)

    def _run_codec(self, addr: int, data: bytes, *, checksum: bool) -> bytes:
        src, dst, cksum = _SCRATCH, _SCRATCH + 0x10000, _SCRATCH + 0x3F000
        self.uc.mem_write(src, data)
        self.uc.mem_write(cksum, b"\0")
        args = (src, dst, len(data), cksum) if checksum else (src, dst, len(data))
        n = self.call(addr, *args)
        return bytes(self.uc.mem_read(dst, n))

    def rle_preencode(self, data: bytes) -> bytes:
        """Stage 1 of the Fine codec, as the binary computes it."""
        return self._run_codec(RLE_PREENCODE, data, checksum=False)

    def pattern_encode(self, data: bytes) -> bytes:
        """Stage 2 of the Fine codec, as the binary computes it."""
        return self._run_codec(PATTERN_ENCODE, data, checksum=True)


def random_line(rng: random.Random, n: int = 2380) -> bytes:
    """A 4bpp scanline mixing white/black runs, saturated nibbles and noise."""
    palette = [0x00, 0xFF, 0x0F, 0xF0, 0x0E, 0xE0, 0xF6, 0x6F, 0x44, 0x20, 0x02]
    out = bytearray()
    while len(out) < n:
        kind = rng.random()
        if kind < 0.3:
            out += bytes([rng.choice((0x00, 0xFF))]) * rng.randint(1, 40)
        elif kind < 0.6:
            unit = bytes(rng.choice(palette) for _ in range(rng.randint(1, 6)))
            out += unit * rng.randint(1, 12)
        else:
            out += bytes(rng.choice(palette) if rng.random() < 0.7 else rng.randrange(256) for _ in range(8))
    return bytes(out[:n])


def main() -> int:
    """Fuzz both Fine codec stages against the binary."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("count", nargs="?", type=int, default=2000, help="random scanlines to test (default 2000)")
    parser.add_argument("--cython", action="store_true", help="test the _fine_fast codec instead of fine_encoder")
    parser.add_argument("--seed", type=int, default=1, help="seed for the random scanlines (default 1)")
    args = parser.parse_args()
    count = args.count
    rle_preencode, pattern_encode = compress_rle_preencode, compress_pattern_encode
    if args.cython:
        fast = importlib.import_module("_fine_fast")
        rle_preencode, pattern_encode = fast.rle_preencode, fast.pattern_encode
    native = NativeFilter()
    rng = random.Random(args.seed)  # noqa: S311 - reproducible test data
    bad = {"stage 1": 0, "stage 2": 0}
    for i in range(count):
        line = random_line(rng)
        s1 = native.rle_preencode(line)
        if rle_preencode(line) != s1:
            if not bad["stage 1"]:
                print(f"stage 1 differs on case {i}: input {line[:24].hex(' ')} ...")
            bad["stage 1"] += 1
        s2 = native.pattern_encode(s1)
        ours = pattern_encode(s1)
        if ours != s2:
            if not bad["stage 2"]:
                at = next(
                    (j for j, (a, b) in enumerate(zip(ours, s2, strict=False)) if a != b), min(len(ours), len(s2))
                )
                print(f"stage 2 differs on case {i} at byte {at} (len native {len(s2)}, ours {len(ours)})")
                print(f"  native {s2[max(0, at - 8) : at + 12].hex(' ')}")
                print(f"  ours   {ours[max(0, at - 8) : at + 12].hex(' ')}")
            bad["stage 2"] += 1
    for stage, n in bad.items():
        print(f"{stage}: {count - n} of {count} identical")
    return 1 if any(bad.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
