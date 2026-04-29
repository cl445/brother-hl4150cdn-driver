"""
Capture parser for XL2HB test fixtures.

Parses original driver captures at test time, extracting per-block compressed
plane data for comparison against our framing layer.  No fixture files needed.
"""

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BlockFixture:
    plane_id: int  # 0=K, 1=C, 2=M, 3=Y
    start_line: int
    block_height: int
    entries: list[bytes]  # compressed data per scanline (raw bytes)
    expected_blob: bytes  # entire extended_data blob (header + body + checksum)


@dataclass
class CaptureFixture:
    filename: str
    pjl_header: bytes
    pjl_footer: bytes
    blocks: list[BlockFixture]
    raw: bytes  # entire file for byte-for-byte comparison


# ---------------------------------------------------------------------------
# PCL-XL data type tags (same as in xl2hb.py)
# ---------------------------------------------------------------------------
_TAG_UBYTE = 0xC0
_TAG_UINT16 = 0xC1
_TAG_UINT32 = 0xC2
_TAG_SINT16 = 0xC3
_TAG_SINT32 = 0xC4
_TAG_REAL32 = 0xC5
_TAG_UBYTE_ARRAY = 0xC8
_TAG_UINT16_ARRAY = 0xC9
_TAG_UINT32_ARRAY = 0xCA
_TAG_SINT16_ARRAY = 0xCB
_TAG_SINT32_ARRAY = 0xCC
_TAG_REAL32_ARRAY = 0xCD
_TAG_UBYTE_XY = 0xD0
_TAG_UINT16_XY = 0xD1
_TAG_UINT32_XY = 0xD2
_TAG_SINT16_XY = 0xD3
_TAG_SINT32_XY = 0xD4
_TAG_REAL32_XY = 0xD5
_TAG_UBYTE_BOX = 0xE0
_TAG_UINT16_BOX = 0xE1
_TAG_REAL32_BOX = 0xE5
_TAG_ATTR = 0xF8
_TAG_EXT_DATA_UINT32 = 0xFA
_TAG_EXT_DATA_UINT8 = 0xFB

# Scalar tags: tag -> (struct_fmt, byte_count)
_SCALAR_TAGS = {
    _TAG_UBYTE: ("B", 1),
    _TAG_UINT16: ("<H", 2),
    _TAG_UINT32: ("<I", 4),
    _TAG_SINT16: ("<h", 2),
    _TAG_SINT32: ("<i", 4),
    _TAG_REAL32: ("<f", 4),
}

# XY tags: tag -> (elem_fmt, elem_size)
_XY_TAGS = {
    _TAG_UBYTE_XY: ("B", 1),
    _TAG_UINT16_XY: ("<H", 2),
    _TAG_UINT32_XY: ("<I", 4),
    _TAG_SINT16_XY: ("<h", 2),
    _TAG_SINT32_XY: ("<i", 4),
    _TAG_REAL32_XY: ("<f", 4),
}

# Array tags: tag -> elem_size
_ARRAY_TAGS = {
    _TAG_UBYTE_ARRAY: 1,
    _TAG_UINT16_ARRAY: 2,
    _TAG_UINT32_ARRAY: 4,
    _TAG_SINT16_ARRAY: 2,
    _TAG_SINT32_ARRAY: 4,
    _TAG_REAL32_ARRAY: 4,
}

# Box tags: tag -> (elem_fmt, elem_size)
_BOX_TAGS = {
    _TAG_UBYTE_BOX: ("B", 1),
    _TAG_UINT16_BOX: ("<H", 2),
    _TAG_REAL32_BOX: ("<f", 4),
}

# Opcodes we care about
_OP_READ_IMAGE = 0xB1

# Attribute IDs
_ATTR_START_LINE = 0x6D
_ATTR_BLOCK_HEIGHT = 0x63
_ATTR_COLOR_TREATMENT = 0x81

# Range for valid opcodes
_OP_RANGE = range(0x41, 0xC0)

PLANE_NAMES = {0: "K", 1: "C", 2: "M", 3: "Y"}


def _parse_blob_entries(blob: bytes) -> list[bytes]:
    """Parse compressed entries from a plane blob.

    Blob layout:
        14B header
        N x entry: [size_BE:2B] [data:size] [pad to 4B] [trailing_size_BE:2B]
        [00 00] terminator
        [1B] checksum
    """
    entries = []
    pos = 14  # skip header
    while pos + 2 <= len(blob):
        size = struct.unpack_from(">H", blob, pos)[0]
        pos += 2
        if size == 0:
            break  # terminator
        # Read compressed data
        data = blob[pos : pos + size]
        entries.append(bytes(data))
        # Skip past data + padding
        padded = (size + 3) & ~3
        pos += padded
        # Skip trailing size
        pos += 2
    return entries


def parse_xl2hb_capture(source: Path | str | bytes) -> CaptureFixture:
    """Parse an XL2HB capture file into a CaptureFixture.

    Walks the binary payload, extracts PJL header/footer and all
    ReadImage blocks with their compressed plane data.

    Args:
        source: Path to an ``.xl2hb`` file, or raw bytes.
    """
    if isinstance(source, bytes):
        raw = source
        filename = "<bytes>"
    else:
        path = Path(source)
        raw = path.read_bytes()
        filename = path.stem

    # --- Find PJL header boundary ---
    marker = b") BROTHER XL2HB"
    marker_pos = raw.find(marker)
    if marker_pos < 0:
        msg = f"No XL2HB marker in {filename}"
        raise ValueError(msg)

    # PJL header = everything before the stream header line
    pjl_header = raw[:marker_pos]

    # Stream header ends at the newline after the marker
    stream_hdr_end = raw.index(b"\n", marker_pos) + 1

    # --- Find PJL footer (double UEL) ---
    # Footer = last two UEL sequences
    uel = b"\x1b%-12345X"
    # Find the start of the double UEL at the end
    # The footer is exactly 2 x UEL = 2 x 9 bytes = 18 bytes
    footer_len = len(uel) * 2
    pjl_footer = raw[-footer_len:]
    payload_end = len(raw) - footer_len

    # --- Walk binary payload ---
    data = raw
    pos = stream_hdr_end
    end = payload_end

    pending_attrs: dict[int, Any] = {}
    blocks: list[BlockFixture] = []
    last_value = None  # tracks the most recently parsed value

    while pos < end:
        byte = data[pos]

        # --- Scalar value ---
        if byte in _SCALAR_TAGS:
            fmt, size = _SCALAR_TAGS[byte]
            pos += 1
            last_value = struct.unpack_from(fmt, data, pos)[0]
            pos += size
            # Check for attribute tag
            if pos < end and data[pos] == _TAG_ATTR:
                pos += 1
                attr_id = data[pos]
                pos += 1
                pending_attrs[attr_id] = last_value
            continue

        # --- XY pair ---
        if byte in _XY_TAGS:
            elem_fmt, elem_size = _XY_TAGS[byte]
            pos += 1
            x = struct.unpack_from(elem_fmt, data, pos)[0]
            y = struct.unpack_from(elem_fmt, data, pos + elem_size)[0]
            last_value = (x, y)
            pos += elem_size * 2
            if pos < end and data[pos] == _TAG_ATTR:
                pos += 1
                attr_id = data[pos]
                pos += 1
                pending_attrs[attr_id] = last_value
            continue

        # --- Box ---
        if byte in _BOX_TAGS:
            elem_fmt, elem_size = _BOX_TAGS[byte]
            pos += 1
            vals = struct.unpack_from(
                f"{elem_fmt[:-1]}4{elem_fmt[-1]}" if len(elem_fmt) > 1 else f"4{elem_fmt}", data, pos
            )
            last_value = vals
            pos += elem_size * 4
            if pos < end and data[pos] == _TAG_ATTR:
                pos += 1
                attr_id = data[pos]
                pos += 1
                pending_attrs[attr_id] = last_value
            continue

        # --- Array ---
        if byte in _ARRAY_TAGS:
            elem_size = _ARRAY_TAGS[byte]
            pos += 1
            # Read array length (next tag determines length encoding)
            len_tag = data[pos]
            pos += 1
            if len_tag == _TAG_UBYTE:
                arr_len = data[pos]
                pos += 1
            elif len_tag == _TAG_UINT16:
                arr_len = struct.unpack_from("<H", data, pos)[0]
                pos += 2
            elif len_tag == _TAG_UINT32:
                arr_len = struct.unpack_from("<I", data, pos)[0]
                pos += 4
            else:
                arr_len = len_tag
            total_bytes = arr_len * elem_size
            arr_data = data[pos : pos + total_bytes]
            pos += total_bytes
            last_value = arr_data
            if pos < end and data[pos] == _TAG_ATTR:
                pos += 1
                attr_id = data[pos]
                pos += 1
                pending_attrs[attr_id] = last_value
            continue

        # --- Extended data ---
        if byte == _TAG_EXT_DATA_UINT32:
            pos += 1
            ext_len = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            ext_data = bytes(data[pos : pos + ext_len])
            pos += ext_len

            # Associate with the most recent ReadImage block
            if blocks and blocks[-1].expected_blob == b"":
                entries = _parse_blob_entries(ext_data)
                blocks[-1] = BlockFixture(
                    plane_id=blocks[-1].plane_id,
                    start_line=blocks[-1].start_line,
                    block_height=blocks[-1].block_height,
                    entries=entries,
                    expected_blob=ext_data,
                )
            continue

        if byte == _TAG_EXT_DATA_UINT8:
            pos += 1
            ext_len = data[pos]
            pos += 1
            ext_data = bytes(data[pos : pos + ext_len])
            pos += ext_len

            if blocks and blocks[-1].expected_blob == b"":
                entries = _parse_blob_entries(ext_data)
                blocks[-1] = BlockFixture(
                    plane_id=blocks[-1].plane_id,
                    start_line=blocks[-1].start_line,
                    block_height=blocks[-1].block_height,
                    entries=entries,
                    expected_blob=ext_data,
                )
            continue

        # --- Opcode ---
        if byte in _OP_RANGE:
            if byte == _OP_READ_IMAGE:
                start_line = pending_attrs.get(_ATTR_START_LINE, 0)
                block_height = pending_attrs.get(_ATTR_BLOCK_HEIGHT, 0)
                plane_id = pending_attrs.get(_ATTR_COLOR_TREATMENT, 0)
                blocks.append(
                    BlockFixture(
                        plane_id=plane_id,
                        start_line=start_line,
                        block_height=block_height,
                        entries=[],
                        expected_blob=b"",  # filled when extended data follows
                    )
                )
            pending_attrs.clear()
            pos += 1
            continue

        # --- Attribute tag (standalone, shouldn't happen but handle) ---
        if byte == _TAG_ATTR:
            pos += 2  # skip attr tag + attr_id
            continue

        # Unknown byte, skip
        pos += 1

    return CaptureFixture(
        filename=filename,
        pjl_header=pjl_header,
        pjl_footer=pjl_footer,
        blocks=blocks,
        raw=raw,
    )
