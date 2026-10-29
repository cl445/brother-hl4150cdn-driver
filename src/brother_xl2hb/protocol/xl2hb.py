"""XL2HB protocol emitter.

Wraps compressed plane data in the XL2HB byte stream (a PCL-XL
variant) Brother's colour lasers accept. Operators, attribute IDs and
the band-config encoding are documented inline; what a model writes for
its paper sizes and media is in `brother_xl2hb.models`.
"""

import struct
from typing import BinaryIO

# ---------------------------------------------------------------------------
# PCL-XL data type tags
# ---------------------------------------------------------------------------
TAG_UBYTE = 0xC0
TAG_UINT16 = 0xC1
TAG_SINT16_XY = 0xD3
TAG_UINT16_XY = 0xD1
TAG_UBYTE_ARRAY = 0xC8
TAG_UINT16_ARRAY = 0xC9
TAG_ATTR = 0xF8
TAG_EXTENDED_DATA_UINT32 = 0xFA
TAG_EXTENDED_DATA_UINT8 = 0xFB

# ---------------------------------------------------------------------------
# Attribute IDs
# ---------------------------------------------------------------------------
# BeginSession
ATTR_MEASURE = 0x86
ATTR_UNITS_PER_MEASURE = 0x89

# OpenDataSource (opcode 0x48)
ATTR_SOURCE_TYPE = 0x88
ATTR_DATA_ORG = 0x82

# Page setup
ATTR_ORIENTATION = 0x28
ATTR_MEDIA_SOURCE = 0x26
ATTR_MEDIA_SIZE = 0x25
ATTR_MEDIA_TYPE = 0x27
ATTR_SIMPLEX_PAGE_MODE = 0x34
ATTR_DUPLEX_PAGE_MODE = 0x35
# Brother emits attr 0x81 = 0x10 ahead of every long-edge duplex back page.
ATTR_DUPLEX_BACK_SIDE = 0x81
DUPLEX_BACK_SIDE_MARKER = 0x10
ATTR_PAGE_ORIGIN = 0x2A

# BeginImage
ATTR_COLOR_MAPPING = 0x64
ATTR_COLOR_DEPTH = 0x62
ATTR_SOURCE_WIDTH = 0x6C
ATTR_SOURCE_HEIGHT = 0x6B
ATTR_DESTINATION_SIZE = 0x67
ATTR_COLOR_TREATMENT = 0x81
ATTR_PAGE_COPIES = 0x31

# ReadImage
ATTR_START_LINE = 0x6D
ATTR_BLOCK_HEIGHT = 0x63
ATTR_COMPRESS_MODE = 0x65

# ---------------------------------------------------------------------------
# Opcodes
# ---------------------------------------------------------------------------
OP_BEGIN_SESSION = 0x41
OP_END_SESSION = 0x42
OP_BEGIN_PAGE = 0x43
OP_END_PAGE = 0x44
OP_OPEN_DATA_SOURCE = 0x48  # 0x48 instead of the standard PCL-XL 0x47
OP_CLOSE_DATA_SOURCE = 0x49
OP_SET_PAGE_ORIGIN = 0x75
OP_BEGIN_IMAGE = 0xB0
OP_READ_IMAGE = 0xB1
OP_END_IMAGE = 0xB2


# ---------------------------------------------------------------------------
# Plane parameters
# ---------------------------------------------------------------------------
# (quant_type, comp_size) per plane — from band config and captures
PLANE_PARAMS = {
    0: (2, 12),  # K: type=2, comp_size=12
    1: (2, 20),  # C: type=2, comp_size=20
    2: (4, 10),  # M: type=4, comp_size=10
    3: (2, 12),  # Y: type=2, comp_size=12
}

# Fine mode: all planes use quant_type=0, comp_size=0
PLANE_PARAMS_FINE = {
    0: (0, 0),  # K
    1: (0, 0),  # C
    2: (0, 0),  # M
    3: (0, 0),  # Y
}

# Plane flush order: C, M, Y, K.
FLUSH_ORDER = (1, 2, 3, 0)


# PlaneBuffer sizing constants.
PLANE_BUF_INIT_FREE = 0x7FF2  # 32754 bytes: 0x8000 minus the 14-byte plane header
PLANE_BUF_FLUSH_THRESH = 0x1688  # 5768 bytes


# ---------------------------------------------------------------------------
# PCL-XL primitive emitters
# ---------------------------------------------------------------------------


def emit_ubyte_attr(buf: bytearray, val: int, attr_id: int) -> None:
    """Emit: C0 val F8 attr_id."""
    buf.append(TAG_UBYTE)
    buf.append(val & 0xFF)
    buf.append(TAG_ATTR)
    buf.append(attr_id)


def emit_uint16_attr(buf: bytearray, val: int, attr_id: int) -> None:
    """Emit: C1 lo hi F8 attr_id."""
    buf.append(TAG_UINT16)
    buf.extend(struct.pack("<H", val & 0xFFFF))
    buf.append(TAG_ATTR)
    buf.append(attr_id)


def emit_sint16_xy_attr(buf: bytearray, x: int, y: int, attr_id: int) -> None:
    """Emit: D3 x_lo x_hi y_lo y_hi F8 attr_id."""
    buf.append(TAG_SINT16_XY)
    buf.extend(struct.pack("<hh", x, y))
    buf.append(TAG_ATTR)
    buf.append(attr_id)


def emit_uint16_xy_attr(buf: bytearray, x: int, y: int, attr_id: int) -> None:
    """Emit: D1 x_lo x_hi y_lo y_hi F8 attr_id."""
    buf.append(TAG_UINT16_XY)
    buf.extend(struct.pack("<HH", x, y))
    buf.append(TAG_ATTR)
    buf.append(attr_id)


def emit_ubyte_array_attr(buf: bytearray, data: bytes, attr_id: int) -> None:
    """Emit: C8 C0 len data... F8 attr_id.

    Raises:
        ValueError: If `data` exceeds 255 bytes (the ubyte length limit).
    """
    if len(data) > 255:
        msg = f"ubyte array too long: {len(data)} > 255"
        raise ValueError(msg)
    buf.append(TAG_UBYTE_ARRAY)
    buf.append(TAG_UBYTE)
    buf.append(len(data))
    buf.extend(data)
    buf.append(TAG_ATTR)
    buf.append(attr_id)


def emit_uint16_array_attr(buf: bytearray, values: list[int], attr_id: int) -> None:
    """Emit: C9 C1 count_lo count_hi data... F8 attr_id."""
    buf.append(TAG_UINT16_ARRAY)
    buf.append(TAG_UINT16)
    buf.extend(struct.pack("<H", len(values)))
    for v in values:
        buf.extend(struct.pack("<H", v & 0xFFFF))
    buf.append(TAG_ATTR)
    buf.append(attr_id)


def emit_opcode(buf: bytearray, op: int) -> None:
    """Emit a single opcode byte."""
    buf.append(op)


def emit_extended_data(buf: bytearray, data: bytes) -> None:
    """Emit extended data: FA len32_LE data (or FB len8 data for small)."""
    dlen = len(data)
    if dlen <= 255:
        buf.append(TAG_EXTENDED_DATA_UINT8)
        buf.append(dlen & 0xFF)
    else:
        buf.append(TAG_EXTENDED_DATA_UINT32)
        buf.extend(struct.pack("<I", dlen))
    buf.extend(data)


# ---------------------------------------------------------------------------
# Band config
# ---------------------------------------------------------------------------


def build_band_config(*, color: bool = True) -> list[int]:
    """Build the 23 x uint16 band config array for Normal mode.

    Verified byte-for-byte against captures. The third word is the colour
    flag: 1 for colour, 0 for grayscale (only the K plane carries data).

    Returns:
        23 uint16 values describing the per-plane band configuration.
    """
    return [
        0x0000,
        0x0003,
        int(color),
        0x0001,
        0x0005,
        0x0000,
        0x0004,
        0x020C,  # K: type=2, comp_size=12
        0x0001,
        0x0005,
        0x0001,
        0x0004,
        0x0214,  # C: type=2, comp_size=20
        0x0001,
        0x0005,
        0x0002,
        0x0004,
        0x040A,  # M: type=4, comp_size=10
        0x0001,
        0x0005,
        0x0003,
        0x0004,
        0x020C,  # Y: type=2, comp_size=12
    ]


def build_band_config_fine(*, color: bool = True) -> list[int]:
    """Build the 23 x uint16 band config array for Fine mode.

    Verified byte-for-byte against captures. Each plane gets [0x000C, 0x0000]
    instead of Normal's [0x0004, 0xTTCC] per-plane parameters. The third
    word is the colour flag, as in `build_band_config`.

    Returns:
        23 uint16 values describing the Fine-mode band configuration.
    """
    return [
        0x0000,
        0x0003,
        int(color),
        0x0001,
        0x0005,
        0x0000,
        0x000C,  # K
        0x0000,
        0x0001,
        0x0005,
        0x0001,
        0x000C,  # C
        0x0000,
        0x0001,
        0x0005,
        0x0002,
        0x000C,  # M
        0x0000,
        0x0001,
        0x0005,
        0x0003,
        0x000C,  # Y
        0x0000,
    ]


# ---------------------------------------------------------------------------
# PlaneBuffer — accumulates compressed scanline entries for one plane
# ---------------------------------------------------------------------------


class PlaneBuffer:
    """Accumulates compressed scanline entries for one color plane.

    Entry format:
        [leading_size_BE:2B] [data:N bytes] [zero-pad N to 4B] [trailing_size_BE:2B]
        Total per entry = ceil4(N) + 4

    After all entries: [00 00] terminator
    Then: [checksum_byte] chosen so sum(ALL bytes in block) = 0 mod 256

    Plane header (14 bytes, big-endian multi-byte fields):
        +0x00  1B  comp_type = 0 (RLE)
        +0x01  1B  bit_depth (4 in Normal mode, 12 in Fine mode)
        +0x02  1B  quant_type (2 or 4)
        +0x03  1B  comp_size (10, 12, or 20)
        +0x04  2B  row_width_BE (bytes per line; 596 for A4 on the HL-4150CDN)
        +0x06  2B  line_count_BE
        +0x08  4B  data_size_BE (= total extended data length)
        +0x0C  2B  always 0x0000
    """

    def __init__(self, plane_id: int, bpl: int, fine: bool = False) -> None:
        """Configure the encoder for the given plane and resolution mode."""
        self.plane_id = plane_id
        params = PLANE_PARAMS_FINE if fine else PLANE_PARAMS
        qt, cs = params[plane_id]
        self.quant_type = qt
        self.comp_size = cs
        self.bit_depth = 12 if fine else 4
        self.bpl = bpl
        self.entries = bytearray()
        self.line_count = 0
        self.start_line = 0
        self._first = True

    def reset(self, start_line: int) -> None:
        """Clear accumulated entries and set the next start line."""
        self.entries = bytearray()
        self.line_count = 0
        self.start_line = start_line
        self._first = True

    @property
    def free_space(self) -> int:
        """Remaining bytes before the buffer reaches its maximum capacity."""
        return PLANE_BUF_INIT_FREE - len(self.entries)

    def is_nearly_full(self) -> bool:
        """True when free space drops below the flush threshold.

        Returns:
            True if a flush should be triggered before the next entry.
        """
        return self.free_space < PLANE_BUF_FLUSH_THRESH

    def append_scanline(self, compressed: bytes, line_idx: int = -1) -> None:
        """Add a scanline entry if compressed data is non-empty.

        Empty compressed data (all-zero scanline) is skipped — the driver
        only emits entries for lines that have actual data.
        """
        if not compressed:
            return
        if self._first:
            if line_idx >= 0:
                self.start_line = line_idx
            self._first = False
        comp_len = len(compressed)
        # Leading size (big-endian)
        self.entries.extend(struct.pack(">H", comp_len))
        # Compressed data
        self.entries.extend(compressed)
        # Pad data to 4-byte alignment
        padded = (comp_len + 3) & ~3
        self.entries.extend(b"\x00" * (padded - comp_len))
        # Trailing size (big-endian)
        self.entries.extend(struct.pack(">H", comp_len))
        self.line_count += 1

    def flush(self) -> tuple[int, int, bytes] | None:
        """Flush the buffer.

        The plane_blob includes: 14B header + entries + 2B terminator + 1B checksum.

        Returns:
            Tuple `(start_line, line_count, plane_blob)`, or None if there
            are no entries to emit.
        """
        if self.line_count == 0:
            return None

        # Build the complete blob
        # Terminator: 00 00
        body = bytes(self.entries) + b"\x00\x00"

        # Total extended data size = 14 (header) + len(body) + 1 (checksum)
        total_size = 14 + len(body) + 1

        # Build 14-byte header
        header = struct.pack(
            ">BBBBHHI2s",
            0,  # comp_type = 0 (RLE)
            self.bit_depth,  # bit_depth: 4 for Normal, 12 for Fine
            self.quant_type,
            self.comp_size,
            self.bpl,  # row_width
            self.line_count,
            total_size,  # data_size = total extended data length
            b"\x00\x00",  # always zero
        )

        blob = header + body

        # Compute checksum: sum of all bytes must be 0 mod 256
        byte_sum = sum(blob) & 0xFF
        checksum = (256 - byte_sum) & 0xFF
        blob += bytes([checksum])

        start = self.start_line
        count = self.line_count
        return (start, count, blob)


# ---------------------------------------------------------------------------
# XL2HBWriter — emits the complete XL2HB binary stream
# ---------------------------------------------------------------------------


class XL2HBWriter:
    """Emits the XL2HB protocol stream."""

    STREAM_HEADER = b") BROTHER XL2HB;1;0\n"

    def __init__(self, output: BinaryIO) -> None:
        """Wrap a writable binary stream and prepare an output buffer."""
        self.out = output
        self.buf = bytearray()

    def _flush(self) -> None:
        """Write buffered bytes to output."""
        if self.buf:
            self.out.write(self.buf)
            self.buf.clear()

    def write_stream_header(self) -> None:
        """Emit the XL2HB stream identifier line."""
        self.out.write(self.STREAM_HEADER)

    def write_begin_session(self, y_resolution: int = 600) -> None:
        """Emit BeginSession with inch units.

        Args:
            y_resolution: Vertical resolution in dpi. 600 for Normal, 1200 for Fine.
        """
        emit_ubyte_attr(self.buf, 0, ATTR_MEASURE)  # eInch
        emit_uint16_xy_attr(self.buf, 600, y_resolution, ATTR_UNITS_PER_MEASURE)
        emit_opcode(self.buf, OP_BEGIN_SESSION)
        self._flush()

    def write_open_data_source(self) -> None:
        """Emit OpenDataSource (Brother opcode 0x48)."""
        emit_ubyte_attr(self.buf, 0, ATTR_SOURCE_TYPE)
        emit_ubyte_attr(self.buf, 1, ATTR_DATA_ORG)
        emit_opcode(self.buf, OP_OPEN_DATA_SOURCE)
        self._flush()

    def write_begin_page(
        self,
        media_size: int | bytes = 2,
        media_source: int = 1,
        orientation: int = 0,
        media_type: bytes = b"dRegular",
        duplex_mode: int | None = None,
        back_side_marker: bool = False,
    ) -> None:
        """Emit BeginPage with media size, source, type, and simplex/duplex mode.

        `media_size` is a PCL XL media-size enum value or a name string and
        `media_type` the MediaType string, as the model writes them (see
        `models.hl4150cdn.media_size_code` / `media_type_code`); the
        defaults are A4 and plain paper. `duplex_mode` None writes
        SimplexPageMode=0; otherwise DuplexPageMode carries the value (0x00
        long edge, 0x81 short edge, as Brother's filter). `back_side_marker`
        prefixes the long-edge back-page marker attribute.
        """
        if back_side_marker:
            emit_ubyte_attr(self.buf, DUPLEX_BACK_SIDE_MARKER, ATTR_DUPLEX_BACK_SIDE)
        emit_ubyte_attr(self.buf, orientation, ATTR_ORIENTATION)
        if media_source > 0xFF:
            emit_uint16_attr(self.buf, media_source, ATTR_MEDIA_SOURCE)
        else:
            emit_ubyte_attr(self.buf, media_source, ATTR_MEDIA_SOURCE)
        if isinstance(media_size, bytes):
            emit_ubyte_array_attr(self.buf, media_size, ATTR_MEDIA_SIZE)
        else:
            emit_ubyte_attr(self.buf, media_size, ATTR_MEDIA_SIZE)
        emit_ubyte_array_attr(self.buf, media_type, ATTR_MEDIA_TYPE)
        if duplex_mode is None:
            emit_ubyte_attr(self.buf, 0, ATTR_SIMPLEX_PAGE_MODE)
        else:
            emit_ubyte_attr(self.buf, duplex_mode, ATTR_DUPLEX_PAGE_MODE)
        emit_opcode(self.buf, OP_BEGIN_PAGE)
        self._flush()

    def write_set_page_origin(self, x: int, y: int) -> None:
        """Emit SetPageOrigin: where the image starts on the sheet, in pixels."""
        emit_sint16_xy_attr(self.buf, x, y, ATTR_PAGE_ORIGIN)
        emit_opcode(self.buf, OP_SET_PAGE_ORIGIN)
        self._flush()

    def write_begin_image(
        self,
        source_width: int,
        source_height: int,
        copies: int = 1,
        dest_width: int | None = None,
        dest_height: int | None = None,
        fine: bool = False,
        color: bool = True,
    ) -> None:
        """Emit BeginImage with dimensions, band config, and copy count.

        dest_width/dest_height override the physical output size (in session units).
        When None, they default to source dimensions (correct for 600 dpi).
        For Fine mode, uses build_band_config_fine() and Fine dimensions.
        `color` False marks a grayscale (K-only) image.
        """
        emit_ubyte_attr(self.buf, 0, ATTR_COLOR_MAPPING)  # Direct
        color_depth = 1 if fine else 0  # Fine=1, Normal=0
        emit_ubyte_attr(self.buf, color_depth, ATTR_COLOR_DEPTH)
        emit_uint16_attr(self.buf, source_width, ATTR_SOURCE_WIDTH)
        emit_uint16_attr(self.buf, source_height, ATTR_SOURCE_HEIGHT)
        dw = dest_width if dest_width is not None else source_width
        dh = dest_height if dest_height is not None else source_height
        emit_uint16_xy_attr(self.buf, dw, dh, ATTR_DESTINATION_SIZE)
        band_config = build_band_config_fine(color=color) if fine else build_band_config(color=color)
        emit_uint16_array_attr(self.buf, band_config, ATTR_COLOR_TREATMENT)
        emit_uint16_attr(self.buf, copies, ATTR_PAGE_COPIES)
        emit_opcode(self.buf, OP_BEGIN_IMAGE)
        self._flush()

    def write_read_image(self, start_line: int, block_height: int, plane_id: int, data: bytes) -> None:
        """Emit ReadImage with one plane's compressed data block."""
        emit_uint16_attr(self.buf, start_line, ATTR_START_LINE)
        emit_uint16_attr(self.buf, block_height, ATTR_BLOCK_HEIGHT)
        emit_ubyte_attr(self.buf, 1, ATTR_COMPRESS_MODE)  # RLE
        emit_uint16_attr(self.buf, plane_id, ATTR_COLOR_TREATMENT)
        emit_opcode(self.buf, OP_READ_IMAGE)
        emit_extended_data(self.buf, data)
        self._flush()

    def write_end_image(self) -> None:
        """Emit EndImage opcode."""
        emit_opcode(self.buf, OP_END_IMAGE)
        self._flush()

    def write_end_page(self, copies: int = 1) -> None:
        """Emit EndPage with copy count."""
        emit_uint16_attr(self.buf, copies, ATTR_PAGE_COPIES)
        emit_opcode(self.buf, OP_END_PAGE)
        self._flush()

    def write_close_data_source(self) -> None:
        """Emit CloseDataSource opcode."""
        emit_opcode(self.buf, OP_CLOSE_DATA_SOURCE)
        self._flush()

    def write_end_session(self) -> None:
        """Emit EndSession opcode."""
        emit_opcode(self.buf, OP_END_SESSION)
        self._flush()
