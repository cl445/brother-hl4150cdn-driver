"""PrintSettings model: enums, dataclass, and the RC- and CUPS-option parsers."""

import configparser
import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Self

logger = logging.getLogger(__name__)


# --- Enums ---


class MediaType(StrEnum):
    """Paper/media type selection (maps to XL2HB MediaType attribute).

    Values are this driver's PPD choices, matched without regard to case
    (Brother's BOND, PostCard); a model's other spellings are its
    `PrinterModel.aliases`.
    """

    PLAIN = "Plain"
    THIN = "Thin"
    THICK = "Thick"
    THICKER = "Thicker"
    BOND = "Bond"
    ENVELOPE = "Envelope"
    ENV_THIN = "EnvThin"
    ENV_THICK = "EnvThick"
    RECYCLED = "Recycled"
    POSTCARD = "Postcard"
    LABEL = "Label"
    GLOSSY = "Glossy"

    @classmethod
    def _missing_(cls, value: object) -> "MediaType | None":
        if isinstance(value, str):
            for member in cls:
                if member.value.lower() == value.lower():
                    return member
        return None


class PageSize(StrEnum):
    """Page size selection, by PPD PageSize; a model's other spellings are its `PrinterModel.aliases`."""

    A4 = "A4"
    LETTER = "Letter"
    LEGAL = "Legal"
    EXECUTIVE = "Executive"
    A5 = "A5"
    A5_ROTATED = "PRA5Rotated"
    A6 = "A6"
    ISOB5 = "ISOB5"
    ISOB6 = "ISOB6"
    JISB5 = "JISB5"
    JISB6 = "JISB6"
    POSTCARD = "Postcard"
    ENV_DL = "EnvDL"
    ENV_C5 = "EnvC5"
    ENV_10 = "Env10"
    ENV_MONARCH = "EnvMonarch"
    BR_3X5 = "Br3x5"
    FOLIO = "FanFoldGermanLegal"
    ENV_DL_LONG_EDGE = "EnvPRC5Rotated"
    ENV_YOU4 = "EnvYou4"
    ENV_CHOU3 = "EnvChou3"


class Resolution(StrEnum):
    """Print quality: Normal (600 dpi) or Fine (600 dpi raster + 2400 dpi-class dithering)."""

    NORMAL = "Normal"
    FINE = "Fine"


class DuplexMode(StrEnum):
    """Duplex printing mode (maps to XL2HB DuplexPageMode attribute)."""

    NONE = "None"
    NO_TUMBLE = "DuplexNoTumble"
    TUMBLE = "DuplexTumble"


class MonoColor(StrEnum):
    """Color mode: Auto (detect), FullColor, or Mono (K-only grayscale)."""

    AUTO = "Auto"
    FULL_COLOR = "FullColor"
    MONO = "Mono"


class ColorMatching(StrEnum):
    """Color matching profile for RGB-to-CMYK conversion."""

    NORMAL = "Normal"
    VIVID = "Vivid"
    NONE = "None"


class ImproveOutput(StrEnum):
    """Output improvement mode (PJL LESSPAPERCURL / FIXINTENSITYUP)."""

    OFF = "OFF"
    LESS_PAPER_CURL = "BRLessPaperCurl"
    FIX_INTENSITY = "BRFixIntensity"


class InputSlot(StrEnum):
    """Paper input tray selection (maps to the XL2HB MediaSource attribute)."""

    AUTO = "AutoSelect"
    TRAY1 = "Tray1"
    TRAY2 = "Tray2"
    MP_TRAY = "MPTray"
    MANUAL = "Manual"


Aliases = Mapping[str, Mapping[str, str]]
"""A model's spellings of option values: settings field -> {spelling: value}.

Spellings match without regard to case, e.g. {"media_type": {"Env": "Envelope"}}.
"""


def alias_lookup(aliases: Aliases | None) -> Callable[[str, str], str]:
    """Return `spelled(field, raw)`: `raw` translated through `aliases[field]`, else unchanged.

    Returns:
        The lookup function.
    """
    folded = {attr: {k.casefold(): v for k, v in table.items()} for attr, table in (aliases or {}).items()}

    def spelled(attr: str, raw: str) -> str:
        return folded.get(attr, {}).get(str(raw).casefold(), raw)

    return spelled


# --- Print settings ---


@dataclass
class PrintSettings:
    """Print settings, corresponding to brhl4150cdnrc."""

    media_type: MediaType = MediaType.PLAIN
    page_size: PageSize = PageSize.A4
    input_slot: InputSlot = InputSlot.AUTO
    resolution: Resolution = Resolution.NORMAL
    copies: int = 1
    duplex: DuplexMode = DuplexMode.NONE
    mono_color: MonoColor = MonoColor.AUTO
    color_matching: ColorMatching = ColorMatching.NORMAL
    improve_gray: bool = False
    enhance_black: bool = False
    toner_save: bool = False
    improve_output: ImproveOutput = ImproveOutput.OFF
    brightness: int = 0
    contrast: int = 0
    red: int = 0
    green: int = 0
    blue: int = 0
    saturation: int = 0
    skip_blank: bool = False
    reverse: bool = False

    @classmethod
    def from_rc_file(cls, path: str, aliases: Aliases | None = None) -> Self:
        """Read settings from Brother's rc file (brhl4150cdnrc).

        `aliases` are the model's spellings of option values (see
        `PrinterModel.aliases`).

        Values are read as brhl4150cdnfilter reads them (checked against
        it in Docker): an unknown choice keeps the default, with a
        warning, and only an exact ON is on; a BRResolution other than
        Normal is Fine; numbers are C `atoi`, so "50%" is 50, "abc" is 0
        and they are not limited to the PPD's -20..20 (only to
        ±`_RC_INT_LIMIT`, where the colour maths still fits 32 bits, as
        in the original). Copies outside 1..`MAX_COPIES` are limited,
        unlike the original, which sends PageCopies 0 for 0 and 65535
        for -1.

        Returns:
            Populated settings instance (defaults if the file has no sections).
        """
        spelled = alias_lookup(aliases)
        settings = cls()
        config = configparser.ConfigParser(interpolation=None)
        config.read(path)
        if not config.sections():
            return settings
        s = config[config.sections()[0]]

        enum_map: dict[str, tuple[str, type[StrEnum]]] = {
            "MediaType": ("media_type", MediaType),
            "PageSize": ("page_size", PageSize),
            "InputSlot": ("input_slot", InputSlot),
            "Duplex": ("duplex", DuplexMode),
            "BRMonoColor": ("mono_color", MonoColor),
            "BRColorMatching": ("color_matching", ColorMatching),
            "BRImproveOutput": ("improve_output", ImproveOutput),
        }
        for rc_key, (attr, enum_cls) in enum_map.items():
            if rc_key in s:
                try:
                    setattr(settings, attr, enum_cls(spelled(attr, s[rc_key])))
                except ValueError:
                    logger.warning("Unknown %s value %r, keeping default", rc_key, s[rc_key])

        if "BRResolution" in s:
            if s["BRResolution"] not in tuple(Resolution):
                logger.warning("Unknown BRResolution value %r, using Fine", s["BRResolution"])
            settings.resolution = Resolution.NORMAL if s["BRResolution"] == Resolution.NORMAL else Resolution.FINE

        if "Copies" in s:
            settings.copies = valid_copies(_atoi(s["Copies"]))

        bool_map: dict[str, str] = {
            "BRGray": "improve_gray",
            "BREnhanceBlkPrt": "enhance_black",
            "TonerSaveMode": "toner_save",
            "BRSkipBlank": "skip_blank",
            "BRReverse": "reverse",
        }
        for rc_key, attr in bool_map.items():
            setattr(settings, attr, s.get(rc_key, "OFF") == "ON")

        int_map: dict[str, str] = {
            "Brightness": "brightness",
            "Contrast": "contrast",
            "RedKey": "red",
            "GreenKey": "green",
            "BlueKey": "blue",
            "Saturation": "saturation",
        }
        for rc_key, attr in int_map.items():
            if rc_key in s:
                value = _atoi(s[rc_key])
                if abs(value) > _RC_INT_LIMIT:
                    value = max(-_RC_INT_LIMIT, min(_RC_INT_LIMIT, value))
                    logger.warning("%s=%s out of range, using %d", rc_key, s[rc_key], value)
                setattr(settings, attr, value)
        return settings

    @classmethod
    def from_cups_options(
        cls,
        options_str: str,
        copies: int = 1,
        ppd_defaults: Mapping[str, str] | None = None,
        aliases: Aliases | None = None,
    ) -> Self:
        """Parse a CUPS option string into PrintSettings.

        CUPS passes options as space-separated Key=Value pairs, e.g.:
        ``"PageSize=A4 Duplex=DuplexNoTumble BRBrightness=5"``, quoted as
        `parse_cups_options` reads them. It only passes what the job
        carries, so the queue's PPD defaults (`ppd_defaults`, see
        `read_ppd_defaults`) apply first and the job's options override
        them, as Brother's cupswrapper does. The IPP options of generic
        clients, `sides` and `print-color-mode=monochrome`, beat the PPD
        defaults but not the job's own Duplex or BRMonoColor. IPP
        `print-quality` is not mapped: Fine is a separate, much slower
        mode, chosen with BRResolution only.

        `aliases` are the model's spellings of option values (see
        `PrinterModel.aliases`).

        Returns:
            Populated settings instance with the parsed values.
        """
        spelled = alias_lookup(aliases)
        settings = cls()
        settings.copies = valid_copies(copies)

        job_opts = parse_cups_options(options_str)
        opts = {**(ppd_defaults or {}), **job_opts}

        # Enum options
        enum_map: dict[str, tuple[str, type]] = {
            "PageSize": ("page_size", PageSize),
            "BRDuplex": ("duplex", DuplexMode),  # pre-2026-09 PPD name
            "Duplex": ("duplex", DuplexMode),
            "BRInputSlot": ("input_slot", InputSlot),
            "BRMonoColor": ("mono_color", MonoColor),
            "BRMediaType": ("media_type", MediaType),
            "BRColorMatching": ("color_matching", ColorMatching),
            "BRImproveOutput": ("improve_output", ImproveOutput),
        }
        for cups_key, (attr, enum_cls) in enum_map.items():
            if cups_key in opts:
                try:
                    setattr(settings, attr, enum_cls(spelled(attr, opts[cups_key])))
                except ValueError:
                    logger.warning("Unknown %s value %r, keeping default", cups_key, opts[cups_key])

        # IPP "sides" when the job has no PPD duplex option.
        if "Duplex" not in job_opts and "BRDuplex" not in job_opts and "sides" in opts:
            sides = _SIDES_TO_DUPLEX.get(opts["sides"])
            if sides is None:
                logger.warning("Unknown sides value %r, keeping default", opts["sides"])
            else:
                settings.duplex = sides

        # IPP "print-color-mode" when the job has no BRMonoColor: monochrome
        # prints K-only; color and auto leave the PPD's choice (Auto by
        # default), since FullColor would turn off the colour detection.
        if "BRMonoColor" not in job_opts and "print-color-mode" in opts:
            mode = opts["print-color-mode"]
            if mode in _MONOCHROME_COLOR_MODES:
                settings.mono_color = MonoColor.MONO
            elif mode not in _COLOR_MODES:
                logger.warning("Unknown print-color-mode value %r, keeping default", mode)

        # Resolution: "600x2400dpi" -> Fine, else Normal
        if "BRResolution" in opts:
            settings.resolution = Resolution.FINE if opts["BRResolution"] == "600x2400dpi" else Resolution.NORMAL

        # Boolean options (ON/OFF, also true/false and yes/no in any case:
        # `-o noBRGray` reaches the filter as BRGray=false)
        bool_map: dict[str, str] = {
            "BRTonerSaveMode": "toner_save",
            "BRSkipBlank": "skip_blank",
            "BRGray": "improve_gray",
            "BREnhanceBlkPrt": "enhance_black",
            "BRReverse": "reverse",
        }
        for cups_key, attr in bool_map.items():
            if cups_key in opts:
                value = _BOOLEANS.get(opts[cups_key].lower())
                if value is None:
                    logger.warning("Unknown %s value %r, keeping default", cups_key, opts[cups_key])
                else:
                    setattr(settings, attr, value)

        # Integer options (-20..+20)
        int_map: dict[str, str] = {
            "BRBrightness": "brightness",
            "BRContrast": "contrast",
            "BRRed": "red",
            "BRGreen": "green",
            "BRBlue": "blue",
            "BRSaturation": "saturation",
        }
        for cups_key, attr in int_map.items():
            if cups_key in opts:
                try:
                    val = int(opts[cups_key])
                    setattr(settings, attr, max(-20, min(20, val)))
                except ValueError:
                    logger.warning("Non-integer value %r for %s, keeping default", opts[cups_key], cups_key)

        return settings


_PPD_DEFAULT = re.compile(r"^\*Default(\w+):\s*(\S+)", re.MULTILINE)


def read_ppd_defaults(path: str | Path) -> dict[str, str]:
    """Read the `*DefaultKey: Value` choices of a PPD file as CUPS-style options.

    Returns:
        Option name → default choice; empty if the file cannot be read.
    """
    try:
        text = Path(path).read_text(encoding="latin-1")
    except OSError as exc:
        logger.warning("Cannot read PPD %s: %s", path, exc)
        return {}
    return dict(_PPD_DEFAULT.findall(text))


def parse_cups_options(options: str) -> dict[str, str]:
    """Split a CUPS option string into options as `cupsParseOptions` does.

    Options are separated by whitespace. A value may be quoted with '…' or
    "…" (quotes removed), a backslash takes the next character literally,
    and a {…} collection is kept as it is. A name without `=` is a
    boolean: ``noName`` gives Name=false, any other name=true. A later
    option of the same name replaces an earlier one.

    Returns:
        Option name → value.
    """
    opts: dict[str, str] = {}
    pos, end = 0, len(options)
    while pos < end:
        while pos < end and options[pos].isspace():
            pos += 1
        start = pos
        while pos < end and not options[pos].isspace() and options[pos] != "=":
            pos += 1
        name = options[start:pos]
        if not name:
            break
        while pos < end and options[pos].isspace():
            pos += 1
        if pos >= end or options[pos] != "=":
            if name[:2].lower() == "no":
                opts[name[2:]] = "false"
            else:
                opts[name] = "true"
            continue
        pos += 1
        value: list[str] = []
        while pos < end and not options[pos].isspace():
            char = options[pos]
            if char in "'\"":
                pos += 1
                while pos < end and options[pos] != char:
                    if options[pos] == "\\" and pos + 1 < end:
                        pos += 1
                    value.append(options[pos])
                    pos += 1
                pos += 1
            elif char == "{":
                depth = 0
                while pos < end:
                    if options[pos] == "\\" and pos + 1 < end:
                        pos += 1
                    elif options[pos] == "{":
                        depth += 1
                    elif options[pos] == "}":
                        depth -= 1
                    value.append(options[pos])
                    pos += 1
                    if depth == 0:
                        break
            else:
                if char == "\\" and pos + 1 < end:
                    pos += 1
                value.append(options[pos])
                pos += 1
        opts[name] = "".join(value)
    return opts


MAX_COPIES = 9999
"""Most copies a job may ask for: CUPS's default MaxCopies; PageCopies is 16 bits."""


def valid_copies(copies: int) -> int:
    """`copies` limited to 1..`MAX_COPIES`, with a warning if it was outside.

    Returns:
        The number of copies to print.
    """
    if 1 <= copies <= MAX_COPIES:
        return copies
    limited = max(1, min(MAX_COPIES, copies))
    logger.warning("Copies=%d out of range, printing %d", copies, limited)
    return limited


_RC_INT_LIMIT = 1_000_000
"""Largest rc colour adjustment kept; far past where the colour transforms saturate."""

# ASCII only: C atoi skips C-locale whitespace and reads 0-9, no other digits.
_ATOI = re.compile(r"\s*([+-]?\d+)", re.ASCII)


def _atoi(text: str) -> int:
    """`text` read as C's `atoi` reads it: the leading integer, else 0.

    Returns:
        The integer.
    """
    match = _ATOI.match(text)
    return int(match.group(1)) if match else 0


_SIDES_TO_DUPLEX = {
    "one-sided": DuplexMode.NONE,
    "two-sided-long-edge": DuplexMode.NO_TUMBLE,
    "two-sided-short-edge": DuplexMode.TUMBLE,
}

_MONOCHROME_COLOR_MODES = frozenset({"monochrome", "bi-level", "process-monochrome", "process-bi-level"})
_COLOR_MODES = frozenset({"color", "auto", "auto-monochrome", "highlight"})

_BOOLEANS = {"on": True, "true": True, "yes": True, "off": False, "false": False, "no": False}
