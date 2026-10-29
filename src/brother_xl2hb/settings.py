"""PrintSettings model: enums, dataclass, and the RC- and CUPS-option parsers."""

import configparser
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Self

logger = logging.getLogger(__name__)


# --- Enums ---


class MediaType(StrEnum):
    """Paper/media type selection (maps to XL2HB MediaType attribute).

    Values are this driver's PPD choices; Brother's own spellings from
    brhl4150cdnrc and its PPD (BOND, Env, PostCard) are accepted as aliases.
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
        alias = {"env": cls.ENVELOPE}
        if isinstance(value, str):
            for member in cls:
                if member.value.lower() == value.lower():
                    return member
            return alias.get(value.lower())
        return None


class PageSize(StrEnum):
    """Page size selection (maps to XL2HB MediaSize enum and PAPER_SIZES dimensions)."""

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

    @classmethod
    def _missing_(cls, value: object) -> "PageSize | None":
        # PPD keys of Brother's own cupswrapper PPD (its B5/B6 are ISO sizes).
        alias = {"A5Rotated": cls.A5_ROTATED, "B5": cls.ISOB5, "B6": cls.ISOB6}
        return alias.get(value) if isinstance(value, str) else None


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
    def from_rc_file(cls, path: str) -> Self:
        """Read settings from brhl4150cdnrc.

        Returns:
            Populated settings instance (defaults if the file has no sections).
        """
        settings = cls()
        config = configparser.ConfigParser()
        config.read(path)
        if not config.sections():
            return settings
        s = config[config.sections()[0]]
        settings.media_type = MediaType(s.get("MediaType", settings.media_type))
        settings.page_size = PageSize(s.get("PageSize", settings.page_size))
        settings.input_slot = InputSlot(s.get("InputSlot", settings.input_slot))
        settings.resolution = Resolution(s.get("BRResolution", settings.resolution))
        settings.copies = int(s.get("Copies", str(settings.copies)))
        settings.duplex = DuplexMode(s.get("Duplex", settings.duplex))
        settings.mono_color = MonoColor(s.get("BRMonoColor", settings.mono_color))
        settings.color_matching = ColorMatching(s.get("BRColorMatching", settings.color_matching))
        settings.improve_gray = s.get("BRGray", "OFF") == "ON"
        settings.enhance_black = s.get("BREnhanceBlkPrt", "OFF") == "ON"
        settings.toner_save = s.get("TonerSaveMode", "OFF") == "ON"
        settings.improve_output = ImproveOutput(s.get("BRImproveOutput", "OFF"))
        settings.brightness = int(s.get("Brightness", "0"))
        settings.contrast = int(s.get("Contrast", "0"))
        settings.red = int(s.get("RedKey", "0"))
        settings.green = int(s.get("GreenKey", "0"))
        settings.blue = int(s.get("BlueKey", "0"))
        settings.saturation = int(s.get("Saturation", "0"))
        settings.skip_blank = s.get("BRSkipBlank", "OFF") == "ON"
        settings.reverse = s.get("BRReverse", "OFF") == "ON"
        return settings

    @classmethod
    def from_cups_options(
        cls, options_str: str, copies: int = 1, ppd_defaults: Mapping[str, str] | None = None
    ) -> Self:
        """Parse a CUPS option string into PrintSettings.

        CUPS passes options as space-separated Key=Value pairs, e.g.:
        ``"PageSize=A4 Duplex=DuplexNoTumble BRBrightness=5"``. It only
        passes what the job carries, so the queue's PPD defaults
        (`ppd_defaults`, see `read_ppd_defaults`) apply first and the job's
        options override them, as Brother's cupswrapper does. An IPP
        `sides` in the job still beats a PPD default duplex.

        Returns:
            Populated settings instance with the parsed values.
        """
        settings = cls()
        settings.copies = copies

        job_opts: dict[str, str] = {}
        for token in options_str.split():
            if "=" in token:
                key, value = token.split("=", 1)
                job_opts[key] = value
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
                    setattr(settings, attr, enum_cls(opts[cups_key]))
                except ValueError:
                    logger.warning("Unknown %s value %r, keeping default", cups_key, opts[cups_key])

        # IPP "sides" when the job has no PPD duplex option.
        if "Duplex" not in job_opts and "BRDuplex" not in job_opts and "sides" in opts:
            sides = _SIDES_TO_DUPLEX.get(opts["sides"])
            if sides is None:
                logger.warning("Unknown sides value %r, keeping default", opts["sides"])
            else:
                settings.duplex = sides

        # Resolution: "600x2400dpi" -> Fine, else Normal
        if "BRResolution" in opts:
            settings.resolution = Resolution.FINE if opts["BRResolution"] == "600x2400dpi" else Resolution.NORMAL

        # Boolean options (ON/OFF)
        bool_map: dict[str, str] = {
            "BRTonerSaveMode": "toner_save",
            "BRSkipBlank": "skip_blank",
            "BRGray": "improve_gray",
            "BREnhanceBlkPrt": "enhance_black",
            "BRReverse": "reverse",
        }
        for cups_key, attr in bool_map.items():
            if cups_key in opts:
                setattr(settings, attr, opts[cups_key] == "ON")

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


_SIDES_TO_DUPLEX = {
    "one-sided": DuplexMode.NONE,
    "two-sided-long-edge": DuplexMode.NO_TUMBLE,
    "two-sided-short-edge": DuplexMode.TUMBLE,
}

# XL2HB DuplexPageMode values as written by Brother's filter; None = simplex.
DUPLEX_MAP: dict[DuplexMode, int | None] = {
    DuplexMode.NONE: None,
    DuplexMode.NO_TUMBLE: 0x00,
    DuplexMode.TUMBLE: 0x81,
}

# XL2HB MediaSource values as brhl4150cdnfilter writes them. The tray goes
# only into BeginPage; the original never emits @PJL SET SOURCETRAY for
# this model.
MEDIA_SOURCE: dict[InputSlot, int] = {
    InputSlot.AUTO: 1,
    InputSlot.TRAY1: 1001,
    InputSlot.TRAY2: 1005,
    InputSlot.MP_TRAY: 1004,
    InputSlot.MANUAL: 2,
}
