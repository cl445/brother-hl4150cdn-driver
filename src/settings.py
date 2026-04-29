"""PrintSettings model: enums, dataclass, and the RC- and CUPS-option parsers."""

import configparser
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Self

logger = logging.getLogger(__name__)


# --- Enums ---


class MediaType(StrEnum):
    """Paper/media type selection (maps to XL2HB MediaType attribute)."""

    PLAIN = "Plain"
    THIN = "Thin"
    THICK = "Thick"
    THICKER = "Thicker"
    BOND = "Bond"
    ENVELOPE = "Envelope"
    ENV_THICK = "EnvThick"
    RECYCLED = "Recycled"
    LABEL = "Label"
    GLOSSY = "Glossy"


class PageSize(StrEnum):
    """Page size selection (maps to XL2HB MediaSize enum and PAPER_SIZES dimensions)."""

    A4 = "A4"
    LETTER = "Letter"
    LEGAL = "Legal"
    EXECUTIVE = "Executive"
    A5 = "A5"
    JISB5 = "JISB5"
    POSTCARD = "Postcard"
    ENV_DL = "EnvDL"
    ENV_C5 = "EnvC5"
    ENV_10 = "Env10"
    ENV_MONARCH = "EnvMonarch"


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
    """Paper input tray selection (maps to PJL SOURCETRAY)."""

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
    gamma_select: int | None = None
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
        gamma_str = s.get("GammaSelect", None)
        if gamma_str is not None:
            settings.gamma_select = int(gamma_str)
        settings.red = int(s.get("RedKey", "0"))
        settings.green = int(s.get("GreenKey", "0"))
        settings.blue = int(s.get("BlueKey", "0"))
        settings.saturation = int(s.get("Saturation", "0"))
        settings.skip_blank = s.get("BRSkipBlank", "OFF") == "ON"
        settings.reverse = s.get("BRReverse", "OFF") == "ON"
        return settings

    @classmethod
    def from_cups_options(cls, options_str: str, copies: int = 1) -> Self:
        """Parse a CUPS option string into PrintSettings.

        CUPS passes options as space-separated Key=Value pairs, e.g.:
        ``"PageSize=A4 BRDuplex=DuplexNoTumble BRBrightness=5"``.

        Returns:
            Populated settings instance with the parsed values.
        """
        settings = cls()
        settings.copies = copies

        opts: dict[str, str] = {}
        for token in options_str.split():
            if "=" in token:
                key, value = token.split("=", 1)
                opts[key] = value

        # Enum options
        enum_map: dict[str, tuple[str, type]] = {
            "PageSize": ("page_size", PageSize),
            "BRDuplex": ("duplex", DuplexMode),
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

        # Gamma select (0 or 1, None = disabled)
        if "BRGammaSelect" in opts:
            try:
                settings.gamma_select = int(opts["BRGammaSelect"])
            except ValueError:
                logger.warning("Non-integer value %r for BRGammaSelect, keeping default", opts["BRGammaSelect"])

        return settings


DUPLEX_MAP = {
    DuplexMode.NONE: 0,
    DuplexMode.NO_TUMBLE: 1,
    DuplexMode.TUMBLE: 2,
}

_TRAY_MAP = {"Tray1": "TRAY1", "Tray2": "TRAY2"}


def input_slot_to_tray(slot: InputSlot) -> str | None:
    """Map InputSlot enum to PJL SOURCETRAY value, or None to skip.

    Returns:
        PJL SOURCETRAY string, or None if the slot has no PJL mapping.
    """
    return _TRAY_MAP.get(slot)
