"""
CUPS option parsing tests.

Tests PrintSettings.from_cups_options() for all supported option types:
enum options, resolution, booleans, integers, and edge cases.
"""

import pytest

from brfilter import (
    ColorMatching,
    DuplexMode,
    ImproveOutput,
    InputSlot,
    MediaType,
    MonoColor,
    PageSize,
    PrintSettings,
    Resolution,
)


class TestCupsOptionsDefaults:
    def test_empty_string(self):
        s = PrintSettings.from_cups_options("")
        assert s.page_size == PageSize.A4
        assert s.duplex == DuplexMode.NONE
        assert s.copies == 1

    def test_copies_from_arg(self):
        s = PrintSettings.from_cups_options("", copies=5)
        assert s.copies == 5


class TestCupsOptionsEnums:
    @pytest.mark.parametrize(
        ("option", "expected"),
        [
            ("PageSize=A4", PageSize.A4),
            ("PageSize=Letter", PageSize.LETTER),
            ("PageSize=Legal", PageSize.LEGAL),
            ("PageSize=Executive", PageSize.EXECUTIVE),
            ("PageSize=A5", PageSize.A5),
            ("PageSize=JISB5", PageSize.JISB5),
            ("PageSize=Env10", PageSize.ENV_10),
            ("PageSize=EnvMonarch", PageSize.ENV_MONARCH),
        ],
    )
    def test_page_size(self, option, expected):
        s = PrintSettings.from_cups_options(option)
        assert s.page_size == expected

    @pytest.mark.parametrize(
        ("option", "expected"),
        [
            ("BRDuplex=None", DuplexMode.NONE),
            ("BRDuplex=DuplexNoTumble", DuplexMode.NO_TUMBLE),
            ("BRDuplex=DuplexTumble", DuplexMode.TUMBLE),
        ],
    )
    def test_duplex(self, option, expected):
        s = PrintSettings.from_cups_options(option)
        assert s.duplex == expected

    @pytest.mark.parametrize(
        ("option", "expected"),
        [
            ("BRInputSlot=AutoSelect", InputSlot.AUTO),
            ("BRInputSlot=Tray1", InputSlot.TRAY1),
            ("BRInputSlot=Tray2", InputSlot.TRAY2),
            ("BRInputSlot=MPTray", InputSlot.MP_TRAY),
        ],
    )
    def test_input_slot(self, option, expected):
        s = PrintSettings.from_cups_options(option)
        assert s.input_slot == expected

    @pytest.mark.parametrize(
        ("option", "expected"),
        [
            ("BRMonoColor=Auto", MonoColor.AUTO),
            ("BRMonoColor=FullColor", MonoColor.FULL_COLOR),
            ("BRMonoColor=Mono", MonoColor.MONO),
        ],
    )
    def test_mono_color(self, option, expected):
        s = PrintSettings.from_cups_options(option)
        assert s.mono_color == expected

    @pytest.mark.parametrize(
        ("option", "expected"),
        [
            ("BRMediaType=Plain", MediaType.PLAIN),
            ("BRMediaType=Thick", MediaType.THICK),
            ("BRMediaType=Envelope", MediaType.ENVELOPE),
            ("BRMediaType=Glossy", MediaType.GLOSSY),
        ],
    )
    def test_media_type(self, option, expected):
        s = PrintSettings.from_cups_options(option)
        assert s.media_type == expected

    @pytest.mark.parametrize(
        ("option", "expected"),
        [
            ("BRColorMatching=Normal", ColorMatching.NORMAL),
            ("BRColorMatching=Vivid", ColorMatching.VIVID),
            ("BRColorMatching=None", ColorMatching.NONE),
        ],
    )
    def test_color_matching(self, option, expected):
        s = PrintSettings.from_cups_options(option)
        assert s.color_matching == expected

    @pytest.mark.parametrize(
        ("option", "expected"),
        [
            ("BRImproveOutput=OFF", ImproveOutput.OFF),
            ("BRImproveOutput=BRLessPaperCurl", ImproveOutput.LESS_PAPER_CURL),
            ("BRImproveOutput=BRFixIntensity", ImproveOutput.FIX_INTENSITY),
        ],
    )
    def test_improve_output(self, option, expected):
        s = PrintSettings.from_cups_options(option)
        assert s.improve_output == expected


class TestCupsOptionsResolution:
    def test_normal(self):
        s = PrintSettings.from_cups_options("BRResolution=600dpi")
        assert s.resolution == Resolution.NORMAL

    def test_fine(self):
        s = PrintSettings.from_cups_options("BRResolution=600x2400dpi")
        assert s.resolution == Resolution.FINE


class TestCupsOptionsBooleans:
    @pytest.mark.parametrize(
        ("option", "attr", "on_value"),
        [
            ("BRTonerSaveMode=ON", "toner_save", True),
            ("BRTonerSaveMode=OFF", "toner_save", False),
            ("BRSkipBlank=ON", "skip_blank", True),
            ("BRSkipBlank=OFF", "skip_blank", False),
            ("BRGray=ON", "improve_gray", True),
            ("BRGray=OFF", "improve_gray", False),
            ("BREnhanceBlkPrt=ON", "enhance_black", True),
            ("BREnhanceBlkPrt=OFF", "enhance_black", False),
            ("BRReverse=ON", "reverse", True),
            ("BRReverse=OFF", "reverse", False),
        ],
    )
    def test_boolean_option(self, option, attr, on_value):
        s = PrintSettings.from_cups_options(option)
        assert getattr(s, attr) == on_value


class TestCupsOptionsIntegers:
    @pytest.mark.parametrize(
        ("option", "attr", "expected"),
        [
            ("BRBrightness=5", "brightness", 5),
            ("BRBrightness=-10", "brightness", -10),
            ("BRBrightness=0", "brightness", 0),
            ("BRContrast=20", "contrast", 20),
            ("BRContrast=-20", "contrast", -20),
            ("BRRed=3", "red", 3),
            ("BRGreen=-2", "green", -2),
            ("BRBlue=1", "blue", 1),
            ("BRSaturation=8", "saturation", 8),
        ],
    )
    def test_integer_option(self, option, attr, expected):
        s = PrintSettings.from_cups_options(option)
        assert getattr(s, attr) == expected

    def test_integer_clamping_high(self):
        s = PrintSettings.from_cups_options("BRBrightness=50")
        assert s.brightness == 20

    def test_integer_clamping_low(self):
        s = PrintSettings.from_cups_options("BRContrast=-100")
        assert s.contrast == -20


class TestCupsOptionsInvalid:
    def test_unknown_enum_value_keeps_default(self):
        s = PrintSettings.from_cups_options("PageSize=Tabloid")
        assert s.page_size == PageSize.A4  # default

    def test_non_integer_keeps_default(self):
        s = PrintSettings.from_cups_options("BRBrightness=abc")
        assert s.brightness == 0  # default

    def test_unknown_option_key_ignored(self):
        s = PrintSettings.from_cups_options("UnknownOption=foo")
        assert s.page_size == PageSize.A4  # defaults preserved

    def test_malformed_token_ignored(self):
        s = PrintSettings.from_cups_options("noequals PageSize=Letter")
        assert s.page_size == PageSize.LETTER


class TestCupsOptionsMultiple:
    def test_full_options_string(self):
        opts = (
            "PageSize=Letter BRDuplex=DuplexNoTumble BRResolution=600x2400dpi "
            "BRMonoColor=FullColor BRMediaType=Thick BRColorMatching=Vivid "
            "BRGray=ON BREnhanceBlkPrt=ON BRTonerSaveMode=ON "
            "BRImproveOutput=BRLessPaperCurl BRBrightness=5 BRContrast=-3 "
            "BRRed=2 BRGreen=-1 BRBlue=4 BRSaturation=10 "
            "BRSkipBlank=ON BRReverse=ON BRInputSlot=Tray1"
        )
        s = PrintSettings.from_cups_options(opts, copies=3)
        assert s.page_size == PageSize.LETTER
        assert s.duplex == DuplexMode.NO_TUMBLE
        assert s.resolution == Resolution.FINE
        assert s.mono_color == MonoColor.FULL_COLOR
        assert s.media_type == MediaType.THICK
        assert s.color_matching == ColorMatching.VIVID
        assert s.improve_gray is True
        assert s.enhance_black is True
        assert s.toner_save is True
        assert s.improve_output == ImproveOutput.LESS_PAPER_CURL
        assert s.brightness == 5
        assert s.contrast == -3
        assert s.red == 2
        assert s.green == -1
        assert s.blue == 4
        assert s.saturation == 10
        assert s.skip_blank is True
        assert s.reverse is True
        assert s.input_slot == InputSlot.TRAY1
        assert s.copies == 3
