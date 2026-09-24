"""
PJL header/footer tests.

Verifies our PJL generation matches the format in all captures.
"""

import pytest

from xl2hb import generate_pjl_footer, generate_pjl_header


class TestPJLHeader:
    def test_header_starts_with_uel(self):
        header = generate_pjl_header()
        assert header.startswith(b"\x1b%-12345X"), "Header must start with UEL"

    def test_header_ends_with_enter_language(self):
        header = generate_pjl_header()
        assert header.endswith(b"@PJL ENTER LANGUAGE=XL2HB\n")

    def test_header_default_settings(self):
        header = generate_pjl_header()
        text = header.decode("ascii")
        assert "ECONOMODE=OFF" in text
        assert "RENDERMODE=COLOR" in text
        assert "COLORADAPT=ON" in text
        assert "LESSPAPERCURL=OFF" in text
        assert "FIXINTENSITYUP=OFF" in text
        assert "APTMODE=OFF" in text
        assert "RESOLUTION=600" in text

    def test_header_economode_on(self):
        header = generate_pjl_header(economode=True)
        assert b"ECONOMODE=ON" in header

    def test_header_grayscale(self):
        header = generate_pjl_header(color=False)
        assert b"RENDERMODE=GRAYSCALE" in header

    def test_coloradapt_on_when_color(self):
        header = generate_pjl_header(color=True)
        assert b"COLORADAPT=ON" in header

    def test_coloradapt_off_when_grayscale(self):
        header = generate_pjl_header(color=False)
        assert b"COLORADAPT=OFF" in header

    def test_aptmode_on4(self):
        header = generate_pjl_header(apt_mode=True)
        assert b"APTMODE=ON4" in header

    def test_aptmode_off(self):
        header = generate_pjl_header(apt_mode=False)
        assert b"APTMODE=OFF" in header

    def test_improvegray_on_color(self):
        header = generate_pjl_header(color=True, improve_gray=True)
        assert b"IMPROVEGRAY=ON" in header

    def test_improvegray_absent_grayscale(self):
        header = generate_pjl_header(color=False, improve_gray=True)
        assert b"IMPROVEGRAY" not in header

    def test_improvegray_absent_by_default(self):
        header = generate_pjl_header()
        assert b"IMPROVEGRAY" not in header

    def test_ucrgcr_on_color(self):
        header = generate_pjl_header(color=True, ucrgcr=True)
        assert b"UCRGCRFORIMAGE=ON" in header

    def test_ucrgcr_absent_grayscale(self):
        header = generate_pjl_header(color=False, ucrgcr=True)
        assert b"UCRGCRFORIMAGE" not in header

    def test_ucrgcr_absent_by_default(self):
        header = generate_pjl_header()
        assert b"UCRGCRFORIMAGE" not in header

    @pytest.mark.parametrize("command", [b"SOURCETRAY", b"RET=", b"PAGEPROTECT", b"MANUALDPX"])
    def test_commands_the_original_never_sends(self, command):
        """pjl.c knows these commands, but brhl4150cdnfilter never emits them for this model."""
        assert command not in generate_pjl_header()

    def test_command_order_matches_original(self):
        """All commands present, in original pjl.c order."""
        header = generate_pjl_header(
            color=True,
            apt_mode=True,
            improve_gray=True,
            ucrgcr=True,
        )
        text = header.decode("ascii")
        commands = [
            "ECONOMODE=",
            "RENDERMODE=",
            "COLORADAPT=",
            "LESSPAPERCURL=",
            "FIXINTENSITYUP=",
            "APTMODE=",
            "IMPROVEGRAY=",
            "UCRGCRFORIMAGE=",
            "RESOLUTION=",
            "ENTER LANGUAGE=",
        ]
        positions = []
        for cmd in commands:
            pos = text.find(cmd)
            assert pos != -1, f"{cmd} not found in header"
            positions.append(pos)
        assert positions == sorted(positions), (
            f"Commands not in original order: {list(zip(commands, positions, strict=True))}"
        )


class TestPJLFooter:
    def test_footer_double_uel(self):
        footer = generate_pjl_footer()
        uel = b"\x1b%-12345X"
        assert footer == uel + uel

    def test_footer_length(self):
        footer = generate_pjl_footer()
        assert len(footer) == 18  # 2 * 9 bytes


class TestPJLMatchesCaptures:
    def test_pjl_matches_all_captures(self, all_captures):
        """Verify our PJL header/footer matches every capture."""
        expected_header = generate_pjl_header()
        expected_footer = generate_pjl_footer()

        for name, cap in all_captures.items():
            assert cap.pjl_header == expected_header, f"PJL header mismatch in {name}"
            assert cap.pjl_footer == expected_footer, f"PJL footer mismatch in {name}"
