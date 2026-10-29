"""PJL header and footer around the XL2HB stream."""


def generate_pjl_header(
    resolution: int = 600,
    color: bool = True,
    color_adapt: bool = True,
    economode: bool = False,
    less_paper_curl: bool = False,
    fix_intensity: bool = False,
    apt_mode: bool = False,
    improve_gray: bool = False,
    ucrgcr: bool = False,
) -> bytes:
    """Generate the PJL header.

    Lines are LF-terminated and emitted in the order of the original's
    pjl.c. The PAGEPROTECT, RET, SOURCETRAY and MANUALDPX commands it also
    knows are never sent for this model.

    Returns:
        Encoded PJL header bytes (ASCII).
    """
    lines = [
        "\x1b%-12345X@PJL \n",
        f"@PJL SET ECONOMODE={'ON' if economode else 'OFF'}\n",
        f"@PJL SET RENDERMODE={'COLOR' if color else 'GRAYSCALE'}\n",
        f"@PJL SET COLORADAPT={'ON' if color and color_adapt else 'OFF'}\n",
        f"@PJL SET LESSPAPERCURL={'ON' if less_paper_curl else 'OFF'}\n",
        f"@PJL SET FIXINTENSITYUP={'ON' if fix_intensity else 'OFF'}\n",
    ]
    lines.append(f"@PJL SET APTMODE={'ON4' if apt_mode else 'OFF'}\n")
    if color:
        if improve_gray:
            lines.append("@PJL SET IMPROVEGRAY=ON\n")
        if ucrgcr:
            lines.append("@PJL SET UCRGCRFORIMAGE=ON\n")
    lines.append(f"@PJL SET RESOLUTION={resolution}\n")
    lines.append("@PJL ENTER LANGUAGE=XL2HB\n")
    return "".join(lines).encode("ascii")


def generate_pjl_footer() -> bytes:
    """Generate the PJL footer: two UEL escapes, no @PJL RESET.

    Returns:
        Encoded PJL footer bytes.
    """
    return b"\x1b%-12345X\x1b%-12345X"
