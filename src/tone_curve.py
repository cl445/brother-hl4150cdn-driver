"""Tone curve utilities for the Brother HL-4150CDN.

Provides:
- 256-byte gamma LUT loading (`gamma_curve_{0,1}.bin`).
- Natural cubic spline interpolation via the Thomas algorithm.
- Tone curve builder combining brightness, contrast, and an optional
  gamma stage.
- Tone curve application to CMYK channels.

`rgb_default_lut.bin` already has gamma baked in, so these helpers are
only invoked when `gamma_select` is explicitly set in `PrintSettings`.
"""

import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Data directory containing extracted binary tables
_DATA_DIR = Path(__file__).resolve().parent / "color_data"

# Module-level cache for gamma curves
_gamma_cache: dict[int, npt.NDArray[np.uint8]] = {}


def load_gamma_curve(curve_id: int) -> npt.NDArray[np.uint8]:
    """Load a 256-byte gamma LUT from binary file.

    Falls back to parametric generation if the binary file is missing.

    Args:
        curve_id: 0 or 1, selects gamma_curve_{0,1}.bin

    Returns:
        np.ndarray shape (256,) dtype uint8

    Raises:
        ValueError: If `curve_id` is not 0 or 1, or the binary file is corrupt.
    """
    if curve_id not in (0, 1):
        msg = f"Invalid gamma curve_id {curve_id}, must be 0 or 1"
        raise ValueError(msg)
    if curve_id in _gamma_cache:
        return _gamma_cache[curve_id]

    path = _DATA_DIR / f"gamma_curve_{curve_id}.bin"
    if not path.exists():
        logger.info("Gamma curve %d binary not found, generating", curve_id)
        from color_lut_gen import generate_gamma_curve

        curve = generate_gamma_curve(curve_id)
        _gamma_cache[curve_id] = curve
        return curve
    data = path.read_bytes()
    if len(data) != 256:
        msg = f"Gamma curve {curve_id}: expected 256 bytes, got {len(data)}"
        raise ValueError(msg)
    curve = np.frombuffer(data, dtype=np.uint8).copy()
    _gamma_cache[curve_id] = curve
    return curve


def cubic_spline_interpolate(
    control_points: npt.NDArray[np.float64], output_count: int = 256
) -> npt.NDArray[np.float64]:
    """Natural cubic spline interpolation through evenly-spaced control points.

    Control points are assumed to be evenly spaced across [0, output_count-1].
    Natural boundary conditions: second derivatives = 0 at endpoints.

    Args:
        control_points: 1D array of y-values at evenly-spaced x-positions.
        output_count: Number of output samples (default 256).

    Returns:
        float64 array of length output_count, clamped to [0, output_count-1].
    """
    n = len(control_points)
    max_idx = output_count - 1

    cp = np.asarray(control_points, dtype=np.float64)

    # Build x-coordinates (evenly spaced)
    x = np.array([max_idx * i / (n - 1) for i in range(n)], dtype=np.float64)
    y = cp.copy()

    if n < 3:
        # Linear interpolation for 2 or fewer points
        slope = (y[1] - y[0]) / (x[1] - x[0])
        intercept = y[1] - slope * x[1]
        result = np.array([slope * i + intercept for i in range(output_count)], dtype=np.float64)
        return np.clip(result, 0.0, float(max_idx))

    # Natural cubic spline via Thomas algorithm
    # Solve tridiagonal system for second derivatives (spline_coeffs)
    m = n - 1  # number of intervals

    # Interval widths and divided differences
    h = np.empty(m, dtype=np.float64)
    d = np.empty(m + 1, dtype=np.float64)  # d[1..m] used
    for i in range(m):
        h[i] = x[i + 1] - x[i]
        d[i + 1] = (y[i + 1] - y[i]) / h[i]

    # Spline coefficients (second derivatives / 6, matching driver convention)
    s = np.zeros(n, dtype=np.float64)  # s[0] = s[n-1] = 0 (natural)

    # Forward elimination
    # diag[i] stores the diagonal element after elimination
    diag = np.empty(n, dtype=np.float64)
    diag[1] = 2.0 * (x[2] - x[0])
    s[1] = d[2] - d[1] - s[0] * h[0]

    for i in range(1, m - 1):
        ratio = h[i] / diag[i]
        s[i + 1] = d[i + 2] - d[i + 1] - s[i] * ratio
        diag[i + 1] = 2.0 * (x[i + 2] - x[i]) - h[i] * ratio

    # Boundary: incorporate s[n-1] = 0
    s[m - 1] = s[m - 1] - s[m] * h[m - 1]

    # Back substitution
    for i in range(m - 1, 0, -1):
        s[i] = (s[i] - s[i + 1] * h[i]) / diag[i]

    # Evaluate spline at each output point
    result = np.empty(output_count, dtype=np.float64)
    for idx in range(output_count):
        # Binary search for interval (matching driver's search)
        lo = 0
        hi = m
        while lo < hi:
            mid = (lo + hi) // 2
            if x[mid] < float(idx):
                lo = mid + 1
                mid = hi
            hi = mid
        if lo > 0:
            lo -= 1

        # Evaluate cubic polynomial in interval [x[lo], x[lo+1]]
        si = s[lo]
        si1 = s[lo + 1]
        dx = float(idx) - x[lo]
        hh = x[lo + 1] - x[lo]

        val = y[lo] + (((si1 - si) * dx / hh + si * 3.0) * dx + ((y[lo + 1] - y[lo]) / hh - (si + si + si1) * hh)) * dx

        if val > float(max_idx):
            val = float(max_idx)
        elif val < 0.0:
            val = 0.0

        result[idx] = val

    return result


def build_tone_curve(
    brightness: int = 0,
    contrast: int = 0,
    gamma_select: int | None = None,
) -> npt.NDArray[np.uint8]:
    """Build a 256-entry tone curve LUT.

    Applies brightness, then contrast, then optional gamma correction.

    Args:
        brightness: -20..+20
        contrast: -20..+20 (mapped to a percentage delta around 100)
        gamma_select: 0 or 1 to apply gamma curve, None to skip

    Returns:
        np.ndarray shape (256,) dtype uint8
    """
    gamma_curve = load_gamma_curve(gamma_select) if gamma_select is not None else None

    # Driver formula: brightness * 255, then arithmetic right shift by 7
    b_scaled = brightness * 255
    if b_scaled < 0:
        b_scaled += 127
    b_offset = b_scaled >> 7

    # Driver uses tc->contrast directly (e.g. brightness=5 → contrast passed as-is)
    # The contrast field in the driver struct maps to the RC/CUPS value
    # which ranges -20..+20. The division by 100 means contrast acts as a
    # percentage modifier: contrast=0 → no change, contrast=20 → 20% boost.
    c_val = contrast

    v = np.arange(256, dtype=np.int32)

    # Brightness
    adjusted = np.clip(v + b_offset, 0, 255)

    # Contrast formula: (adjusted - 128) * c_val / 100 + adjusted
    adjusted = np.clip(((adjusted - 128) * c_val) // 100 + adjusted, 0, 255)

    # Gamma
    if gamma_curve is not None:
        adjusted = gamma_curve[adjusted]

    return adjusted.astype(np.uint8)


def apply_tone_curve(
    k_arr: bytes,
    c_arr: bytes,
    m_arr: bytes,
    y_arr: bytes,
    curve: npt.NDArray[np.uint8],
) -> tuple[bytes, bytes, bytes, bytes]:
    """Apply a 256-entry tone curve LUT to CMYK intensity arrays.

    Args:
        k_arr: K-channel byte array (brightness convention).
        c_arr: C-channel byte array (brightness convention).
        m_arr: M-channel byte array (brightness convention).
        y_arr: Y-channel byte array (brightness convention).
        curve: 256-entry uint8 LUT from build_tone_curve().

    Returns:
        Tuple of 4 bytes objects (k, c, m, y) with curve applied.
    """

    def apply(arr: bytes) -> bytes:
        return np.take(curve, np.frombuffer(arr, dtype=np.uint8)).tobytes()

    return apply(k_arr), apply(c_arr), apply(m_arr), apply(y_arr)
