"""Fit a parametric 3-layer decomposition to the Brother HL-4150CDN color LUT.

Decomposes the 17x17x17 -> CMYK LUT into three physically interpretable layers:
  1. 1D tone curves: per-channel transfer functions (4 CMYK x 3 RGB x 17 points)
  2. 2D pair interactions: bilinear correction surfaces for RG, RB, GB pairs
  3. 3D trilinear residual: low-res correction grid for 3-way interactions

Joint least-squares fit of all layers simultaneously.
Exports coefficients as Python source code for embedding in color_lut_gen.py.

Usage:
    uv run python scripts/fit_lut_coefficients.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LUT_DIM = 17
PAIR_DIM = 9   # 2D interaction grid resolution
TRI_DIM = 5    # 3D residual grid resolution

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "color_data"
OUTPUT_FILE = Path(__file__).resolve().parent / "fitted_coefficients.py"


# ---------------------------------------------------------------------------
# Load original LUT
# ---------------------------------------------------------------------------
def load_original_lut() -> np.ndarray:
    """Load the binary LUT as (4913, 4) int32 array [C, M, Y, K]."""
    path = DATA_DIR / "rgb_default_lut.bin"
    data = path.read_bytes()
    assert len(data) == LUT_DIM**3 * 8, f"Unexpected LUT size: {len(data)}"
    packed = np.frombuffer(data, dtype=np.int32).reshape(-1, 2)
    unpacked = np.empty((LUT_DIM**3, 4), dtype=np.int32)
    unpacked[:, 0] = packed[:, 0] & 0xFFFF   # C
    unpacked[:, 1] = packed[:, 0] >> 16       # M
    unpacked[:, 2] = packed[:, 1] & 0xFFFF   # Y
    unpacked[:, 3] = packed[:, 1] >> 16       # K
    return unpacked


# ---------------------------------------------------------------------------
# Build basis matrices
# ---------------------------------------------------------------------------
def build_1d_basis(grid: np.ndarray) -> np.ndarray:
    """Build 1D tone curve basis: for each RGB input, 17-point piecewise-linear.

    Each grid point in each input channel gets a hat function basis.
    Result: (N, 3*17) matrix where N = number of LUT entries.
    """
    n = len(grid)
    n_basis = 3 * LUT_DIM
    basis = np.zeros((n, n_basis), dtype=np.float64)

    for ch in range(3):
        vals = grid[:, ch].astype(np.float64)
        # Normalize to [0, LUT_DIM-1]
        t = vals / (255.0 / (LUT_DIM - 1))

        for k in range(LUT_DIM):
            # Hat function centered at k
            col = ch * LUT_DIM + k
            if k == 0:
                mask = t < 1.0
                basis[mask, col] = 1.0 - t[mask]
            elif k == LUT_DIM - 1:
                mask = t > (LUT_DIM - 2)
                basis[mask, col] = t[mask] - (LUT_DIM - 2)
            else:
                # Left side
                mask_l = (t >= (k - 1)) & (t < k)
                basis[mask_l, col] = t[mask_l] - (k - 1)
                # Right side
                mask_r = (t >= k) & (t < (k + 1))
                basis[mask_r, col] = (k + 1) - t[mask_r]
            # Exact grid point
            mask_exact = np.isclose(t, float(k))
            basis[mask_exact, col] = 1.0

    return basis


def build_2d_basis(grid: np.ndarray, ch_a: int, ch_b: int, dim: int) -> np.ndarray:
    """Build 2D bilinear interaction basis for a pair of channels.

    Result: (N, dim*dim) matrix.
    """
    n = len(grid)
    basis = np.zeros((n, dim * dim), dtype=np.float64)

    a = grid[:, ch_a].astype(np.float64) / (255.0 / (dim - 1))
    b = grid[:, ch_b].astype(np.float64) / (255.0 / (dim - 1))

    a_lo = np.floor(a).astype(int)
    b_lo = np.floor(b).astype(int)
    a_lo = np.clip(a_lo, 0, dim - 2)
    b_lo = np.clip(b_lo, 0, dim - 2)
    a_frac = a - a_lo
    b_frac = b - b_lo

    # Bilinear weights for 4 corners
    for da in range(2):
        for db in range(2):
            wa = (1 - a_frac) if da == 0 else a_frac
            wb = (1 - b_frac) if db == 0 else b_frac
            idx = (a_lo + da) * dim + (b_lo + db)
            for i in range(n):
                basis[i, idx[i]] += wa[i] * wb[i]

    return basis


def build_3d_basis(grid: np.ndarray, dim: int) -> np.ndarray:
    """Build 3D trilinear correction basis.

    Result: (N, dim*dim*dim) matrix.
    """
    n = len(grid)
    basis = np.zeros((n, dim**3), dtype=np.float64)

    coords = []
    for ch in range(3):
        v = grid[:, ch].astype(np.float64) / (255.0 / (dim - 1))
        lo = np.floor(v).astype(int)
        lo = np.clip(lo, 0, dim - 2)
        frac = v - lo
        coords.append((lo, frac))

    # Trilinear weights for 8 corners
    for dr in range(2):
        for dg in range(2):
            for db in range(2):
                wr = (1 - coords[0][1]) if dr == 0 else coords[0][1]
                wg = (1 - coords[1][1]) if dg == 0 else coords[1][1]
                wb = (1 - coords[2][1]) if db == 0 else coords[2][1]
                w = wr * wg * wb
                ri = coords[0][0] + dr
                gi = coords[1][0] + dg
                bi = coords[2][0] + db
                idx = ri * dim * dim + gi * dim + bi
                for i in range(n):
                    basis[i, idx[i]] += w[i]

    return basis


# ---------------------------------------------------------------------------
# Joint fit
# ---------------------------------------------------------------------------
def fit_all_channels(lut: np.ndarray) -> dict:
    """Perform joint least-squares fit for all 4 CMYK channels."""
    # Build RGB grid coordinates (17x17x17)
    grid = np.zeros((LUT_DIM**3, 3), dtype=np.float64)
    for idx in range(LUT_DIM**3):
        r = idx // (LUT_DIM * LUT_DIM)
        g = (idx // LUT_DIM) % LUT_DIM
        b = idx % LUT_DIM
        grid[idx] = [r * 255 / (LUT_DIM - 1), g * 255 / (LUT_DIM - 1), b * 255 / (LUT_DIM - 1)]

    # Round grid to nearest integer (they're all multiples of 255/16 = 15.9375)
    # The actual LUT grid points are at 0, 16, 32, ..., 240, 255
    grid = np.round(grid).astype(np.float64)

    print("Building basis matrices...")
    basis_1d = build_1d_basis(grid)
    basis_rg = build_2d_basis(grid, 0, 1, PAIR_DIM)
    basis_rb = build_2d_basis(grid, 0, 2, PAIR_DIM)
    basis_gb = build_2d_basis(grid, 1, 2, PAIR_DIM)
    basis_3d = build_3d_basis(grid, TRI_DIM)

    # Concatenate all basis matrices
    A = np.hstack([basis_1d, basis_rg, basis_rb, basis_gb, basis_3d])

    n_1d = basis_1d.shape[1]       # 51
    n_rg = basis_rg.shape[1]       # 81
    n_rb = basis_rb.shape[1]       # 81
    n_gb = basis_gb.shape[1]       # 81
    n_3d = basis_3d.shape[1]       # 125
    total = A.shape[1]

    print(f"Basis dimensions: 1D={n_1d}, RG={n_rg}, RB={n_rb}, GB={n_gb}, 3D={n_3d}, total={total}")
    print(f"Matrix shape: {A.shape}")

    results = {}
    channel_names = ["C", "M", "Y", "K"]

    for ch_idx, ch_name in enumerate(channel_names):
        target = lut[:, ch_idx].astype(np.float64)

        # Solve least squares
        coeffs, _, rank, _ = np.linalg.lstsq(A, target, rcond=None)

        # Predict and compute errors
        pred = A @ coeffs
        err = np.abs(pred - target)

        print(f"\n{ch_name}: rank={rank}, max_err={err.max():.1f}, "
              f"mean_err={err.mean():.1f}, p95={np.percentile(err, 95):.1f}")

        # Split coefficients
        offset = 0
        tone = coeffs[offset:offset + n_1d]
        offset += n_1d
        pair_rg = coeffs[offset:offset + n_rg]
        offset += n_rg
        pair_rb = coeffs[offset:offset + n_rb]
        offset += n_rb
        pair_gb = coeffs[offset:offset + n_gb]
        offset += n_gb
        tri = coeffs[offset:offset + n_3d]

        results[ch_name] = {
            "tone": tone,                          # (51,)
            "pair_rg": pair_rg.reshape(PAIR_DIM, PAIR_DIM),  # (9, 9)
            "pair_rb": pair_rb.reshape(PAIR_DIM, PAIR_DIM),  # (9, 9)
            "pair_gb": pair_gb.reshape(PAIR_DIM, PAIR_DIM),  # (9, 9)
            "tri": tri.reshape(TRI_DIM, TRI_DIM, TRI_DIM),  # (5, 5, 5)
            "max_err": float(err.max()),
            "mean_err": float(err.mean()),
            "p95": float(np.percentile(err, 95)),
        }

    return results


# ---------------------------------------------------------------------------
# Generate interpolation tables analytically
# ---------------------------------------------------------------------------
def generate_interp_tables_reference() -> np.ndarray:
    """Generate the tetrahedral interpolation tables analytically.

    These are standard tetrahedral interpolation weights for a unit cube
    partitioned into 6 tetrahedra. The weights depend on the ordering
    of the fractional RGB components.

    Returns: (17, 289, 9) uint8 array matching interp_tables.bin
    """
    tables = np.zeros((17, 17 * 17, 9), dtype=np.uint8)

    for b_frac in range(17):
        for g_frac in range(17):
            for r_frac in range(17):
                row_idx = g_frac * 17 + r_frac
                total = 16  # Common denominator

                # Fractional values (0-16 range)
                rf, gf, bf = r_frac, g_frac, b_frac

                # The 8 corner weights for tetrahedral interpolation
                # Corner order: [0,17,1,18,289,306,290,307] in the flat array
                # which maps to: (r,g,b) offsets:
                #   0: (0,0,0)  1: (0,1,0)  2: (1,0,0)  3: (1,1,0)
                #   4: (0,0,1)  5: (0,1,1)  6: (1,0,0)  7: (1,1,1)
                # Actually in the driver order:
                #   corner 0: base            = (r_lo, g_lo, b_lo)
                #   corner 1: base+17         = (r_lo, g_lo+1, b_lo)
                #   corner 2: base+1          = (r_lo+1, g_lo, b_lo)
                #   corner 3: base+18         = (r_lo+1, g_lo+1, b_lo)
                #   corner 4: base+289        = (r_lo, g_lo, b_lo+1)
                #   corner 5: base+306        = (r_lo, g_lo+1, b_lo+1)
                #   corner 6: base+290        = (r_lo+1, g_lo, b_lo+1)
                #   corner 7: base+307        = (r_lo+1, g_lo+1, b_lo+1)

                # Tetrahedral partition based on fractional ordering
                w = [0] * 8

                if rf >= gf and gf >= bf:
                    # r >= g >= b: tetrahedron through corners 0,2,3,7
                    w[0] = total - rf
                    w[2] = rf - gf
                    w[3] = gf - bf
                    w[7] = bf
                elif rf >= bf and bf >= gf:
                    # r >= b >= g: tetrahedron through 0,2,6,7
                    w[0] = total - rf
                    w[2] = rf - bf
                    w[6] = bf - gf
                    w[7] = gf
                elif gf >= rf and rf >= bf:
                    # g >= r >= b: tetrahedron through 0,1,3,7
                    w[0] = total - gf
                    w[1] = gf - rf
                    w[3] = rf - bf
                    w[7] = bf
                elif gf >= bf and bf >= rf:
                    # g >= b >= r: tetrahedron through 0,1,5,7
                    w[0] = total - gf
                    w[1] = gf - bf
                    w[5] = bf - rf
                    w[7] = rf
                elif bf >= rf and rf >= gf:
                    # b >= r >= g: tetrahedron through 0,4,6,7
                    w[0] = total - bf
                    w[4] = bf - rf
                    w[6] = rf - gf
                    w[7] = gf
                elif bf >= gf and gf >= rf:
                    # b >= g >= r: tetrahedron through 0,4,5,7
                    w[0] = total - bf
                    w[4] = bf - gf
                    w[5] = gf - rf
                    w[7] = rf
                else:
                    # Shouldn't happen, but fallback
                    w[0] = total

                tables[b_frac, row_idx, 0] = total
                for c in range(8):
                    tables[b_frac, row_idx, c + 1] = w[c]

    return tables


# ---------------------------------------------------------------------------
# Generate gamma curves analytically
# ---------------------------------------------------------------------------
def analyze_gamma_curves():
    """Analyze the gamma curves and find a parametric representation."""
    for curve_id in range(2):
        path = DATA_DIR / f"gamma_curve_{curve_id}.bin"
        data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        print(f"\nGamma curve {curve_id}:")
        print(f"  First 20: {data[:20].tolist()}")
        print(f"  Last 20:  {data[236:].tolist()}")

        # Check if it's identity
        identity = np.arange(256, dtype=np.uint8)
        if np.array_equal(data, identity):
            print("  -> Identity curve")
            continue

        # Check if it's piecewise linear with 17 control points
        # Grid points at 0, 16, 32, ..., 240, 255
        grid_indices = [i * 16 for i in range(16)] + [255]
        grid_values = data[grid_indices]
        print(f"  Grid values (17 pts): {grid_values.tolist()}")

        # Reconstruct via piecewise linear interpolation
        reconstructed = np.zeros(256, dtype=np.float64)
        for i in range(256):
            if i <= 240:
                seg = i // 16
                frac = (i % 16) / 16.0
            else:
                seg = 15
                frac = (i - 240) / 15.0

            reconstructed[i] = grid_values[seg] * (1 - frac) + grid_values[min(seg + 1, 16)] * frac

        err = np.abs(reconstructed - data.astype(np.float64))
        print(f"  Piecewise linear error: max={err.max():.1f}, mean={err.mean():.2f}")


# ---------------------------------------------------------------------------
# Export coefficients
# ---------------------------------------------------------------------------
def format_array(arr: np.ndarray, name: str, indent: int = 0) -> str:
    """Format a numpy array as a Python literal for embedding."""
    prefix = " " * indent
    flat = arr.flatten()
    # Use repr format for each number
    nums = [f"{x:.10g}" for x in flat]
    # Group into lines of ~80 chars
    lines = []
    current = []
    current_len = 0
    for n in nums:
        if current_len + len(n) + 2 > 72:
            lines.append(", ".join(current) + ",")
            current = [n]
            current_len = len(n)
        else:
            current.append(n)
            current_len += len(n) + 2
    if current:
        lines.append(", ".join(current) + ",")

    shape_str = repr(arr.shape)
    result = f"{prefix}# shape: {shape_str}\n"
    result += f"{prefix}{name} = np.array([\n"
    for line in lines:
        result += f"{prefix}    {line}\n"
    result += f"{prefix}], dtype=np.float64).reshape({shape_str})\n"
    return result


def export_coefficients(results: dict, output_path: Path):
    """Export fitted coefficients as a Python module."""
    lines = [
        '"""Fitted LUT coefficients — auto-generated by fit_lut_coefficients.py."""',
        "",
        "import numpy as np",
        "",
        "# Fitting accuracy per channel:",
    ]

    for ch in ["C", "M", "Y", "K"]:
        r = results[ch]
        lines.append(f"# {ch}: max_err={r['max_err']:.1f}, mean_err={r['mean_err']:.1f}, p95={r['p95']:.1f}")
    lines.append("")

    # Export tone curves: (4, 3, 17)
    tone_all = np.zeros((4, 3, LUT_DIM), dtype=np.float64)
    for i, ch in enumerate(["C", "M", "Y", "K"]):
        tone_all[i] = results[ch]["tone"].reshape(3, LUT_DIM)
    lines.append(format_array(tone_all, "TONE_CURVES"))

    # Export pair interactions
    for pair_name in ["pair_rg", "pair_rb", "pair_gb"]:
        pair_all = np.zeros((4, PAIR_DIM, PAIR_DIM), dtype=np.float64)
        for i, ch in enumerate(["C", "M", "Y", "K"]):
            pair_all[i] = results[ch][pair_name]
        var_name = f"PAIR_{pair_name[5:].upper()}"
        lines.append(format_array(pair_all, var_name))

    # Export trilinear corrections
    tri_all = np.zeros((4, TRI_DIM, TRI_DIM, TRI_DIM), dtype=np.float64)
    for i, ch in enumerate(["C", "M", "Y", "K"]):
        tri_all[i] = results[ch]["tri"]
    lines.append(format_array(tri_all, "TRILINEAR"))

    # Also export gamma curve control points
    lines.append("")
    lines.append("# Gamma curve control points (17 points at grid positions 0,16,32,...,240,255)")
    for curve_id in range(2):
        path = DATA_DIR / f"gamma_curve_{curve_id}.bin"
        data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        grid_indices = [i * 16 for i in range(16)] + [255]
        grid_values = data[grid_indices]
        lines.append(f"GAMMA_{curve_id}_POINTS = np.array({grid_values.tolist()}, dtype=np.uint8)")

    lines.append("")
    output_path.write_text("\n".join(lines) + "\n")
    print(f"\nCoefficients written to {output_path}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(results: dict, lut: np.ndarray):
    """Validate the fit by reconstructing the full LUT."""
    grid = np.zeros((LUT_DIM**3, 3), dtype=np.float64)
    for idx in range(LUT_DIM**3):
        r = idx // (LUT_DIM * LUT_DIM)
        g = (idx // LUT_DIM) % LUT_DIM
        b = idx % LUT_DIM
        grid[idx] = [r * 255 / (LUT_DIM - 1), g * 255 / (LUT_DIM - 1), b * 255 / (LUT_DIM - 1)]
    grid = np.round(grid).astype(np.float64)

    basis_1d = build_1d_basis(grid)
    basis_rg = build_2d_basis(grid, 0, 1, PAIR_DIM)
    basis_rb = build_2d_basis(grid, 0, 2, PAIR_DIM)
    basis_gb = build_2d_basis(grid, 1, 2, PAIR_DIM)
    basis_3d = build_3d_basis(grid, TRI_DIM)

    A = np.hstack([basis_1d, basis_rg, basis_rb, basis_gb, basis_3d])

    print("\n=== Validation ===")
    for ch_idx, ch_name in enumerate(["C", "M", "Y", "K"]):
        r = results[ch_name]
        coeffs = np.concatenate([
            r["tone"],
            r["pair_rg"].flatten(),
            r["pair_rb"].flatten(),
            r["pair_gb"].flatten(),
            r["tri"].flatten(),
        ])
        pred = A @ coeffs
        pred_clamped = np.clip(np.round(pred), 0, 255).astype(np.int32)
        target = lut[:, ch_idx]
        err = np.abs(pred_clamped - target)
        print(f"{ch_name}: max_err={err.max()}, mean_err={err.mean():.2f}, "
              f"p95={np.percentile(err, 95):.1f}, p99={np.percentile(err, 99):.1f}")

    # Validate interpolation tables
    print("\n=== Interpolation Tables ===")
    path = DATA_DIR / "interp_tables.bin"
    original = np.frombuffer(path.read_bytes(), dtype=np.uint8).reshape(17, 17 * 17, 9)
    generated = generate_interp_tables_reference()
    diff = np.abs(original.astype(np.int16) - generated.astype(np.int16))
    print(f"Interp tables: max_diff={diff.max()}, mean_diff={diff.mean():.4f}")
    if diff.max() > 0:
        mismatch_count = np.sum(diff > 0)
        print(f"  Mismatches: {mismatch_count}/{original.size}")

    # Validate gamma curves
    print("\n=== Gamma Curves ===")
    analyze_gamma_curves()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Fit polynomial coefficients to the original LUT and export them."""
    print("Loading original LUT...")
    lut = load_original_lut()
    print(f"LUT shape: {lut.shape}, range: [{lut.min()}, {lut.max()}]")

    results = fit_all_channels(lut)
    validate(results, lut)

    export_coefficients(results, OUTPUT_FILE)
    print("\nDone!")


if __name__ == "__main__":
    main()
