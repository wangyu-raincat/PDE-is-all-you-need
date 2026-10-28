"""Independent check of the Gaussian derivative correction.

Only the mathematical Gaussian and its derivatives are implemented here. No
third-party source is imported or executed. This diagnostic is not evidence of
PDE structure recovery.
"""
from __future__ import annotations

import json

import numpy as np


SOURCE_COMMIT = "271c6ab556c0788cad5d60e86537c8b0d6dab9e0"
SOURCE_FILE = "redo_two.py"
SOURCE_LINE = 84
SEED = 20261028


def gaussian_derivatives(x, t, sigma0, velocity=.4, diffusivity=.05):
    """Return u, u_xx, and corrected u_xxx; complex x supports differentiation."""
    if sigma0 <= 0 or diffusivity < 0:
        raise ValueError("Need positive initial width and nonnegative diffusivity.")
    x, t = np.asarray(x), np.asarray(t)
    variance = sigma0**2 + 2 * diffusivity * t
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError("Gaussian variance must be finite and positive.")
    shifted = x - velocity * t
    u = np.exp(-shifted**2 / (2 * variance)) / np.sqrt(2 * np.pi * variance)
    second = u * (shifted**2 / variance**2 - 1 / variance)
    third = u * (3 * shifted / variance**2 - shifted**3 / variance**3)
    return u, second, third


def incorrect_reference_third(x, t, sigma0, velocity=.4, diffusivity=.05):
    """Algebraic expansion of the erroneous reference line, for a negative check."""
    u, _, _ = gaussian_derivatives(x, t, sigma0, velocity, diffusivity)
    variance = sigma0**2 + 2 * diffusivity * np.asarray(t)
    shifted = np.asarray(x) - velocity * np.asarray(t)
    return u * (5 * shifted / variance**2 - 3 * shifted**3 / variance**3)


def verify():
    """Check 200 points for each of two widths against complex-step d_x(u_xx)."""
    rng = np.random.default_rng(SEED)
    x = rng.uniform(-3, 4, 200)
    t = rng.uniform(0, 1, 200)
    h = 1e-25
    records = []
    for sigma0 in (.5, .9):
        corrected = gaussian_derivatives(x, t, sigma0)[2]
        independently_differentiated = gaussian_derivatives(x + 1j*h, t, sigma0)[1].imag / h
        erroneous = incorrect_reference_third(x, t, sigma0)
        corrected_error = float(np.max(np.abs(corrected - independently_differentiated)))
        erroneous_error = float(np.max(np.abs(erroneous - independently_differentiated)))
        records.append({
            "sigma0": sigma0,
            "points": len(x),
            "corrected_complex_step_max_abs_error": corrected_error,
            "original_complex_step_max_abs_error": erroneous_error,
            "original_relative_l2_error": float(
                np.linalg.norm(erroneous-independently_differentiated)
                / np.linalg.norm(independently_differentiated)),
        })
    passed = all(r["corrected_complex_step_max_abs_error"] < 5e-13
                 and r["original_complex_step_max_abs_error"] > .1 for r in records)
    return {
        "source_repository": "https://github.com/changsongjin/pde-l1-analytic",
        "source_commit": SOURCE_COMMIT,
        "source_file": SOURCE_FILE,
        "source_line": SOURCE_LINE,
        "seed": SEED,
        "points_checked": 400,
        "reference": "complex-step d/dx of independently evaluated second derivative",
        "step": h,
        "corrected_formula": "u*(3*xi/s**2-xi**3/s**3)",
        "wrong_minus_corrected": "2*u*xi*(s-xi**2)/s**3",
        "checks": records,
        "max_corrected_absolute_error": max(r["corrected_complex_step_max_abs_error"] for r in records),
        "max_original_absolute_error": max(r["original_complex_step_max_abs_error"] for r in records),
        "passed": passed,
        "third_party_code_executed": False,
        "is_pde_recovery_experiment": False,
    }


def main():
    result = verify()
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
