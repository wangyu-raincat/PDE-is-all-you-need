"""Independent synthetic experiment provider for the PDE-discovery demo.

The discovery engine never imports this module.  Observations contain no true
equation or coefficients.  ``get_truth`` is for benchmark reporting only.
All quantities are dimensionless and the spatial boundary is periodic.
"""
from __future__ import annotations

import numpy as np

_TRUTH = {
    "diffusion": {"u_xx": 0.2},
    "advection_diffusion": {"u_x": -0.7, "u_xx": 0.12},
    "ambiguity": {"u_xx": 1.0},
}


def get_truth(scenario: str) -> dict[str, float]:
    if scenario not in _TRUTH:
        raise ValueError("未知演示场景。")
    return dict(_TRUTH[scenario])


def make_dataset(scenario: str, seed: int = 7, noise: float = 0.0,
                 initial_condition: str = "train") -> dict:
    """Generate observations using a hidden, exact Fourier propagator.

    ``noise`` is Gaussian sensor noise relative to clean trajectory RMS.
    train/probe/validation are separate experiments, not adjacent data windows.
    """
    truth = get_truth(scenario)
    if not np.isfinite(noise) or not 0 <= noise <= 0.1:
        raise ValueError("噪声比例须在 0 到 0.1 之间。")
    x = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    t = np.linspace(0, 1.2 if scenario == "ambiguity" else 2.0, 101)
    if initial_condition == "train":
        u0 = np.sin(x) if scenario == "ambiguity" else (
            np.sin(x) + 0.5 * np.sin(2 * x + 0.3) + 0.3 * np.cos(3 * x))
    elif initial_condition in ("probe", "probe2"):
        u0 = np.sin(2 * x)
    elif initial_condition == "probe3":
        u0 = np.sin(3 * x)
    elif initial_condition == "amplitude":
        u0 = 1.5 * np.sin(x)
    elif initial_condition == "validation":
        u0 = (0.7 * np.cos(x + 0.4) + 0.6 * np.sin(2 * x - 0.2)
              + 0.35 * np.cos(4 * x + 0.1))
    else:
        raise ValueError("未知初始条件；支持 train/probe/probe2/probe3/amplitude/validation。")
    k = 2 * np.pi * np.fft.fftfreq(len(x), d=x[1] - x[0])
    eigenvalues = (1j * truth.get("u_x", 0) * k
                   - truth.get("u_xx", 0) * k**2
                   + truth.get("u", 0))
    u = np.fft.ifft(np.exp(t[:, None] * eigenvalues) * np.fft.fft(u0), axis=1).real
    rng = np.random.default_rng(seed)
    u = u + noise * float(np.sqrt(np.mean(u**2))) * rng.standard_normal(u.shape)
    return {"x": x.tolist(), "t": t.tolist(), "u": u.tolist(),
            "boundary": "periodic", "units": "dimensionless"}
