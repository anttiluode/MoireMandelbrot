"""
A tiny observability/discrimination playground built on MoireMandelbrot.

Two dynamical laws (lambda_A, lambda_B) are treated as competing "worlds".
For every parameter-space point c, both worlds start from z=0 and emit a
chosen observation y(t). The accumulated squared separation

    D_T^2(c) = sum_{t=1..T} || y_A(c,t) - y_B(c,t) ||^2

is the finite-horizon evidence available at that probe.

The first threshold-crossing time is also recorded:

    T_star(c) = min { T : sqrt(D_T^2(c)) / sigma >= threshold }

Pixels that never cross within the requested horizon are NaN in T_star.  This
is intentionally a visual demo of standard signal discrimination /
observability ideas, not a new information-theory claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from moire_mandelbrot import mixed_step


READOUTS = {
    "Complex state + escape": "complex",
    "Magnitude + escape": "magnitude",
    "Phase + escape": "phase",
    "Escape only": "escape",
}


@dataclass
class TwoWorldResult:
    d2: np.ndarray
    escape_a: np.ndarray
    escape_b: np.ndarray
    escaped_a: np.ndarray
    escaped_b: np.ndarray
    fraction_curve: np.ndarray
    median_dprime_curve: np.ndarray
    commitment_time: np.ndarray
    xs: np.ndarray
    ys: np.ndarray


def _observe(z: np.ndarray, escaped: np.ndarray, mode: str) -> np.ndarray:
    """Bounded observation vector so escaped orbits cannot numerically explode."""
    mag = np.abs(z)
    squashed = z / (1.0 + mag)
    flag = escaped.astype(np.float64)

    if mode == "complex":
        return np.stack([squashed.real, squashed.imag, flag], axis=-1)
    if mode == "magnitude":
        return np.stack([np.abs(squashed), flag], axis=-1)
    if mode == "phase":
        unit = np.zeros_like(z)
        valid = mag > 1e-9
        unit[valid] = z[valid] / mag[valid]
        return np.stack([unit.real, unit.imag, flag], axis=-1)
    if mode == "escape":
        return flag[..., None]
    raise ValueError(f"unknown readout mode: {mode}")


def accumulate_discrimination(
    lambda_a: float,
    lambda_b: float,
    *,
    horizon: int = 50,
    resolution: int = 300,
    x_range: tuple[float, float] = (-2.0, 1.0),
    y_range: tuple[float, float] = (-1.2, 1.2),
    temperature: float = 0.5,
    escape: float = 20.0,
    readout: str = "complex",
    noise_sigma: float = 0.20,
    dprime_threshold: float = 3.0,
) -> TwoWorldResult:
    """Accumulate receiver-relative separation for two Moire-Mandelbrot worlds."""
    horizon = int(horizon)
    resolution = int(resolution)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if resolution < 8:
        raise ValueError("resolution must be >= 8")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if noise_sigma <= 0:
        raise ValueError("noise_sigma must be > 0")
    if dprime_threshold <= 0:
        raise ValueError("dprime_threshold must be > 0")

    xs = np.linspace(float(x_range[0]), float(x_range[1]), resolution)
    ys = np.linspace(float(y_range[0]), float(y_range[1]), resolution)
    X, Y = np.meshgrid(xs, ys)
    c = X + 1j * Y

    za = np.zeros_like(c)
    zb = np.zeros_like(c)
    escaped_a = np.zeros(c.shape, dtype=bool)
    escaped_b = np.zeros(c.shape, dtype=bool)
    escape_a = np.full(c.shape, float(horizon), dtype=np.float32)
    escape_b = np.full(c.shape, float(horizon), dtype=np.float32)
    d2 = np.zeros(c.shape, dtype=np.float64)

    fraction_curve = np.zeros(horizon, dtype=np.float64)
    median_dprime_curve = np.zeros(horizon, dtype=np.float64)
    commitment_time = np.full(c.shape, np.nan, dtype=np.float32)

    for it in range(horizon):
        next_a = mixed_step(za, c, float(lambda_a), temperature=float(temperature))
        next_b = mixed_step(zb, c, float(lambda_b), temperature=float(temperature))

        # Once an orbit has escaped, stop evolving its raw z to avoid overflow.
        big_a = np.abs(next_a) > float(escape)
        big_b = np.abs(next_b) > float(escape)
        new_a = big_a & ~escaped_a
        new_b = big_b & ~escaped_b
        escape_a[new_a] = float(it + 1)
        escape_b[new_b] = float(it + 1)
        escaped_a |= big_a
        escaped_b |= big_b
        za = np.where(escaped_a, 0.0, next_a)
        zb = np.where(escaped_b, 0.0, next_b)

        ya = _observe(za, escaped_a, readout)
        yb = _observe(zb, escaped_b, readout)
        d2 += np.sum((ya - yb) ** 2, axis=-1)

        dprime = np.sqrt(d2) / float(noise_sigma)
        crossed = np.isnan(commitment_time) & (dprime >= float(dprime_threshold))
        commitment_time[crossed] = float(it + 1)
        fraction_curve[it] = np.mean(dprime >= float(dprime_threshold))
        median_dprime_curve[it] = np.median(dprime)

    return TwoWorldResult(
        d2=d2,
        escape_a=escape_a,
        escape_b=escape_b,
        escaped_a=escaped_a,
        escaped_b=escaped_b,
        fraction_curve=fraction_curve,
        median_dprime_curve=median_dprime_curve,
        commitment_time=commitment_time,
        xs=xs,
        ys=ys,
    )
