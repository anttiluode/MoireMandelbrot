"""
moire_mandelbrot.py
===================

The Moiré-Mandelbrot: a continuous family of fractals that interpolates
between the classical Mandelbrot set (z -> z^2 + c) and a dynamical system
whose update is governed by Fubini-Study / Moiré attention.

The interpolation parameter λ ∈ [0, 1]:
    λ = 0  -> pure z^2 + c                  (classical Mandelbrot)
    λ = 1  -> pure attention dynamics       (the "Moiré fractal")
    λ ∈ (0, 1) -> continuous hybrid         (the "liquid" regime)

The two systems live on the same stage — the Riemann sphere CP¹ with its
Fubini-Study metric — and this module renders the parameter-space fractal
(escape-time plot of c) as λ varies.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------

def _softmax_last(s: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax along the last axis."""
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-30)


def mandelbrot_step(z: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Classical Mandelbrot update: z -> z^2 + c."""
    return z * z + c


def fs_attention_step(z: np.ndarray, c: np.ndarray,
                      temperature: float = 0.5) -> np.ndarray:
    """
    One step of the Fubini-Study / Moiré attention dynamical system.

    Keys are three canonical reference points built from c:

        k1 = 1                       (constant)
        k2 = c                       (linear)
        k3 = c · conj(c)             (quadratic amplitude)

    Scoring is the canonical Hermitian pairing on CP^(d-1), normalized
    to the chordal Fubini-Study similarity:

        score(q, k) = Re<q, k>_H / (|q| |k|)   ∈ [-1, 1]

    Softmaxed with temperature T to get attention weights w, the
    "attention direction" is the weighted sum of keys. Magnitude
    expansion comes from the  |z|²  term (the same term that gives
    z² + c its escape dynamics); phase comes from the attention.

        r_new    = |z|² + |direction|
        φ_new    = 2·arg(direction) + arg(c)
        z'       = r_new · exp(i·φ_new) + c
    """
    k1 = np.ones_like(c)
    k2 = c
    k3 = c * np.conj(c) + 1e-12
    K = np.stack([k1, k2, k3], axis=-1)     # [..., 3]
    q = z[..., None]                         # [..., 1]

    num = np.real(np.conj(q) * K)
    denom = (np.abs(q) + 1e-12) * (np.abs(K) + 1e-12)
    score = num / denom / temperature

    w = _softmax_last(score)
    direction = (w * K).sum(axis=-1)

    r = (np.abs(z) ** 2) + np.abs(direction)
    phi = 2.0 * np.angle(direction) + np.angle(c + 1e-12)
    return r * np.exp(1j * phi) + c


def mixed_step(z: np.ndarray, c: np.ndarray, lam: float,
               temperature: float = 0.5) -> np.ndarray:
    """Interpolated update between Mandelbrot and Moiré attention."""
    a = mandelbrot_step(z, c)
    b = fs_attention_step(z, c, temperature=temperature)
    return (1.0 - lam) * a + lam * b


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(
    lam: float,
    resolution: int = 600,
    iters: int = 80,
    x_range: tuple[float, float] = (-2.0, 1.0),
    y_range: tuple[float, float] = (-1.2, 1.2),
    temperature: float = 0.5,
    escape: float = 20.0,
) -> np.ndarray:
    """
    Render the Moiré-Mandelbrot at interpolation λ.

    Returns a 2D float array of smooth iteration counts (useful for
    coloring with any matplotlib colormap).
    """
    xs = np.linspace(x_range[0], x_range[1], resolution)
    ys = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(xs, ys)
    c = X + 1j * Y
    z = np.zeros_like(c)

    escape_time = np.full(c.shape, float(iters), dtype=np.float32)
    escaped = np.zeros(c.shape, dtype=bool)

    for it in range(iters):
        z = mixed_step(z, c, lam, temperature=temperature)
        big = np.abs(z) > escape
        newly = big & ~escaped
        with np.errstate(invalid="ignore", divide="ignore"):
            mag = np.abs(z[newly])
            escape_time[newly] = it + 1 - np.log2(np.log2(mag + 1e-12) + 1e-12)
        escaped |= big
        z = np.where(escaped, 0, z)

    return escape_time
