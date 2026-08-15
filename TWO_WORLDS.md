# Two Worlds

A small observability / experiment-design demo built directly on the existing Moire-Mandelbrot dynamics.

Run it with:

```bash
pip install -r requirements.txt
python two_worlds_app.py
```

## The picture

Choose two candidate dynamical laws:

```text
World A: lambda_A
World B: lambda_B
```

Every point `c` in the rendered complex plane is treated as a possible probe. Both candidate worlds start from the same `z=0`. The demo exposes only the first `T` updates of each response history.

For a chosen readout `C`, it accumulates

```text
D_T^2(c) = sum_{t=1..T} || y_A(c,t) - y_B(c,t) ||^2
```

where `y` is one of four deliberately simple observations:

```text
complex state + escape flag
magnitude + escape flag
phase + escape flag
escape flag only
```

The lower-left panel visualizes

```text
log10(1 + D_T^2 / sigma^2)
```

for a user-controlled toy white-noise scale `sigma`. A white contour marks a chosen `d-prime = sqrt(D_T^2)/sigma` decision line. The cyan x marks the most discriminating probe on the current grid.

The lower-right panel shows how the fraction of available probes above the decision threshold grows as the observation horizon increases.

## Why this exists

The point is not to claim a new fractal metric or new information theory.

It is a visual version of a standard finite-horizon discrimination question:

> At this readout, by this moment, how distinguishable have two possible causes/worlds become?

The useful interactions are:

```text
WAIT       increase T
ROUTE      change the readout C
PROBE      use a bright / high-information c
NOISE      change the practical distinguishability floor
```

The same two worlds can be easy to tell apart under one readout and difficult under another.

This is the intentionally weird public-demo cousin of the measurement-capability work in TransientWaveCompiler, where the competing "worlds" are candidate physical circuit explanations and the available actions are real measurement channels / controlled perturbations.

## Calibrations

`test_two_worlds.py` freezes three boring checks:

1. `lambda_A == lambda_B` gives exactly zero discrimination everywhere.
2. accumulated evidence is nonnegative and threshold-crossing fraction cannot decrease with `T`.
3. all four readout modes execute with finite outputs.

Run:

```bash
python -m unittest test_two_worlds -v
```

## What this is not

- not a claim that Mandelbrot dynamics are a physical measurement device;
- not Clockfield or black-hole physics;
- not Connes distance;
- not thermodynamic entropy;
- not evidence that phase is intrinsically superior to magnitude;
- not a replacement for the original Moire-Mandelbrot morph demo.

It is simply a small application where you can watch an experiment acquire the ability to tell two worlds apart.
