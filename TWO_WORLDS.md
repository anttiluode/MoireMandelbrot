# Two Worlds

A small observability / experiment-design demo built directly on the existing Moire-Mandelbrot dynamics.

Run the original interactive experiment with:

```bash
pip install -r requirements.txt
python two_worlds_app.py
```

The repo now also contains **Commitment Dissolve**, a video effect driven by the first threshold-crossing time of the same experiment:

```bash
python commitment_dissolve_app.py
```

On Windows you can double-click:

```text
run_two_worlds.bat
run_commitment_dissolve.bat
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

## Commitment time T*

`two_worlds.py` now also records, per probe location, the first time the selected decision line is crossed:

```text
T*(c) = min { T : sqrt(D_T^2(c)) / sigma >= threshold }
```

Locations that never cross within the requested horizon are stored as `NaN` and shown black in the commitment-time map.

This is simply the inverse view of the existing acquisition curve: instead of asking "how much is decidable at time T?", ask "when did this location first become decidable?"

## Commitment Dissolve

`commitment_dissolve.py` treats `T*(x,y)` as a **temporal alpha mask** between any two images A and B.

Early-crossing regions switch first. Late-crossing regions switch later. Regions that never cross stay in A throughout the measured part of the clip. By default the final 12% is an explicitly artistic global completion tail so the MP4 can end exactly on B; set `finish_tail=0` to keep scientifically unresolved pixels in A.

CLI, built-in fractal images:

```bash
python commitment_dissolve.py -o commitment_demo.mp4
```

Any two images:

```bash
python commitment_dissolve.py \
  --image-a human.png \
  --image-b robot.png \
  --lambda-a 0.13 \
  --lambda-b 0.48 \
  --horizon 80 \
  -o human_to_robot.mp4
```

The exporter writes both the MP4 and a sibling `*_commitment.png` showing the `T*` field that drove it.

This is intentionally an **effect**, not a claim that Moire-Mandelbrot is a video model. It is a first experiment in using a measured/constructed time field as a transition geometry.

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

`test_two_worlds.py` freezes the boring checks:

1. `lambda_A == lambda_B` gives exactly zero discrimination everywhere and no finite `T*`.
2. accumulated evidence is nonnegative and threshold-crossing fraction cannot decrease with `T`.
3. all four readout modes execute with finite outputs.
4. unresolved `T*` pixels remain dark in the scientific transition mask.
5. an enabled artistic finish-tail really does end exactly on image B.

Run:

```bash
python -m unittest test_two_worlds -v
```

The same tests run in GitHub Actions via `two-worlds-ci`.

## What this is not

- not a claim that Mandelbrot dynamics are a physical measurement device;
- not Clockfield or black-hole physics;
- not Connes distance;
- not thermodynamic entropy;
- not evidence that phase is intrinsically superior to magnitude;
- not a replacement for the original Moire-Mandelbrot morph demo;
- not a claim that denoising/generative time is physical scene time.

It is simply a small application where you can watch an experiment acquire the ability to tell two worlds apart — and now use the resulting first-crossing field as a weird clock for an actual video transition.
