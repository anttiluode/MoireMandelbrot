"""
Interactive "Two Worlds" observability toy built on MoireMandelbrot.

Run:
    python two_worlds_app.py

Drag the horizon and rerun the experiment. The bottom-left panel is not a
final-image subtraction: it is accumulated trajectory discrimination.
"""
from __future__ import annotations

import io
import time

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from two_worlds import READOUTS, accumulate_discrimination


PRESETS = {
    "Nearby liquid worlds": dict(lambda_a=0.42, lambda_b=0.48, temperature=0.50),
    "Classic vs first deformation": dict(lambda_a=0.00, lambda_b=0.10, temperature=0.50),
    "Liquid vs butterfly": dict(lambda_a=0.40, lambda_b=0.75, temperature=0.50),
    "Mandelbrot vs pure Moire": dict(lambda_a=0.00, lambda_b=1.00, temperature=0.50),
}


def _extent(cx: float, cy: float, zoom: float):
    span_x = 1.5 / float(zoom)
    span_y = 1.2 / float(zoom)
    return (
        (float(cx) - span_x, float(cx) + span_x),
        (float(cy) - span_y, float(cy) + span_y),
    )


def _fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, facecolor="#05070b",
                bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def run_experiment(
    lambda_a: float,
    lambda_b: float,
    horizon: int,
    readout_label: str,
    noise_sigma: float,
    dprime_threshold: float,
    temperature: float,
    resolution: int,
    cx: float,
    cy: float,
    zoom: float,
):
    readout = READOUTS[readout_label]
    x_range, y_range = _extent(cx, cy, zoom)

    t0 = time.perf_counter()
    result = accumulate_discrimination(
        lambda_a=float(lambda_a),
        lambda_b=float(lambda_b),
        horizon=int(horizon),
        resolution=int(resolution),
        x_range=x_range,
        y_range=y_range,
        temperature=float(temperature),
        readout=readout,
        noise_sigma=float(noise_sigma),
        dprime_threshold=float(dprime_threshold),
    )
    elapsed = time.perf_counter() - t0

    dprime = np.sqrt(result.d2) / float(noise_sigma)
    evidence = np.log10(1.0 + dprime * dprime)
    best_flat = int(np.argmax(result.d2))
    iy, ix = np.unravel_index(best_flat, result.d2.shape)
    best_x = float(result.xs[ix])
    best_y = float(result.ys[iy])
    final_fraction = float(result.fraction_curve[-1])
    median_dprime = float(np.median(dprime))

    extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
    fig, axs = plt.subplots(2, 2, figsize=(11, 9), facecolor="#05070b")
    for ax in axs.flat:
        ax.set_facecolor("#05070b")
        ax.tick_params(colors="0.7", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("0.25")

    axs[0, 0].imshow(
        result.escape_a, origin="lower", extent=extent,
        cmap="twilight_shifted", interpolation="bilinear"
    )
    axs[0, 0].set_title(
        f"WORLD A  lambda={lambda_a:.3f}\nwhat has happened by T={int(horizon)}",
        color="white", fontsize=11
    )
    axs[0, 1].imshow(
        result.escape_b, origin="lower", extent=extent,
        cmap="twilight_shifted", interpolation="bilinear"
    )
    axs[0, 1].set_title(
        f"WORLD B  lambda={lambda_b:.3f}\nwhat has happened by T={int(horizon)}",
        color="white", fontsize=11
    )

    im2 = axs[1, 0].imshow(
        evidence, origin="lower", extent=extent,
        cmap="magma", interpolation="bilinear"
    )
    if np.nanmax(dprime) >= float(dprime_threshold):
        try:
            axs[1, 0].contour(
                result.xs, result.ys, dprime,
                levels=[float(dprime_threshold)],
                colors="white", linewidths=0.75, alpha=0.75,
            )
        except ValueError:
            pass
    axs[1, 0].plot(best_x, best_y, marker="x", markersize=9,
                   markeredgewidth=2, color="cyan")
    axs[1, 0].set_title(
        "WHAT THE EXPERIMENT CAN TELL APART YET\n"
        r"$\log_{10}(1 + D_T^2/\sigma^2)$  -  white = threshold",
        color="white", fontsize=10
    )
    cb = fig.colorbar(im2, ax=axs[1, 0], fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors="0.7", labelsize=8)

    ts = np.arange(1, int(horizon) + 1)
    axs[1, 1].plot(ts, 100.0 * result.fraction_curve, linewidth=2)
    axs[1, 1].axvline(int(horizon), linestyle="--", linewidth=1, alpha=0.6)
    axs[1, 1].set_ylim(0, 100)
    axs[1, 1].set_xlim(1, max(2, int(horizon)))
    axs[1, 1].grid(alpha=0.15)
    axs[1, 1].set_xlabel("elapsed observation / iterations", color="0.8")
    axs[1, 1].set_ylabel("% probes above d-prime threshold", color="0.8")
    axs[1, 1].set_title(
        "THE EXPERIMENT ACQUIRES THE ABILITY TO DECIDE",
        color="white", fontsize=10
    )

    fig.suptitle(
        f"TWO WORLDS - same stage, different law - readout: {readout_label}",
        color="white", fontsize=14, y=0.995
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    image = _fig_to_pil(fig)

    info = f"""
### At this moment

**World A:** lambda = `{float(lambda_a):.3f}`  
**World B:** lambda = `{float(lambda_b):.3f}`  
**Readout C:** `{readout_label}`  
**Elapsed observation T:** `{int(horizon)}` iterations

With the toy white-noise scale **sigma = {float(noise_sigma):.3f}** and
decision line **d-prime >= {float(dprime_threshold):.2f}**:

- **{100.0 * final_fraction:.1f}%** of probe locations can distinguish the two worlds by this horizon.
- Median probe **d-prime = {median_dprime:.2f}**.
- Strongest current probe is near **c = {best_x:+.3f} {best_y:+.3f}i** (cyan x).
- Computed at {int(resolution)}x{int(resolution)} in **{elapsed:.2f}s**.

The important panel is the lower-left one. It is **not** the difference
between the two final fractal pictures. Each pixel accumulates the difference
between the two *response histories* up to the selected horizon:

`D_T^2(c) = sum_t || y_A(c,t) - y_B(c,t) ||^2`

Change the **readout** without changing either world. If the map changes,
the observer changed what distinctions were available.
"""
    return image, info


def apply_preset(name):
    p = PRESETS[name]
    return p["lambda_a"], p["lambda_b"], p["temperature"]


def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown(
            """
# Two Worlds - when does an experiment know the difference?

This is the old **Moire-Mandelbrot** toy turned into an observability toy.

Pick two nearby dynamical laws. Every point `c` in the image is treated as a
possible probe. Both worlds start from the same `z=0`. We reveal only the first
**T** steps of the probe response and accumulate how different those histories
have become.

At `T=1`, many probes cannot tell the worlds apart. Increase the horizon and
watch the lower-left map ignite. Then change the **readout C**: magnitude,
phase, full complex state, or merely "has it escaped?". Same worlds. Different
observer. Different distinguishability.

Nothing here is new information theory. It is a deliberately weird picture of
finite-horizon signal discrimination.
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                preset = gr.Dropdown(
                    list(PRESETS.keys()),
                    value="Nearby liquid worlds",
                    label="World pair",
                )
                lambda_a = gr.Slider(
                    0.0, 1.0, value=0.42, step=0.01, label="World A lambda"
                )
                lambda_b = gr.Slider(
                    0.0, 1.0, value=0.48, step=0.01, label="World B lambda"
                )
                horizon = gr.Slider(
                    1, 100, value=35, step=1,
                    label="elapsed observation T (iterations)"
                )
                readout = gr.Radio(
                    list(READOUTS.keys()),
                    value="Complex state + escape",
                    label="readout / receiver C",
                )
                noise_sigma = gr.Slider(
                    0.02, 1.0, value=0.20, step=0.01,
                    label="toy observation-noise sigma"
                )
                dprime_threshold = gr.Slider(
                    0.5, 10.0, value=3.0, step=0.25,
                    label="decision threshold d-prime"
                )
                temperature = gr.Slider(
                    0.1, 3.0, value=0.5, step=0.05,
                    label="attention temperature"
                )
                resolution = gr.Slider(
                    120, 500, value=280, step=20, label="resolution"
                )

                with gr.Accordion("view", open=False):
                    cx = gr.Slider(-2.5, 2.5, value=-0.5, step=0.01, label="center x")
                    cy = gr.Slider(-2.0, 2.0, value=0.0, step=0.01, label="center y")
                    zoom = gr.Slider(0.3, 20.0, value=1.0, step=0.1, label="zoom")

                run = gr.Button("Run the experiment", variant="primary")

            with gr.Column(scale=2):
                image = gr.Image(type="pil", label="Two Worlds", height=760)
                info = gr.Markdown()

        preset.change(
            apply_preset,
            inputs=[preset],
            outputs=[lambda_a, lambda_b, temperature],
        )

        inputs = [
            lambda_a, lambda_b, horizon, readout, noise_sigma,
            dprime_threshold, temperature, resolution, cx, cy, zoom,
        ]
        run.click(run_experiment, inputs=inputs, outputs=[image, info])
        demo.load(run_experiment, inputs=inputs, outputs=[image, info])

        gr.Markdown(
            """
---
### Things to try

- **Nearby liquid worlds**, then move `T` from 3 -> 10 -> 30 -> 80.
- Keep the worlds fixed and switch **Complex -> Magnitude -> Phase -> Escape**.
  This is the readout/receiver experiment.
- Put lambda A and lambda B almost on top of each other. The "best probe" moves
  and the acquisition curve slows.
- Set lambda A = lambda B. The evidence map should go black. That is the calibration.

The cyan x is the best probe *within this toy grid at this horizon*. It is the
little seed of the practical TWC idea: if two explanations are still hard to
distinguish, **where should I measure next?**
"""
        )
    return demo


if __name__ == "__main__":
    build_ui().launch()
