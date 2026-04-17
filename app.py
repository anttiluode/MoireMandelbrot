"""
Gradio app for the Moiré-Mandelbrot.

Runs on CPU. At 500×500 / 80 iterations each frame is ~1 second on
a modest machine. Drag the λ slider to watch the Mandelbrot melt into
the Moiré fractal and back.
"""

import io
import time

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from moire_mandelbrot import render


CMAP_CHOICES = ["twilight_shifted", "magma", "inferno", "plasma", "viridis", "cividis"]


def _render_to_image(
    lam: float,
    temperature: float,
    resolution: int,
    iters: int,
    cx: float,
    cy: float,
    zoom: float,
    cmap: str,
) -> tuple[Image.Image, str]:
    span_x = 1.5 / zoom
    span_y = 1.2 / zoom
    x_range = (cx - span_x, cx + span_x)
    y_range = (cy - span_y, cy + span_y)

    t0 = time.perf_counter()
    et = render(
        lam=lam,
        resolution=int(resolution),
        iters=int(iters),
        x_range=x_range,
        y_range=y_range,
        temperature=temperature,
    )
    dt = time.perf_counter() - t0

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), facecolor="black")
    ax.imshow(
        et,
        cmap=cmap,
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        origin="lower",
        interpolation="bilinear",
    )
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, facecolor="black",
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).copy()

    info = (
        f"**λ** = {lam:.3f}   |   **T** = {temperature:.2f}   |   "
        f"zoom = {zoom:.2f}   |   {int(resolution)}×{int(resolution)}   |   "
        f"{int(iters)} iter   |   rendered in {dt:.1f}s"
    )
    return img, info


PRESETS = {
    "Classic Mandelbrot (λ=0)":
        dict(lam=0.00, temperature=0.5, cx=-0.5, cy=0.0, zoom=1.0),
    "First morph (λ=0.25)":
        dict(lam=0.25, temperature=0.5, cx=-0.5, cy=0.0, zoom=1.0),
    "Liquid funky (λ=0.50)":
        dict(lam=0.50, temperature=0.5, cx=-0.5, cy=0.0, zoom=1.0),
    "Spiked flower (λ=0.40)":
        dict(lam=0.40, temperature=0.5, cx=-0.5, cy=0.0, zoom=1.0),
    "Butterfly regime (λ=0.75)":
        dict(lam=0.75, temperature=0.5, cx=-0.5, cy=0.0, zoom=1.0),
    "Pure Moiré fractal (λ=1.0)":
        dict(lam=1.00, temperature=0.5, cx=-0.5, cy=0.0, zoom=1.4),
    "Tendril zoom (λ=0.40)":
        dict(lam=0.40, temperature=0.5, cx=-1.10, cy=0.0, zoom=3.0),
}


def apply_preset(name):
    p = PRESETS[name]
    return p["lam"], p["temperature"], p["cx"], p["cy"], p["zoom"]


def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # 🌊 The Moiré-Mandelbrot

            A continuous family of fractals that interpolates between the
            classical Mandelbrot set and a dynamical system whose update
            is Fubini-Study / Moiré attention. Drag **λ** to watch it morph.

            - **λ = 0** — pure `z → z² + c` (classical Mandelbrot)
            - **λ = 1** — pure attention dynamics (three-lobed "Moiré fractal")
            - **λ ≈ 0.4–0.6** — the **liquid regime**: petaled, fractal-filigreed

            Both systems live on the same stage: the Riemann sphere CP¹
            with the Fubini-Study metric. The deformation is continuous
            because it is geometrically natural — the Mandelbrot iteration
            and phase-coherence attention are two dynamical systems on the
            same Kähler manifold.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                preset = gr.Dropdown(
                    choices=list(PRESETS.keys()),
                    value="Liquid funky (λ=0.50)",
                    label="Presets",
                )
                lam = gr.Slider(0.0, 1.0, value=0.5, step=0.01,
                                label="λ  (Mandelbrot ↔ Moiré)")
                temperature = gr.Slider(0.1, 3.0, value=0.5, step=0.05,
                                        label="Attention temperature T")
                cx = gr.Slider(-2.5, 2.5, value=-0.5, step=0.01,
                               label="center x")
                cy = gr.Slider(-2.0, 2.0, value=0.0, step=0.01,
                               label="center y")
                zoom = gr.Slider(0.3, 30.0, value=1.0, step=0.1,
                                 label="zoom")
                resolution = gr.Slider(200, 900, value=500, step=50,
                                       label="resolution (px)")
                iters = gr.Slider(20, 200, value=80, step=5,
                                  label="iterations")
                cmap = gr.Dropdown(CMAP_CHOICES, value="twilight_shifted",
                                   label="colormap")
                btn = gr.Button("Render", variant="primary")
                info = gr.Markdown()

            with gr.Column(scale=2):
                img = gr.Image(label="Moiré-Mandelbrot", type="pil", height=640)

        # preset dropdown wires into the sliders
        preset.change(
            apply_preset, inputs=[preset],
            outputs=[lam, temperature, cx, cy, zoom],
        )

        inputs = [lam, temperature, resolution, iters, cx, cy, zoom, cmap]
        btn.click(_render_to_image, inputs=inputs, outputs=[img, info])

        # render once on load
        demo.load(_render_to_image, inputs=inputs, outputs=[img, info])

        gr.Markdown(
            """
            ---
            **What you are looking at.** For each point *c* in the plane
            we iterate a map starting from z = 0. Dark = orbit stays
            bounded; bright = orbit escapes (colored by how fast). The
            Mandelbrot set is the λ = 0 bounded region; the "Moiré
            fractal" is the λ = 1 bounded region. At λ = 0.5 the two
            fight, and the fight produces filigree.

            **Why this is not just an effect.** The Moiré attention score
            is `Re<q,k>_H` — the real part of the canonical Hermitian
            pairing on CP^(d-1). The Mandelbrot map `z² + c` is a
            degree-2 rational map on CP¹. Same manifold, different dynamics.
            See [REPO_LINK] for the math.
            """
        )

    return demo


if __name__ == "__main__":
    build_ui().launch()
