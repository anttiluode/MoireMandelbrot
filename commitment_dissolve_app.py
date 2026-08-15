"""Gradio front end for the Two Worlds Commitment Dissolve video effect.

Run:
    python commitment_dissolve_app.py

Upload any A/B pair or leave both empty to use the two fractal worlds.
"""
from __future__ import annotations

import os
import tempfile

import gradio as gr
import numpy as np
from PIL import Image

from commitment_dissolve import export_commitment_dissolve
from two_worlds import READOUTS


PRESETS = {
    "Fast jagged front": dict(lambda_a=0.13, lambda_b=0.48, temperature=1.95),
    "Nearby liquid worlds": dict(lambda_a=0.42, lambda_b=0.48, temperature=0.50),
    "Classic vs deformation": dict(lambda_a=0.00, lambda_b=0.10, temperature=0.50),
    "Mandelbrot vs Moire": dict(lambda_a=0.00, lambda_b=1.00, temperature=0.50),
}


def apply_preset(name):
    p = PRESETS[name]
    return p["lambda_a"], p["lambda_b"], p["temperature"]


def make_video(
    image_a,
    image_b,
    lambda_a,
    lambda_b,
    horizon,
    readout_label,
    noise_sigma,
    dprime_threshold,
    temperature,
    resolution,
    seconds,
    softness,
    finish_tail,
):
    fd, path = tempfile.mkstemp(prefix="two_worlds_", suffix=".mp4")
    os.close(fd)
    out, cmap, result = export_commitment_dissolve(
        path,
        image_a=image_a,
        image_b=image_b,
        lambda_a=float(lambda_a),
        lambda_b=float(lambda_b),
        horizon=int(horizon),
        resolution=int(resolution),
        temperature=float(temperature),
        readout=READOUTS[readout_label],
        noise_sigma=float(noise_sigma),
        dprime_threshold=float(dprime_threshold),
        fps=30,
        seconds=float(seconds),
        softness=float(softness),
        finish_tail=float(finish_tail),
    )
    resolved = float(np.mean(np.isfinite(result.commitment_time)))
    map_img = Image.open(cmap).copy()
    note = f"""
### Commitment Dissolve

- **{100.0 * resolved:.1f}%** of the Two Worlds field crossed the chosen threshold by T={int(horizon)}.
- Colour in the map is the **first threshold-crossing time** `T*(x,y)`; black means unresolved by the measured horizon.
- The movie uses that field as an alpha clock. Regions that become decidable earlier transition earlier.
- The final **{100.0 * float(finish_tail):.0f}%** of the clip is an explicitly artistic completion tail for pixels that never crossed. Set **finish tail = 0** if you want unresolved pixels to remain in image A.

This is a temporal-mask effect driven by the Two Worlds field. It is not a claim that the fractal itself is a video model.
"""
    return str(out), map_img, note


def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown(
            """
# Commitment Dissolve — let one world invade another in its own time

Upload **Image A** and **Image B**. The Two Worlds experiment supplies a spatial
commitment time `T*(x,y)`: when each location first crosses the selected
finite-horizon discrimination threshold. That field becomes the transition
clock for a real MP4.

Leave both images empty and the app uses the two fractal worlds themselves.
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_a = gr.Image(type="pil", label="Image A (optional)")
                image_b = gr.Image(type="pil", label="Image B (optional)")
                preset = gr.Dropdown(list(PRESETS), value="Fast jagged front", label="Two Worlds field")
                lambda_a = gr.Slider(0.0, 1.0, value=0.13, step=0.01, label="World A lambda")
                lambda_b = gr.Slider(0.0, 1.0, value=0.48, step=0.01, label="World B lambda")
                horizon = gr.Slider(8, 120, value=68, step=1, label="field horizon T")
                readout = gr.Radio(list(READOUTS), value="Complex state + escape", label="readout / receiver C")
                noise_sigma = gr.Slider(0.02, 1.0, value=0.20, step=0.01, label="toy noise sigma")
                dprime_threshold = gr.Slider(0.5, 10.0, value=5.75, step=0.25, label="decision threshold d-prime")
                temperature = gr.Slider(0.1, 3.0, value=1.95, step=0.05, label="attention temperature")
                resolution = gr.Slider(120, 500, value=320, step=20, label="field resolution")
                seconds = gr.Slider(2.0, 10.0, value=5.0, step=0.5, label="video seconds")
                softness = gr.Slider(0.1, 5.0, value=1.5, step=0.1, label="front softness (iterations)")
                finish_tail = gr.Slider(0.0, 0.4, value=0.12, step=0.01, label="artistic finish tail")
                run = gr.Button("Render the transition", variant="primary")

            with gr.Column(scale=2):
                video = gr.Video(label="Commitment Dissolve", autoplay=True)
                cmap = gr.Image(type="pil", label="T* commitment-time map")
                info = gr.Markdown()

        preset.change(apply_preset, inputs=[preset], outputs=[lambda_a, lambda_b, temperature])
        inputs = [
            image_a, image_b, lambda_a, lambda_b, horizon, readout,
            noise_sigma, dprime_threshold, temperature, resolution,
            seconds, softness, finish_tail,
        ]
        run.click(make_video, inputs=inputs, outputs=[video, cmap, info])

    return demo


if __name__ == "__main__":
    build_ui().launch()
