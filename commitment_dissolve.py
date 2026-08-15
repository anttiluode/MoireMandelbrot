"""Turn a Two Worlds commitment-time field into an A -> B video transition.

This is deliberately an *effect*, not a claim that the Moire-Mandelbrot toy is
a video model.  Two Worlds supplies a spatial first-crossing time T*(x,y).
That scalar field becomes a moving alpha mask: pixels/regions whose probe has
crossed the discrimination threshold transition first, unresolved regions stay
in A until an optional short finish-tail completes the dissolve.

Examples
--------
Built-in fractal pair, no input images required::

    python commitment_dissolve.py -o commitment_demo.mp4

Use any two images::

    python commitment_dissolve.py --image-a human.png --image-b robot.png \
        --lambda-a 0.13 --lambda-b 0.48 --horizon 80 -o human_to_robot.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageOps

from two_worlds import TwoWorldResult, accumulate_discrimination


def _rgb_u8(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        scale = 255.0 if float(np.nanmax(arr)) <= 1.5 else 1.0
        arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _fit_pair(
    image_a: Image.Image | np.ndarray,
    image_b: Image.Image | np.ndarray,
    size: Optional[tuple[int, int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    a = Image.fromarray(_rgb_u8(image_a))
    b = Image.fromarray(_rgb_u8(image_b))
    target = size or a.size
    a = ImageOps.fit(a, target, method=Image.Resampling.LANCZOS)
    b = ImageOps.fit(b, target, method=Image.Resampling.LANCZOS)
    return np.asarray(a, np.uint8), np.asarray(b, np.uint8)


def commitment_alpha(
    commitment_time: np.ndarray,
    moment: float,
    *,
    softness: float = 1.5,
) -> np.ndarray:
    """Smooth alpha field for one Two Worlds observation moment.

    NaN means "not decided within the measured horizon" and remains zero here.
    The optional end-of-video completion is handled separately by
    :func:`dissolve_frames` so the distinction is visible rather than silently
    pretending unresolved pixels crossed the scientific threshold.
    """
    tstar = np.asarray(commitment_time, dtype=np.float32)
    soft = max(1e-6, float(softness))
    u = (float(moment) - (tstar - soft)) / (2.0 * soft)
    u = np.clip(u, 0.0, 1.0)
    alpha = u * u * (3.0 - 2.0 * u)
    alpha[~np.isfinite(tstar)] = 0.0
    return alpha.astype(np.float32)


def commitment_map_image(result: TwoWorldResult, horizon: int) -> Image.Image:
    """Colour = first threshold crossing; black = unresolved by horizon."""
    import matplotlib

    tstar = result.commitment_time.astype(np.float32)
    finite = np.isfinite(tstar)
    norm = np.zeros_like(tstar, dtype=np.float32)
    if np.any(finite):
        norm[finite] = np.clip((tstar[finite] - 1.0) / max(1.0, float(horizon - 1)), 0, 1)
    rgba = matplotlib.colormaps["turbo"](norm)
    rgb = np.clip(rgba[..., :3] * 255.0, 0, 255).astype(np.uint8)
    rgb[~finite] = 0
    return Image.fromarray(rgb, mode="RGB")


def _default_world_images(result: TwoWorldResult, horizon: int) -> tuple[Image.Image, Image.Image]:
    """Use the two escape-time worlds themselves when no A/B images are supplied."""
    import matplotlib

    def paint(field: np.ndarray) -> Image.Image:
        x = np.clip(field.astype(np.float32) / max(1.0, float(horizon)), 0.0, 1.0)
        rgba = matplotlib.colormaps["twilight_shifted"](x)
        return Image.fromarray(np.clip(rgba[..., :3] * 255.0, 0, 255).astype(np.uint8))

    return paint(result.escape_a), paint(result.escape_b)


def dissolve_frames(
    image_a: Image.Image | np.ndarray,
    image_b: Image.Image | np.ndarray,
    commitment_time: np.ndarray,
    *,
    horizon: int,
    frame_count: int = 150,
    softness: float = 1.5,
    finish_tail: float = 0.12,
) -> list[np.ndarray]:
    """Create RGB uint8 frames driven by the commitment front.

    ``finish_tail`` is explicitly an artistic completion tail.  During the
    measured part of the movie, unresolved pixels remain in image A.  In the
    final fraction of the movie they cross-fade globally so an exported effect
    can end exactly on B.  Set it to 0 to leave unresolved pixels untouched.
    """
    frame_count = max(2, int(frame_count))
    horizon = max(1, int(horizon))
    finish_tail = float(np.clip(finish_tail, 0.0, 0.95))

    a, b = _fit_pair(image_a, image_b)
    h, w = a.shape[:2]
    tstar = np.asarray(
        Image.fromarray(np.asarray(commitment_time, np.float32), mode="F").resize(
            (w, h), resample=Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )
    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)

    frames: list[np.ndarray] = []
    scientific_end = 1.0 - finish_tail if finish_tail > 0 else 1.0
    for k in range(frame_count):
        progress = k / float(frame_count - 1)
        if scientific_end > 0:
            measured_progress = min(progress / scientific_end, 1.0)
        else:
            measured_progress = 1.0
        moment = 1.0 + measured_progress * max(0.0, float(horizon - 1))
        alpha = commitment_alpha(tstar, moment, softness=softness)

        if finish_tail > 0 and progress > scientific_end:
            u = (progress - scientific_end) / max(1e-6, finish_tail)
            u = float(np.clip(u, 0.0, 1.0))
            u = u * u * (3.0 - 2.0 * u)
            alpha = alpha + (1.0 - alpha) * u

        frame = a_f + (b_f - a_f) * alpha[..., None]
        frames.append(np.clip(frame, 0, 255).astype(np.uint8))
    return frames


def export_commitment_dissolve(
    output: str | Path,
    *,
    image_a: Optional[Image.Image | np.ndarray] = None,
    image_b: Optional[Image.Image | np.ndarray] = None,
    lambda_a: float = 0.13,
    lambda_b: float = 0.48,
    horizon: int = 80,
    resolution: int = 320,
    temperature: float = 0.5,
    readout: str = "complex",
    noise_sigma: float = 0.20,
    dprime_threshold: float = 3.0,
    fps: int = 30,
    seconds: float = 5.0,
    softness: float = 1.5,
    finish_tail: float = 0.12,
) -> tuple[Path, Path, TwoWorldResult]:
    """Render the Two Worlds field, write MP4, and save the T* map beside it."""
    try:
        import imageio.v2 as imageio
    except Exception as exc:  # pragma: no cover - dependency message for users
        raise RuntimeError(
            "MP4 export needs imageio + imageio-ffmpeg; run pip install -r requirements.txt"
        ) from exc

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() != ".mp4":
        output = output.with_suffix(".mp4")

    result = accumulate_discrimination(
        float(lambda_a),
        float(lambda_b),
        horizon=int(horizon),
        resolution=int(resolution),
        temperature=float(temperature),
        readout=str(readout),
        noise_sigma=float(noise_sigma),
        dprime_threshold=float(dprime_threshold),
    )

    if (image_a is None) != (image_b is None):
        raise ValueError("provide both image_a and image_b, or neither")
    if image_a is None:
        image_a, image_b = _default_world_images(result, int(horizon))

    frame_count = max(2, int(round(float(seconds) * int(fps))))
    frames = dissolve_frames(
        image_a,
        image_b,
        result.commitment_time,
        horizon=int(horizon),
        frame_count=frame_count,
        softness=float(softness),
        finish_tail=float(finish_tail),
    )

    with imageio.get_writer(
        str(output), fps=int(fps), codec="libx264", quality=8,
        macro_block_size=None,
    ) as writer:
        for frame in frames:
            writer.append_data(frame)

    map_path = output.with_name(output.stem + "_commitment.png")
    commitment_map_image(result, int(horizon)).save(map_path)
    return output, map_path, result


def _load(path: Optional[str]) -> Optional[Image.Image]:
    return None if not path else Image.open(path).convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser(description="Two Worlds commitment-front video effect")
    ap.add_argument("--image-a")
    ap.add_argument("--image-b")
    ap.add_argument("-o", "--output", default="commitment_dissolve.mp4")
    ap.add_argument("--lambda-a", type=float, default=0.13)
    ap.add_argument("--lambda-b", type=float, default=0.48)
    ap.add_argument("--horizon", type=int, default=80)
    ap.add_argument("--resolution", type=int, default=320)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--readout", choices=("complex", "magnitude", "phase", "escape"), default="complex")
    ap.add_argument("--noise-sigma", type=float, default=0.20)
    ap.add_argument("--threshold", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--softness", type=float, default=1.5)
    ap.add_argument("--finish-tail", type=float, default=0.12,
                    help="artistic final fraction used to fade unresolved pixels to B; 0 disables")
    args = ap.parse_args()

    out, cmap, result = export_commitment_dissolve(
        args.output,
        image_a=_load(args.image_a),
        image_b=_load(args.image_b),
        lambda_a=args.lambda_a,
        lambda_b=args.lambda_b,
        horizon=args.horizon,
        resolution=args.resolution,
        temperature=args.temperature,
        readout=args.readout,
        noise_sigma=args.noise_sigma,
        dprime_threshold=args.threshold,
        fps=args.fps,
        seconds=args.seconds,
        softness=args.softness,
        finish_tail=args.finish_tail,
    )
    resolved = float(np.mean(np.isfinite(result.commitment_time)))
    print(f"wrote {out}")
    print(f"wrote {cmap}")
    print(f"resolved by horizon: {100.0 * resolved:.1f}%")


if __name__ == "__main__":
    main()
