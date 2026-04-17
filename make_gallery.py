"""
Regenerate all the gallery images shown in README.md.

Usage:  python make_gallery.py
Output: gallery/*.png
"""

import os
import matplotlib.pyplot as plt

from moire_mandelbrot import render


OUT_DIR = "gallery"
os.makedirs(OUT_DIR, exist_ok=True)


def save(et, path, extent, dpi=140, cmap="twilight_shifted"):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), facecolor="black")
    ax.imshow(et, cmap=cmap, extent=extent, origin="lower",
              interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, facecolor="black", bbox_inches="tight",
                pad_inches=0)
    plt.close(fig)
    print(f"  saved {path}")


def make_morph_grid():
    lams = [0.00, 0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 1.00]
    fig, axes = plt.subplots(3, 3, figsize=(15, 15), facecolor="black")
    for ax, lam in zip(axes.flatten(), lams):
        print(f"  rendering λ={lam}")
        et = render(lam=lam, resolution=500, iters=80)
        ax.imshow(et, cmap="twilight_shifted",
                  extent=[-2, 1, -1.2, 1.2], origin="lower",
                  interpolation="bilinear")
        ax.set_title(f"λ={lam:.3f}", color="white", fontsize=14)
        ax.axis("off")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "morph_grid.png")
    plt.savefig(path, dpi=130, facecolor="black", bbox_inches="tight",
                pad_inches=0.1)
    plt.close(fig)
    print(f"  saved {path}")


def main():
    print("Rendering liquid funky Mandelbrot (λ=0.5)...")
    et = render(lam=0.5, resolution=1000, iters=100)
    save(et, os.path.join(OUT_DIR, "liquid.png"), [-2, 1, -1.2, 1.2])

    print("Rendering tendril zoom (λ=0.4)...")
    et = render(lam=0.4, resolution=900, iters=140,
                x_range=(-1.6, -0.6), y_range=(-0.5, 0.5))
    save(et, os.path.join(OUT_DIR, "tendrils.png"),
         [-1.6, -0.6, -0.5, 0.5])

    print("Rendering pure Moiré fractal (λ=1.0)...")
    et = render(lam=1.0, resolution=900, iters=100,
                x_range=(-1.5, 0.5), y_range=(-0.8, 0.8))
    save(et, os.path.join(OUT_DIR, "moire_pure.png"),
         [-1.5, 0.5, -0.8, 0.8])

    print("Rendering morph grid...")
    make_morph_grid()


if __name__ == "__main__":
    main()
