"""
benchmark_complexity.py  --  Member A

Measures build time of:
    original   : ImageCompressor (std/mean recomputed over pixels, SortedList)
    fast+sorted: integral image + SortedList
    fast+heap  : integral image + heapq

Experiment 1: time vs image size N at fixed iterations k
Experiment 2: time vs iterations k at fixed image size

Usage:
    python benchmark_complexity.py                       # synthetic test image
    python benchmark_complexity.py --image input/mountain.jpg
    python benchmark_complexity.py --max-side 2048 --repeats 3 --memory

Requires: numpy, matplotlib (pip install matplotlib). Pillow only if --image is used.
Outputs : results_size.csv, results_iters.csv, plot_time_vs_size.png, plot_time_vs_iters.png
"""

import argparse
import csv
import time
import tracemalloc

import numpy as np

from quad_tree_compression import ImageCompressor          # the ORIGINAL implementation
from fast_quad_tree import FastImageCompressor


# ----------------------------------------------------------------------
def synthetic_image(side: int, seed: int = 0) -> np.ndarray:
    """Gradient + noise + a few hard-edged blocks: has both smooth and busy areas."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:side, 0:side]
    img = np.stack([xx * 255 // side, yy * 255 // side, (xx + yy) * 255 // (2 * side)], axis=-1)
    img = img + rng.integers(0, 40, img.shape)
    for _ in range(12):
        x, y = rng.integers(0, side * 3 // 4, 2)
        s = rng.integers(side // 16 + 1, side // 4 + 2)
        img[y:y + s, x:x + s] = rng.integers(0, 255, 3)
    return np.clip(img, 0, 255).astype(np.uint8)


def load_image(path: str, side: int):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im = im.resize((side, side))
    return np.array(im)


# ----------------------------------------------------------------------
# Each variant is a function image, k -> compressor after add_detail(k).
# Build time INCLUDES the constructor (so the integral-image pre-processing
# is counted) and add_detail. draw()/encode are excluded: they are O(N)/O(k)
# and identical in both versions.
VARIANTS = {
    "original":    lambda img, k: _run(ImageCompressor(img), k),
    "fast+sorted": lambda img, k: _run(FastImageCompressor(img, queue="sorted"), k),
    "fast+heap":   lambda img, k: _run(FastImageCompressor(img, queue="heap"), k),
}


def _run(comp, k):
    comp.add_detail(k, 10)
    return comp


def time_variant(name, img, k, repeats):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        VARIANTS[name](img, k)
        best = min(best, time.perf_counter() - t0)       # min = least noisy
    return best


def peak_memory_mb(name, img, k):
    tracemalloc.start()
    VARIANTS[name](img, k)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1e6


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="optional real image (resized to a square)")
    ap.add_argument("--max-side", type=int, default=2048, help="largest side for experiment 1")
    ap.add_argument("--fixed-k", type=int, default=5000, help="iterations used in experiment 1")
    ap.add_argument("--fixed-side", type=int, default=1024, help="image side used in experiment 2")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--memory", action="store_true", help="also record peak memory (slower)")
    args = ap.parse_args()

    make = (lambda s: load_image(args.image, s)) if args.image else synthetic_image

    # ---- Experiment 1: vary image size, fixed k -----------------------
    sides = [s for s in (128, 256, 512, 1024, 2048, 4096) if s <= args.max_side]
    rows1 = []
    print(f"\nExperiment 1: time vs image size (k = {args.fixed_k})")
    for side in sides:
        img = make(side)
        row = {"side": side, "pixels": side * side}
        for name in VARIANTS:
            row[name] = time_variant(name, img, args.fixed_k, args.repeats)
            if args.memory:
                row[name + "_MB"] = peak_memory_mb(name, img, args.fixed_k)
        rows1.append(row)
        print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

    # ---- Experiment 2: vary k, fixed image ---------------------------
    img = make(args.fixed_side)
    ks = [100, 300, 1000, 3000, 10000, 30000]
    rows2 = []
    print(f"\nExperiment 2: time vs iterations (image {args.fixed_side}x{args.fixed_side})")
    for k in ks:
        row = {"k": k}
        for name in VARIANTS:
            row[name] = time_variant(name, img, k, args.repeats)
        rows2.append(row)
        print("  ", {a: (round(b, 4) if isinstance(b, float) else b) for a, b in row.items()})

    for fname, rows in (("results_size.csv", rows1), ("results_iters.csv", rows2)):
        with open(fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    plot(rows1, rows2, args)


def plot(rows1, rows2, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {"original": "o-", "fast+sorted": "s--", "fast+heap": "^-"}

    fig, ax = plt.subplots(figsize=(6, 4.2))
    for name, st in styles.items():
        ax.plot([r["pixels"] for r in rows1], [r[name] for r in rows1], st, label=name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("image size N (pixels)"); ax.set_ylabel("build time (s)")
    ax.set_title(f"Time vs image size  (k = {args.fixed_k})")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig("plot_time_vs_size.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    for name, st in styles.items():
        ax.plot([r["k"] for r in rows2], [r[name] for r in rows2], st, label=name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("iterations k"); ax.set_ylabel("build time (s)")
    ax.set_title(f"Time vs iterations  ({args.fixed_side}x{args.fixed_side})")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig("plot_time_vs_iters.png", dpi=150)
    print("\nSaved: results_size.csv, results_iters.csv, plot_time_vs_size.png, plot_time_vs_iters.png")


if __name__ == "__main__":
    main()
