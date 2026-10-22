"""
benchmark_representation.py  --  Member B

Compares three tree representations on the same image:
    original      : QuadTreeNode objects + per-node image views  (original repo)
    pointer(A)    : FastNode objects with __slots__               (Member A)
    array(B)      : struct-of-arrays ArrayImageCompressor         (Member B)

Experiment 1: build time and TREE memory vs iterations k
Experiment 2: pruning trade-off (tolerance vs leaves, file size, error)

Tree memory = tracemalloc "current" after add_detail minus before add_detail.
The integral image (A's and B's) is built in the constructor, i.e. BEFORE the
baseline is taken, so it is excluded: this isolates the tree structure itself.

Usage:
    python benchmark_representation.py
    python benchmark_representation.py --image input/mountain.jpg --side 1024
Outputs: results_repr.csv, results_prune.csv, plot_repr.png, plot_prune.png
"""

import argparse
import csv
import gc
import time
import tracemalloc

import numpy as np

from quad_tree_compression import ImageCompressor, reconstruct_image_data
from fast_quad_tree import FastImageCompressor
from array_quad_tree import ArrayImageCompressor, ArrayQuadTree
from benchmark_complexity import synthetic_image, load_image      # Member A's helpers

THRESH = 10

VARIANTS = {
    "original":   lambda img: ImageCompressor(img),
    "pointer(A)": lambda img: FastImageCompressor(img, queue="heap"),
    "array(B)":   lambda img: ArrayImageCompressor(img),
}


def node_count(comp):
    if isinstance(comp, ArrayQuadTree):
        return comp.n
    if isinstance(comp, FastImageCompressor):
        return 4 * comp.splits + 1
    return _count_original(comp.root_node)


def _count_original(node):
    stack, c = [node], 0
    while stack:
        n = stack.pop(); c += 1
        if n.is_subdivided:
            stack.extend((n.bottom_left_node, n.bottom_right_node, n.top_left_node, n.top_right_node))
    return c


def measure(name, img, k, repeats):
    # time (min of repeats, constructor + add_detail, same convention as Member A)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        c = VARIANTS[name](img); c.add_detail(k, THRESH)
        best = min(best, time.perf_counter() - t0)
    del c

    # memory of the tree structure only
    gc.collect()
    tracemalloc.start()
    c = VARIANTS[name](img)
    before, _ = tracemalloc.get_traced_memory()
    c.add_detail(k, THRESH)
    after, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    nodes = node_count(c)
    mem = after - before
    return {"time_s": best, "tree_MB": mem / 1e6, "nodes": nodes, "bytes_per_node": mem / max(nodes, 1)}


def mae(a, b):
    return float(np.abs(a.astype(np.int32) - b.astype(np.int32)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--side", type=int, default=1024)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--prune-k", type=int, default=20000)
    args = ap.parse_args()

    img = load_image(args.image, args.side) if args.image else synthetic_image(args.side)
    print(f"image: {img.shape[1]}x{img.shape[0]}")

    # ---- Experiment 1 -------------------------------------------------
    ks = [100, 1000, 5000, 20000, 50000]
    rows = []
    print("\nExperiment 1: representation comparison")
    print(f"{'k':>7} {'variant':<11} {'time(s)':>8} {'tree MB':>8} {'nodes':>9} {'B/node':>8}")
    for k in ks:
        for name in VARIANTS:
            r = measure(name, img, k, args.repeats)
            r.update(k=k, variant=name)
            rows.append(r)
            print(f"{k:>7} {name:<11} {r['time_s']:>8.3f} {r['tree_MB']:>8.2f} {r['nodes']:>9} {r['bytes_per_node']:>8.1f}")
    write_csv("results_repr.csv", rows)

    # ---- Correctness / round trip ------------------------------------
    comp = ArrayImageCompressor(img); comp.add_detail(5000, THRESH)
    data = comp.encode_to_binary()
    ok_orig = np.array_equal(reconstruct_image_data(data), comp.draw())
    ok_arr = np.array_equal(ArrayQuadTree.from_encoded(data).draw(), comp.draw())
    print(f"\nround trip: original decoder {'OK' if ok_orig else 'FAIL'}, array decoder {'OK' if ok_arr else 'FAIL'}")

    # ---- Experiment 2: pruning ---------------------------------------
    print(f"\nExperiment 2: pruning at k = {args.prune_k}")
    print(f"{'tol':>4} {'leaves':>8} {'file KB':>8} {'MAE vs original image':>22} {'prune time(s)':>14}")
    prows = []
    for tol in (None, 0, 2, 4, 8, 16, 32):
        c = ArrayImageCompressor(img); c.add_detail(args.prune_k, THRESH)
        t = 0.0
        if tol is not None:
            t0 = time.perf_counter(); c.prune(tol); t = time.perf_counter() - t0
        row = {"tolerance": "none" if tol is None else tol, "leaves": c.leaf_count(),
               "file_KB": len(c.encode_to_binary()) / 1024, "MAE": mae(img, c.draw()), "prune_s": t}
        prows.append(row)
        print(f"{str(row['tolerance']):>4} {row['leaves']:>8} {row['file_KB']:>8.1f} {row['MAE']:>22.3f} {row['prune_s']:>14.3f}")
    write_csv("results_prune.csv", prows)

    plot(rows, prows)


def write_csv(name, rows):
    with open(name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def plot(rows, prows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, st in (("original", "o-"), ("pointer(A)", "s--"), ("array(B)", "^-")):
        pts = [r for r in rows if r["variant"] == name]
        axes[0].plot([r["k"] for r in pts], [r["time_s"] for r in pts], st, label=name)
        axes[1].plot([r["k"] for r in pts], [r["tree_MB"] for r in pts], st, label=name)
    for ax, yl, t in ((axes[0], "build time (s)", "Build time"), (axes[1], "tree memory (MB)", "Tree memory")):
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("iterations k")
        ax.set_ylabel(yl); ax.set_title(t); ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig("plot_repr.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot([r["file_KB"] for r in prows], [r["MAE"] for r in prows], "o-")
    for r in prows:
        ax.annotate(str(r["tolerance"]), (r["file_KB"], r["MAE"]), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("encoded file size (KB)"); ax.set_ylabel("MAE vs original image")
    ax.set_title("Pruning trade-off (labels = tolerance)"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig("plot_prune.png", dpi=150)
    print("\nSaved: results_repr.csv, results_prune.csv, plot_repr.png, plot_prune.png")


if __name__ == "__main__":
    main()
