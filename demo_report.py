"""
demo_report.py  --  one command that shows the evaluator everything.

    python demo_report.py                         # synthetic test image
    python demo_report.py --image input/mountain.jpg
    python demo_report.py --quick                 # smaller and faster (for rehearsal)
    python demo_report.py --skip-tests            # if pytest is not installed

Sections
    1. Test cases: original code vs our code on edge cases (same output?)
    2. Inputs / files the original code cannot handle, and how we handle them
    3. Speed and memory: original vs our versions (measured live)
    4. Quality vs size: iterations, pruning and target modes
    5. The automated test suite (pytest)
    6. A picture for the screen: output/demo_montage.png

Everything printed here is measured on THIS machine, nothing is hard-coded.
"""

import argparse
import os
import subprocess
import sys

import numpy as np
from PIL import Image

import qt_core as qt
import qt_eval as ev
from qt_eval import synthetic_image


# ----------------------------------------------------------------------
def title(text):
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def table(rows, cols):
    widths = [max(len(c), *(len(str(r[c])) for r in rows)) for c in cols]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(str(r[c]).ljust(w) for c, w in zip(cols, widths)))


def section_edge_cases():
    title("1. TEST CASES - original code vs our array engine (same settings)")
    rows, all_ok = ev.edge_cases()
    table(rows, ["test case", "original code", "our code", "output vs original", "leaves", "PSNR (dB)"])
    print("\n=> " + ("every case matches the original" if all_ok else "SOME CASES DIFFER - investigate before the evaluation!"))
    print("   " + ev.TIE_NOTE)


def section_robustness():
    title("2. INPUTS THE ORIGINAL CODE WAS NOT BUILT FOR - what happens?")
    input_rows, bad_rows = ev.robustness()
    table(input_rows, ["input", "original code", "our code"])
    print()
    table(bad_rows, ["bad file", "original code", "our code"])


def section_speed(img, k, repeats):
    title(f"3. SPEED AND MEMORY - image {img.shape[1]}x{img.shape[0]}, {k} splits")
    rows = ev.speed_memory(img, k, repeats)
    table(rows, ["version", "time (s)", "speedup", "tree memory (MB)", "bytes/node"])
    print("\n(time = build only: integral image + splitting; same output in every version)")


def section_quality(img):
    title("4. QUALITY vs SIZE")
    print("a) more iterations -> better quality, bigger file\n")
    rows = []
    for k in (100, 1000, 5000, 20000):
        r = qt.compress(img, k, "array")
        rows.append({"iterations": k, "leaves": r.leaves, "size (KB)": f"{r.compressed_bytes / 1024:.1f}",
                     "ratio vs raw": f"{r.ratio:.0f}x", "PSNR (dB)": f"{r.psnr:.2f}", "MAE": f"{r.mae:.2f}"})
    table(rows, ["iterations", "leaves", "size (KB)", "ratio vs raw", "PSNR (dB)", "MAE"])

    print("\nb) pruning near-identical siblings at 20000 iterations (our addition)\n")
    rows = []
    for tol in (None, 4, 8, 16):
        r = qt.compress(img, 20000, "array", prune_tol=tol)
        rows.append({"prune tolerance": "off" if tol is None else tol, "leaves": r.leaves,
                     "size (KB)": f"{r.compressed_bytes / 1024:.1f}", "PSNR (dB)": f"{r.psnr:.2f}"})
    table(rows, ["prune tolerance", "leaves", "size (KB)", "PSNR (dB)"])

    print("\nc) ask for a goal instead of an iteration count (our addition)\n")
    rows = []
    p = qt.compress_to_psnr(img, 30.0)
    rows.append({"goal": "reach 30 dB", "splits used": p.splits, "size (KB)": f"{p.compressed_bytes / 1024:.1f}",
                 "PSNR (dB)": f"{p.psnr:.2f}", "met": p.reached})
    kb = max(1, p.compressed_bytes // 1024 // 2) * 1024
    s = qt.compress_to_size(img, kb)
    rows.append({"goal": f"fit in {kb // 1024} KB", "splits used": s.splits, "size (KB)": f"{s.compressed_bytes / 1024:.1f}",
                 "PSNR (dB)": f"{s.psnr:.2f}", "met": s.reached})
    table(rows, ["goal", "splits used", "size (KB)", "PSNR (dB)", "met"])


def section_tests():
    title("5. AUTOMATED TEST SUITE")
    status, text = ev.run_tests()
    lines = [l for l in text.splitlines() if l.strip()]
    print("\n".join(lines[-6:]))


def section_montage(img, path):
    title("6. PICTURE FOR THE SCREEN")
    panels = [("Original", img)]
    for k in (200, 2000):
        panels.append((f"{k} splits", qt.compress(img, k).image))
    r = qt.compress(img, 2000)
    panels.append(("2000 splits + cell borders", qt.draw_overlay(r.data, r.image)))
    height = 360
    tiles = []
    for _, a in panels:
        im = Image.fromarray(a)
        w = max(1, int(im.width * height / im.height))
        tiles.append(im.resize((w, height), Image.NEAREST))
    sheet = Image.new("RGB", (sum(t.width for t in tiles) + 8 * (len(tiles) - 1), height), (255, 255, 255))
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0))
        x += t.width + 8
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sheet.save(path)
    print(f"saved {path}  (left to right: " + ", ".join(n for n, _ in panels) + ")")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="photo to use (default: synthetic test image)")
    ap.add_argument("--max-pixels", type=int, default=2_000_000)
    ap.add_argument("--iterations", type=int, default=20000)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--quick", action="store_true", help="smaller image and fewer iterations")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--montage", default="output/demo_montage.png")
    a = ap.parse_args()

    if a.image:
        img, notes = qt.load_image(a.image, max_pixels=250_000 if a.quick else a.max_pixels)
        for n in notes:
            print("note:", n)
    else:
        img = synthetic_image(512 if a.quick else 1024)
    k = 5000 if a.quick else a.iterations

    section_edge_cases()
    section_robustness()
    section_speed(img, k, 1 if a.quick else a.repeats)
    section_quality(img)
    if not a.skip_tests:
        section_tests()
    section_montage(img, a.montage)
    print("\nDone.")


if __name__ == "__main__":
    main()
