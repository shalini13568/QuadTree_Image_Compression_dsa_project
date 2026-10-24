"""
qt_eval.py  --  Member C: the evidence for the evaluator, as DATA.

Every function returns plain Python data (lists of dicts, numbers, strings), so the
same code feeds the web app (app.py tabs) and the terminal report (demo_report.py).
Nothing here is hard-coded: every number is measured when the function runs.

    edge_cases()      original vs our array engine on tricky images
    robustness()      inputs / files the original code was not built for
    speed_memory()    time and tree memory: original vs our versions
    scaling()         time vs image size and vs iterations (Big-O evidence)
    run_tests()       runs the pytest suite and returns its output
"""

import io
import math
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

import qt_core as qt
from quad_tree_compression import ImageCompressor, reconstruct_image_data


# ----------------------------------------------------------------------
# Small image helpers
# ----------------------------------------------------------------------
def synthetic_image(side: int, seed: int = 0) -> np.ndarray:
    """Gradient + noise + hard-edged blocks: smooth and busy areas in one picture."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:side, 0:side]
    img = np.stack([xx * 255 // side, yy * 255 // side, (xx + yy) * 255 // (2 * side)], axis=-1)
    img = img + rng.integers(0, 40, img.shape)
    for _ in range(12):
        x, y = rng.integers(0, side * 3 // 4, 2)
        s = rng.integers(side // 16 + 1, side // 4 + 2)
        img[y:y + s, x:x + s] = rng.integers(0, 255, 3)
    return np.clip(img, 0, 255).astype(np.uint8)


def _noisy(h, w, seed=1):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _gradient(h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.stack([xx * 255 // max(w - 1, 1), yy * 255 // max(h - 1, 1),
                     (xx + yy) * 255 // max(h + w - 2, 1)], -1).astype(np.uint8)


def _pil_bytes(arr, mode, fmt="PNG"):
    buf = io.BytesIO()
    Image.fromarray(arr).convert(mode).save(buf, format=fmt)
    buf.seek(0)
    return buf


def _run_original(img, k):
    c = ImageCompressor(img)
    c.add_detail(k, 10)
    return c.draw(), c.encode_to_binary()


# ----------------------------------------------------------------------
# 1. Edge cases: original vs ours
# ----------------------------------------------------------------------
TIE_NOTE = ("IDENTICAL = same pixels as the original. 'equal quality (tie order)' = two regions had the "
            "same detail up to float rounding (~1e-13); the original's np.std and our integral-image "
            "formula round differently, so a different one of the two is split first. Same number of "
            "leaves and same quality; only the choice between two equally good splits differs.")


def edge_case_inputs():
    """The tricky images used by edge_cases() and edge_case_gallery(): list of (name, array)."""
    return [
        ("1 x 1 pixel", _noisy(1, 1)),
        ("1 x 40 strip", _noisy(1, 40)),
        ("40 x 1 strip", _noisy(40, 1)),
        ("odd size 33 x 129", _noisy(33, 129)),
        ("uniform colour 64 x 64", np.full((64, 64, 3), 128, dtype=np.uint8)),
        ("pure noise 128 x 128", _noisy(128, 128)),
        ("smooth gradient 128 x 128", _gradient(128, 128)),
        ("photo-like 256 x 256", synthetic_image(256)),
    ]


def edge_cases(k: int = 500):
    """Returns (rows, all_ok). One row per tricky image."""
    cases = edge_case_inputs()
    rows, all_ok = [], True
    for name, img in cases:
        try:
            ref, _ = _run_original(img, k)
            original = "ok"
        except Exception as e:                                   # noqa: BLE001
            ref, original = None, f"fails ({type(e).__name__})"
        ours = qt.compress(img, k, "array")
        if ref is None:
            same = "-"
        elif np.array_equal(ref, ours.image):
            same = "IDENTICAL"
        else:
            d = abs(qt.psnr(img, ref) - qt.psnr(img, ours.image))
            same = f"equal quality (tie order, {d:.3f} dB)" if d < 0.05 else "DIFFERENT"
        all_ok &= same != "DIFFERENT"
        rows.append({"test case": name, "original code": original, "our code": "ok",
                     "output vs original": same, "leaves": ours.leaves,
                     "PSNR (dB)": "inf" if math.isinf(ours.psnr) else round(ours.psnr, 1)})
    return rows, all_ok


def edge_case_gallery(k: int = 500):
    """For each edge case: (name, input image, compressed image, division picture, leaves)."""
    out = []
    for name, img in edge_case_inputs():
        res = qt.compress(img, k, "array")
        out.append((name, img, res.image, qt.draw_division(res.data, "only"), res.leaves))
    return out


# ----------------------------------------------------------------------
# 2. Robustness: inputs the original was not built for
# ----------------------------------------------------------------------
def robustness():
    """Returns (input_rows, bad_file_rows)."""
    base = synthetic_image(64)
    rgba = np.dstack([base, np.full(base.shape[:2], 128, dtype=np.uint8)])
    inputs = [("grayscale image", base, "L"),
              ("PNG with transparency (RGBA)", rgba, "RGBA"),
              ("palette image (GIF-style)", base, "P")]
    input_rows = []
    for name, arr, mode in inputs:
        raw = np.array(Image.open(_pil_bytes(arr, mode)))        # what the original code would get
        try:
            _run_original(raw, 100)
            original = "ok"
        except Exception as e:                                   # noqa: BLE001
            original = f"crashes ({type(e).__name__})"
        loaded, _ = qt.load_image(_pil_bytes(arr, mode))
        res = qt.compress(loaded, 100)
        input_rows.append({"input": name, "original code": original,
                           "our code": f"ok -> {res.image.shape[1]}x{res.image.shape[0]} RGB"})

    good = qt.compress(synthetic_image(64), 200)
    png_file = _pil_bytes(base, "RGB").getvalue()
    bad = [("garbage bytes", b"hello world"), ("empty file", b""),
           ("a PNG image given as .qid", png_file), ("truncated .qid file", good.data[:-10])]
    bad_rows = []
    for name, payload in bad:
        try:
            reconstruct_image_data(payload)
            original = "no error"
        except Exception as e:                                   # noqa: BLE001
            original = f"raw {type(e).__name__} traceback"
        try:
            qt.decode_bytes(payload)
            ours = "no error"
        except qt.QTError as e:
            ours = f"clear message: {e}"
        bad_rows.append({"bad file": name, "original code": original, "our code": ours})
    return input_rows, bad_rows


# ----------------------------------------------------------------------
# 3. Speed and memory
# ----------------------------------------------------------------------
def speed_memory(img: np.ndarray, k: int = 20000, repeats: int = 2):
    """Original vs integral image + heap vs array layout: build time and tree memory."""
    from benchmark_representation import measure, VARIANTS
    labels = {"original": qt.ENGINES["original"][0],            # same ladder names as the compare modes
              "pointer(A)": qt.ENGINES["pointer-heap"][0],
              "array(B)": qt.ENGINES["array"][0]}
    rows, base = [], None
    for name in VARIANTS:
        r = measure(name, img, k, repeats)
        base = base or r["time_s"]
        rows.append({"version": labels.get(name, name), "time (s)": round(r["time_s"], 3),
                     "speedup": round(base / r["time_s"], 2),
                     "tree memory (MB)": round(r["tree_MB"], 1),
                     "nodes": r["nodes"], "bytes/node": round(r["bytes_per_node"])})
    return rows


# ----------------------------------------------------------------------
# 4. Scaling (Big-O evidence)
# ----------------------------------------------------------------------
def _time_build(key, img, k, repeats):
    best = float("inf")
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        comp = qt.ENGINES[key][1](img)
        comp.add_detail(k, 10)
        best = min(best, time.perf_counter() - t0)
    return best


def scaling(sides=(128, 256, 512, 1024), fixed_k=3000, ks=(100, 300, 1000, 3000, 10000),
            fixed_side=512, repeats=1, progress=None):
    """
    Returns (rows_vs_size, rows_vs_iterations); every row has one time per engine key.
    progress: optional callable(fraction) for a progress bar.
    """
    keys = list(qt.ENGINES)
    total = (len(sides) + len(ks)) * len(keys)
    done = 0
    rows_size, rows_iter = [], []
    for side in sides:
        img = synthetic_image(side)
        row = {"side": side, "pixels": side * side}
        for key in keys:
            row[key] = _time_build(key, img, fixed_k, repeats)
            done += 1
            if progress:
                progress(done / total)
        rows_size.append(row)
    img = synthetic_image(fixed_side)
    for k in ks:
        row = {"k": k}
        for key in keys:
            row[key] = _time_build(key, img, k, repeats)
            done += 1
            if progress:
                progress(done / total)
        rows_iter.append(row)
    return rows_size, rows_iter


# ----------------------------------------------------------------------
# 5. The pytest suite
# ----------------------------------------------------------------------
def run_tests(timeout: int = 900):
    """Run pytest in the project folder. Returns (status, output_text);
    status is True (all passed), False (failures) or None (pytest unavailable)."""
    folder = os.path.dirname(os.path.abspath(__file__))
    try:
        out = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"],
                             capture_output=True, text=True, timeout=timeout, cwd=folder)
    except subprocess.TimeoutExpired:
        return False, "pytest timed out"
    text = (out.stdout + "\n" + out.stderr).strip()
    if "No module named pytest" in text:
        return None, "pytest is not installed.  Run:  pip install pytest"
    return out.returncode == 0, text
