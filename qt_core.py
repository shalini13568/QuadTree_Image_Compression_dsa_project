"""
qt_core.py  --  Member C: robustness, metrics and the shared API used by the
CLI (qt_cli.py) and the web demo (app.py).

Builds on
    quad_tree_compression.py   original repo (Inspiaaa, MIT licence)
    fast_quad_tree.py          Member A: integral image + heap / sorted queue
    array_quad_tree.py         Member B: array-based tree, pruning, decoding

What this module adds
---------------------
* load_image(): turns almost any image file into a clean uint8 RGB array
  (RGBA, grayscale, palette, 1-bit, 16-bit, CMYK, EXIF rotation, animated GIF,
  huge images) and reports what it had to change.
* mae / psnr metrics.
* compress(): one call for all four engines (original, pointer+sorted,
  pointer+heap, array) with the same result object.
* compress_to_psnr() / compress_to_size(): "give me the smallest tree that
  reaches X dB" / "the best quality that fits in N bytes", found by BINARY SEARCH
  over a single build (see _state_at for the trick).
* decode_bytes(): reads a .qid file with validation, so corrupt input gives a
  clear error instead of a crash or a hang.
* draw_overlay(): shows the quadtree boundaries on the image.
"""

import io
import lzma
import math
import struct
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from array_quad_tree import ArrayImageCompressor, ArrayQuadTree
from fast_quad_tree import FastImageCompressor

# The integral image needs ~125 bytes per pixel at peak, so cap the input size.
DEFAULT_MAX_PIXELS = 8_000_000
MAX_DECODE_PIXELS = 100_000_000          # sanity limit when decoding a .qid file


class QTError(ValueError):
    """Any problem caused by the user's input (bad file, bad option...)."""


# ----------------------------------------------------------------------
# Engines: all expose add_detail(k, threshold), draw(), encode_to_binary()
# ----------------------------------------------------------------------
def _original(img):
    from quad_tree_compression import ImageCompressor
    return ImageCompressor(img)


# Each version changes ONE thing compared with the one below it (an ablation ladder):
#   original        -> pixel scan for mean/std, sorted list, node objects
#   pointer-sorted  -> + integral image (O(1) mean/std)           [Member A]
#   pointer-heap    -> + heap instead of sorted list              [Member A]
#   array           -> + array layout instead of node objects     [Member B]
ENGINES = {
    "array":          ("4. + array layout (integral image + heap)",   lambda img: ArrayImageCompressor(img)),
    "pointer-heap":   ("3. + heap (integral image + node objects)",   lambda img: FastImageCompressor(img, queue="heap")),
    "pointer-sorted": ("2. + integral image (sorted list, node objects)", lambda img: FastImageCompressor(img, queue="sorted")),
    "original":       ("1. Original repo (pixel scan, sorted list)",  _original),
}


# ----------------------------------------------------------------------
# Loading and validating images
# ----------------------------------------------------------------------
def load_image(source, max_pixels: int = DEFAULT_MAX_PIXELS):
    """
    Open `source` (path or file-like object) and return (array, notes):
    array = uint8 array of shape (H, W, 3); notes = list of human-readable
    messages describing anything that was changed.
    """
    notes = []
    try:
        im = Image.open(source)
        im.load()
    except FileNotFoundError:
        raise QTError(f"file not found: {source}") from None
    except Image.DecompressionBombError:
        raise QTError("image is far too large to process safely") from None
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise QTError("this file is not a readable image") from None

    if getattr(im, "n_frames", 1) > 1:
        im.seek(0)
        notes.append("animated image: using the first frame")

    try:
        im = ImageOps.exif_transpose(im)                 # honour phone-camera rotation
    except Exception:
        pass

    mode = im.mode
    if mode in ("I", "I;16", "I;16L", "I;16B", "F"):     # 16-bit / float grayscale
        a = np.asarray(im).astype(np.float64)
        hi = float(a.max()) if a.size else 0.0
        if hi > 255:
            a = a * (255.0 / hi)
            notes.append(f"{mode} image rescaled to 8 bits")
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    elif mode in ("RGBA", "LA", "PA") or (mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        im = Image.alpha_composite(background, rgba)
        notes.append("transparency flattened onto a white background")

    if im.mode != "RGB":
        if im.mode not in ("L", "1"):
            notes.append(f"converted from {im.mode} to RGB")
        im = im.convert("RGB")

    w, h = im.size
    if w * h > max_pixels:
        scale = math.sqrt(max_pixels / (w * h))
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        im = im.resize(new_size, Image.LANCZOS)
        notes.append(f"image downscaled from {w}x{h} to {new_size[0]}x{new_size[1]} "
                     f"(limit {max_pixels:,} pixels)")
    return np.array(im, dtype=np.uint8), notes


def _check_array(img):
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise QTError("image must be a uint8 array of shape (height, width, 3); use load_image()")
    if img.shape[0] < 1 or img.shape[1] < 1:
        raise QTError("image is empty")


def _check_options(iterations, threshold, prune_tol):
    if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)) or iterations < 0:
        raise QTError("iterations must be a whole number >= 0")
    if threshold < 0:
        raise QTError("threshold must be >= 0")
    if prune_tol is not None and (prune_tol < 0 or prune_tol > 255):
        raise QTError("prune tolerance must be between 0 and 255")


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def mae(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute error over all pixels and channels (0-255 scale)."""
    return float(np.abs(a.astype(np.int32) - b.astype(np.int32)).mean())


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB. Higher = closer. inf if identical."""
    diff = a.astype(np.float64) - b.astype(np.float64)
    mse = float(np.mean(diff * diff))
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10(255.0 ** 2 / mse)


# ----------------------------------------------------------------------
# Result object
# ----------------------------------------------------------------------
@dataclass
class Result:
    image: np.ndarray                 # the compressed (reconstructed) picture
    data: bytes                       # the encoded .qid bytes
    engine: str
    splits: int
    leaves: int
    seconds: float
    mae: float
    psnr: float
    raw_bytes: int                    # H * W * 3
    png_bytes: Optional[int] = None   # PNG size of the input, if measured
    reached: Optional[bool] = None    # target modes: was the target met?
    note: str = ""

    @property
    def compressed_bytes(self) -> int:
        return len(self.data)

    @property
    def ratio(self) -> float:
        return self.raw_bytes / max(self.compressed_bytes, 1)

    def summary(self) -> dict:
        d = {
            "engine": self.engine,
            "leaves": self.leaves,
            "compressed size (KB)": round(self.compressed_bytes / 1024, 2),
            "raw size (KB)": round(self.raw_bytes / 1024, 1),
            "ratio vs raw": round(self.ratio, 1),
            "MAE": round(self.mae, 3),
            "PSNR (dB)": "inf (identical)" if math.isinf(self.psnr) else round(self.psnr, 2),
            "time (s)": round(self.seconds, 3),
        }
        if self.png_bytes:
            d["PNG size (KB)"] = round(self.png_bytes / 1024, 1)
        return d


def _png_size(img: np.ndarray) -> int:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.tell()


def _make_result(img, comp, engine, seconds, png_baseline, **extra) -> Result:
    out = comp.draw()
    data = comp.encode_to_binary()
    if hasattr(comp, "leaf_count"):
        leaves = comp.leaf_count()
    else:
        leaves = len(comp.extract_data()[1])
    return Result(image=out, data=data, engine=engine, splits=(leaves - 1) // 3, leaves=leaves,
                  seconds=seconds, mae=mae(img, out), psnr=psnr(img, out),
                  raw_bytes=img.size, png_bytes=_png_size(img) if png_baseline else None, **extra)


# ----------------------------------------------------------------------
# Mode 1: fixed number of iterations
# ----------------------------------------------------------------------
def compress(img: np.ndarray, iterations: int = 20000, engine: str = "array",
             threshold: float = 10, prune_tol: Optional[int] = None,
             png_baseline: bool = False) -> Result:
    _check_array(img)
    _check_options(iterations, threshold, prune_tol)
    if engine not in ENGINES:
        raise QTError(f"unknown engine '{engine}' (choose from {', '.join(ENGINES)})")
    if prune_tol is not None and engine != "array":
        raise QTError("pruning is only available with the 'array' engine")

    t0 = time.perf_counter()
    comp = ENGINES[engine][1](img)
    comp.add_detail(int(iterations), threshold)
    if prune_tol is not None:
        comp.prune(prune_tol)
    seconds = time.perf_counter() - t0
    return _make_result(img, comp, engine, seconds, png_baseline)


# ----------------------------------------------------------------------
# Comparing engines on the same image and settings
# ----------------------------------------------------------------------
def compare_engines(img: np.ndarray, iterations: int = 20000, threshold: float = 10,
                    engines=None, repeats: int = 2):
    """
    Run several engines on the SAME image with the SAME settings.

    Returns (results, rows): results maps engine key -> Result (best-of-`repeats`
    by time); rows is a list of dicts ready to print or show as a table.
    The engines implement the same algorithm, so quality columns should match;
    what differs is speed. "max pixel diff" is measured against the original engine.
    """
    keys = list(engines or ENGINES)
    for k in keys:
        if k not in ENGINES:
            raise QTError(f"unknown engine '{k}'")
    results = {}
    for k in keys:
        best = None
        for _ in range(max(1, repeats)):
            r = compress(img, iterations, k, threshold)
            if best is None or r.seconds < best.seconds:
                best = r
        results[k] = best

    base = results.get("original")
    rows = []
    for k, r in results.items():
        rows.append({
            "engine": ENGINES[k][0],
            "key": k,
            "time (s)": round(r.seconds, 3),
            "speedup vs original": round(base.seconds / r.seconds, 2) if base else None,
            "leaves": r.leaves,
            "size (KB)": round(r.compressed_bytes / 1024, 2),
            "PSNR (dB)": "inf" if math.isinf(r.psnr) else round(r.psnr, 2),
            "max pixel diff vs original": int(np.abs(r.image.astype(int) - base.image.astype(int)).max()) if base else None,
        })
    return results, rows


# ----------------------------------------------------------------------
# Every variation on one image
# ----------------------------------------------------------------------
def compare_all(img: np.ndarray, iterations: int = 20000, threshold: float = 10,
                target_db: float = 30.0, max_kb: float = 50.0,
                prune_tols=(4, 8, 16), repeats: int = 1):
    """
    Run EVERY variation on the same image and return (variations, rows).

    variations: list of (group, label, Result)
        group "engine"  - the four implementations, same settings (speed comparison)
        group "pruning" - array engine + prune at each tolerance (size/quality trade-off)
        group "goal"    - array engine asked for a PSNR / a file size (search modes)
    rows: list of dicts ready for a table.
    """
    _check_array(img)
    _check_options(int(iterations), threshold, None)
    variations = []

    for key in ENGINES:                                   # 1. engines
        best = None
        for _ in range(max(1, repeats)):
            r = compress(img, iterations, key, threshold)
            if best is None or r.seconds < best.seconds:
                best = r
        variations.append(("engine", ENGINES[key][0], best))

    for tol in prune_tols:                                # 2. pruning levels
        variations.append(("pruning", f"Array + prune (tolerance {tol})",
                           compress(img, iterations, "array", threshold, tol)))

    variations.append(("goal", f"Array, target {target_db:g} dB",    # 3. goal modes
                       compress_to_psnr(img, target_db, iterations, threshold)))
    variations.append(("goal", f"Array, fit in {max_kb:g} KB",
                       compress_to_size(img, int(max_kb * 1024), iterations, threshold)))

    original = next(r for g, l, r in variations if g == "engine" and l == ENGINES["original"][0])
    rows = []
    for group, label, r in variations:
        row = {
            "group": group,
            "variation": label,
            "time (s)": round(r.seconds, 3),
            "leaves": r.leaves,
            "size (KB)": round(r.compressed_bytes / 1024, 2),
            "ratio vs raw": round(r.ratio, 1),
            "PSNR (dB)": "inf" if math.isinf(r.psnr) else round(r.psnr, 2),
            "MAE": round(r.mae, 3),
        }
        if group == "engine":
            row["speedup vs original"] = round(original.seconds / r.seconds, 2)
            row["same picture as original"] = bool(np.array_equal(r.image, original.image))
        else:
            row["speedup vs original"] = None
            row["same picture as original"] = None
        if group == "goal" and r.reached is False:
            row["variation"] += " (target not reached)"
        rows.append(row)
    return variations, rows


# ----------------------------------------------------------------------
# Modes 2 and 3: target quality / target size, by binary search
# ----------------------------------------------------------------------
@contextmanager
def _state_at(tree: ArrayQuadTree, splits: int, prune_tol: Optional[int] = None):
    """
    Temporarily turn a finished ArrayImageCompressor into the tree it was after
    only its first `splits` splits (optionally pruned), then restore it.

    WHY THIS WORKS (the benefit of Member B's array layout):
    split number j (1-based) allocates nodes 4j-3 .. 4j, so a node that was split
    in the first `splits` splits has first_child <= 4*splits - 3. Cutting every
    larger first_child back to -1 gives exactly the tree that `splits` iterations
    would have produced. One build, many cheap "what if" evaluations.
    """
    n = tree.n
    saved = (tree.first_child[:n].copy(), tree.splits, tree._heap)
    try:
        fc = tree.first_child[:n]
        fc[fc > 4 * splits - 3] = -1
        tree.splits = splits
        if prune_tol is not None:
            tree.prune(prune_tol)
        yield tree
    finally:
        tree.first_child[:n] = saved[0]
        tree.splits, tree._heap = saved[1], saved[2]


def _build_full(img, max_splits, threshold):
    comp = ArrayImageCompressor(img)
    comp.add_detail(max_splits, threshold)
    return comp


def _finish(img, comp, s, prune_tol, seconds, png_baseline, reached, note, label):
    with _state_at(comp, s, prune_tol) as t:
        return _make_result(img, t, label, seconds, png_baseline, reached=reached, note=note)


def compress_to_psnr(img: np.ndarray, target_db: float, max_iterations: int = 60000,
                     threshold: float = 10, prune_tol: Optional[int] = None,
                     png_baseline: bool = False) -> Result:
    """Smallest tree (fewest splits) whose PSNR is >= target_db."""
    _check_array(img)
    _check_options(0, threshold, prune_tol)
    if target_db <= 0:
        raise QTError("target PSNR must be greater than 0 dB")

    t0 = time.perf_counter()
    comp = _build_full(img, max_iterations, threshold)
    top = comp.splits

    def psnr_at(s):
        with _state_at(comp, s, prune_tol) as t:
            return psnr(img, t.draw())

    if psnr_at(top) < target_db:
        best = psnr_at(top)
        return _finish(img, comp, top, prune_tol, time.perf_counter() - t0, png_baseline, False,
                       f"target {target_db} dB not reachable with these limits; best is {best:.2f} dB",
                       "array (target PSNR)")
    lo, hi = 0, top
    while lo < hi:                                   # smallest s with psnr_at(s) >= target
        mid = (lo + hi) // 2
        if psnr_at(mid) >= target_db:
            hi = mid
        else:
            lo = mid + 1
    return _finish(img, comp, lo, prune_tol, time.perf_counter() - t0, png_baseline, True,
                   f"reached {target_db} dB with {lo} splits", "array (target PSNR)")


def compress_to_size(img: np.ndarray, max_bytes: int, max_iterations: int = 60000,
                     threshold: float = 10, prune_tol: Optional[int] = None,
                     png_baseline: bool = False) -> Result:
    """Best quality (most splits) whose encoded file is <= max_bytes."""
    _check_array(img)
    _check_options(0, threshold, prune_tol)
    if max_bytes < 1:
        raise QTError("size limit must be at least 1 byte")

    t0 = time.perf_counter()
    comp = _build_full(img, max_iterations, threshold)
    top = comp.splits

    def size_at(s):
        with _state_at(comp, s, prune_tol) as t:
            return len(t.encode_to_binary())

    if size_at(0) > max_bytes:
        return _finish(img, comp, 0, prune_tol, time.perf_counter() - t0, png_baseline, False,
                       f"even the smallest possible file is {size_at(0)} bytes (> {max_bytes})",
                       "array (target size)")
    lo, hi = 0, top
    while lo < hi:                                   # largest s with size_at(s) <= max_bytes
        mid = (lo + hi + 1) // 2
        if size_at(mid) <= max_bytes:
            lo = mid
        else:
            hi = mid - 1
    note = (f"fits in {max_bytes} bytes with {lo} splits" if lo < top
            else f"all {top} available splits fit in {max_bytes} bytes (raise --iterations for more detail)")
    return _finish(img, comp, lo, prune_tol, time.perf_counter() - t0, png_baseline, True, note,
                   "array (target size)")


# ----------------------------------------------------------------------
# Decoding with validation
# ----------------------------------------------------------------------
def decode_bytes(data: bytes) -> np.ndarray:
    """Decode a .qid file into an RGB array. Raises QTError for invalid files."""
    from quad_tree_compression import decode_image_data
    try:
        raw = lzma.decompress(data)
    except (lzma.LZMAError, EOFError, ValueError):
        raise QTError("not a valid .qid file (decompression failed)") from None
    if len(raw) < 12:
        raise QTError("not a valid .qid file (header too short)")

    width, height, flag_count = struct.unpack("<III", raw[:12])
    if width < 1 or height < 1 or width * height > MAX_DECODE_PIXELS:
        raise QTError("not a valid .qid file (impossible image size)")
    if flag_count < 1 or 12 + math.ceil(flag_count / 8) > len(raw):
        raise QTError("not a valid .qid file (corrupt tree section)")

    try:
        _, _, flags, colors = decode_image_data(data)
    except Exception:
        raise QTError("not a valid .qid file (cannot decode)") from None

    pending = 1                                      # nodes still expected in preorder
    for f in flags:
        if pending == 0:
            raise QTError("not a valid .qid file (inconsistent tree)")
        pending += 3 if f else -1
    if pending != 0:
        raise QTError("not a valid .qid file (incomplete tree)")
    if len(raw) != 12 + math.ceil(flag_count / 8) + 3 * flags.count(False):
        raise QTError("not a valid .qid file (colour data has the wrong length)")

    return ArrayQuadTree.from_encoded(data).draw()


# ----------------------------------------------------------------------
# Showing how the image is divided
# ----------------------------------------------------------------------
def _border_mask(tree: ArrayQuadTree) -> np.ndarray:
    """Boolean (H, W) mask: True on the 1-px border of every leaf cell (and the image frame)."""
    mask = np.zeros((tree.height, tree.width), dtype=bool)
    rect = tree.rect[:tree.n].tolist()
    for i in tree._leaf_indices():
        x, y, w, h = rect[i]
        mask[y, x:x + w] = True          # top edge of the cell
        mask[y:y + h, x] = True          # left edge of the cell
    mask[-1, :] = True                   # close the frame on the bottom ...
    mask[:, -1] = True                   # ... and right
    return mask


DIVISION_MODES = ("only", "on_picture")


def tree_division(tree: ArrayQuadTree, mode: str = "only") -> np.ndarray:
    """
    Picture of the cell division of `tree`.
      mode "only"       : dark cell borders on a white background (the pure partition)
      mode "on_picture" : red cell borders drawn on the compressed picture
    """
    if mode not in DIVISION_MODES:
        raise QTError(f"unknown division view '{mode}'")
    mask = _border_mask(tree)
    if mode == "only":
        out = np.full((tree.height, tree.width, tree.channels), 255, dtype=np.uint8)
        out[mask] = (40, 40, 40)
    else:
        out = tree.draw()
        out[mask] = (255, 0, 0)
    return out


def draw_division(data: bytes, mode: str = "only") -> np.ndarray:
    """Division picture for an encoded .qid payload (works for every engine's output)."""
    return tree_division(ArrayQuadTree.from_encoded(data), mode)


def division_at(comp: ArrayImageCompressor, splits: int, mode: str = "only"):
    """
    How the image looked after only its first `splits` splits, from ONE finished build.
    Returns (compressed_picture, division_picture, leaves).
    """
    splits = max(0, min(int(splits), comp.splits))
    with _state_at(comp, splits) as t:
        return t.draw(), tree_division(t, mode), t.leaf_count()


def build_division_tree(img: np.ndarray, max_splits: int = 5000, threshold: float = 10) -> ArrayImageCompressor:
    """Build once; then call division_at() for any number of splits up to comp.splits."""
    _check_array(img)
    _check_options(int(max_splits), threshold, None)
    return _build_full(img, int(max_splits), threshold)


def draw_overlay(data: bytes, base: np.ndarray, color=(255, 0, 0)) -> np.ndarray:
    """Draw the quadtree cell borders (1 px) onto `base`. `data` is a .qid payload."""
    tree = ArrayQuadTree.from_encoded(data)
    if base.shape[:2] != (tree.height, tree.width):
        raise QTError("overlay base image has a different size than the compressed image")
    out = base.copy()
    out[_border_mask(tree)] = color
    return out


def save_image(array: np.ndarray, path: str):
    try:
        Image.fromarray(array).save(path)
    except (OSError, ValueError, KeyError) as e:
        raise QTError(f"could not save image to {path}: {e}") from None
