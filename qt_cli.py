"""
qt_cli.py  --  Member C: command-line interface

Examples
--------
  python qt_cli.py compress input/mountain.jpg -o out/mountain.qid --iterations 20000 --preview out/mountain.png
  python qt_cli.py compress photo.png --psnr 30                 # smallest tree reaching 30 dB
  python qt_cli.py compress photo.png --max-kb 50               # best quality within 50 KB
  python qt_cli.py compress photo.png --iterations 5000 --prune 8 --overlay out/cells.png
  python qt_cli.py compress photo.png --engine original        # run the original repo's code
  python qt_cli.py compare photo.png -n 20000                  # original vs our engines, side by side
  python qt_cli.py decompress out/mountain.qid -o out/restored.png
"""

import argparse
import math
import os
import sys

import qt_core as qt


def _ensure_parent(path):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


def cmd_compress(a) -> int:
    img, notes = qt.load_image(a.input, max_pixels=a.max_pixels)
    for n in notes:
        print(f"note: {n}")

    chosen = [m for m in (a.psnr, a.max_kb) if m is not None]
    if len(chosen) > 1:
        raise qt.QTError("use only one of --psnr and --max-kb")

    if a.psnr is not None:
        if a.engine != "array":
            raise qt.QTError("--psnr needs the 'array' engine")
        res = qt.compress_to_psnr(img, a.psnr, a.iterations, a.threshold, a.prune, a.png_baseline)
    elif a.max_kb is not None:
        if a.engine != "array":
            raise qt.QTError("--max-kb needs the 'array' engine")
        res = qt.compress_to_size(img, int(a.max_kb * 1024), a.iterations, a.threshold, a.prune, a.png_baseline)
    else:
        res = qt.compress(img, a.iterations, a.engine, a.threshold, a.prune, a.png_baseline)

    out = a.output or os.path.splitext(a.input)[0] + ".qid"
    _ensure_parent(out)
    with open(out, "wb") as f:
        f.write(res.data)
    if a.preview:
        _ensure_parent(a.preview)
        qt.save_image(res.image, a.preview)
    if a.overlay:
        _ensure_parent(a.overlay)
        qt.save_image(qt.draw_overlay(res.data, res.image), a.overlay)

    print(f"saved {out}")
    for k, v in res.summary().items():
        print(f"  {k:<22} {v}")
    if res.note:
        print(f"  {'result':<22} {res.note}")
    return 0 if res.reached in (None, True) else 3        # 3 = target not reachable


def cmd_decompress(a) -> int:
    try:
        with open(a.input, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        raise qt.QTError(f"file not found: {a.input}") from None
    img = qt.decode_bytes(data)
    out = a.output or os.path.splitext(a.input)[0] + "_restored.png"
    _ensure_parent(out)
    qt.save_image(img, out)
    print(f"saved {out}  ({img.shape[1]}x{img.shape[0]})")
    return 0


def cmd_compare(a) -> int:
    img, notes = qt.load_image(a.input, max_pixels=a.max_pixels)
    for n in notes:
        print(f"note: {n}")
    print(f"image {img.shape[1]}x{img.shape[0]}, {a.iterations} iterations, best of {a.repeats} run(s)\n")
    _, rows = qt.compare_engines(img, a.iterations, a.threshold, a.engines, a.repeats)
    cols = ["engine", "time (s)", "speedup vs original", "leaves", "size (KB)", "PSNR (dB)", "max pixel diff vs original"]
    widths = [max(len(c), *(len(str(r[c])) for r in rows)) for c in cols]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for r in rows:
        print("  ".join(str(r[c]).ljust(w) for c, w in zip(cols, widths)))
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="qt_cli", description="Quadtree image compression")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("compress", help="compress an image to a .qid file")
    c.add_argument("input")
    c.add_argument("-o", "--output", help="output .qid path (default: next to the input)")
    c.add_argument("-n", "--iterations", type=int, default=20000,
                   help="number of splits (default 20000); upper limit in --psnr / --max-kb modes")
    c.add_argument("--psnr", type=float, help="stop as soon as this PSNR (dB) is reached")
    c.add_argument("--max-kb", type=float, help="largest allowed file size in KB")
    c.add_argument("--engine", default="array", choices=list(qt.ENGINES))
    c.add_argument("--threshold", type=float, default=10, help="minimum detail worth splitting (default 10)")
    c.add_argument("--prune", type=int, metavar="TOL", help="merge near-identical siblings (0-255, array engine)")
    c.add_argument("--max-pixels", type=int, default=qt.DEFAULT_MAX_PIXELS,
                   help="downscale larger images to this many pixels (memory safety)")
    c.add_argument("--png-baseline", action="store_true", help="also report the PNG size of the input")
    c.add_argument("--preview", help="save the reconstructed image here (png/jpg)")
    c.add_argument("--overlay", help="save the reconstructed image with quad borders drawn on it")
    c.set_defaults(func=cmd_compress)

    m = sub.add_parser("compare", help="run several engines on one image and compare them")
    m.add_argument("input")
    m.add_argument("-n", "--iterations", type=int, default=20000)
    m.add_argument("--threshold", type=float, default=10)
    m.add_argument("--engines", nargs="+", choices=list(qt.ENGINES), help="default: all four")
    m.add_argument("--repeats", type=int, default=2, help="runs per engine; the fastest is reported")
    m.add_argument("--max-pixels", type=int, default=qt.DEFAULT_MAX_PIXELS)
    m.set_defaults(func=cmd_compare)

    d = sub.add_parser("decompress", help="rebuild an image from a .qid file")
    d.add_argument("input")
    d.add_argument("-o", "--output")
    d.set_defaults(func=cmd_decompress)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except qt.QTError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except MemoryError:
        print("error: not enough memory; try --max-pixels with a smaller value", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
