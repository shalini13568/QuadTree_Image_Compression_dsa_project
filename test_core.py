"""
Member C tests: input handling, metrics, all engines, target modes, decoding
validation, overlay and the CLI.     Run from the repo root:   pytest -q
"""
import io
import math
import os
import tempfile

import numpy as np
import pytest
from PIL import Image

import qt_cli
import qt_core as qt
from array_quad_tree import ArrayImageCompressor


def make_image(h, w, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([xx * 255 // max(w - 1, 1), yy * 255 // max(h - 1, 1),
                    (xx + yy) * 255 // max(h + w - 2, 1)], -1)
    img = img + rng.integers(0, 30, img.shape)
    img[h // 4: h // 2, w // 4: w // 2] = [200, 30, 30]
    return np.clip(img, 0, 255).astype(np.uint8)


def as_file(pil_image, fmt="PNG", **kw):
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt, **kw)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------- loading
def test_load_rgb_unchanged():
    arr = make_image(20, 30)
    out, notes = qt.load_image(as_file(Image.fromarray(arr)))
    assert np.array_equal(out, arr) and notes == []


@pytest.mark.parametrize("mode", ["L", "1", "P", "CMYK", "LA", "RGBA"])
def test_load_other_modes_become_rgb(mode):
    base = Image.fromarray(make_image(17, 23)).convert(mode)
    fmt = "TIFF" if mode == "CMYK" else "PNG"
    out, _ = qt.load_image(as_file(base, fmt))
    assert out.shape == (17, 23, 3) and out.dtype == np.uint8


def test_transparency_is_flattened_on_white():
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)          # fully transparent black
    out, notes = qt.load_image(as_file(Image.fromarray(rgba, "RGBA")))
    assert (out == 255).all() and any("transparency" in n for n in notes)


def test_16_bit_grayscale():
    a = (np.arange(64 * 64, dtype=np.uint32).reshape(64, 64) * 16).astype(np.uint16)
    out, notes = qt.load_image(as_file(Image.fromarray(a)))
    assert out.shape == (64, 64, 3) and out.max() == 255 and notes


@pytest.mark.parametrize("h,w", [(1, 1), (1, 50), (50, 1), (2, 3), (7, 5)])
def test_tiny_images_compress_and_decode(h, w):
    img = make_image(h, w)
    res = qt.compress(img, iterations=100)
    assert res.image.shape == img.shape
    assert np.array_equal(qt.decode_bytes(res.data), res.image)


def test_large_image_is_downscaled():
    big = Image.fromarray(make_image(200, 300))
    out, notes = qt.load_image(as_file(big), max_pixels=10_000)
    assert out.shape[0] * out.shape[1] <= 10_000 and any("downscaled" in n for n in notes)


def test_animated_gif_uses_first_frame():
    frames = [Image.fromarray(make_image(16, 16, seed=s)) for s in (1, 2)]
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:])
    buf.seek(0)
    out, notes = qt.load_image(buf)
    assert out.shape == (16, 16, 3) and any("first frame" in n for n in notes)


def test_missing_and_non_image_files():
    with pytest.raises(qt.QTError, match="not found"):
        qt.load_image("definitely_not_here.png")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "fake.png")
        open(p, "wb").write(b"this is just text")
        with pytest.raises(qt.QTError, match="not a readable image"):
            qt.load_image(p)


# ---------------------------------------------------------------- options
@pytest.mark.parametrize("bad", [-1, 2.5, True])
def test_bad_iterations(bad):
    with pytest.raises(qt.QTError):
        qt.compress(make_image(8, 8), iterations=bad)


def test_bad_engine_and_array():
    with pytest.raises(qt.QTError):
        qt.compress(make_image(8, 8), engine="nope")
    with pytest.raises(qt.QTError):
        qt.compress(np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(qt.QTError):
        qt.compress(make_image(8, 8), engine="original", prune_tol=4)


def test_zero_iterations_gives_flat_colour():
    img = make_image(32, 32)
    res = qt.compress(img, iterations=0)
    assert res.leaves == 1 and len(np.unique(res.image.reshape(-1, 3), axis=0)) == 1


# ---------------------------------------------------------------- metrics
def test_psnr_known_value_and_identity():
    a = np.full((10, 10, 3), 100, dtype=np.uint8)
    assert qt.psnr(a, a) == math.inf and qt.mae(a, a) == 0
    b = a + 10                                   # MSE = 100
    assert qt.psnr(a, b) == pytest.approx(10 * math.log10(255 ** 2 / 100))
    assert qt.mae(a, b) == 10


# ---------------------------------------------------------------- engines
@pytest.mark.parametrize("h,w", [(64, 64), (50, 77)])
def test_all_engines_agree(h, w):
    img = make_image(h, w)
    outs = {name: qt.compress(img, iterations=300, engine=name) for name in qt.ENGINES}
    ref = outs["original"]
    for name, r in outs.items():
        assert np.abs(r.image.astype(int) - ref.image.astype(int)).mean() < 0.5, name
        assert r.leaves == ref.leaves, name


def test_more_iterations_is_never_worse():
    img = make_image(100, 100)
    p = [qt.compress(img, iterations=k).psnr for k in (10, 100, 1000)]
    assert p[0] < p[1] < p[2]


def test_uniform_image():
    img = np.full((40, 40, 3), 77, dtype=np.uint8)
    res = qt.compress(img, iterations=500)
    assert np.array_equal(res.image, img) and math.isinf(res.psnr)


# ---------------------------------------------------------------- target modes
def test_state_at_equals_direct_build():
    img = make_image(120, 120)
    big = ArrayImageCompressor(img); big.add_detail(3000, 10)
    for s in (0, 1, 7, 100, 999):
        direct = ArrayImageCompressor(img); direct.add_detail(s, 10)
        assert direct.splits == s
        with qt._state_at(big, s) as t:
            assert np.array_equal(t.draw(), direct.draw())
            assert t.leaf_count() == direct.leaf_count()
    full = ArrayImageCompressor(img); full.add_detail(3000, 10)
    assert np.array_equal(big.draw(), full.draw())                   # restored after the with-blocks


def test_target_psnr_is_met_with_minimal_splits():
    img = make_image(128, 128)
    res = qt.compress_to_psnr(img, 30.0)
    assert res.reached and res.psnr >= 30.0
    fewer = qt.compress(img, iterations=res.splits - 1)              # one split less misses the target
    assert fewer.psnr < 30.0


def test_target_psnr_unreachable():
    img = make_image(64, 64)
    res = qt.compress_to_psnr(img, 90.0, max_iterations=50)
    assert res.reached is False and "not reachable" in res.note


def test_target_size_is_respected_and_tight():
    img = make_image(128, 128)
    limit = 3000
    res = qt.compress_to_size(img, limit)
    assert res.reached and res.compressed_bytes <= limit
    more = qt.compress(img, iterations=res.splits + 1)
    assert more.compressed_bytes > limit or res.splits >= 60000


def test_target_size_impossible():
    res = qt.compress_to_size(make_image(32, 32), 5)
    assert res.reached is False


def test_target_modes_with_pruning():
    img = make_image(128, 128)
    res = qt.compress_to_psnr(img, 28.0, prune_tol=6)
    assert res.reached and res.psnr >= 28.0


# ---------------------------------------------------------------- decoding
def test_round_trip_all_engines():
    img = make_image(90, 120)
    for name in qt.ENGINES:
        res = qt.compress(img, iterations=400, engine=name)
        assert np.array_equal(qt.decode_bytes(res.data), res.image), name


@pytest.mark.parametrize("payload", [b"", b"hello", b"\x00" * 50])
def test_garbage_is_rejected(payload):
    with pytest.raises(qt.QTError):
        qt.decode_bytes(payload)


def test_truncated_and_tampered_files_are_rejected():
    import lzma
    res = qt.compress(make_image(64, 64), iterations=200)
    with pytest.raises(qt.QTError):
        qt.decode_bytes(res.data[:-10])
    raw = bytearray(lzma.decompress(res.data))
    raw[0:4] = (10 ** 9).to_bytes(4, "little")                      # absurd width
    with pytest.raises(qt.QTError):
        qt.decode_bytes(lzma.compress(bytes(raw)))
    with pytest.raises(qt.QTError):                                  # missing colour bytes
        qt.decode_bytes(lzma.compress(bytes(raw[:-9])))


# ---------------------------------------------------------------- overlay
def test_overlay_marks_borders_only():
    img = make_image(64, 64)
    res = qt.compress(img, iterations=50)
    ov = qt.draw_overlay(res.data, res.image, color=(255, 0, 0))
    assert ov.shape == img.shape
    changed = (ov != res.image).any(axis=2)
    assert changed.any() and changed.mean() < 0.5
    assert (ov[0, :, :] == [255, 0, 0]).all()                       # top row is a cell border


def test_overlay_size_mismatch():
    res = qt.compress(make_image(32, 32), iterations=10)
    with pytest.raises(qt.QTError):
        qt.draw_overlay(res.data, make_image(16, 16))


# ---------------------------------------------------------------- CLI
def test_cli_round_trip_and_errors(capsys):
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "in.png")
        Image.fromarray(make_image(80, 60)).save(src)
        out = os.path.join(d, "sub", "x.qid")
        prev = os.path.join(d, "sub", "p.png")
        ovl = os.path.join(d, "sub", "o.png")
        assert qt_cli.main(["compress", src, "-o", out, "-n", "300", "--preview", prev, "--overlay", ovl]) == 0
        assert all(os.path.exists(p) for p in (out, prev, ovl))
        back = os.path.join(d, "back.png")
        assert qt_cli.main(["decompress", out, "-o", back]) == 0
        assert np.array_equal(np.array(Image.open(back)), np.array(Image.open(prev).convert("RGB")))

        assert qt_cli.main(["compress", src, "-o", out, "--psnr", "28"]) == 0
        assert qt_cli.main(["compress", src, "-o", out, "--max-kb", "2"]) == 0
        assert qt_cli.main(["compress", src, "--psnr", "30", "--max-kb", "5"]) == 2     # both modes
        assert qt_cli.main(["compress", "missing.png"]) == 2
        assert qt_cli.main(["decompress", src]) == 2                                      # PNG is not a .qid
        assert qt_cli.main(["compress", src, "-o", out, "--psnr", "90", "-n", "20"]) == 3  # unreachable


# ---------------------------------------------------------------- comparison
def test_compare_engines_rows():
    img = make_image(80, 80)
    results, rows = qt.compare_engines(img, iterations=300, repeats=1)
    assert [r["key"] for r in rows] == list(qt.ENGINES)
    assert rows[[r["key"] for r in rows].index("original")]["speedup vs original"] == 1.0
    assert all(r["max pixel diff vs original"] == 0 for r in rows)       # same algorithm, same picture
    assert len({r["leaves"] for r in rows}) == 1


def test_compare_engines_subset_and_errors():
    img = make_image(40, 40)
    _, rows = qt.compare_engines(img, 100, engines=["array", "pointer-heap"], repeats=1)
    assert len(rows) == 2 and rows[0]["speedup vs original"] is None
    with pytest.raises(qt.QTError):
        qt.compare_engines(img, 100, engines=["nope"])


def test_cli_compare(capsys):
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "in.png")
        Image.fromarray(make_image(60, 60)).save(src)
        assert qt_cli.main(["compare", src, "-n", "200", "--repeats", "1"]) == 0


def test_compare_all_covers_every_variation():
    img = make_image(96, 96)
    variations, rows = qt.compare_all(img, iterations=400, target_db=26.0, max_kb=2.0, repeats=1)
    groups = [g for g, _, _ in variations]
    assert groups.count("engine") == len(qt.ENGINES) and groups.count("pruning") == 3 and groups.count("goal") == 2
    assert len(rows) == len(variations)
    engine_rows = [r for r in rows if r["group"] == "engine"]
    assert all(r["same picture as original"] for r in engine_rows)
    by = {l: r for _, l, r in variations}
    plain = by[qt.ENGINES["array"][0]]
    assert by["Array + prune (tolerance 16)"].leaves <= by["Array + prune (tolerance 4)"].leaves <= plain.leaves
    goal_psnr = by["Array, target 26 dB"]
    assert goal_psnr.reached and goal_psnr.psnr >= 26.0
    goal_size = by["Array, fit in 2 KB"]
    assert goal_size.reached and goal_size.compressed_bytes <= 2048


# ---------------------------------------------------------------- division pictures
def _count_regions(free):
    """Number of 4-connected regions of True pixels (simple flood fill, no scipy)."""
    free = free.copy()
    h, w = free.shape
    regions = 0
    for y in range(h):
        for x in range(w):
            if free[y, x]:
                regions += 1
                stack = [(y, x)]
                free[y, x] = False
                while stack:
                    cy, cx = stack.pop()
                    for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                        if 0 <= ny < h and 0 <= nx < w and free[ny, nx]:
                            free[ny, nx] = False
                            stack.append((ny, nx))
    return regions


def test_division_only_is_white_with_dark_borders():
    img = make_image(64, 64)
    res = qt.compress(img, iterations=40)
    div = qt.draw_division(res.data, "only")
    assert div.shape == img.shape
    on_border = (div != 255).any(axis=2)
    assert 0 < on_border.mean() < 0.6                          # some borders, mostly white
    assert on_border[0, :].all() and on_border[:, 0].all()     # the image frame is drawn


def test_division_one_cell_is_just_the_frame():
    res = qt.compress(make_image(32, 48), iterations=0)
    mask = (qt.draw_division(res.data, "only") != 255).any(axis=2)
    assert mask[0, :].all() and mask[-1, :].all() and mask[:, 0].all() and mask[:, -1].all()
    assert not mask[1:-1, 1:-1].any()                          # nothing inside: a single cell


def test_division_cell_count_matches_leaves():
    # Count the white regions left after removing the borders: one region per cell.
    res = qt.compress(make_image(96, 96), iterations=12)
    interior = (qt.draw_division(res.data, "only") == 255).all(axis=2)
    assert _count_regions(interior) == res.leaves


def test_division_on_picture_keeps_the_picture_inside_the_cells():
    res = qt.compress(make_image(64, 64), iterations=60)
    on = qt.draw_division(res.data, "on_picture")
    inside = (on != [255, 0, 0]).any(axis=2)
    assert np.array_equal(on[inside], res.image[inside])


def test_division_bad_mode():
    res = qt.compress(make_image(16, 16), iterations=5)
    with pytest.raises(qt.QTError):
        qt.draw_division(res.data, "nope")


def test_division_at_matches_a_direct_build():
    img = make_image(100, 100)
    comp = qt.build_division_tree(img, max_splits=800)
    for s in (0, 1, 10, 100, 800):
        pic, div, leaves = qt.division_at(comp, s, "only")
        direct = qt.compress(img, iterations=s)
        assert np.array_equal(pic, direct.image) and leaves == direct.leaves
        assert np.array_equal(div, qt.draw_division(direct.data, "only"))
    assert qt.division_at(comp, 10 ** 6)[2] == comp.leaf_count()         # clamps to what exists
    full = qt.compress(img, iterations=800)
    assert np.array_equal(comp.draw(), full.image)                      # build untouched afterwards
