"""
Member A tests: FastImageCompressor must behave like the original ImageCompressor.
Run from the repo root:   pytest -q
"""
import numpy as np
import pytest

from quad_tree_compression import ImageCompressor, reconstruct_image_data
from fast_quad_tree import FastImageCompressor, IntegralImage


def make_image(h, w, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([xx * 255 // max(w - 1, 1),
                    yy * 255 // max(h - 1, 1),
                    (xx + yy) * 255 // max(h + w - 2, 1)], -1)
    img = img + rng.integers(0, 30, img.shape)
    img[h // 4: h // 2, w // 4: w // 2] = [200, 30, 30]
    return np.clip(img, 0, 255).astype(np.uint8)


SIZES = [(64, 64), (100, 77), (33, 129), (200, 300), (1, 40), (40, 1), (2, 2), (3, 5)]


@pytest.mark.parametrize("h,w", SIZES)
@pytest.mark.parametrize("k", [0, 1, 10, 200, 3000])
@pytest.mark.parametrize("queue", ["heap", "sorted"])
def test_output_matches_original(h, w, k, queue):
    img = make_image(h, w)
    ref = ImageCompressor(img)
    ref.add_detail(k, 10)
    fast = FastImageCompressor(img, queue=queue)
    fast.add_detail(k, 10)
    # Exact equality is expected; ties in detail can in principle be broken
    # differently by the two queues, so allow a tiny mean difference.
    diff = np.abs(ref.draw().astype(int) - fast.draw().astype(int)).mean()
    assert diff < 0.5


def test_integral_image_matches_numpy():
    img = make_image(57, 91)
    ig = IntegralImage(img)
    rng = np.random.default_rng(1)
    for _ in range(200):
        x, y = rng.integers(0, 80), rng.integers(0, 50)
        w, h = rng.integers(1, 91 - x), rng.integers(1, 57 - y)
        mean, std = ig.region_stats(x, y, w, h)
        region = img[y:y + h, x:x + w].astype(float)
        assert np.allclose(mean, region.mean(axis=(0, 1)))
        assert np.allclose(std, region.std(axis=(0, 1)), atol=1e-6)


def test_encode_decode_round_trip():
    img = make_image(120, 90)
    comp = FastImageCompressor(img)
    comp.add_detail(800, 10)
    assert np.array_equal(reconstruct_image_data(comp.encode_to_binary()), comp.draw())


def test_uniform_image_stops_early():
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    comp = FastImageCompressor(img)
    comp.add_detail(1000, 10)
    # The root is always queued (as in the original), so exactly one split
    # happens; its children have detail 0 and are never queued again.
    assert comp.splits == 1
    assert np.array_equal(comp.draw(), img)


def test_leaf_count_formula():
    comp = FastImageCompressor(make_image(100, 100))
    comp.add_detail(250, 10)
    assert comp.leaf_count() == len(comp.extract_data()[1])


# ---- images with many EXACTLY equal detail values (ties) ------------------
def _gradient(h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.stack([xx * 255 // max(w - 1, 1), yy * 255 // max(h - 1, 1),
                     (xx + yy) * 255 // max(h + w - 2, 1)], -1).astype(np.uint8)


def _checker(n, cell=8):
    yy, xx = np.mgrid[0:n, 0:n]
    v = (((yy // cell) + (xx // cell)) % 2 * 200 + 20).astype(np.uint8)
    return np.stack([v, v, 255 - v], -1)


def _mirror(n):
    half = make_image(n, n // 2, seed=5)
    return np.concatenate([half, half[:, ::-1]], axis=1)


# EXACT ties (symmetric / repeated content of power-of-two size): must match the original exactly.
TIE_IMAGES = {"gradient128": _gradient(128, 128), "checker64": _checker(64), "mirror64": _mirror(64)}

# NEAR ties: congruent regions whose details agree to ~13 digits but differ in the last bits
# because np.std and the integral-image formula round differently. Which of the two splits
# first is then decided by rounding noise, so the trees may differ; quality does not.
NEAR_TIE_IMAGES = {"gradient100x77": _gradient(100, 77), "gradient33x129": _gradient(33, 129)}


def _psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


@pytest.mark.parametrize("name", list(TIE_IMAGES))
@pytest.mark.parametrize("k", [50, 500, 3000])
@pytest.mark.parametrize("queue", ["heap", "sorted"])
def test_ties_are_broken_like_the_original(name, k, queue):
    img = TIE_IMAGES[name]
    ref = ImageCompressor(img); ref.add_detail(k, 10)
    fast = FastImageCompressor(img, queue=queue); fast.add_detail(k, 10)
    assert np.array_equal(ref.draw(), fast.draw())          # exact, not just "close"


@pytest.mark.parametrize("name", list(NEAR_TIE_IMAGES))
@pytest.mark.parametrize("k", [50, 500, 3000])
@pytest.mark.parametrize("queue", ["heap", "sorted"])
def test_near_ties_give_equivalent_quality(name, k, queue):
    img = NEAR_TIE_IMAGES[name]
    ref = ImageCompressor(img); ref.add_detail(k, 10)
    fast = FastImageCompressor(img, queue=queue); fast.add_detail(k, 10)
    assert fast.leaf_count() == len(ref.extract_data()[1])      # same number of leaves
    assert abs(_psnr(img, ref.draw()) - _psnr(img, fast.draw())) < 0.05
