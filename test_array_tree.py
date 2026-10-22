"""
Member B tests: ArrayImageCompressor must match the original and Member A's code,
decode correctly, and prune sensibly.   Run from the repo root:   pytest -q
"""
import numpy as np
import pytest

from quad_tree_compression import ImageCompressor, reconstruct_image_data
from fast_quad_tree import FastImageCompressor
from array_quad_tree import ArrayImageCompressor, ArrayQuadTree


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
def test_matches_original_and_member_a(h, w, k):
    img = make_image(h, w)
    ref = ImageCompressor(img); ref.add_detail(k, 10)
    fast = FastImageCompressor(img); fast.add_detail(k, 10)
    arr = ArrayImageCompressor(img); arr.add_detail(k, 10)
    out = arr.draw()
    assert np.abs(ref.draw().astype(int) - out.astype(int)).mean() < 0.5
    assert np.array_equal(fast.draw(), out)          # same algorithm as A -> identical


def test_array_grows_past_initial_capacity():
    img = make_image(256, 256)
    arr = ArrayImageCompressor(img)
    assert arr.capacity == 1024
    arr.add_detail(3000, 10)                          # > 12000 nodes -> several doublings
    assert arr.capacity > 1024 and arr.n <= arr.capacity
    assert arr.n == 4 * arr.splits + 1


def test_children_have_larger_index_than_parent():
    arr = ArrayImageCompressor(make_image(128, 128))
    arr.add_detail(500, 10)
    parents = np.nonzero(arr.first_child[:arr.n] >= 0)[0]
    assert (arr.first_child[parents] > parents).all()      # needed by prune()'s single pass


def test_counts():
    arr = ArrayImageCompressor(make_image(100, 100))
    arr.add_detail(250, 10)
    assert arr.leaf_count() == 3 * arr.splits + 1
    assert arr.leaf_count() == len(arr.extract_data()[1])
    assert arr.node_count() == arr.n


@pytest.mark.parametrize("h,w", [(120, 90), (65, 33), (4, 4)])
def test_round_trip_through_original_format(h, w):
    img = make_image(h, w)
    arr = ArrayImageCompressor(img); arr.add_detail(800, 10)
    data = arr.encode_to_binary()
    assert np.array_equal(reconstruct_image_data(data), arr.draw())      # original decoder
    assert np.array_equal(ArrayQuadTree.from_encoded(data).draw(), arr.draw())   # our decoder


def test_prune_zero_tolerance_changes_nothing_visible():
    arr = ArrayImageCompressor(make_image(128, 128))
    arr.add_detail(1500, 10)
    before = arr.draw()
    arr.prune(0)
    assert np.array_equal(arr.draw(), before)


def test_prune_shrinks_tree_and_file_with_bounded_error():
    img = make_image(256, 256)
    base = ArrayImageCompressor(img); base.add_detail(4000, 10)
    size0, leaves0, out0 = len(base.encode_to_binary()), base.leaf_count(), base.draw()

    pruned = ArrayImageCompressor(img); pruned.add_detail(4000, 10)
    merged = pruned.prune(8)
    assert merged > 0
    assert pruned.leaf_count() == leaves0 - 3 * merged
    assert len(pruned.encode_to_binary()) < size0
    # every merged leaf is within 8 levels of the original colour -> error is bounded
    assert np.abs(pruned.draw().astype(int) - out0.astype(int)).max() <= 8


def test_prune_is_monotonic_in_tolerance():
    img = make_image(200, 200)
    counts = []
    for tol in (0, 2, 6, 12, 30):
        a = ArrayImageCompressor(img); a.add_detail(3000, 10); a.prune(tol)
        counts.append(a.leaf_count())
    assert counts == sorted(counts, reverse=True)


def test_uniform_image():
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    arr = ArrayImageCompressor(img); arr.add_detail(1000, 10)
    assert arr.splits == 1
    assert np.array_equal(arr.draw(), img)


# ---- tie-heavy images: must match the original EXACTLY --------------------
from test_fast_equivalence import TIE_IMAGES, NEAR_TIE_IMAGES, _psnr


@pytest.mark.parametrize("name", list(TIE_IMAGES))
@pytest.mark.parametrize("k", [50, 500, 3000])
def test_array_ties_are_broken_like_the_original(name, k):
    img = TIE_IMAGES[name]
    ref = ImageCompressor(img); ref.add_detail(k, 10)
    arr = ArrayImageCompressor(img); arr.add_detail(k, 10)
    assert np.array_equal(ref.draw(), arr.draw())


@pytest.mark.parametrize("name", list(NEAR_TIE_IMAGES))
@pytest.mark.parametrize("k", [50, 500, 3000])
def test_array_near_ties_give_equivalent_quality(name, k):
    img = NEAR_TIE_IMAGES[name]
    ref = ImageCompressor(img); ref.add_detail(k, 10)
    arr = ArrayImageCompressor(img); arr.add_detail(k, 10)
    assert arr.leaf_count() == len(ref.extract_data()[1])
    assert abs(_psnr(img, ref.draw()) - _psnr(img, arr.draw())) < 0.05
