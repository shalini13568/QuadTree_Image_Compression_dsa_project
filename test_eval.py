"""Member C tests for qt_eval.py (the data behind the app's evaluator tabs).  Run: pytest -q"""
import qt_core as qt
import qt_eval as ev


def test_edge_cases_match_original():
    rows, all_ok = ev.edge_cases(k=200)
    assert len(rows) == 8 and all_ok
    assert all(r["our code"] == "ok" for r in rows)
    assert all(r["output vs original"] in ("IDENTICAL", "-") or r["output vs original"].startswith("equal quality")
               for r in rows)


def test_robustness_ours_never_crashes():
    inputs, bad = ev.robustness()
    assert len(inputs) == 3 and len(bad) == 4
    assert all(r["our code"].startswith("ok") for r in inputs)
    assert all(r["our code"].startswith("clear message") for r in bad)   # no raw tracebacks


def test_speed_memory_rows():
    rows = ev.speed_memory(ev.synthetic_image(128), k=500, repeats=1)
    assert [r["version"] for r in rows] == [qt.ENGINES["original"][0], qt.ENGINES["pointer-heap"][0],
                                            qt.ENGINES["array"][0]]
    assert rows[0]["speedup"] == 1.0
    assert len({r["nodes"] for r in rows}) == 1                           # same tree size everywhere


def test_scaling_shape_and_progress():
    seen = []
    size_rows, iter_rows = ev.scaling(sides=(32, 64), fixed_k=100, ks=(10, 50), fixed_side=64,
                                      repeats=1, progress=seen.append)
    assert [r["side"] for r in size_rows] == [32, 64]
    assert [r["k"] for r in iter_rows] == [10, 50]
    assert all(key in size_rows[0] for key in qt.ENGINES)
    assert seen and abs(seen[-1] - 1.0) < 1e-9


def test_synthetic_image_is_deterministic():
    a, b = ev.synthetic_image(64), ev.synthetic_image(64)
    assert (a == b).all() and a.shape == (64, 64, 3)


def test_edge_case_gallery_pictures_match_their_inputs():
    gallery = ev.edge_case_gallery(k=100)
    assert len(gallery) == len(ev.edge_case_inputs()) == 8
    for name, original, compressed, division, leaves in gallery:
        assert original.shape == compressed.shape == division.shape, name
        assert leaves >= 1 and leaves % 3 == 1, name              # leaves = 3 * splits + 1
