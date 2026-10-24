"""
app.py  --  web demo and evaluator dashboard

Run:   python -m streamlit run app.py

Tabs
  1. Compress & compare       : upload an image; fixed iterations / target PSNR / target size /
                                compare versions / ALL variations. Every result is shown with
                                the way the image was divided.
  2. How the image is divided : watch the quadtree split the image step by step (slider + filmstrip)
  3. Test cases               : original vs our code on edge cases, with the division of each test
                                image, and on inputs / files the original was not built for
  4. Speed & memory           : the versions compared, measured live
  5. Scaling (Big-O)          : time vs image size and vs iterations, measured live
  6. Test suite               : runs the pytest suite

Everything shown is measured when you press the button - nothing is hard-coded.
"""

import io
import math
import os

import numpy as np
import streamlit as st
from PIL import Image

import qt_core as qt
import qt_eval as ev

st.set_page_config(page_title="Quadtree image compression", layout="wide")
st.title("Quadtree image compression")
st.caption("Based on Inspiaaa/QuadTreeImageCompression (MIT). Team changes: integral image, "
           "heap queue, array-based tree, pruning, target modes, tests, this interface.")


def _clear():
    for key in ("state", "compare", "all"):
        st.session_state.pop(key, None)


def _chart_num(v):                                   # chart-friendly number
    return 100.0 if v == "inf" else float(v)


VIEWS = {"Cell borders only": "only", "Cell borders on the picture": "on_picture"}


@st.cache_data(show_spinner=False, max_entries=64)
def division_cached(data: bytes, mode: str):
    """The division picture for a .qid payload (cached: reruns do not redraw it)."""
    return qt.draw_division(data, mode)


def division_of(comp, n_splits, mode):
    """Division picture of a finished build after only `n_splits` splits."""
    return qt.division_at(comp, n_splits, mode)[1]


def fit(arr, min_side=140, max_side=700):
    """Enlarge tiny images (nearest neighbour) so a 1x1 or 1x40 test image is visible."""
    h, w = arr.shape[:2]
    scale = min(max(1, -(-min_side // min(h, w))), max(1, max_side // max(h, w)))
    if scale <= 1:
        return arr
    return np.array(Image.fromarray(arr).resize((w * scale, h * scale), Image.NEAREST))


# ---------------------------------------------------------------- sidebar (tab 1 settings)
with st.sidebar.form("options"):
    st.caption("These settings drive tab 1 (Compress & compare).")
    mode = st.radio("Mode", ["Fixed iterations", "Target PSNR", "Target file size",
                             "Compare engines", "All variations"])
    iterations = st.number_input("Iterations (upper limit in target modes)", 0, 200_000, 20_000, step=1_000)
    target_db = st.number_input("Target PSNR (dB)", 10.0, 80.0, 30.0, step=1.0)
    max_kb = st.number_input("Maximum size (KB)", 0.1, 5_000.0, 50.0, step=5.0)
    engine = st.selectbox("Engine (fixed iterations only)", list(qt.ENGINES),
                          format_func=lambda k: qt.ENGINES[k][0])
    threshold = st.number_input("Detail threshold", 0.0, 10_000.0, 10.0, step=5.0)
    prune_on = st.checkbox("Prune near-identical siblings (array engine)")
    prune_tol = st.slider("Prune tolerance", 0, 64, 4)
    max_pixels = st.number_input("Max pixels (larger images are downscaled)", 10_000, 50_000_000,
                                 2_000_000, step=500_000)
    png_baseline = st.checkbox("Also measure PNG size of the input", value=True)
    repeats = st.slider("Runs per engine (compare modes; fastest is shown)", 1, 5, 2)
    go = st.form_submit_button("Compress")

upload = st.file_uploader("Image (used by tabs 1, 3 and 4; the other tabs use built-in test images)",
                          type=["png", "jpg", "jpeg", "bmp", "gif", "tiff", "webp"])


def eval_image(limit=1_000_000):
    """Image for the benchmark tabs: the upload (downscaled for speed) or a built-in one."""
    if upload is not None:
        try:
            img, notes = qt.load_image(io.BytesIO(upload.getvalue()), max_pixels=limit)
            return img, f"using your uploaded image at {img.shape[1]}x{img.shape[0]}" + \
                (f" ({'; '.join(notes)})" if notes else "")
        except qt.QTError as e:
            st.error(str(e))
    img = ev.synthetic_image(512)
    return img, "no image uploaded: using a built-in 512x512 test image"


tabs = st.tabs(["1 Compress & compare", "2 How the image is divided", "3 Test cases",
                "4 Speed & memory", "5 Scaling (Big-O)", "6 Test suite"])

# ================================================================ TAB 1
with tabs[0]:
    if go:
        if upload is None:
            st.warning("Please upload an image first.")
        else:
            try:
                img, notes = qt.load_image(io.BytesIO(upload.getvalue()), max_pixels=int(max_pixels))
                pr = prune_tol if prune_on else None
                with st.spinner("Compressing..." if mode != "All variations" else
                                "Running every variation (the original engine is the slow one)..."):
                    if mode == "All variations":
                        variations, rows = qt.compare_all(img, int(iterations), threshold, target_db, max_kb,
                                                          repeats=int(repeats))
                        _clear()
                        st.session_state["all"] = (img, variations, rows, notes)
                        res = None
                    elif mode == "Compare engines":
                        results, rows = qt.compare_engines(img, int(iterations), threshold, repeats=int(repeats))
                        _clear()
                        st.session_state["compare"] = (img, results, rows, notes)
                        res = None
                    elif mode == "Fixed iterations":
                        if pr is not None and engine != "array":
                            raise qt.QTError("pruning needs the array engine")
                        res = qt.compress(img, int(iterations), engine, threshold, pr, png_baseline)
                    elif mode == "Target PSNR":
                        res = qt.compress_to_psnr(img, target_db, int(iterations), threshold, pr, png_baseline)
                    else:
                        res = qt.compress_to_size(img, int(max_kb * 1024), int(iterations), threshold, pr,
                                                  png_baseline)
                if res is not None:
                    _clear()
                    st.session_state["state"] = (img, res, None, notes)
            except qt.QTError as e:
                _clear()
                st.error(str(e))
            except MemoryError:
                _clear()
                st.error("Not enough memory. Lower 'Max pixels' and try again.")

    if "all" in st.session_state:
        img, variations, rows, notes = st.session_state["all"]
        for n in notes:
            st.info(n)
        st.subheader("Every variation on the same image")
        st.dataframe(rows, hide_index=True, use_container_width=True)
        st.caption("engine = the four implementations with identical settings (same picture, different speed). "
                   "pruning = array engine with near-identical siblings merged (smaller file, small quality "
                   "cost). goal = the array engine searching for a target PSNR / file size. "
                   "Timing is machine-dependent; goal rows include the search time.")
        ch1, ch2, ch3 = st.columns(3)
        ch1.caption("Build time (s) - engines")
        ch1.bar_chart({r["variation"]: r["time (s)"] for r in rows if r["group"] == "engine"})
        ch2.caption("File size (KB) - pruning and goals")
        ch2.bar_chart({r["variation"]: r["size (KB)"] for r in rows if r["group"] != "engine"})
        ch3.caption("PSNR (dB) - pruning and goals")
        ch3.bar_chart({r["variation"]: _chart_num(r["PSNR (dB)"]) for r in rows if r["group"] != "engine"})

        view_label = st.radio("How to draw the division", list(VIEWS), horizontal=True, key="view_all")
        view = VIEWS[view_label]
        tiles = [("Original", img, None)]
        engines = [(l, r) for g, l, r in variations if g == "engine"]
        if all(np.array_equal(engines[0][1].image, r.image) for _, r in engines):
            tiles.append(("All 4 engines (identical picture)", engines[0][1].image, engines[0][1]))
        else:
            tiles += [(l, r.image, r) for l, r in engines]
        tiles += [(l, r.image, r) for g, l, r in variations if g != "engine"]
        for start in range(0, len(tiles), 3):
            cols = st.columns(3)
            for col, (label, image, r) in zip(cols, tiles[start:start + 3]):
                caption = label if r is None else (
                    f"{label} - {r.compressed_bytes / 1024:.1f} KB, "
                    f"{'inf' if math.isinf(r.psnr) else f'{r.psnr:.1f}'} dB")
                col.image(image, caption=caption, use_container_width=True)
                if r is not None:
                    col.image(division_cached(r.data, view), use_container_width=True,
                              caption=f"how it is divided: {r.leaves:,} cells")

    elif "compare" in st.session_state:
        img, results, rows, notes = st.session_state["compare"]
        for n in notes:
            st.info(n)
        st.subheader("Engine comparison (same image, same settings)")
        st.dataframe(rows, hide_index=True, use_container_width=True)
        st.caption("Time includes building the tree (and the integral image for our engines). "
                   "All engines implement the same algorithm, so leaves, size and PSNR match and "
                   "'max pixel diff' is 0; the improvement is in speed. Timing is machine-dependent.")
        st.bar_chart({r["engine"]: r["time (s)"] for r in rows})
        left, right = st.columns(2)
        left.subheader("Original")
        left.image(img, use_container_width=True)
        first = next(iter(results.values()))
        right.subheader("Compressed (identical for every version)")
        right.image(first.image, use_container_width=True)
        st.subheader("How each version divided the image")
        view_label = st.radio("How to draw the division", list(VIEWS), horizontal=True, key="view_cmp")
        cols = st.columns(len(results))
        for col, (key, r) in zip(cols, results.items()):
            col.image(division_cached(r.data, VIEWS[view_label]), use_container_width=True,
                      caption=f"{qt.ENGINES[key][0]}: {r.leaves:,} cells")

    elif "state" in st.session_state:
        img, res, overlay, notes = st.session_state["state"]
        for n in notes:
            st.info(n)
        if res.note:
            (st.success if res.reached in (True, None) else st.warning)(res.note)
        view_label = st.radio("How to draw the division", list(VIEWS), horizontal=True, key="view_one")
        c1, c2, c3 = st.columns(3)
        c1.subheader("Original")
        c1.image(img, use_container_width=True)
        c2.subheader("Compressed")
        c2.image(res.image, use_container_width=True)
        c3.subheader("How it is divided")
        c3.image(division_cached(res.data, VIEWS[view_label]), use_container_width=True,
                 caption=f"{res.leaves:,} cells: small where the image has detail, large where it is flat")
        m = st.columns(4)
        m[0].metric("Compressed size", f"{res.compressed_bytes / 1024:.1f} KB")
        m[1].metric("Ratio vs raw", f"{res.ratio:.1f}x")
        m[2].metric("PSNR", "inf" if math.isinf(res.psnr) else f"{res.psnr:.2f} dB")
        m[3].metric("Leaves", f"{res.leaves:,}")
        st.table({k: [str(v)] for k, v in res.summary().items()})
        png = io.BytesIO()
        Image.fromarray(res.image).save(png, format="PNG")
        d1, d2 = st.columns(2)
        d1.download_button("Download .qid", res.data, file_name="compressed.qid")
        d2.download_button("Download reconstructed PNG", png.getvalue(), file_name="reconstructed.png")

    else:
        st.write("Upload an image, choose options in the sidebar and press **Compress**.")

# ================================================================ TAB 2
with tabs[1]:
    st.subheader("How the image is divided")
    st.write("The algorithm starts with the whole image as one cell and repeatedly splits the cell with "
             "the most detail into four. Press the button, then drag the slider or read the filmstrip.")
    div_splits = st.number_input("Largest number of splits to explore", 10, 20_000, 3_000, step=500, key="div_n")
    if st.button("Build the division"):
        img_d, msg = eval_image()
        with st.spinner("Building..."):
            comp = qt.build_division_tree(img_d, int(div_splits))
            steps = [s_ for s_ in (0, 1, 4, 16, 64, 256, 1024, 4096, 16384) if s_ <= comp.splits]
            strip = []
            for n_splits in steps:
                pic, _, leaves = qt.division_at(comp, n_splits, "only")
                strip.append((n_splits, leaves, qt.psnr(img_d, pic),
                              division_of(comp, n_splits, "only"),
                              division_of(comp, n_splits, "on_picture")))
        st.session_state["div"] = (comp, img_d, msg, strip)
    if "div" in st.session_state:
        comp, img_d, msg, strip = st.session_state["div"]
        st.caption(msg)
        view_label = st.radio("How to draw the division", list(VIEWS), horizontal=True, key="view_div")
        view_idx = 3 if VIEWS[view_label] == "only" else 4

        st.markdown("**Filmstrip: the same image after more and more splits**")
        for start in range(0, len(strip), 4):
            cols = st.columns(4)
            for col, item in zip(cols, strip[start:start + 4]):
                n_splits, leaves, p, *_ = item
                col.image(item[view_idx], use_container_width=True,
                          caption=f"{n_splits} splits: {leaves:,} cells, "
                                  f"{'inf' if math.isinf(p) else f'{p:.1f}'} dB")

        st.markdown("**Interactive: choose the number of splits**")
        n_pick = st.slider("Number of splits", 0, comp.splits, min(200, comp.splits), key="div_slider")
        pic, div, leaves = qt.division_at(comp, n_pick, VIEWS[view_label])
        p = qt.psnr(img_d, pic)
        m1, m2, m3 = st.columns(3)
        m1.metric("Splits", f"{n_pick:,}")
        m2.metric("Cells (leaves)", f"{leaves:,}")
        m3.metric("PSNR", "inf" if math.isinf(p) else f"{p:.2f} dB")
        c1, c2, c3 = st.columns(3)
        c1.caption("Original")
        c1.image(img_d, use_container_width=True)
        c2.caption("Compressed at this many splits")
        c2.image(pic, use_container_width=True)
        c3.caption("How it is divided")
        c3.image(div, use_container_width=True)

# ================================================================ TAB 3
with tabs[2]:
    st.subheader("Test cases: original code vs our code")
    st.write("Tricky images and inputs, run through the original repo's code and through ours with the "
             "same settings.")
    if st.button("Run test cases"):
        with st.spinner("Running..."):
            case_rows, all_ok = ev.edge_cases()
            input_rows, bad_rows = ev.robustness()
            gallery = ev.edge_case_gallery()
        st.session_state["ev_cases"] = (case_rows, all_ok, input_rows, bad_rows, gallery)
    if "ev_cases" in st.session_state:
        case_rows, all_ok, input_rows, bad_rows, gallery = st.session_state["ev_cases"]
        st.markdown("**A. Edge-case images - is our output the same as the original's?**")
        st.dataframe(case_rows, hide_index=True, use_container_width=True)
        (st.success if all_ok else st.error)(
            "Every case matches the original." if all_ok else "Some cases differ - investigate!")
        st.caption(ev.TIE_NOTE)
        st.markdown("**How each test image was divided**")
        st.caption("Left: the test image. Middle: our compressed picture. Right: how the quadtree divided it "
                   "(tiny images are enlarged so you can see them).")
        for name, original, compressed, division, leaves in gallery:
            g1, g2, g3 = st.columns(3)
            g1.image(fit(original), caption=f"{name}: input")
            g2.image(fit(compressed), caption="compressed")
            g3.image(fit(division), caption=f"division: {leaves:,} cells")
        st.markdown("**B. Image types the original code was not built for**")
        st.dataframe(input_rows, hide_index=True, use_container_width=True)
        st.markdown("**C. Damaged or wrong files**")
        st.dataframe(bad_rows, hide_index=True, use_container_width=True)
        st.caption("'original code' columns show what actually happened when the original repo's code "
                   "was given the same input on this machine.")

# ================================================================ TAB 4
with tabs[3]:
    st.subheader("Speed and memory: original vs our versions")
    col_k, col_r = st.columns(2)
    k_speed = col_k.number_input("Iterations (splits)", 100, 100_000, 20_000, step=1_000, key="k_speed")
    rep_speed = col_r.slider("Runs per version (fastest is shown)", 1, 5, 2, key="rep_speed")
    if st.button("Run speed & memory benchmark"):
        img_b, msg = eval_image()
        st.info(msg)
        with st.spinner("Measuring (the original version is the slow one)..."):
            st.session_state["ev_speed"] = (ev.speed_memory(img_b, int(k_speed), int(rep_speed)), msg)
    if "ev_speed" in st.session_state:
        rows, msg = st.session_state["ev_speed"]
        st.caption(msg)
        st.dataframe(rows, hide_index=True, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.caption("Build time (s)")
        c1.bar_chart({r["version"]: r["time (s)"] for r in rows})
        c2.caption("Tree memory (MB)")
        c2.bar_chart({r["version"]: r["tree memory (MB)"] for r in rows})
        c3.caption("Bytes per node")
        c3.bar_chart({r["version"]: r["bytes/node"] for r in rows})
        st.caption("Versions are numbered as in the compare modes: 1 = original (pixel scan for mean and std, "
                   "sorted list, node objects); 3 = integral image + heap, still node objects; "
                   "4 = integral image + heap + array layout. Each step changes one thing, so the gap between "
                   "neighbouring rows is the effect of that change. Tree memory excludes the integral image "
                   "(shared by 3 and 4). All produce the same picture.")

# ================================================================ TAB 5
with tabs[4]:
    st.subheader("Scaling: does the time grow the way the analysis says?")
    st.write("Expected: the original rescans pixels for every new node, about O(N log k + k log k). "
             "With the integral image: O(N + k log k). Log-log plots; times include building the tree.")
    quick = st.checkbox("Quick run (smaller sizes)", value=True)
    if st.button("Run scaling experiment"):
        bar = st.progress(0.0)
        sides = (64, 128, 256, 512) if quick else (128, 256, 512, 1024)
        ks = (50, 200, 800, 3000) if quick else (100, 300, 1000, 3000, 10000)
        data = ev.scaling(sides=sides, fixed_k=1000 if quick else 3000, ks=ks,
                          fixed_side=256 if quick else 512, repeats=1, progress=bar.progress)
        st.session_state["ev_scaling"] = (data, 1000 if quick else 3000, 256 if quick else 512)
        bar.empty()
    if "ev_scaling" in st.session_state:
        (rows_size, rows_iter), fixed_k, fixed_side = st.session_state["ev_scaling"]
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def loglog(rows, xkey, xlabel, title):
            fig, ax = plt.subplots(figsize=(5.5, 4))
            for key, (label, _) in qt.ENGINES.items():
                ax.plot([r[xkey] for r in rows], [r[key] for r in rows], marker="o", label=label)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("build time (s)")
            ax.set_title(title)
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=7)
            fig.tight_layout()
            return fig

        left, right = st.columns(2)
        fig1 = loglog(rows_size, "pixels", "image size N (pixels)", f"Time vs image size (k = {fixed_k})")
        left.pyplot(fig1)
        plt.close(fig1)
        fig2 = loglog(rows_iter, "k", "iterations k", f"Time vs iterations ({fixed_side}x{fixed_side})")
        right.pyplot(fig2)
        plt.close(fig2)
        with st.expander("Show the raw timings"):
            st.dataframe([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                          for r in rows_size], hide_index=True)
            st.dataframe([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                          for r in rows_iter], hide_index=True)
    saved = [f for f in ("plot_time_vs_size.png", "plot_time_vs_iters.png", "plot_repr.png", "plot_prune.png")
             if os.path.exists(f)]
    if saved:
        with st.expander("Saved benchmark plots from the repo"):
            cols = st.columns(2)
            for i, f in enumerate(saved):
                cols[i % 2].image(f, caption=f, use_container_width=True)

# ================================================================ TAB 6
with tabs[5]:
    st.subheader("Automated test suite")
    st.write("Runs every `test_*.py` file with pytest: equivalence with the original, round trips, edge "
             "cases, pruning, target modes and the command line.")
    if st.button("Run all tests"):
        with st.spinner("Running pytest (can take a minute)..."):
            st.session_state["ev_tests"] = ev.run_tests()
    if "ev_tests" in st.session_state:
        status, text = st.session_state["ev_tests"]
        if status is None:
            st.warning(text)
        else:
            (st.success if status else st.error)("All tests passed." if status else "Some tests failed.")
            st.code(text[-4000:])
