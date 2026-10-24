# Quadtree Image Compression — DSA Lab Evaluation 3

> **Attribution.** This project is built on
> [Inspiaaa/QuadTreeImageCompression](https://github.com/Inspiaaa/QuadTreeImageCompression)
> (MIT licence). The core idea (adaptive quadtree + priority queue + preorder bitset
> + lzma) and the file `quad_tree_compression.py` are the original author's work.
> Everything listed under **"What we changed"** is ours. The original licence is kept in `LICENSE`.

## How it works (short)
1. Start with the whole image as one rectangle (the root of a quadtree).
2. Repeatedly take the rectangle with the highest *detail* (std-dev of its colours × area) from a
   **priority queue**, and split it into 4 quadrants.
3. Each rectangle is drawn in its mean colour. Busy areas end up with small cells, flat areas with large ones.
4. The tree is saved as a preorder bit-flag list ("is this node split?") plus one RGB colour per leaf,
   then compressed with lzma (`.qid` file).

## What we changed

| Area | Original | Our version | File | Owner |
|---|---|---|---|---|
| Region statistics | `np.std`/`np.mean` over the node's pixels: O(area) per node | Integral images: O(N) once, then O(1) per region | `fast_quad_tree.py` | Member A |
| Priority queue | `SortedListWithKey` | `heapq` max-heap, benchmarked against the original | `fast_quad_tree.py` | Member A |
| Tree storage | Node objects + per-node image views | Array-based tree (struct-of-arrays), batched statistics, array decoder | `array_quad_tree.py` | Member B |
| Size reduction | — | Lossy pruning of near-identical siblings with a guaranteed per-pixel error bound (drift-tracked tolerance) | `array_quad_tree.py` | Member B |
| Input handling | RGB images only | RGBA, grayscale, palette, 16-bit, CMYK, EXIF rotation, GIF, huge images, clear errors | `qt_core.py` | Member C |
| Quality metrics | MAE | MAE + PSNR | `qt_core.py` | Member C |
| Stopping criteria | Fixed iteration count | + target PSNR, + target file size (binary search on one build) | `qt_core.py` | Member C |
| Interface | Python calls | CLI + Streamlit dashboard: compress / compare-all modes, division pictures (cell borders), step-by-step division slider, live test cases, speed & memory, scaling plots, test runner | `qt_cli.py`, `app.py` | Member C |
| Verification | none | pytest suites, equivalence with the original, round trips, corrupt-file tests | `test_*.py` | A, B, C |

## Install and run
```bash
pip install -r requirements.txt

# command line
python qt_cli.py compress input/mountain.jpg -o out/mountain.qid -n 20000 --preview out/mountain.png
python qt_cli.py compress photo.png --psnr 30            # smallest tree reaching 30 dB
python qt_cli.py compress photo.png --max-kb 50          # best quality within 50 KB
python qt_cli.py compress photo.png -n 5000 --prune 8 --overlay out/cells.png
python qt_cli.py compare photo.png -n 20000             # original vs our engines, side by side
python qt_cli.py decompress out/mountain.qid -o out/restored.png

# web dashboard (6 tabs: compress & compare, how the image is divided, test cases, speed & memory, scaling, test suite)
python -m streamlit run app.py

# tests and benchmarks
pytest -q
python benchmark_complexity.py --max-side 2048 --memory      # Member A
python benchmark_representation.py --side 1024                # Member B
```

## Data structures and complexity
* **Quadtree** — each node covers a rectangle; a node is either a leaf or has 4 children.
  *Why not a plain grid or JPEG-style fixed blocks?* The quadtree adapts cell size to local detail.
* **Priority queue (binary max-heap)** — we always need "the leaf with the most detail", i.e. only the
  maximum. A heap gives O(log k) push/pop with little overhead; a full sorted order is not needed.
* **Integral image** — `S[y2,x2] − S[y,x2] − S[y2,x] + S[y,x]` gives the sum over any rectangle in O(1).
  Mean and variance (via a second table of squares) follow in O(1).
* **Array-based tree** — children are allocated in groups of 4, so one `first_child` index per node
  replaces 4 pointers; parents always have smaller indices than children, which makes pruning a single
  backwards pass and target-size/PSNR search possible on one build.

Build time for an image of N pixels and k splits:
original ≈ O(N·log k + k log k) (our analysis of the code)  →  ours **O(N + k log k)**.  Encoding and decoding are O(k).
Memory: the integral image costs O(N) extra (about 125 bytes per pixel at peak), which is why images above
8 megapixels are downscaled by default (`--max-pixels`).

## Results (fill in with YOUR measurements)
<!-- Paste the numbers printed by benchmark_complexity.py / benchmark_representation.py. -->

| Variant | Build time, 1024², k=20000 | Tree memory | Bytes/node |
|---|---|---|---|
| Original | TODO | TODO | TODO |
| Pointer + heap (A) | TODO | TODO | TODO |
| Array + heap (B) | TODO | TODO | TODO |

Plots: `plot_time_vs_size.png`, `plot_time_vs_iters.png`, `plot_repr.png`, `plot_prune.png`.

## Limitations
* **Equivalence with the original.** Our engines produce pixel-identical output to the original on all
  our test images, including exact ties (we break ties the same way the original does). The one exception is
  *near-ties*: two congruent regions whose details agree to ~13 digits but differ in the last bits, because
  `np.std` and the integral-image formula round differently. Then a different one of two equally detailed
  regions may be split first. The leaf count is the same and the quality differs by well under 0.05 dB
  (tested in `test_near_ties_give_equivalent_quality`).
* Compression is lossy. Pruning lowers file size further at the cost of quality (see `plot_prune.png`).
* Target PSNR / size modes and pruning need the `array` engine.
* Decoded `.qid` files cannot be pruned (the format stores colours only for leaves).
* The integral image trades memory for speed.

## Team
| Member | GitHub | Contribution |
|---|---|---|
| A | TODO | Algorithm efficiency: integral image, heap queue, timing and complexity analysis |
| B | TODO | Tree representation: array-based tree, decoding, pruning, memory benchmarks |
| C | TODO | Robustness and demo: input handling, metrics, target modes, tests, CLI/web dashboard (`qt_core.py`, `qt_eval.py`, `app.py`), README |
