# Quadtree Image Compression --- DSA Lab Evaluation 3

An image-compression project using **quadtrees, priority queues,
integral images, and array-based tree representations**, with a
Streamlit dashboard for interactive visualization and performance
analysis.

> **Attribution:** This project builds upon
> [Inspiaaa/QuadTreeImageCompression](https://github.com/Inspiaaa/QuadTreeImageCompression),
> licensed under MIT. The original implementation and its licence are
> retained. The additional implementations, optimizations, testing,
> benchmarks, and interface are documented below.

## 1. Problem Statement

Images contain regions with different levels of visual detail. Storing
every pixel independently can require substantial storage, including in
large areas with similar colours.

The objective is to implement a quadtree-based image-compression system
that represents uniform regions efficiently while preserving more detail
in complex regions.

## 2. Proposed Solution

The image is represented as a quadtree. The algorithm repeatedly selects
a region with the highest detail score and divides it into four
quadrants. Each leaf region is represented by its mean colour.

The resulting tree can be serialized into a `.qid` file and later
reconstructed into an image. Since each region is represented by a
representative colour, the compression is lossy.

## 3. Key Features

-   Adaptive image subdivision using quadtrees.
-   Multiple implementations for comparing data structures and
    algorithms.
-   Integral-image preprocessing for efficient region statistics.
-   Binary heap-based priority queue.
-   Array-based tree representation.
-   Lossy pruning of similar sibling regions.
-   Image reconstruction and quality evaluation.
-   MAE and PSNR image-quality metrics.
-   Fixed-iteration, target-PSNR, and target-file-size modes.
-   Handling for multiple image formats and image edge cases.
-   Interactive visualization of quadtree subdivision.
-   Performance, memory, and scaling comparisons.
-   Automated tests using `pytest`.

## 4. Technologies Used

  -----------------------------------------------------------------------
  Technology                          Purpose
  ----------------------------------- -----------------------------------
  Python                              Core implementation

  NumPy                               Array operations and image
                                      calculations

  Pillow                              Image loading, processing, and
                                      saving

  Streamlit                           Interactive web interface

  Matplotlib                          Benchmark plots

  pytest                              Automated testing

  `heapq`                             Heap-based priority queue

  `sortedcontainers`                  Sorted priority queue used by the
                                      original implementation
  -----------------------------------------------------------------------

## 5. Data Structures and Justification

### 5.1 Quadtree

A quadtree recursively divides a two-dimensional region into four
quadrants.

-   The root represents the entire image.
-   An internal node represents a region that has been subdivided.
-   A leaf node represents a region stored using a representative
    colour.

**Why a quadtree?** Unlike a uniform grid, a quadtree adapts subdivision
to local image detail. Uniform regions can remain large, while complex
regions can be divided into smaller cells.

### 5.2 Priority Queue

The algorithm repeatedly selects the region with the highest priority
for subdivision. The project compares sorted-list-based priority
management with a binary heap.

For a binary heap containing at most (O(k)) regions:

-   Insert: (O(`\log `{=tex}k))
-   Extract the highest-priority item: (O(`\log `{=tex}k))
-   Peek at the highest-priority item: (O(1))

Here, (k) is the number of split iterations. A heap is suitable because
the algorithm needs the next highest-priority region, not a fully sorted
collection.

### 5.3 Integral Images

An integral image stores cumulative pixel sums, allowing rectangular
region sums to be calculated using four table lookups. Summed values and
squared values can be used to calculate region means and variances.

-   Preprocessing time: (O(N))
-   Region-sum query after preprocessing: (O(1))
-   Additional memory: (O(N))

Here, (N) is the number of pixels. Integral images reduce repeated pixel
scanning but require additional memory.

### 5.4 Array-Based Tree

The array-based implementation stores tree information in arrays rather
than creating a separate object for every node. This can reduce per-node
object and pointer overhead and supports efficient traversal and
pruning.

**Trade-off:** Array-based representations require careful index
management and can be more complex to implement than pointer-based
trees.

## 6. Implementation Strategy

  -----------------------------------------------------------------------
  File                                Responsibility
  ----------------------------------- -----------------------------------
  `qt_core.py`                        Shared compression, reconstruction,
                                      image handling, and evaluation
                                      functionality

  `fast_quad_tree.py`                 Integral-image statistics and
                                      heap-based quadtree construction

  `array_quad_tree.py`                Array-based tree implementation and
                                      pruning

  `quad_tree_compression.py`          Original implementation retained
                                      for comparison

  `qt_cli.py`                         Command-line interface

  `qt_eval.py`                        Evaluation helpers and test-image
                                      support

  `app.py`                            Interactive Streamlit dashboard

  `benchmark_complexity.py`           Build-time experiments varying
                                      image size and iteration count

  `benchmark_representation.py`       Tree representation, memory, and
                                      pruning experiments

  `test_*.py`                         Automated correctness, edge-case,
                                      and regression tests
  -----------------------------------------------------------------------

### Compression Workflow

1.  Load the input image and convert it to a supported representation.
2.  Precompute integral-image tables for implementations that use them.
3.  Create the root region of the quadtree.
4.  Calculate region detail scores.
5.  Select the highest-priority region using the priority queue.
6.  Subdivide that region into four quadrants.
7.  Repeat until the iteration limit or selected stopping condition is
    reached.
8.  Optionally prune similar sibling regions.
9.  Encode the tree and leaf colours into a compressed file.
10. Decode the file to reconstruct the image and evaluate reconstruction
    quality.

## 7. Time Complexity Analysis

Let:

-   \(N\) = number of image pixels.
-   \(k\) = number of split iterations.

### Original Implementation

The original implementation recalculates region statistics by scanning
pixels and uses a sorted priority queue. The project's working analysis
estimates the build time as approximately:

\[ O(N`\log `{=tex}k + k`\log `{=tex}k) \]

This is an implementation-dependent estimate. The precise bound depends
on the total number of pixels scanned across all generated regions and
the cost of sorted-list operations. The benchmark code should be used
alongside the source-code loops to justify the bound.

### Optimized Implementation

The optimized implementation builds integral-image tables in (O(N)) time
and uses a binary heap to manage regions. Region-statistics queries take
(O(1)) time after preprocessing, and heap operations take
(O(`\log `{=tex}k)) time.

Assuming (k) split iterations, a constant number of heap operations per
iteration, and constant-time region-statistics queries, the expected
build-time complexity is:

\[ `\boxed{O(N + k\log k)}`{=tex} \]

The array-based implementation aims to reduce per-node storage overhead
while retaining the same asymptotic build-time bound under these
assumptions.

**Memory consideration:** Integral-image tables require additional
(O(N)) memory. Faster execution does not necessarily mean lower total
memory usage. Encoding, decoding, and writing reconstructed pixels have
separate costs and should be considered when analysing the full
pipeline.

## 8. Benchmarking and Results

The project compares the following implementations:

1.  Original pixel-scan implementation with a sorted priority queue.
2.  Integral images with a sorted priority queue.
3.  Integral images with a heap and node objects.
4.  Integral images with a heap and array-based tree storage.

### Experiments

-   **Time vs image size:** Varies the number of input pixels while
    holding the split count fixed.
-   **Time vs iterations:** Varies the number of split iterations for a
    fixed image size.
-   **Representation and memory:** Compares build time and tree-storage
    overhead.
-   **Pruning:** Evaluates the trade-off between encoded file size and
    reconstruction error.

### Recorded Representation Benchmark

The following values are the measurements recorded in the project's
benchmark table. The benchmark script measures the best build time
across repeated runs. Its tree-memory measurement isolates memory
allocated during tree construction and excludes integral-image tables
constructed before the memory baseline.

  Implementation                           Build time (s)   Tree memory (MB)   Bytes per node
  -------------------------------------- ---------------- ------------------ ----------------
  Original                                          5.715               43.9              557
  Integral image + heap + node objects              2.361               27.9              354
  Integral image + heap + array layout              1.775               10.1              128

**Benchmark configuration:** The table is intended to represent the 1024
× 1024 image and 20,000 split-iteration run. Confirm these settings
against the generated `results_repr.csv` before submission. The memory
figures above are tree memory, not total process memory.

For a reproducible run, execute:

``` bash
python benchmark_representation.py --side 1024 --repeats 3
```

This script generates `results_repr.csv`, `results_prune.csv`,
`plot_repr.png`, and `plot_prune.png`. Use the row where `k = 20000`
when reporting the table above.

The complexity benchmark generates `results_size.csv`,
`results_iters.csv`, `plot_time_vs_size.png`, and
`plot_time_vs_iters.png`:

``` bash
python benchmark_complexity.py --max-side 2048 --fixed-k 5000 --repeats 3 --memory
```

The exact timings depend on hardware, Python version, image content,
image dimensions, and iteration count. Include the benchmark
configuration when presenting results.

## 9. Image Quality Metrics

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between original and
reconstructed pixel values. Lower MAE indicates a smaller average pixel
error.

### Peak Signal-to-Noise Ratio (PSNR)

For 8-bit images:

\[ `\operatorname{PSNR}`{=tex} =
10`\log`{=tex}\_{10}`\left`{=tex}(`\frac{255^2}{\operatorname{MSE}}`{=tex}`\right`{=tex})
\]

Higher PSNR generally indicates a closer reconstruction. An exact
reconstruction has zero MSE and infinite PSNR. Compression is lossy, so
reducing file size may reduce reconstruction quality.

## 10. Running the Project

### Requirements

-   Python version compatible with the dependencies in
    `requirements.txt`.
-   `pip`.
-   The packages listed in `requirements.txt`.

### Install Dependencies

Run these commands from the project directory:

``` bash
python -m pip install -r requirements.txt
```

### Run the Streamlit Dashboard

``` bash
python -m streamlit run app.py
```

The dashboard opens in a local web browser.

The dashboard includes these tabs:

1.  **Compress & Compare:** Upload an image, configure compression, and
    compare implementations.
2.  **How the Image Is Divided:** Explore quadtree subdivision step by
    step.
3.  **Test Cases:** Examine supported image types and edge cases.
4.  **Speed & Memory:** Compare implementation performance and memory.
5.  **Scaling (Big-O):** View build time against image size and split
    iterations.
6.  **Test Suite:** Run automated tests from the interface.

### Command-Line Usage

Compress an image and save a reconstructed preview:

``` bash
python qt_cli.py compress input/mountain.jpg -o output/demo.qid -n 20000 --preview output/demo.png
```

Compress using a target PSNR:

``` bash
python qt_cli.py compress input/mountain.jpg --psnr 30
```

Compress under a target file size:

``` bash
python qt_cli.py compress input/mountain.jpg --max-kb 50
```

Compress with pruning and save a subdivision overlay:

``` bash
python qt_cli.py compress input/mountain.jpg -n 5000 --prune 8 --overlay output/cells.png
```

Compare implementations:

``` bash
python qt_cli.py compare input/mountain.jpg -n 20000
```

Reconstruct an image from a compressed file:

``` bash
python qt_cli.py decompress output/demo.qid -o output/restored.png
```

### Run Automated Tests

``` bash
python -m pytest -q
```

Run tests separately from the live demo so test output does not
interrupt the presentation.

## 11. Testing and Verification

The test suite is intended to verify:

-   Compression and reconstruction correctness.
-   Equivalence between implementations where expected.
-   Supported image formats and edge cases, including small and
    grayscale images.
-   Handling of malformed or corrupt compressed files.
-   Image-quality calculations.
-   Pruning and target-based stopping modes.
-   Regression behaviour after implementation changes.

Run the test suite before the evaluation and report the actual result
from your final code version.

## 12. Limitations

-   Compression is lossy, so reconstruction quality may decrease as the
    tree is simplified.
-   Integral images require additional memory proportional to the image
    size.
-   Floating-point differences can change the subdivision order of
    nearly equal-detail regions.
-   Target-PSNR, target-size, and pruning modes may have engine-specific
    constraints.
-   Large images may need downscaling to fit available memory.
-   Benchmark timings vary between machines.
-   The `.qid` format stores colours for leaf regions, so pruning a
    decoded file is not equivalent to pruning a live tree.

## 13. Team Contributions and GitHub Proof

Each member should be prepared to explain their own contribution and
show their commit history. The descriptions below should match the
actual work contributed by each member.

  -----------------------------------------------------------------------
  Team member             GitHub username         Contribution
  ----------------------- ----------------------- -----------------------
  Member A                `Shalini13568`          Integral-image
                                                  optimization,
                                                  heap-based priority
                                                  queue, timing, and
                                                  complexity analysis

  Member B                `Poojitha7513`          Array-based tree
                                                  representation,
                                                  decoding, pruning, and
                                                  memory benchmarks

  Member C                `Veni0406`              Image handling, quality
                                                  metrics, target modes,
                                                  CLI, Streamlit
                                                  interface, testing, and
                                                  documentation
  -----------------------------------------------------------------------

**Repository:**
[shalini13568/QuadTree_Image_Compression_dsa_project](https://github.com/shalini13568/QuadTree_Image_Compression_dsa_project)

Each team member must contribute using their own GitHub account. The
repository's commit history should provide evidence of individual
contributions.

## 14. References and Attribution

-   Original project:
    [Inspiaaa/QuadTreeImageCompression](https://github.com/Inspiaaa/QuadTreeImageCompression)
-   The original MIT licence is retained in the repository.

## Conclusion

This project demonstrates how data-structure and algorithm choices
affect image-compression performance. The quadtree provides adaptive
spatial subdivision, integral images accelerate region-statistics
queries, a binary heap improves priority management, and an array-based
representation reduces per-node storage overhead.

The Streamlit dashboard makes these choices observable through image
reconstruction, subdivision visualization, performance comparisons, and
scaling experiments.
