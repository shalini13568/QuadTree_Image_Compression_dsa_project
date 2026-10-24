Quadtree Image Compression — DSA Lab Evaluation 3

An image-compression project demonstrating quadtrees, priority queues, integral images, and array-based tree representations, with a Streamlit dashboard for interactive visualization and performance analysis.

«Attribution: This project builds upon "Inspiaaa/QuadTreeImageCompression" (https://github.com/Inspiaaa/QuadTreeImageCompression), licensed under the MIT License. The original implementation and its license are retained. Additional implementations, optimizations, tests, benchmarks, and the dashboard are documented below.»

1. Problem Statement

Images contain regions with different levels of visual detail. Storing every pixel independently can require substantial storage. This project explores quadtree-based image compression, representing relatively uniform regions efficiently while subdividing more detailed regions.

2. Proposed Solution

The image is represented as a quadtree. The algorithm selects a region according to a detail or priority score and divides it into four quadrants. Each leaf region is represented by a representative colour, such as its mean colour.

The tree can be serialized into a ".qid" file and later reconstructed into an image. Since regions are represented by representative colours, the compression is lossy.

3. Key Features

- Adaptive image subdivision using quadtrees.
- Multiple implementations for comparing data structures and algorithms.
- Integral-image preprocessing for efficient region statistics.
- Priority queues using sorted priority management and a binary heap.
- Array-based tree representation.
- Lossy pruning of similar sibling regions.
- Image reconstruction and quality evaluation.
- MAE and PSNR image-quality metrics.
- Fixed-iteration, target-PSNR, and target-file-size modes.
- Interactive visualization of quadtree subdivision.
- Performance, memory, and scaling comparisons.
- Automated tests using "pytest".

4. Technologies Used

Technology| Purpose
Python| Core implementation
NumPy| Array operations and image calculations
Pillow| Image loading and processing
Streamlit| Interactive web dashboard
Matplotlib| Benchmark plots
pytest| Automated testing
"heapq"| Heap-based priority queue
"sortedcontainers"| Sorted priority queue used by the original implementation

5. Data Structures and Justification

5.1 Quadtree

A quadtree recursively divides a two-dimensional region into four quadrants.

- The root represents the entire image.
- An internal node represents a subdivided region.
- A leaf represents a region stored using a representative colour.

Why a quadtree? Unlike a uniform grid, a quadtree adapts subdivision to local image detail. Uniform regions can remain large, while complex regions can be divided into smaller cells.

5.2 Priority Queue

The algorithm repeatedly selects the region with the highest priority for subdivision. The project compares sorted-list-based priority management with a binary heap.

For a binary heap containing at most (O(k)) regions:

- Insert: (O(log k))
- Extract the highest-priority item: (O(log k))
- Peek at the highest-priority item: (O(1))

Here, (k) is the number of split iterations. A heap is suitable because the algorithm needs the next highest-priority region rather than a fully sorted collection.

5.3 Integral Images

An integral image stores cumulative pixel sums, allowing rectangular region sums to be calculated using four table lookups. Summed values and squared values can be used to calculate region means and variances.

- Preprocessing time: (O(N))
- Region-sum query after preprocessing: (O(1))
- Additional memory: (O(N))

Here, (N) is the number of pixels. Integral images reduce repeated pixel scanning but require additional memory.

5.4 Array-Based Tree

The array-based implementation stores tree information in arrays rather than creating a separate object for every node. This can reduce per-node object and pointer overhead and support efficient traversal and pruning.

Trade-off: Array-based representations require careful index management and can be more complex to implement than pointer-based trees.

6. Implementation Strategy

File| Responsibility
"qt_core.py"| Shared compression, reconstruction, image handling, and evaluation functionality
"fast_quad_tree.py"| Integral-image statistics and optimized quadtree construction
"array_quad_tree.py"| Array-based tree implementation and pruning
"quad_tree_compression.py"| Original implementation retained for comparison
"qt_cli.py"| Command-line interface
"qt_eval.py"| Evaluation helpers and test-image support
"app.py"| Interactive Streamlit dashboard
"benchmark_complexity.py"| Experiments varying image size and iteration count
"benchmark_representation.py"| Tree-representation, memory, and pruning experiments
"test_*.py"| Automated correctness, edge-case, and regression tests

Compression Workflow

1. Load the input image and convert it to a supported representation.
2. Precompute integral-image tables for implementations that use them.
3. Create the root region of the quadtree.
4. Calculate region detail scores.
5. Select the highest-priority region using the priority queue.
6. Subdivide that region into four quadrants.
7. Repeat until the iteration limit or selected stopping condition is reached.
8. Optionally prune similar sibling regions.
9. Encode the tree and leaf colours into a compressed file.
10. Decode the file to reconstruct the image and evaluate reconstruction quality.

7. Time Complexity Analysis

Let:

- (N) = number of image pixels.
- (k) = number of split iterations.

Original Implementation

The original implementation recalculates region statistics by scanning pixels and uses sorted priority management. A working estimate for comparison is:

[
O(Nlog k + klog k)
]

This is an implementation-dependent estimate rather than a universal bound. The precise cost depends on the total number of pixels scanned across generated regions and the operations used by the sorted priority structure.

Optimized Implementation

The optimized implementation builds integral-image tables in (O(N)) time and uses a binary heap to manage regions. Region-statistics queries take (O(1)) time after preprocessing, and heap operations take (O(log k)) time.

Assuming (k) split iterations, a constant number of heap operations per iteration, and constant-time region-statistics queries, the expected build-time complexity is:

O(N + klog k)

The array-based implementation aims to reduce per-node storage overhead while retaining the same asymptotic build-time bound under these assumptions.

Memory consideration: Integral-image tables require additional (O(N)) memory. Faster execution does not necessarily mean lower total memory usage. Encoding, decoding, and writing reconstructed pixels have separate costs.

8. Benchmarking and Results

The project compares four implementations:

1. Original pixel-scan implementation with sorted priority management.
2. Integral images with sorted priority management.
3. Integral images with a heap and node objects.
4. Integral images with a heap and array-based tree storage.

Experiments

- Time vs image size: Varies the number of input pixels while holding the split count fixed.
- Time vs iterations: Varies the number of split iterations for a fixed image size.
- Representation and memory: Compares build time and tree-storage overhead.
- Pruning: Evaluates the trade-off between encoded file size and reconstruction error.

8.1 Representation Benchmark

The following results come from the terminal output supplied for the 1024 × 1024 image at (k=20{,}000).

Implementation| Build time (s)| Tree memory (MB)| Bytes per node
Original| 6.762| 44.35| 554.4
Pointer-based| 2.598| 28.43| 355.4
Array-based| 2.109| 10.52| 131.6

In this run, the array-based implementation used approximately 76.3% less tree memory than the original implementation:

[
\frac{44.35-10.52}{44.35}\times100 \approx 76.3%
]

These are tree-memory measurements, not total process memory. Integral-image tables and other data structures may consume additional memory.

8.2 Complexity Benchmark

Run the following command to measure performance across image sizes and split counts:

python benchmark_complexity.py --max-side 2048 --fixed-k 5000 --repeats 3 --memory

The supplied terminal output included these timing results:

Experiment| Original (s)| Fast + sorted (s)| Fast + heap (s)
1024 × 1024 image, (k=5{,}000)| 1.9401| 0.8614| 0.6495
1024 × 1024 image, (k=30{,}000)| 8.0773| 3.8826| 3.9121
2048 × 2048 image, (k=5{,}000)| 3.1251| 2.1975| 2.6140

The optimized implementations improved build time in these examples, but their relative performance varied with the workload.

8.3 Representation and Pruning Experiment

Run:

python benchmark_representation.py --side 1024 --repeats 3

The supplied terminal output showed the following results at (k=20{,}000) for pruning:

Tolerance| Leaves| File size (KB)| MAE vs original
None| 60,001| 105.3| 6.217
0| 59,995| 105.4| 6.217
2| 58,996| 104.3| 6.219
4| 47,335| 86.4| 6.270
8| 29,455| 53.7| 6.405
16| 14,944| 24.3| 6.557
32| 8,326| 10.5| 6.901

In this run, increasing the pruning tolerance reduced the number of leaves and compressed file size, while MAE increased slightly. This illustrates the trade-off between compactness and reconstruction accuracy.

The scripts save benchmark data and plots when they complete successfully:

- "results_repr.csv"
- "results_prune.csv"
- "plot_repr.png"
- "plot_prune.png"
- "results_size.csv"
- "results_iters.csv"
- "plot_time_vs_size.png"
- "plot_time_vs_iters.png"

Important: The dashboard and terminal scripts may use different input images, image dimensions, iteration counts, implementations, repetition policies, or timing methods. Their results can therefore differ. In the presentation, use the dashboard results for the dashboard demonstration and explain that benchmark timings depend on the image and configuration. Compare results only when the configurations and measurement methods are equivalent.

9. Image Quality Metrics

Mean Absolute Error (MAE)

MAE measures the average absolute difference between original and reconstructed pixel values. Lower MAE indicates a smaller average pixel error.

Peak Signal-to-Noise Ratio (PSNR)

For 8-bit images:

PSNR = 10log10(((255)^2)/100)

Higher PSNR generally indicates a closer reconstruction. An exact reconstruction has zero MSE and infinite PSNR. Compression is lossy, so reducing file size may reduce reconstruction quality.

10. Running the Project

Install Dependencies

Run from the project directory:

python -m pip install -r requirements.txt

Run the Streamlit Dashboard

python -m streamlit run app.py

The dashboard includes the following sections:

1. Compress & Compare: Upload an image, configure compression, and compare implementations.
2. How the Image Is Divided: Explore quadtree subdivision step by step.
3. Test Cases: Examine supported image types and edge cases.
4. Speed & Memory: Compare implementation performance and memory.
5. Scaling (Big-O): View build time against image size and split iterations.
6. Test Suite: Run automated tests from the interface.

Command-Line Examples

Compress an image and save a reconstructed preview:

python qt_cli.py compress input/mountain.jpg -o output/demo.qid -n 20000 --preview output/demo.png

Compress using a target PSNR:

python qt_cli.py compress input/mountain.jpg --psnr 30

Compress under a target file size:

python qt_cli.py compress input/mountain.jpg --max-kb 50

Compress with pruning and save a subdivision overlay:

python qt_cli.py compress input/mountain.jpg -n 5000 --prune 8 --overlay output/cells.png

Compare implementations:

python qt_cli.py compare input/mountain.jpg -n 20000

Reconstruct an image from a compressed file:

python qt_cli.py decompress output/demo.qid -o output/restored.png

Check the CLI's "--help" output if an option differs in the final version.

Run Automated Tests

python -m pytest -q

Run the test suite separately from the live demo so test output does not interrupt the presentation. Report the actual result from the final integrated code version.

11. Testing and Verification

The test suite is intended to verify:

- Compression and reconstruction correctness.
- Equivalence between implementations where expected.
- Supported image formats and edge cases, including small and grayscale images.
- Handling of malformed or corrupt compressed files.
- Image-quality calculations.
- Pruning and target-based stopping modes.
- Regression behaviour after implementation changes.

The optimized implementation's equivalence test was run separately and reported 114 passed in the development environment. This does not confirm that the complete final test suite has passed. Run "python -m pytest -q" after integrating all members' changes and report the actual result.

12. Limitations

- Compression is lossy, so reconstruction quality may decrease as the tree is simplified.
- Integral images require additional memory proportional to image size.
- Floating-point differences can change subdivision order when region-detail scores are nearly equal.
- Target-PSNR, target-size, and pruning modes may have implementation-specific constraints.
- Large images may need downscaling to fit available memory.
- Benchmark timings vary between machines and configurations.
- The ".qid" format stores colours for leaf regions, so pruning a decoded file is not equivalent to pruning a live tree.

13. Team Contributions and GitHub Proof

The descriptions below should match each member's actual work and be supported by the repository's commit history.

Team member| GitHub username| Contribution
Member A| "Shalini13568"| Integral-image optimization, heap-based priority queue, equivalence testing, timing and complexity benchmarks
Member B| "Poojitha7513"| Array-based tree representation, decoding, pruning, representation and memory benchmarks
Member C| "Veni0406"| Image handling, quality metrics, target modes, CLI, Streamlit interface, testing and documentation

Repository: "shalini13568/QuadTree_Image_Compression_dsa_project" (https://github.com/shalini13568/QuadTree_Image_Compression_dsa_project)

Each team member should contribute using their own GitHub account. The commit history should provide evidence of individual contributions. Update the table if the actual division of work differs.

14. References and Attribution

- Original project: "Inspiaaa/QuadTreeImageCompression" (https://github.com/Inspiaaa/QuadTreeImageCompression).
- The original MIT licence is retained in the repository.

Conclusion

This project demonstrates how data-structure and algorithm choices affect image-compression performance. The quadtree provides adaptive spatial subdivision, integral images accelerate region-statistics queries, a binary heap manages priority selection, and an array-based representation reduces per-node storage overhead in the measured experiments.

The Streamlit dashboard makes these choices observable through image reconstruction, subdivision visualization, implementation comparisons, pruning, quality metrics, and scaling experiments.
