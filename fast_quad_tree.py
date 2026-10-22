"""
fast_quad_tree.py  --  Member A: algorithm efficiency

Drop-in alternative to ImageCompressor from quad_tree_compression.py
(original work: Inspiaaa/QuadTreeImageCompression, MIT licence).

What is different from the original
-----------------------------------
1. Integral images (summed-area tables).
   Original: every new CompressNode calls np.std / np.mean over its pixels
             -> O(area of node) per node.
   Here:     one O(N) pre-processing pass, then the mean and std of ANY
             rectangle take O(1) (4 table look-ups).

2. Pluggable priority queue.
   "heap"   -> heapq (binary max-heap via negated keys)
   "sorted" -> sortedcontainers.SortedListWithKey (what the original uses)
   so the two can be benchmarked on the same algorithm.

3. Light-weight nodes (__slots__, no per-node image slice) and iterative
   draw / extract instead of recursion.

The output of this class is meant to be identical (up to float rounding in
tie-breaks) to the original ImageCompressor for the same input.
"""

import heapq
import itertools

import numpy as np


# ----------------------------------------------------------------------
# 1. Integral image
# ----------------------------------------------------------------------
class IntegralImage:
    """
    Summed-area tables for an image of shape (H, W, C).

    s[y, x, c]  = sum of pixel values in rows < y and columns < x
    s2[y, x, c] = same, but for squared pixel values

    Both tables have an extra leading row/column of zeros (shape H+1, W+1),
    which removes all special cases at the image border.

    Sum over rectangle [x, x+w) x [y, y+h):
        S = t[y+h, x+w] - t[y, x+w] - t[y+h, x] + t[y, x]
    """

    def __init__(self, image: np.ndarray):
        if image.ndim == 2:                       # grayscale -> (H, W, 1)
            image = image[:, :, None]
        self.height, self.width, self.channels = image.shape

        # int64 so sums of squares never overflow and results are exact.
        img = image.astype(np.int64)
        self.s = np.zeros((self.height + 1, self.width + 1, self.channels), dtype=np.int64)
        self.s2 = np.zeros_like(self.s)
        self.s[1:, 1:] = img.cumsum(axis=0).cumsum(axis=1)
        self.s2[1:, 1:] = (img * img).cumsum(axis=0).cumsum(axis=1)

    def region_stats(self, x: int, y: int, w: int, h: int):
        """Return (mean, std) per channel of the rectangle, in O(1)."""
        x2, y2 = x + w, y + h
        n = w * h
        s, s2 = self.s, self.s2

        total = s[y2, x2] - s[y, x2] - s[y2, x] + s[y, x]
        total_sq = s2[y2, x2] - s2[y, x2] - s2[y2, x] + s2[y, x]

        mean = total / n
        # population variance: E[x^2] - (E[x])^2   (same as np.std, ddof=0)
        var = total_sq / n - mean * mean
        np.maximum(var, 0.0, out=var)             # guard tiny negative round-off
        return mean, np.sqrt(var)


# ----------------------------------------------------------------------
# 2. Priority queues (same interface, different data structure)
# ----------------------------------------------------------------------
class HeapQueue:
    """Max-priority queue on a binary heap.  push/pop: O(log n)."""

    def __init__(self):
        self._heap = []
        self._counter = itertools.count()   # tie-breaker; also stops heapq comparing nodes

    def push(self, node):
        # heapq is a MIN-heap, so negate the detail to get max-first behaviour.
        # Tie-break: among EQUAL details the most recently pushed node comes out
        # first (-counter). This is exactly what the original's sorted list does
        # (equal keys keep insertion order, pop() takes the last), so both
        # versions split the same node when details tie and give identical trees.
        heapq.heappush(self._heap, (-node.detail, -next(self._counter), node))

    def pop(self):
        return heapq.heappop(self._heap)[2]

    def __len__(self):
        return len(self._heap)


class SortedQueue:
    """Max-priority queue on a sorted list (what the original repo uses)."""

    def __init__(self):
        from sortedcontainers import SortedListWithKey
        self._list = SortedListWithKey(key=lambda node: node.detail)

    def push(self, node):
        self._list.add(node)

    def pop(self):
        return self._list.pop()             # largest key is at the end

    def __len__(self):
        return len(self._list)


QUEUES = {"heap": HeapQueue, "sorted": SortedQueue}


# ----------------------------------------------------------------------
# 3. Node + compressor
# ----------------------------------------------------------------------
class FastNode:
    """One rectangle of the image. Stores NO pixel data, only numbers."""

    __slots__ = ("x", "y", "w", "h", "detail", "color", "children")

    def __init__(self, x, y, w, h, integral: IntegralImage):
        self.x, self.y, self.w, self.h = x, y, w, h
        mean, std = integral.region_stats(x, y, w, h)
        # Same formula as the original: sum of per-channel std * (pixels * channels)
        self.detail = float(std.sum()) * (w * h * integral.channels)
        self.color = mean.astype(np.uint8)       # truncation, same as original
        self.children = None                     # None => leaf

    def can_subdivide(self):
        return self.children is None and self.w > 1 and self.h > 1


class FastImageCompressor:
    """
    Same public behaviour as the original ImageCompressor:
        add_detail(iterations, detail_error_threshold)
        draw()
        extract_data() / encode_to_binary()
    """

    def __init__(self, image_data: np.ndarray, queue: str = "heap"):
        if image_data.ndim == 2:
            image_data = image_data[:, :, None]
        self._shape = image_data.shape
        self.height, self.width, self.channels = self._shape

        self.integral = IntegralImage(image_data)          # O(N)
        self.root_node = FastNode(0, 0, self.width, self.height, self.integral)

        self.queue_name = queue
        self.areas = QUEUES[queue]()
        self.areas.push(self.root_node)

        # Counters -- handy for the complexity write-up.
        self.splits = 0
        self.region_queries = 1                            # the root

    # -- subdivision ---------------------------------------------------
    def _subdivide(self, node: FastNode):
        if not node.can_subdivide():
            return []
        x, y, w, h = node.x, node.y, node.w, node.h
        sw, sh = w // 2, h // 2
        ig = self.integral
        # Same child order as the original: BL, BR, TL, TR
        node.children = (
            FastNode(x,      y,      sw,     sh,     ig),
            FastNode(x + sw, y,      w - sw, sh,     ig),
            FastNode(x,      y + sh, sw,     h - sh, ig),
            FastNode(x + sw, y + sh, w - sw, h - sh, ig),
        )
        self.splits += 1
        self.region_queries += 4
        return node.children

    def add_detail(self, max_iterations: int = 1, detail_error_threshold: float = 100):
        for _ in range(max_iterations):
            if not len(self.areas):
                break
            node = self.areas.pop()                         # O(log k)
            for child in self._subdivide(node):             # 4 x O(1)
                if child.detail > detail_error_threshold:
                    self.areas.push(child)                  # O(log k)

    # -- output --------------------------------------------------------
    def draw(self) -> np.ndarray:
        out = np.zeros(self._shape, dtype=np.uint8)
        stack = [self.root_node]
        while stack:
            n = stack.pop()
            if n.children is not None:
                stack.extend(n.children)
            else:
                out[n.y:n.y + n.h, n.x:n.x + n.w] = n.color
        return out

    def extract_data(self):
        """Preorder is_subdivided flags + leaf colours (original's format)."""
        flags, colors = [], []
        stack = [self.root_node]
        while stack:
            n = stack.pop()
            flags.append(n.children is not None)
            if n.children is not None:
                stack.extend(reversed(n.children))          # keep BL, BR, TL, TR order
            else:
                colors.append(tuple(int(c) for c in n.color))
        return flags, colors

    def encode_to_binary(self) -> bytes:
        from quad_tree_compression import encode_image_data   # reuse original encoder
        flags, colors = self.extract_data()
        return encode_image_data(self.width, self.height, flags, colors)

    def leaf_count(self) -> int:
        return 3 * self.splits + 1
