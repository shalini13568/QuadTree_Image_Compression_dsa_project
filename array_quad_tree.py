"""
array_quad_tree.py  --  Member B: tree representation and storage

Array-based (struct-of-arrays) quadtree, as an alternative to the pointer-based
trees (QuadTreeNode objects in the original, FastNode objects in Member A's code).

Needs: fast_quad_tree.IntegralImage (Member A) for O(1) region statistics.
Original work this builds on: Inspiaaa/QuadTreeImageCompression (MIT licence).

Memory layout: node i lives at index i of parallel NumPy arrays
-----------------------------------------------------------------
    rect[i]         = (x, y, w, h)           int32 x 4
    first_child[i]  = index of child BL, or -1 for a leaf          int32
    color[i]        = mean colour of the region                    uint8 x C
    detail[i]       = priority (std * area)                        float64

The 4 children of a node are ALWAYS allocated next to each other, so one
integer (first_child) is enough:
        BL = first_child[i]      BR = +1      TL = +2      TR = +3
(same child order as the original: bottom-left, bottom-right, top-left, top-right)

Two properties follow from "children are appended after their parent":
  * every child index is greater than its parent's index
        -> iterating i = n-1 ... 0 visits all descendants before their ancestor
           (used by prune(): bottom-up merging in ONE pass, no recursion)
  * growth is "append at the end"
        -> a dynamic array that doubles when full (amortised O(1) per node)
"""

import heapq

import numpy as np

from fast_quad_tree import IntegralImage


# ----------------------------------------------------------------------
# Storage + output (no image needed)
# ----------------------------------------------------------------------
class ArrayQuadTree:
    """The tree itself: storage, traversal, drawing, encoding, pruning."""

    def __init__(self, width: int, height: int, channels: int = 3, capacity: int = 1024):
        self.width, self.height, self.channels = width, height, channels
        self.n = 0                      # number of nodes in use (incl. pruned-away ones)
        self.splits = 0
        self._alloc(capacity)

    # -- dynamic array handling --------------------------------------
    def _alloc(self, cap: int):
        self.capacity = cap
        self.rect = np.zeros((cap, 4), dtype=np.int32)
        self.first_child = np.full(cap, -1, dtype=np.int32)
        self.color = np.zeros((cap, self.channels), dtype=np.uint8)
        self.detail = np.zeros(cap, dtype=np.float64)

    def _ensure(self, extra: int):
        """Make room for `extra` more nodes; double the capacity when full."""
        need = self.n + extra
        if need <= self.capacity:
            return
        old = (self.rect, self.first_child, self.color, self.detail)
        n = self.n
        self._alloc(max(need, self.capacity * 2))
        self.rect[:n], self.first_child[:n], self.color[:n], self.detail[:n] = (a[:n] for a in old)

    def _new_nodes(self, count: int) -> int:
        self._ensure(count)
        base = self.n
        self.n += count
        return base

    # -- traversal ------------------------------------------------------
    def _leaf_indices(self):
        """Indices of the reachable leaves, in preorder (BL, BR, TL, TR)."""
        fc = self.first_child[:self.n].tolist()      # python ints: faster than numpy scalars
        out, stack = [], [0]
        while stack:
            i = stack.pop()
            b = fc[i]
            if b < 0:
                out.append(i)
            else:
                stack.extend((b + 3, b + 2, b + 1, b))
        return out

    def leaf_count(self) -> int:
        return len(self._leaf_indices())

    def node_count(self) -> int:
        """Reachable nodes (leaves + internal), ignoring pruned-away ones."""
        return 4 * (self.leaf_count() - 1) // 3 + 1      # leaves = 3*internal + 1

    def memory_bytes(self) -> int:
        """Bytes actually used by the node data (not the spare capacity)."""
        n = self.n
        return (self.rect[:n].nbytes + self.first_child[:n].nbytes
                + self.color[:n].nbytes + self.detail[:n].nbytes)

    # -- output ----------------------------------------------------------
    def draw(self) -> np.ndarray:
        out = np.zeros((self.height, self.width, self.channels), dtype=np.uint8)
        rect = self.rect[:self.n].tolist()
        color = self.color
        for i in self._leaf_indices():
            x, y, w, h = rect[i]
            out[y:y + h, x:x + w] = color[i]
        return out

    def extract_data(self):
        """Preorder is_subdivided flags + leaf colours (original file format)."""
        fc = self.first_child[:self.n].tolist()
        flags, colors = [], []
        stack = [0]
        while stack:
            i = stack.pop()
            b = fc[i]
            flags.append(b >= 0)
            if b >= 0:
                stack.extend((b + 3, b + 2, b + 1, b))
            else:
                colors.append(tuple(int(c) for c in self.color[i]))
        return flags, colors

    def encode_to_binary(self) -> bytes:
        from quad_tree_compression import encode_image_data     # reuse original encoder
        flags, colors = self.extract_data()
        return encode_image_data(self.width, self.height, flags, colors)

    # -- decoding straight into the array layout ----------------------
    @classmethod
    def from_encoded(cls, data: bytes) -> "ArrayQuadTree":
        """Rebuild an ArrayQuadTree from the original binary format."""
        from quad_tree_compression import decode_image_data
        width, height, flags, colors = decode_image_data(data)
        channels = len(colors[0]) if colors else 3
        tree = cls(width, height, channels, capacity=max(16, len(flags)))
        root = tree._new_nodes(1)
        tree.rect[root] = (0, 0, width, height)

        fi = ci = 0                                   # cursors into flags / colors
        stack = [root]
        while stack:
            i = stack.pop()
            if flags[fi]:                             # internal node
                fi += 1
                x, y, w, h = tree.rect[i].tolist()
                sw, sh = w // 2, h // 2
                base = tree._new_nodes(4)
                tree.rect[base:base + 4] = ((x, y, sw, sh), (x + sw, y, w - sw, sh),
                                            (x, y + sh, sw, h - sh), (x + sw, y + sh, w - sw, h - sh))
                tree.first_child[i] = base
                tree.splits += 1
                stack.extend((base + 3, base + 2, base + 1, base))   # BL is popped first
            else:                                     # leaf
                fi += 1
                tree.color[i] = colors[ci]
                ci += 1
        return tree

    # -- pruning (lossy merge) ------------------------------------------
    def prune(self, tolerance: int = 4) -> int:
        """
        Merge 4 sibling LEAVES back into their parent when doing so changes no
        leaf colour by more than `tolerance` (per channel, 0-255) compared with
        the tree before pruning.

        Merges cascade (parent becomes a leaf, then ITS parent may merge, ...),
        so errors could add up level by level. To keep `tolerance` a real
        guarantee, `drift[i]` stores how far the colours below node i have
        already been moved from their original values; a merge is allowed only if
            max over children c of ( drift[c] + |color[c] - color[parent]| ) <= tolerance
        (triangle inequality => no pixel ends up more than `tolerance` away).

        Because children always have larger indices than their parent, ONE pass
        from the last node to the first is enough: when we reach a node, all of
        its descendants have already been considered.

        Cost: O(n) time, O(n) temporary memory for `drift`.
        Returns the number of merges. Pruned children stay in the arrays but
        become unreachable.
        """
        fc, color = self.first_child, self.color
        drift = np.zeros(self.n, dtype=np.int16)
        merged = 0
        for i in range(self.n - 1, -1, -1):
            b = fc[i]
            if b < 0:
                continue
            if (fc[b:b + 4] >= 0).any():              # a child is still internal
                continue
            step = np.abs(color[b:b + 4].astype(np.int16) - color[i].astype(np.int16)).max(axis=1)
            worst = int((drift[b:b + 4] + step).max())
            if worst <= tolerance:
                fc[i] = -1
                drift[i] = worst
                merged += 1
        self.splits -= merged
        return merged


# ----------------------------------------------------------------------
# The compressor: adds the image, integral image and priority queue
# ----------------------------------------------------------------------
class ArrayImageCompressor(ArrayQuadTree):
    """
    Same public behaviour as ImageCompressor / FastImageCompressor:
        add_detail(iterations, detail_error_threshold), draw(), encode_to_binary()
    """

    def __init__(self, image_data: np.ndarray):
        if image_data.ndim == 2:
            image_data = image_data[:, :, None]
        height, width, channels = image_data.shape
        super().__init__(width, height, channels)

        self.integral = IntegralImage(image_data)               # O(N), from Member A
        self._heap = []                                         # entries: (-detail, -node_index)

        root = self._new_nodes(1)
        self.rect[root] = (0, 0, width, height)
        colors, details = self._stats(self.rect[root:root + 1])
        self.color[root], self.detail[root] = colors[0], details[0]
        heapq.heappush(self._heap, (-float(details[0]), -root))  # root is always queued

    def _stats(self, rects: np.ndarray):
        """
        Mean colour and detail of m rectangles AT ONCE (rects has shape (m, 4)).
        One vectorised NumPy call handles all four children of a split, instead
        of four separate calls -- less Python/NumPy call overhead.
        """
        s, s2 = self.integral.s, self.integral.s2
        xs, ys, ws, hs = (rects[:, j].astype(np.int64) for j in range(4))
        x2, y2 = xs + ws, ys + hs
        area = ws * hs

        tot = s[y2, x2] - s[ys, x2] - s[y2, xs] + s[ys, xs]            # (m, C)
        tot2 = s2[y2, x2] - s2[ys, x2] - s2[y2, xs] + s2[ys, xs]
        mean = tot / area[:, None]
        var = np.maximum(tot2 / area[:, None] - mean * mean, 0.0)
        detail = np.sqrt(var).sum(axis=1) * (area * self.channels)     # same formula as original
        return mean.astype(np.uint8), detail

    def add_detail(self, max_iterations: int = 1, detail_error_threshold: float = 100):
        heap = self._heap
        for _ in range(max_iterations):
            if not heap:
                break
            _, neg_i = heapq.heappop(heap)
            i = -neg_i
            x, y, w, h = self.rect[i].tolist()
            if self.first_child[i] >= 0 or w <= 1 or h <= 1:    # cannot split (as in original)
                continue

            sw, sh = w // 2, h // 2
            child_rects = np.array([(x, y, sw, sh), (x + sw, y, w - sw, sh),
                                    (x, y + sh, sw, h - sh), (x + sw, y + sh, w - sw, h - sh)],
                                   dtype=np.int32)
            colors, details = self._stats(child_rects)

            base = self._new_nodes(4)                            # may reallocate the arrays
            self.rect[base:base + 4] = child_rects
            self.color[base:base + 4] = colors
            self.detail[base:base + 4] = details
            self.first_child[i] = base
            self.splits += 1

            for j, d in enumerate(details.tolist()):
                if d > detail_error_threshold:
                    heapq.heappush(heap, (-d, -(base + j)))

    def prune(self, tolerance: int = 4) -> int:
        self._heap = []              # queued nodes may no longer be leaves: stop refining
        return super().prune(tolerance)
