"""Build the LOD pyramid: isotropic 2x downsample per level + surface culling.

Two things matter here and both differ from the reference encoder.

**Isotropic.** `voxelize_big_gb.py`'s medium tiers coarsen X and Z only
(`scale: [4, 1, 4]`) and keep Y at full resolution, because its viewer draws
flat slabs. A 3D Tiles pyramid needs *cubic* cells at every level, otherwise
`geometricError` means something different horizontally than vertically and the
client's screen-space-error test picks the wrong level. So each level here
halves all three axes.

**Surface culling per level, not once.** A voxel is only drawn if at least one
of its 6 face neighbours is empty *at that level's resolution*. Coarsening makes
the model more solid, so the fraction culled grows as you go up -- which is
exactly the right direction: coarse tiles cover more ground but hold
proportionally fewer cubes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_NEIGHBOURS = np.array(
    [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
    dtype=np.int64,
)


@dataclass
class Level:
    """One resolution of the pyramid."""

    #: 0 = finest (1 voxel = `voxel_size` metres).
    depth_from_finest: int
    #: Metres per cell at this level.
    cell_size: float
    #: (M, 3) int32 cell indices in this level's own grid.
    index: np.ndarray
    #: (M, 3) uint8 colour.
    rgb: np.ndarray
    #: (nx, ny, nz) of this level's grid.
    dims: tuple[int, int, int]
    #: Boolean mask over `index`: True where the cell has an exposed face.
    surface: np.ndarray

    @property
    def visible_count(self) -> int:
        return int(self.surface.sum())

    def visible(self) -> tuple[np.ndarray, np.ndarray]:
        return self.index[self.surface], self.rgb[self.surface]


def _pack(index: np.ndarray, dims: tuple[int, int, int]) -> np.ndarray:
    nx, ny, nz = dims
    i = index.astype(np.int64)
    return (i[:, 0] * np.int64(ny) + i[:, 1]) * np.int64(nz) + i[:, 2]


def exposed_faces(index: np.ndarray, dims: tuple[int, int, int]) -> np.ndarray:
    """(M, 6) bool: which of each cell's faces touch empty space.

    Face order matches `_NEIGHBOURS`: +X, -X, +Y, -Y, +Z, -Z.

    Set-membership via sorted search rather than a dense grid, so memory scales
    with occupied cells instead of the bounding volume -- which matters a lot
    here, where occupancy measured ~3% of the bounding box.
    """
    nx, ny, nz = dims
    occupied = np.sort(_pack(index, dims))
    i = index.astype(np.int64)
    faces = np.zeros((len(index), 6), dtype=bool)

    for f, (dx, dy, dz) in enumerate(_NEIGHBOURS):
        nb = i + (dx, dy, dz)
        # A neighbour outside the grid counts as empty -> this face is exposed.
        outside = (
            (nb[:, 0] < 0) | (nb[:, 0] >= nx)
            | (nb[:, 1] < 0) | (nb[:, 1] >= ny)
            | (nb[:, 2] < 0) | (nb[:, 2] >= nz)
        )
        keys = (nb[:, 0] * np.int64(ny) + nb[:, 1]) * np.int64(nz) + nb[:, 2]
        pos = np.clip(np.searchsorted(occupied, keys), 0, max(len(occupied) - 1, 0))
        present = (~outside) & (occupied[pos] == keys)
        faces[:, f] = outside | ~present
    return faces


def mark_surface(index: np.ndarray, dims: tuple[int, int, int]) -> np.ndarray:
    """True for cells with at least one empty face neighbour (or on a boundary)."""
    return exposed_faces(index, dims).any(axis=1)


def downsample(index: np.ndarray, rgb: np.ndarray, dims: tuple[int, int, int], counts: np.ndarray | None = None):
    """One isotropic 2x coarsening step.

    A parent cell exists if any of its 8 children is occupied; its colour is the
    child colour averaged weighted by how many finest-level voxels each child
    represents, so a coarse cube's colour reflects the volume it stands for
    rather than whichever child happened to sort first.
    """
    if counts is None:
        counts = np.ones(len(index), dtype=np.int64)

    parent = (index.astype(np.int64) >> 1).astype(np.int64)
    pdims = tuple(max(1, (d + 1) // 2) for d in dims)
    keys = _pack(parent, pdims)

    order = np.argsort(keys, kind="stable")
    keys, parent, rgb, counts = keys[order], parent[order], rgb[order], counts[order]
    uniq, start = np.unique(keys, return_index=True)

    weighted = rgb.astype(np.float64) * counts[:, None]
    sums = np.add.reduceat(weighted, start, axis=0)
    total = np.add.reduceat(counts, start)
    parent_rgb = np.rint(sums / total[:, None]).clip(0, 255).astype(np.uint8)

    return parent[start].astype(np.int32), parent_rgb, total, pdims


def build_pyramid(
    index: np.ndarray,
    rgb: np.ndarray,
    dims: tuple[int, int, int],
    voxel_size: float,
    levels: int,
    verbose: bool = True,
) -> list[Level]:
    """Return `levels` Level objects, index 0 = finest.

    `levels` is the total count including the finest, so `levels=4` gives cell
    sizes 1, 2, 4, 8 x voxel_size.
    """
    out: list[Level] = []
    cur_index, cur_rgb, cur_dims = index, rgb, dims
    counts: np.ndarray | None = None

    for depth in range(levels):
        cell = voxel_size * (2 ** depth)
        surface = mark_surface(cur_index, cur_dims)
        out.append(
            Level(
                depth_from_finest=depth,
                cell_size=cell,
                index=cur_index,
                rgb=cur_rgb,
                dims=cur_dims,
                surface=surface,
            )
        )
        if verbose:
            n, s = len(cur_index), int(surface.sum())
            print(
                f"  L{depth}  cell {cell:6.1f} m  dims {cur_dims[0]:5d}x{cur_dims[1]:4d}x{cur_dims[2]:5d}  "
                f"cells {n:9,}  drawn {s:9,} ({100 * s / max(n, 1):5.1f}%)"
            )
        if depth + 1 < levels:
            cur_index, cur_rgb, counts, cur_dims = downsample(cur_index, cur_rgb, cur_dims, counts)

    return out
