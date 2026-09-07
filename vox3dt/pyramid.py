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

Streaming
---------
`exposed_faces` used to need a whole level's occupied-cell set sorted in RAM.
It still does exactly that computation, but now only over one tile's own
cells plus a thin `extra` apron (see `voxelize.apron_keys`) instead of the
whole level -- so `pyramid_stream` can process (and hand off to the tileset
writer, and fold into the next level) one tile at a time, keeping peak RAM
bounded by one tile's size and `--memory-budget`, not by total voxel count.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .extsort import RunSpiller, merge_stream
from .voxelize import apron_keys, brick_key, decode_brick_key, tile_dims, tile_of_key

_NEIGHBOURS = np.array(
    [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
    dtype=np.int64,
)


def _pack(index: np.ndarray, dims: tuple[int, int, int]) -> np.ndarray:
    nx, ny, nz = dims
    i = index.astype(np.int64)
    return (i[:, 0] * np.int64(ny) + i[:, 1]) * np.int64(nz) + i[:, 2]


def exposed_faces(index: np.ndarray, dims: tuple[int, int, int], extra: np.ndarray | None = None) -> np.ndarray:
    """(M, 6) bool: which of each cell's faces touch empty space.

    Face order matches `_NEIGHBOURS`: +X, -X, +Y, -Y, +Z, -Z.

    `extra` is additional occupied cells (e.g. a tile's apron) folded into the
    membership test but not reported on -- the returned mask has one row per
    row of `index`, exactly as if the whole level's occupied set were
    available. Set-membership via sorted search rather than a dense grid, so
    memory scales with occupied cells instead of the bounding volume.
    """
    nx, ny, nz = dims
    all_index = index if extra is None or len(extra) == 0 else np.concatenate([index, extra])
    occupied = np.sort(_pack(all_index, dims))
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


def mark_surface(index: np.ndarray, dims: tuple[int, int, int], extra: np.ndarray | None = None) -> np.ndarray:
    """True for cells with at least one empty face neighbour (or on a boundary)."""
    return exposed_faces(index, dims, extra).any(axis=1)


@dataclass
class TileBlock:
    """One tile's own cells at one pyramid level, ready to draw and/or coarsen."""

    depth_from_finest: int
    cell_size: float
    dims: tuple[int, int, int]
    tx: int
    tz: int
    #: (M, 3) int32 own-cell grid indices, (x, y, z) lex order.
    index: np.ndarray
    #: (M, 3) uint8 colour.
    rgb: np.ndarray
    #: (M,) int64 downsample weight to carry into the next coarser level.
    count: np.ndarray
    #: (M, 6) bool -- which faces of each cell touch empty space.
    faces: np.ndarray
    #: (M,) bool -- True where the cell has an exposed face.
    surface: np.ndarray

    def visible(self) -> tuple[np.ndarray, np.ndarray]:
        return self.index[self.surface], self.rgb[self.surface]


def level_tile_blocks(spilled, dims: tuple[int, int, int], brick: int, depth: int, cell_size: float, temp_dir: Path):
    """Stream one pyramid level's tiles, each as a `TileBlock`.

    Groups the globally-sorted, globally-unique records `merge_stream` yields
    by `tile_id` (the most-significant part of the brick-major key, so tiles
    arrive in ascending order and each is contiguous even across merge
    blocks), decodes each tile's own cells vs. its neighbour apron, and
    computes colour + face exposure for the own cells only.
    """
    _ntx, ntz = tile_dims(dims, brick)

    def _flush(tile_id: int, parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> TileBlock:
        keys = np.concatenate([p[0] for p in parts])
        sums = np.concatenate([p[1] for p in parts])
        cnts = np.concatenate([p[2] for p in parts])
        x, y, z, lx, lz, _tid = decode_brick_key(keys, dims, brick)
        is_own = (lx >= 1) & (lx <= brick) & (lz >= 1) & (lz <= brick)

        if depth == 0:
            rgb_all = np.rint(sums / (cnts[:, None] * spilled.denom)).clip(0, 255).astype(np.uint8)
            weight_all = np.ones(len(keys), dtype=np.int64)
        else:
            rgb_all = np.rint(sums / cnts[:, None]).clip(0, 255).astype(np.uint8)
            weight_all = cnts

        own_index = np.stack([x[is_own], y[is_own], z[is_own]], axis=1).astype(np.int32)
        own_rgb = rgb_all[is_own]
        own_weight = weight_all[is_own]
        apron_index = np.stack([x[~is_own], y[~is_own], z[~is_own]], axis=1).astype(np.int32)

        faces = exposed_faces(own_index, dims, extra=apron_index)
        surface = faces.any(axis=1)
        return TileBlock(
            depth_from_finest=depth,
            cell_size=cell_size,
            dims=dims,
            tx=tile_id // ntz,
            tz=tile_id % ntz,
            index=own_index,
            rgb=own_rgb,
            count=own_weight,
            faces=faces,
            surface=surface,
        )

    pending_tile: int | None = None
    pending_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for block in merge_stream(spilled, temp_dir):
        if len(block) == 0:
            continue
        keys = block["key"]
        tids = tile_of_key(keys, dims, brick)
        change = np.flatnonzero(np.diff(tids)) + 1
        boundaries = [0, *change.tolist(), len(tids)]
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            tid = int(tids[a])
            if pending_tile is not None and tid != pending_tile:
                yield _flush(pending_tile, pending_parts)
                pending_parts = []
            pending_tile = tid
            pending_parts.append((keys[a:b], block["sum"][a:b], block["cnt"][a:b]))

    if pending_tile is not None and pending_parts:
        yield _flush(pending_tile, pending_parts)


def pyramid_stream(spilled0, dims: tuple[int, int, int], voxel_size: float, brick: int, levels: int,
                    temp_dir: Path, budget_bytes: int, on_tile, say=print):
    """Drive the whole pyramid, level by level, tile by tile.

    For each tile: `on_tile(block)` is called (the caller writes that tile's
    GLB right there -- tile writing is naturally pipelined), and -- unless
    this is the last level -- the tile's cells are coarsened one level and
    folded into that level's `RunSpiller`. Occupancy drops sharply per level
    (this data is a thin surface, not a solid volume), so most levels above
    the finest never cross `budget_bytes` and stay resident in RAM with zero
    disk I/O -- `RunSpiller`/`merge_stream` handle that automatically.
    """
    spilled = spilled0
    cur_dims = dims
    cell = voxel_size
    stats = []

    for depth in range(levels):
        say(f"  Building pyramid: level {depth + 1}/{levels}")
        level_dir = temp_dir / f"L{depth}"
        n_cells = 0
        n_drawn = 0
        pdims = None
        next_spiller = None
        if depth + 1 < levels:
            pdims = tuple(max(1, (d + 1) // 2) for d in cur_dims)
            next_spiller = RunSpiller(temp_dir / f"L{depth + 1}", budget_bytes, denom=1)

        for block in level_tile_blocks(spilled, cur_dims, brick, depth, cell, level_dir):
            n_cells += len(block.index)
            n_drawn += int(block.surface.sum())
            on_tile(block)

            if next_spiller is not None and len(block.index):
                parent_index = (block.index.astype(np.int64) >> 1)
                w = block.count
                psum = block.rgb.astype(np.int64) * w[:, None]
                own_keys = brick_key(parent_index, pdims, brick)
                apr_keys, apr_rows = apron_keys(parent_index, pdims, brick)
                keys = np.concatenate([own_keys, apr_keys])
                sums = np.concatenate([psum, psum[apr_rows]])
                cnts = np.concatenate([w, w[apr_rows]])
                next_spiller.add(keys, sums, cnts)

        pct = 100 * n_drawn / max(n_cells, 1)
        say(f"    cell {cell:6.1f} m  dims {cur_dims[0]:5d}x{cur_dims[1]:4d}x{cur_dims[2]:5d}  "
            f"cells {n_cells:9,}  drawn {n_drawn:9,} ({pct:5.1f}%)")
        stats.append({"depth_from_finest": depth, "cell_size": cell, "cells": n_cells, "drawn": n_drawn})

        if level_dir.exists():
            shutil.rmtree(level_dir, ignore_errors=True)

        if next_spiller is not None:
            spilled = next_spiller.finish()
            cur_dims = pdims
            cell *= 2

    return stats
