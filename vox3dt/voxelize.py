"""LAS/LAZ -> voxel grid, keeping the georeference.

Deliberately close to `voxelize_big_gb.py`'s `voxelize_streaming` so results are
comparable, with one difference that is the whole point of this project: the
source CRS and the real-world coordinate of grid cell (0,0,0) are *returned*
instead of discarded. A 3D Tiles tileset cannot be placed on the globe without
them.

Axis convention (same as the reference encoder, and it turns out to be exactly
what glTF wants -- see gltfwriter):

    grid x = LAS x                       -> easting
    grid y = LAS z                       -> elevation
    grid z = mirrored LAS y              -> northing, DECREASING with z

The mirror exists because swapping (x, y, z) -> (x, z, y) flips handedness;
mirroring one axis restores it. Consequence: grid cell (0,0,0) sits at the
*maximum* northing of the bounding box, not the minimum.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np

CHUNK_POINTS_DEFAULT = 5_000_000


@dataclass
class Georeference:
    """Everything needed to put grid cell (0,0,0) on the planet."""

    epsg: int | None
    crs_name: str | None
    #: Real-world (easting, northing, height) of grid cell (0, 0, 0).
    origin: tuple[float, float, float]
    voxel_size: float
    #: Per-grid-axis sign in world space. z is -1 because of the mirror above.
    signs: tuple[int, int, int] = (1, 1, -1)

    def to_world(self, gx, gy, gz):
        """Grid index -> (easting, northing, height). Vectorized over arrays."""
        ox, oy, oz = self.origin
        vs = self.voxel_size
        return (ox + gx * vs, oy - gz * vs, oz + gy * vs)

    def as_dict(self) -> dict:
        return {
            "horizontal_epsg": self.epsg,
            "horizontal_name": self.crs_name,
            "units": "meters",
            "voxel_size": self.voxel_size,
            "grid_origin_world": {
                "easting": self.origin[0],
                "northing": self.origin[1],
                "height": self.origin[2],
            },
            "axis_mapping": {
                "grid_x": "easting",
                "grid_y": "elevation",
                "grid_z": "northing",
                "grid_z_sign": -1,
            },
            "transform_formula": "E = ox + gx*vs ; N = oy - gz*vs ; H = oz + gy*vs",
        }


@dataclass
class VoxelGrid:
    """Occupied voxels of the finest level."""

    #: (M, 3) int32 grid indices, y-up, origin at (0,0,0).
    index: np.ndarray
    #: (M, 3) uint8 averaged colour.
    rgb: np.ndarray
    #: (nx, ny, nz)
    dims: tuple[int, int, int]
    georeference: Georeference

    def __len__(self) -> int:
        return int(len(self.index))


def _read_crs(header) -> tuple[int | None, str | None]:
    """EPSG + human name from the LAS header VLRs, or (None, None)."""
    try:
        crs = header.parse_crs()
    except Exception:
        return None, None
    if crs is None:
        return None, None
    try:
        return crs.to_epsg(), crs.name
    except Exception:
        return None, getattr(crs, "name", None)


def _merge(parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]]):
    """Merge per-batch (keys, colour_sum, count) aggregates into one."""
    keys = np.concatenate([p[0] for p in parts])
    sums = np.concatenate([p[1] for p in parts])
    counts = np.concatenate([p[2] for p in parts])
    order = np.argsort(keys, kind="stable")
    keys, sums, counts = keys[order], sums[order], counts[order]
    uniq, start = np.unique(keys, return_index=True)
    merged_sums = np.add.reduceat(sums, start, axis=0)
    merged_counts = np.add.reduceat(counts, start)
    return uniq, merged_sums, merged_counts


def voxelize(
    las_path: Path,
    voxel_size: float = 1.0,
    chunk_points: int = CHUNK_POINTS_DEFAULT,
    verbose: bool = True,
) -> VoxelGrid:
    """Stream a LAS/LAZ file into a voxel grid with averaged per-voxel colour.

    Peak memory scales with the number of *unique voxels*, not the number of
    points, so large inputs stream fine.
    """
    with laspy.open(str(las_path)) as reader:
        header = reader.header
        mins = np.asarray(header.mins, dtype=np.float64)
        maxs = np.asarray(header.maxs, dtype=np.float64)
        n_points = header.point_count
        epsg, crs_name = _read_crs(header)
        has_rgb = any(d.name in ("red", "green", "blue") for d in header.point_format.dimensions)

    # Grid space: (x, y, z) = LAS (x, z, y).
    g_min = np.array([mins[0], mins[2], mins[1]], dtype=np.float64)
    g_max = np.array([maxs[0], maxs[2], maxs[1]], dtype=np.float64)
    dims = np.maximum(1, np.floor((g_max - g_min) / voxel_size).astype(np.int64) + 1)
    nx, ny, nz = (int(v) for v in dims)

    # Grid (0,0,0) is at min easting, min height, and -- because z is mirrored
    # -- the *max* northing that the floor() indexing can express.
    georeference = Georeference(
        epsg=epsg,
        crs_name=crs_name,
        origin=(float(mins[0]), float(mins[1] + (nz - 1) * voxel_size), float(mins[2])),
        voxel_size=voxel_size,
    )

    if verbose:
        print(f"Streaming {n_points:,} points from {las_path.name}  (rgb={has_rgb})")
        print(f"  CRS: EPSG:{epsg} ({crs_name})")
        print(f"  grid: {nx} x {ny} x {nz}   voxel_size={voxel_size}")

    stride_x = np.int64(ny) * np.int64(nz)
    y_lo, y_span = float(g_min[1]), max(float(g_max[1] - g_min[1]), 1e-9)
    color_scale: float | None = None
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    processed = 0
    t0 = time.perf_counter()

    with laspy.open(str(las_path)) as reader:
        for batch in reader.chunk_iterator(chunk_points):
            bx = np.asarray(batch.x, dtype=np.float64)
            by = np.asarray(batch.z, dtype=np.float64)  # elevation
            bz = np.asarray(batch.y, dtype=np.float64)  # northing

            ix = np.floor((bx - g_min[0]) / voxel_size).astype(np.int64)
            iy = np.floor((by - g_min[1]) / voxel_size).astype(np.int64)
            iz = np.floor((bz - g_min[2]) / voxel_size).astype(np.int64)
            np.clip(ix, 0, nx - 1, out=ix)
            np.clip(iy, 0, ny - 1, out=iy)
            np.clip(iz, 0, nz - 1, out=iz)
            iz = (nz - 1) - iz  # restore right-handedness

            if has_rgb:
                r = np.asarray(batch.red, dtype=np.float64)
                g = np.asarray(batch.green, dtype=np.float64)
                b = np.asarray(batch.blue, dtype=np.float64)
                if color_scale is None:
                    peak = max(float(r.max(initial=0)), float(g.max(initial=0)), float(b.max(initial=0)))
                    color_scale = 1.0 / 257.0 if peak > 255 else 1.0
                if color_scale != 1.0:
                    r, g, b = r * color_scale, g * color_scale, b * color_scale
                np.clip(r, 0, 255, out=r)
                np.clip(g, 0, 255, out=g)
                np.clip(b, 0, 255, out=b)
            else:
                t = np.clip((by - y_lo) / y_span, 0.0, 1.0)
                r = np.clip(1.5 * t, 0, 1) * 255.0
                g = np.clip(1.0 - np.abs(t - 0.5) * 2.0, 0, 1) * 255.0
                b = np.clip(1.5 * (1.0 - t), 0, 1) * 255.0

            keys = ix * stride_x + iy * np.int64(nz) + iz
            uniq, inv = np.unique(keys, return_inverse=True)
            sums = np.zeros((uniq.size, 3), dtype=np.float64)
            np.add.at(sums, inv, np.stack([r, g, b], axis=1))
            counts = np.bincount(inv, minlength=uniq.size).astype(np.int64)
            parts.append((uniq, sums, counts))

            if len(parts) >= 8:
                parts = [_merge(parts)]
            processed += len(bx)
            if verbose:
                print(f"  {processed:,}/{n_points:,} points  ({time.perf_counter() - t0:.1f}s)", end="\r")

    if verbose:
        print()
    keys, sums, counts = _merge(parts) if len(parts) > 1 else parts[0]

    gx = (keys // stride_x).astype(np.int32)
    rem = keys % stride_x
    gy = (rem // nz).astype(np.int32)
    gz = (rem % nz).astype(np.int32)
    rgb = np.rint(sums / counts[:, None]).clip(0, 255).astype(np.uint8)

    if verbose:
        print(f"Voxelized -> {len(keys):,} voxels in {time.perf_counter() - t0:.1f}s")

    return VoxelGrid(
        index=np.stack([gx, gy, gz], axis=1),
        rgb=rgb,
        dims=(nx, ny, nz),
        georeference=georeference,
    )
