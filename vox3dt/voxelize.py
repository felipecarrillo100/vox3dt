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

Large files
-----------
Peak memory used to scale with the number of unique occupied voxels held in
RAM for the whole run. It now scales with `--memory-budget` instead: cell
aggregates are streamed through a brick-major sort key (see `brick_key`) so
that `pyramid.py` can process one tile at a time, spilling to disk via
`extsort.RunSpiller` whenever a level's working set outgrows the budget.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np

from .extsort import RunSpiller

CHUNK_POINTS_DEFAULT = 5_000_000
MEMORY_BUDGET_DEFAULT = 256 * 1024 * 1024


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
class GridSpec:
    """Everything derivable from the LAS header alone, before any point is read."""

    dims: tuple[int, int, int]
    georeference: Georeference
    n_points: int
    has_rgb: bool
    #: Grid-space (x, y, z) float64 bounds, for the streaming index math.
    g_min: np.ndarray
    g_max: np.ndarray
    y_lo: float
    y_span: float
    voxel_size: float


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


def header_epsg(las_path: Path) -> int | None:
    """Just the header's declared EPSG code, if any -- for a quick pre-check."""
    with laspy.open(str(las_path)) as reader:
        epsg, _crs_name = _read_crs(reader.header)
    return epsg


def epsg_name(epsg: int) -> str:
    """A human-readable CRS name for an EPSG code, best-effort."""
    try:
        from pyproj import CRS

        return CRS.from_epsg(epsg).name
    except Exception:
        return f"EPSG:{epsg}"


def read_grid_spec(las_path: Path, voxel_size: float = 1.0, epsg_override: int | None = None) -> GridSpec:
    """Read only the LAS header -- no point data -- and derive the voxel grid shape.

    If the header declares no CRS and `epsg_override` is given, the returned
    `Georeference` uses it as though the header had declared it. If the header
    *does* declare a CRS, `epsg_override` is ignored (the caller is expected to
    have already warned about that, since this function does not print).
    """
    with laspy.open(str(las_path)) as reader:
        header = reader.header
        mins = np.asarray(header.mins, dtype=np.float64)
        maxs = np.asarray(header.maxs, dtype=np.float64)
        n_points = header.point_count
        epsg, crs_name = _read_crs(header)
        has_rgb = any(d.name in ("red", "green", "blue") for d in header.point_format.dimensions)

    if epsg is None and epsg_override is not None:
        epsg = epsg_override
        crs_name = epsg_name(epsg_override)

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

    return GridSpec(
        dims=(nx, ny, nz),
        georeference=georeference,
        n_points=n_points,
        has_rgb=has_rgb,
        g_min=g_min,
        g_max=g_max,
        y_lo=float(g_min[1]),
        y_span=max(float(g_max[1] - g_min[1]), 1e-9),
        voxel_size=voxel_size,
    )


# ---------------------------------------------------------------------------
# Brick-major spatial key
#
# Sorts voxels by the same brick x brick tile buckets the tileset already
# uses, not by a plain (x, y, z) stride, so `pyramid.exposed_faces` can work
# tile-by-tile instead of needing a whole pyramid level resident in RAM. A
# 1-cell margin in the key layout carries a thin "apron" of each tile's
# boundary-adjacent neighbour cells (see `apron_keys`), which is exactly the
# context `exposed_faces` needs to test face exposure at tile edges.
#
#   NTX = ceil(nx/brick)   NTZ = ceil(nz/brick)
#   tx = x // brick        tz = z // brick
#   lx = x - tx*brick + 1  lz = z - tz*brick + 1     (own cells: 1..brick)
#   tile_id = tx*NTZ + tz
#   W = brick + 2
#   key = ((tile_id*W + lx)*ny + y)*W + lz
#
# Restricted to a tile's own cells (lx, lz in [1, brick]), key order is exactly
# (x, y, z) lexicographic -- identical to the tile-local order the original
# implementation produced -- so GLB vertex order is unchanged.
# ---------------------------------------------------------------------------


def tile_dims(dims: tuple[int, int, int], brick: int) -> tuple[int, int]:
    nx, _ny, nz = dims
    return (nx + brick - 1) // brick, (nz + brick - 1) // brick


def max_brick_key(dims: tuple[int, int, int], brick: int) -> int:
    """Upper bound on any key `brick_key`/`apron_keys` can produce for `dims`."""
    nx, ny, nz = dims
    ntx, ntz = tile_dims(dims, brick)
    w = brick + 2
    return ntx * ntz * w * w * ny


def check_key_capacity(dims: tuple[int, int, int], brick: int) -> None:
    if max_brick_key(dims, brick) >= 2**63:
        nx, ny, nz = dims
        raise ValueError(
            f"Grid {nx}x{ny}x{nz} with --brick {brick} needs keys beyond int64 capacity. "
            "Use a larger --voxel-size or --brick to shrink the grid."
        )


def brick_key(index: np.ndarray, dims: tuple[int, int, int], brick: int) -> np.ndarray:
    """Pack (x, y, z) grid indices (own cells) into the brick-major sort key."""
    _nx, ny, _nz = dims
    _ntx, ntz = tile_dims(dims, brick)
    x = index[:, 0].astype(np.int64)
    y = index[:, 1].astype(np.int64)
    z = index[:, 2].astype(np.int64)
    tx = x // brick
    tz = z // brick
    lx = x - tx * brick + 1
    lz = z - tz * brick + 1
    tile_id = tx * ntz + tz
    w = brick + 2
    return ((tile_id * w + lx) * ny + y) * w + lz


def tile_of_key(keys: np.ndarray, dims: tuple[int, int, int], brick: int) -> np.ndarray:
    """The `tile_id` component of a brick-major key, without a full decode."""
    _nx, ny, _nz = dims
    w = brick + 2
    return keys // (w * ny * w)


def decode_brick_key(keys: np.ndarray, dims: tuple[int, int, int], brick: int):
    """Inverse of `brick_key`/`apron_keys`. Returns (x, y, z, lx, lz, tile_id)."""
    _nx, ny, _nz = dims
    _ntx, ntz = tile_dims(dims, brick)
    w = brick + 2
    r1, lz = np.divmod(keys, w)
    r2, y = np.divmod(r1, ny)
    tile_id, lx = np.divmod(r2, w)
    tx = tile_id // ntz
    tz = tile_id % ntz
    x = tx * brick + lx - 1
    z = tz * brick + lz - 1
    return x, y, z, lx, lz, tile_id


def apron_keys(index: np.ndarray, dims: tuple[int, int, int], brick: int):
    """Apron copies of `index`'s boundary cells, for the 4 face-neighbour tiles.

    Returns `(keys, src_row)`: `keys` are destination (neighbour-tile) brick
    keys, `src_row` indexes into `index` for the row each key was copied from,
    so callers can gather matching colour-sum/count rows via `src_row`.

    Only the 4 axis-aligned (non-diagonal) neighbour tiles are needed, because
    `pyramid.exposed_faces` only ever tests axis-aligned face neighbours.
    """
    nx, ny, nz = dims
    _ntx, ntz = tile_dims(dims, brick)
    x = index[:, 0].astype(np.int64)
    z = index[:, 2].astype(np.int64)
    tx = x // brick
    tz = z // brick
    w = brick + 2
    rows = np.arange(len(index))

    keys_parts: list[np.ndarray] = []
    rows_parts: list[np.ndarray] = []

    # -X: leftmost own column (x % brick == 0, x > 0) -> tile (tx-1, tz), lx' = brick+1
    m = (x % brick == 0) & (x > 0)
    if m.any():
        dest_tile = (tx[m] - 1) * ntz + tz[m]
        lz_m = z[m] - tz[m] * brick + 1
        y_m = index[m, 1].astype(np.int64)
        keys_parts.append(((dest_tile * w + (brick + 1)) * ny + y_m) * w + lz_m)
        rows_parts.append(rows[m])

    # +X: rightmost own column (x % brick == brick-1, x < nx-1) -> tile (tx+1, tz), lx' = 0
    m = (x % brick == brick - 1) & (x < nx - 1)
    if m.any():
        dest_tile = (tx[m] + 1) * ntz + tz[m]
        lz_m = z[m] - tz[m] * brick + 1
        y_m = index[m, 1].astype(np.int64)
        keys_parts.append(((dest_tile * w + 0) * ny + y_m) * w + lz_m)
        rows_parts.append(rows[m])

    # -Z: front own row (z % brick == 0, z > 0) -> tile (tx, tz-1), lz' = brick+1
    m = (z % brick == 0) & (z > 0)
    if m.any():
        dest_tile = tx[m] * ntz + (tz[m] - 1)
        lx_m = x[m] - tx[m] * brick + 1
        y_m = index[m, 1].astype(np.int64)
        keys_parts.append(((dest_tile * w + lx_m) * ny + y_m) * w + (brick + 1))
        rows_parts.append(rows[m])

    # +Z: back own row (z % brick == brick-1, z < nz-1) -> tile (tx, tz+1), lz' = 0
    m = (z % brick == brick - 1) & (z < nz - 1)
    if m.any():
        dest_tile = tx[m] * ntz + (tz[m] + 1)
        lx_m = x[m] - tx[m] * brick + 1
        y_m = index[m, 1].astype(np.int64)
        keys_parts.append(((dest_tile * w + lx_m) * ny + y_m) * w + 0)
        rows_parts.append(rows[m])

    if not keys_parts:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(keys_parts), np.concatenate(rows_parts)



# ---------------------------------------------------------------------------
# Streaming the LAS into brick-major runs
# ---------------------------------------------------------------------------


def stream_level0_runs(
    las_path: Path,
    spec: GridSpec,
    brick: int,
    budget_bytes: int = MEMORY_BUDGET_DEFAULT,
    temp_dir: Path | None = None,
    chunk_points: int = CHUNK_POINTS_DEFAULT,
    verbose: bool = True,
    say=print,
    on_progress=None,
):
    """Stream a LAS/LAZ file into brick-major voxel runs, spilling as needed.

    Colour is accumulated as an exact integer numerator over a shared integer
    denominator (1, 257, or 2**24, matching the three colour sources below) so
    aggregation is exact regardless of chunk/run boundaries.

    `say` receives the milestone lines below (for a caller that wants them
    logged as well as printed); `verbose` controls only the in-place, terminal
    -only percentage tick, which is never passed to `say` -- that is what
    keeps a persistent log file small regardless of input size.
    """
    check_key_capacity(spec.dims, brick)
    nx, ny, nz = spec.dims
    stride_x = np.int64(ny) * np.int64(nz)
    g_min = spec.g_min

    spiller = RunSpiller(temp_dir, budget_bytes, denom=1)
    denom_decided = False
    processed = 0
    t0 = time.perf_counter()

    say(f"Streaming {spec.n_points:,} points from {las_path.name}  (rgb={spec.has_rgb})")
    say(f"  CRS: EPSG:{spec.georeference.epsg} ({spec.georeference.crs_name})")
    say(f"  grid: {nx} x {ny} x {nz}   voxel_size={spec.voxel_size}")

    with laspy.open(str(las_path)) as reader:
        for batch in reader.chunk_iterator(chunk_points):
            bx = np.asarray(batch.x, dtype=np.float64)
            by = np.asarray(batch.z, dtype=np.float64)  # elevation
            bz = np.asarray(batch.y, dtype=np.float64)  # northing

            ix = np.floor((bx - g_min[0]) / spec.voxel_size).astype(np.int64)
            iy = np.floor((by - g_min[1]) / spec.voxel_size).astype(np.int64)
            iz = np.floor((bz - g_min[2]) / spec.voxel_size).astype(np.int64)
            np.clip(ix, 0, nx - 1, out=ix)
            np.clip(iy, 0, ny - 1, out=iy)
            np.clip(iz, 0, nz - 1, out=iz)
            iz = (nz - 1) - iz  # restore right-handedness

            if spec.has_rgb:
                r = np.asarray(batch.red, dtype=np.float64)
                g = np.asarray(batch.green, dtype=np.float64)
                b = np.asarray(batch.blue, dtype=np.float64)
                if not denom_decided:
                    peak = max(float(r.max(initial=0)), float(g.max(initial=0)), float(b.max(initial=0)))
                    spiller.denom = 257 if peak > 255 else 1
                    denom_decided = True
                if spiller.denom == 257:
                    num_r, num_g, num_b = r, g, b  # exact: raw/257 never exceeds [0,255], no clip needed
                else:
                    num_r = np.clip(r, 0, 255)
                    num_g = np.clip(g, 0, 255)
                    num_b = np.clip(b, 0, 255)
            else:
                if not denom_decided:
                    spiller.denom = 1 << 24
                    denom_decided = True
                t = np.clip((by - spec.y_lo) / spec.y_span, 0.0, 1.0)
                r_f = np.clip(1.5 * t, 0, 1) * 255.0
                g_f = np.clip(1.0 - np.abs(t - 0.5) * 2.0, 0, 1) * 255.0
                b_f = np.clip(1.5 * (1.0 - t), 0, 1) * 255.0
                scale = float(spiller.denom)
                num_r = np.rint(r_f * scale)
                num_g = np.rint(g_f * scale)
                num_b = np.rint(b_f * scale)

            # Intra-chunk dedup on a plain global stride key (same shape as the
            # original implementation), then re-key the unique cells brick-major.
            gkey = ix * stride_x + iy * np.int64(nz) + iz
            uniq, inv = np.unique(gkey, return_inverse=True)
            sums = np.zeros((uniq.size, 3), dtype=np.int64)
            np.add.at(sums, inv, np.stack([num_r, num_g, num_b], axis=1).astype(np.int64))
            counts = np.bincount(inv, minlength=uniq.size).astype(np.int64)

            ugx = (uniq // stride_x).astype(np.int64)
            urem = uniq % stride_x
            ugy = (urem // nz).astype(np.int64)
            ugz = (urem % nz).astype(np.int64)
            uindex = np.stack([ugx, ugy, ugz], axis=1)

            own_keys = brick_key(uindex, spec.dims, brick)
            apr_keys, apr_rows = apron_keys(uindex, spec.dims, brick)

            all_keys = np.concatenate([own_keys, apr_keys])
            all_sums = np.concatenate([sums, sums[apr_rows]])
            all_cnts = np.concatenate([counts, counts[apr_rows]])
            spiller.add(all_keys, all_sums, all_cnts)

            processed += len(bx)
            if verbose:
                pct = 100 * processed / max(spec.n_points, 1)
                print(f"  Voxelizing: {pct:5.1f}%  ({processed:,}/{spec.n_points:,} points, {time.perf_counter() - t0:.1f}s)", end="\r")
            if on_progress is not None:
                on_progress(processed, spec.n_points)

    if verbose:
        print()
    return spiller.finish()
