"""Quadtree tileset generation and the georeference transform.

Why a quadtree and not an octree
--------------------------------
Measured on the Yaloch sample: 3.14% of the bounding volume is occupied, and
the median vertical thickness of an occupied footprint cell is **2 voxels**
(mean 2.7, p95 6). The data is a thick skin, not a volume. An octree over that
extent would spend nearly all its nodes on empty air and the vertical splits
would buy almost nothing, so each tile here covers a square of ground and
carries its **full Y range**. 3D Tiles permits arbitrary tree shapes; nothing
obliges an octree.

Coordinate systems, in order
----------------------------
1. **Grid** -- integer voxel indices, y-up, z mirrored (see voxelize.py).
2. **Tile content (glTF)** -- metres, y-up: ``(gx, gy, gz) * cell_size``.
   No remap is needed, because the encoder's z-mirror already produced exactly
   glTF's right-handed y-up convention. glTF x is east, y is up, z is *south*.
3. **Tile space (3D Tiles)** -- the runtime rotates glTF y-up to z-up, giving
   ENU: ``(east, north, up) = (x, -z, y)``. Bounding volumes are declared here.
4. **ECEF** -- the root tile's ``transform`` is the ENU-to-ECEF matrix at the
   dataset origin.

Only the root carries a ``transform``, so every tile's content and bounding
volume is in dataset-local ENU metres. At a 7 km site that costs about 0.4 mm
of float32 precision, which is irrelevant next to a 1 m voxel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .voxelize import Georeference


def enu_to_ecef_matrix(georeference: Georeference) -> list[float]:
    """Column-major 4x4 placing dataset-local ENU metres onto the WGS84 globe.

    The vertical datum is assumed ellipsoidal. If the source LAS uses
    orthometric height (very common for surveyed data) the whole tileset sits
    off by the local geoid separation -- tens of metres in Central America. That
    needs the LAS vertical CRS to resolve properly; it is flagged rather than
    silently corrected.
    """
    from pyproj import Transformer

    epsg = georeference.epsg
    if epsg is None:
        raise ValueError(
            "The source LAS declares no CRS, so the tileset cannot be georeferenced. "
            "Re-run with --epsg to supply one explicitly."
        )
    e, n, h = georeference.origin

    to_geog = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4979", always_xy=True)
    lon, lat, height = to_geog.transform(e, n, h)
    to_ecef = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4978", always_xy=True)
    x0, y0, z0 = to_ecef.transform(e, n, h)

    lam, phi = math.radians(lon), math.radians(lat)
    sl, cl = math.sin(lam), math.cos(lam)
    sp, cp = math.sin(phi), math.cos(phi)

    # Columns: east, north, up, origin.
    return [
        -sl, cl, 0.0, 0.0,
        -sp * cl, -sp * sl, cp, 0.0,
        cp * cl, cp * sl, sp, 0.0,
        x0, y0, z0, 1.0,
    ]


def origin_lonlat(georeference: Georeference) -> tuple[float, float, float]:
    from pyproj import Transformer

    t = Transformer.from_crs(f"EPSG:{georeference.epsg}", "EPSG:4979", always_xy=True)
    return t.transform(*georeference.origin)


@dataclass
class TileMeta:
    """One tile of the quadtree -- metadata only, no cell data.

    Cell data (index/rgb/faces) is consumed and written as a `.glb` at the
    moment `pyramid.pyramid_stream` produces each tile's `TileBlock`; only
    what `build_tree`/`bounding_box` need to assemble `tileset.json` survives
    afterwards.
    """

    level: int          # 0 = root (coarsest)
    tx: int
    tz: int
    cell_size: float
    #: Cell-index range in this level's own grid (full brick square, not a tight fit).
    x0: int
    x1: int
    z0: int
    z1: int
    #: Vertical extent over this tile's *visible* (surface) cells only.
    y_min: int
    y_max: int

    @property
    def key(self) -> str:
        return f"L{self.level}_{self.tx}_{self.tz}"

    def bounding_box(self, y_pad: float = 0.0) -> list[float]:
        """3D Tiles `box`, in dataset-local ENU (east, north, up)."""
        cs = self.cell_size
        e0, e1 = self.x0 * cs, self.x1 * cs
        # ENU north = -gz, so the cell range [z0, z1) maps to [-z1, -z0].
        n0, n1 = -self.z1 * cs, -self.z0 * cs
        u0 = float(self.y_min) * cs - y_pad
        u1 = (float(self.y_max) + 1.0) * cs + y_pad
        return [
            (e0 + e1) / 2, (n0 + n1) / 2, (u0 + u1) / 2,
            (e1 - e0) / 2, 0.0, 0.0,
            0.0, (n1 - n0) / 2, 0.0,
            0.0, 0.0, (u1 - u0) / 2,
        ]


def pyramid_depth(dims: tuple[int, int, int], brick: int) -> int:
    """How many quadtree levels are needed to cover the footprint."""
    extent = max(dims[0], dims[2])
    depth = 0
    while brick * (2 ** depth) < extent:
        depth += 1
    return depth


def build_tree(tiles_by_level: dict[int, list[TileMeta]], depth: int, content_uri) -> dict:
    """Assemble tileset.json. Parent/child links come from quadtree arithmetic."""
    by_key: dict[int, dict[tuple[int, int], TileMeta]] = {
        level: {(t.tx, t.tz): t for t in tiles} for level, tiles in tiles_by_level.items()
    }

    def node(tile: TileMeta) -> dict:
        is_leaf = tile.level == depth
        entry: dict = {
            "boundingVolume": {"box": tile.bounding_box()},
            "geometricError": 0.0 if is_leaf else tile.cell_size,
            "refine": "REPLACE",
            "content": {"uri": content_uri(tile)},
        }
        if not is_leaf:
            children = []
            for dx in (0, 1):
                for dz in (0, 1):
                    child = by_key.get(tile.level + 1, {}).get((tile.tx * 2 + dx, tile.tz * 2 + dz))
                    if child is not None:
                        children.append(node(child))
            if children:
                entry["children"] = children
        return entry

    roots = tiles_by_level.get(0, [])
    if len(roots) == 1:
        root = node(roots[0])
    else:
        # Shouldn't happen (level 0 is sized to hold everything) but stay safe.
        boxes = [t.bounding_box() for t in roots]
        root = {
            "boundingVolume": {"box": _union_box(boxes)},
            "geometricError": roots[0].cell_size * 2,
            "refine": "REPLACE",
            "children": [node(t) for t in roots],
        }
    return root


def _union_box(boxes: list[list[float]]) -> list[float]:
    lo = np.array([min(b[0] - b[3] for b in boxes), min(b[1] - b[7] for b in boxes), min(b[2] - b[11] for b in boxes)])
    hi = np.array([max(b[0] + b[3] for b in boxes), max(b[1] + b[7] for b in boxes), max(b[2] + b[11] for b in boxes)])
    c, h = (lo + hi) / 2, (hi - lo) / 2
    return [c[0], c[1], c[2], h[0], 0, 0, 0, h[1], 0, 0, 0, h[2]]
