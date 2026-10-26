"""Synthetic inputs for the test suite: one small LAS site and two GeoTIFFs.

Nothing here comes from a real dataset. Every coordinate and colour is
integer arithmetic on the cell index -- no random numbers, no trig -- so the
files are identical on every machine and every numpy version.

The site is 300 x 280 m of 1 m cells in EPSG:31983 at a made-up location
(E 500 000, N 7 500 000), one ground point per cell, plus:

- a 3 m step, a sloping plane and triangle-wave bumps (georeference, LOD)
- a flat "lake", a grey road strip and a red-roofed box building
- 55 trees: crown points with up to 4 returns per pulse over a trunk
- RGB stored as 8-bit values x 256, as in the production LAS

The categorical raster stops 20 m short of the east edge and has a nodata
hole, so nearest-pixel fill and cubes outside the raster both occur. The
continuous raster is in EPSG:4326, so the reprojection path is exercised.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

EPSG = 31983
E0 = 500_000.0
N0 = 7_500_000.0
NX, NY = 300, 280  # site size in metres (cells)

LAKE = (60, 210, 25)  # centre i, j and radius, in cells
BUILDING = (200, 230, 150, 170)  # i0, i1, j0, j1
ROAD = (120, 124)  # j0, j1
TREES = [(i, j) for i in range(20, 290, 25) for j in range(15, 140, 25)]
CROWN_R = 4
NODATA_CLASS = 255
NODATA_FLOAT = -9999.0


@dataclass
class Site:
    root: Path
    las: Path  # declares EPSG:31983
    las_no_crs: Path  # same points, no CRS VLR
    landcover: Path  # categorical, uint8, colour table, EPSG:31983
    elevation: Path  # continuous, float32, EPSG:4326
    config: Path  # --properties config with relative paths
    far_raster: Path  # categorical raster that does not overlap the site


def _ground_mm(i: np.ndarray, j: np.ndarray) -> np.ndarray:
    z = 760_000 + 50 * i + 30 * j + 120 * (np.abs(i % 60 - 30) + np.abs(j % 70 - 35))
    z = z + np.where((i >= 180) & (j < 100), 3_000, 0)
    li, lj, lr = LAKE
    return np.where((i - li) ** 2 + (j - lj) ** 2 <= lr * lr, 765_000, z)


def _points() -> dict[str, np.ndarray]:
    """All points as integer millimetre offsets from (E0, N0, 0) plus attributes."""
    cols: dict[str, list[np.ndarray]] = {k: [] for k in ("x", "y", "z", "rgb", "ret", "nret", "cls")}

    def add(i, j, z_mm, rgb, ret=1, nret=1, cls=1):
        i, j, z_mm = (np.asarray(a, dtype=np.int64).ravel() for a in (i, j, z_mm))
        n = len(i)
        # Jitter within the cell, from 0.1 to 0.9 m, so no point sits on a cell edge.
        cols["x"].append(i * 1000 + 100 + (i * 7 + j * 13) % 9 * 100)
        cols["y"].append(j * 1000 + 100 + (i * 11 + j * 5) % 9 * 100)
        cols["z"].append(z_mm)
        cols["rgb"].append(np.broadcast_to(np.asarray(rgb, dtype=np.int64), (n, 3)) if np.ndim(rgb) == 1
                           else np.asarray(rgb, dtype=np.int64).reshape(n, 3))
        cols["ret"].append(np.broadcast_to(np.int64(ret), (n,)))
        cols["nret"].append(np.broadcast_to(np.int64(nret), (n,)))
        cols["cls"].append(np.broadcast_to(np.int64(cls), (n,)))

    ii, jj = (a.ravel() for a in np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij"))
    bi0, bi1, bj0, bj1 = BUILDING
    in_building = (ii >= bi0) & (ii < bi1) & (jj >= bj0) & (jj < bj1)
    li, lj, lr = LAKE
    in_lake = (ii - li) ** 2 + (jj - lj) ** 2 <= lr * lr
    on_road = (jj >= ROAD[0]) & (jj < ROAD[1])

    # Ground: grass shades, exact grey road, exact blue lake.
    g = ~in_building
    grass = np.stack([90 + ii % 20, 140 + jj % 15, 60 + (ii + jj) % 10], axis=1)
    rgb = np.where(in_lake[:, None], [30, 80, 160], np.where(on_road[:, None], [128, 128, 128], grass))
    under_tree = np.zeros(len(ii), dtype=bool)
    for ti, tj in TREES:
        under_tree |= (ii - ti) ** 2 + (jj - tj) ** 2 <= CROWN_R * CROWN_R
    # Under a crown the ground is the last of four returns.
    for mask, ret, nret in ((g & ~under_tree, 1, 1), (g & under_tree, 4, 4)):
        add(ii[mask], jj[mask], _ground_mm(ii[mask], jj[mask]), rgb[mask], ret, nret, cls=2)

    # Building: flat red roof 12 m above its south-west corner, grey walls.
    roof_mm = int(_ground_mm(np.int64(bi0), np.int64(bj0))) + 12_000
    bi, bj = (a.ravel() for a in np.meshgrid(np.arange(bi0, bi1), np.arange(bj0, bj1), indexing="ij"))
    add(bi, bj, np.full(len(bi), roof_mm), [200, 40, 40])
    edge = (bi == bi0) | (bi == bi1 - 1) | (bj == bj0) | (bj == bj1 - 1)
    for k in range(12):
        add(bi[edge], bj[edge], _ground_mm(bi[edge], bj[edge]) + k * 1000, [180, 180, 170])

    # Trees: a dome of crown points, three returns per pulse above the ground return.
    for n_tree, (ti, tj) in enumerate(TREES):
        di, dj = (a.ravel() for a in np.meshgrid(np.arange(-CROWN_R, CROWN_R + 1), np.arange(-CROWN_R, CROWN_R + 1), indexing="ij"))
        keep = di * di + dj * dj <= CROWN_R * CROWN_R
        ci, cj = ti + di[keep], tj + dj[keep]
        base = _ground_mm(ci, cj)
        top = base + 14_000 + (n_tree % 5) * 1_000 - (di[keep] ** 2 + dj[keep] ** 2) * 400
        leaf = np.stack([40 + (ci + n_tree) % 30, 110 + (cj * 3) % 40, np.full_like(ci, 40 + n_tree % 20)], axis=1)
        for ret, drop in ((1, 0), (2, 2_000), (3, 6_000)):
            add(ci, cj, top - drop, leaf, ret, 4)
        trunk = np.arange(0, 8_000, 500)
        add(np.full(len(trunk), ti), np.full(len(trunk), tj), int(_ground_mm(np.int64(ti), np.int64(tj))) + trunk,
            [100, 70, 40])

    return {k: np.concatenate(v) for k, v in cols.items()}


def write_las(path: Path, epsg: int | None) -> None:
    import laspy

    p = _points()
    header = laspy.LasHeader(point_format=8, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([E0, N0, 0.0])
    header.creation_date = date(2026, 1, 1)
    header.system_identifier = "vox3dt test suite"
    header.generating_software = "tests/synth.py"
    if epsg is not None:
        from pyproj import CRS

        header.add_crs(CRS.from_epsg(epsg))
    las = laspy.LasData(header)
    n = len(p["x"])
    las.X = p["x"].astype(np.int32)
    las.Y = p["y"].astype(np.int32)
    las.Z = p["z"].astype(np.int32)
    las.red, las.green, las.blue = (p["rgb"][:, c].astype(np.uint16) * 256 for c in range(3))
    las.nir = np.zeros(n, dtype=np.uint16)
    las.return_number = p["ret"].astype(np.uint8)
    las.number_of_returns = p["nret"].astype(np.uint8)
    las.classification = p["cls"].astype(np.uint8)
    las.intensity = (np.arange(n) * 37 % 65536).astype(np.uint16)
    las.gps_time = np.arange(n, dtype=np.float64) * 0.001
    las.point_source_id = np.ones(n, dtype=np.uint16)
    las.write(str(path))


def _landcover_class(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Class at site-local metres (x east of E0, y north of N0)."""
    cls = np.full(x.shape, 4, dtype=np.uint8)  # grass
    for ti, tj in TREES:
        cls[(x - ti - 0.5) ** 2 + (y - tj - 0.5) ** 2 <= (CROWN_R + 1) ** 2] = 3  # forest
    bi0, bi1, bj0, bj1 = BUILDING
    cls[(x >= bi0) & (x < bi1) & (y >= bj0) & (y < bj1)] = 1  # building
    li, lj, lr = LAKE
    cls[(x - li - 0.5) ** 2 + (y - lj - 0.5) ** 2 <= lr * lr] = 0  # water
    return cls


def write_landcover(path: Path, offset: float = 0.0) -> None:
    """2 m categorical raster ending at E0 + 280, i.e. 20 m short of the site's east edge.

    ``offset`` shifts it that many metres north-east, off the site, for the
    no-overlap error test.
    """
    import rasterio
    from rasterio.transform import from_origin

    px = 2.0
    west, north = E0 - 10 + offset, N0 + NY + 10 + offset
    width, height = int((280 + 10) / px), int((NY + 20) / px)
    cx = west - E0 + (np.arange(width) + 0.5) * px
    cy = north - N0 - (np.arange(height) + 0.5) * px
    x, y = np.meshgrid(cx, cy)
    data = _landcover_class(x, y)
    data[20:30, 100:110] = NODATA_CLASS
    with rasterio.open(
        path, "w", driver="GTiff", width=width, height=height, count=1, dtype="uint8",
        crs=f"EPSG:{EPSG}", transform=from_origin(west, north, px, px), nodata=NODATA_CLASS,
    ) as dst:
        dst.write(data, 1)
        dst.write_colormap(1, {0: (0, 90, 200, 255), 1: (220, 50, 50, 255), 3: (20, 120, 40, 255),
                               4: (150, 200, 90, 255)})


def write_elevation(path: Path) -> None:
    """Float32 raster in EPSG:4326, about 5 m pixels, covering the site with a margin."""
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import from_origin

    to_geo = Transformer.from_crs(f"EPSG:{EPSG}", "EPSG:4326", always_xy=True)
    lon, lat = to_geo.transform([E0 - 20, E0 + NX + 20, E0 - 20, E0 + NX + 20],
                                [N0 - 20, N0 - 20, N0 + NY + 20, N0 + NY + 20])
    west, east, south, north = min(lon), max(lon), min(lat), max(lat)
    px = 5e-5  # degrees, about 5 m
    width, height = int(np.ceil((east - west) / px)), int(np.ceil((north - south) / px))
    rows, cols = np.mgrid[0:height, 0:width]
    data = (100.0 + 0.5 * cols + 0.25 * rows).astype(np.float32)
    data[10:14, 20:24] = NODATA_FLOAT
    with rasterio.open(
        path, "w", driver="GTiff", width=width, height=height, count=1, dtype="float32",
        crs="EPSG:4326", transform=from_origin(west, north, px, px), nodata=NODATA_FLOAT,
    ) as dst:
        dst.write(data, 1)


def build(root: Path) -> Site:
    root.mkdir(parents=True, exist_ok=True)
    site = Site(
        root=root,
        las=root / "site.las",
        las_no_crs=root / "site_no_crs.las",
        landcover=root / "landcover.tif",
        elevation=root / "elevation.tif",
        config=root / "props.json",
        far_raster=root / "far.tif",
    )
    write_las(site.las, EPSG)
    write_las(site.las_no_crs, None)
    write_landcover(site.landcover)
    write_elevation(site.elevation)
    write_landcover(site.far_raster, offset=50_000)
    site.config.write_text(json.dumps({
        "properties": [
            {"name": "Classification", "file": "landcover.tif", "kind": "categorical",
             "labels": {"0": "Water", "1": "Building", "3": "Forest", "4": "Grass"},
             "colors": {"4": "#88cc44"}},
            {"name": "Elevation", "file": "elevation.tif", "kind": "continuous"},
        ]
    }, indent=2))
    return site
