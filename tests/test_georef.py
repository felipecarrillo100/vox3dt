"""Georeference against answers known from how the synthetic site is built.

synth.py jitters points 0.1-0.9 m into their cell, so the LAS bounds start at
(E0 + 0.1, N0 + 0.1) and the grid is 300 x 280 cells of 1 m. Grid cell
(0, 0, 0) is the *north*-west-bottom corner, so its northing is
N0 + 0.1 + 280 -- one voxel less is the "drawn one voxel south" bug.
"""

from __future__ import annotations

import numpy as np
import pytest
from pyproj import Transformer

import synth

WEST = synth.E0 + 0.1
SOUTH = synth.N0 + 0.1
NX, NZ = 300, 280


@pytest.fixture(scope="module")
def draco(results, site):
    import laspy

    r = results("draco")
    bottom = float(laspy.open(str(site.las)).header.mins[2])
    return r.manifest, bottom


def test_grid_origin_is_north_west_bottom_corner(draco):
    m, bottom = draco
    o = m["specs"]["georeference"]["grid_origin_world"]
    assert o["easting"] == pytest.approx(WEST, abs=1e-6)
    assert o["northing"] == pytest.approx(SOUTH + NZ, abs=1e-6)
    assert o["height"] == pytest.approx(bottom, abs=1e-6)


def test_origin_lonlat(draco):
    m, bottom = draco
    lon, lat, h = Transformer.from_crs("EPSG:31983", "EPSG:4979", always_xy=True).transform(WEST, SOUTH + NZ, bottom)
    assert m["report"]["origin_lonlat"] == pytest.approx([lon, lat, h], abs=1e-9)


@pytest.mark.parametrize("u, v, w, east, north", [
    (0, 0, 0, WEST, SOUTH + NZ),  # north-west
    (0, -NZ, 0, WEST, SOUTH),  # south-west: on the LAS min northing, not a voxel south of it
    (NX, -NZ, 0, WEST + NX, SOUTH),  # south-east
    (NX, 0, 30, WEST + NX, SOUTH + NZ),  # north-east, 30 m up
])
def test_root_transform_places_corners(draco, u, v, w, east, north):
    m, bottom = draco
    matrix = np.array(m["tileset"]["root"]["transform"]).reshape(4, 4).T
    x, y, z = (matrix @ [u, v, w, 1.0])[:3]
    e, n, h = Transformer.from_crs("EPSG:4978", "EPSG:31983", always_xy=True).transform(x, y, z)
    assert e == pytest.approx(east, abs=0.01)
    assert n == pytest.approx(north, abs=0.01)
    assert h == pytest.approx(bottom + w, abs=0.01)


def test_placement_residual_is_small(draco):
    m, _ = draco
    residual = m["specs"]["georeference"]["placement_residual"]
    assert residual["max_horizontal_m"] < 0.01
    assert residual["max_vertical_m"] < 0.01
