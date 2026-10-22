"""Per-cube properties sampled from GeoTIFFs (``--property`` / ``--properties``).

Each property is one raster band mapped to one property-table column. A raster is
2D, so every cube in a vertical column gets the same value; what varies per
level is how much ground one cube covers.

Sampling
--------
A cube smaller than a raster pixel reads the pixel under its centre. A cube at
least one pixel wide summarises the valid pixels whose centres fall inside its
footprint -- the **majority** code for a categorical property, the **mean** for
a continuous one. Footprint sums come from summed-area tables (one per class
for categorical rasters), so every cube costs O(1) no matter how much ground it
covers, and nothing has to be carried through the pyramid's sort.

This counts valid *pixels* under a coarse cube rather than occupied voxel
columns. On the Vale site the raster's valid mask and the cube footprint agree
on 98.7% of pixels, so the two are effectively the same, and this way the six
classification years cost nothing in the external sort.

Nodata
------
Every property-table row needs a value -- a binary column has no "missing". Nodata
pixels are filled from the nearest valid pixel (8-neighbour growth, so distance
is Chebyshev, in pixels), and cubes outside the raster read its nearest edge
pixel. The number of cubes that landed on a filled pixel, and the largest fill
distance used, are reported per property.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: numpy dtype -> (EXT_structural_metadata componentType, numpy storage dtype)
_COMPONENT = {
    "uint8": ("UINT8", np.uint8),
    "int8": ("INT8", np.int8),
    "uint16": ("UINT16", np.uint16),
    "int16": ("INT16", np.int16),
    "uint32": ("UINT32", np.uint32),
    "int32": ("INT32", np.int32),
    "float32": ("FLOAT32", np.float32),
    "float64": ("FLOAT32", np.float32),
}

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
RESERVED = {"FeatureID"}


@dataclass
class PropertySpec:
    name: str
    file: Path
    band: int = 1
    kind: str | None = None  # categorical | continuous; None = detect
    labels: dict[str, str] = field(default_factory=dict)
    colors: dict[str, str] = field(default_factory=dict)
    nodata: str = "nearest"


def parse_property_arg(text: str) -> PropertySpec:
    """``NAME=PATH`` from the command line."""
    name, sep, path = text.partition("=")
    if not sep or not name or not path:
        raise ValueError(f"--property expects NAME=PATH, got {text!r}")
    return PropertySpec(name=name.strip(), file=Path(path.strip()).expanduser())


def load_config(path: Path) -> list[PropertySpec]:
    """``--properties config.json``. Relative paths resolve against the config's folder."""
    data = json.loads(path.read_text())
    defaults = data.get("defaults", {})
    specs = []
    for entry in data.get("properties", []):
        merged = {**defaults, **entry}
        unknown = set(merged) - {"name", "file", "band", "kind", "labels", "colors", "nodata"}
        if unknown:
            raise ValueError(f"{path.name}: unsupported property keys {sorted(unknown)}")
        if "name" not in merged or "file" not in merged:
            raise ValueError(f"{path.name}: every property needs 'name' and 'file'")
        file = Path(merged["file"]).expanduser()
        if not file.is_absolute():
            file = (path.parent / file).resolve()
        specs.append(PropertySpec(
            name=merged["name"], file=file, band=int(merged.get("band", 1)),
            kind=merged.get("kind"),
            labels={str(k): v for k, v in merged.get("labels", {}).items()},
            colors={str(k): v for k, v in merged.get("colors", {}).items()},
            nodata=merged.get("nodata", "nearest"),
        ))
    return specs


def validate_specs(specs: list[PropertySpec]) -> None:
    seen = set()
    for s in specs:
        if not _NAME_RE.match(s.name):
            raise ValueError(f"property name {s.name!r} must be a plain identifier")
        if s.name in RESERVED:
            raise ValueError(f"property name {s.name!r} is reserved")
        if s.name in seen:
            raise ValueError(f"property name {s.name!r} given twice")
        seen.add(s.name)
        if s.kind not in (None, "categorical", "continuous"):
            raise ValueError(f"{s.name}: kind must be 'categorical' or 'continuous'")
        if s.nodata != "nearest":
            raise ValueError(f"{s.name}: only nodata 'nearest' is supported")
        if not s.file.exists():
            raise ValueError(f"{s.name}: raster not found: {s.file}")


def _sat(a: np.ndarray) -> np.ndarray:
    """Summed-area table with a zero row/column in front: S[r, c] = sum a[:r, :c]."""
    out = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float64 if a.dtype.kind == "f" else np.int64)
    np.cumsum(np.cumsum(a, axis=0), axis=1, out=out[1:, 1:])
    return out


def _rect(sat: np.ndarray, r0, r1, c0, c1) -> np.ndarray:
    """Sum over rows [r0, r1) and columns [c0, c1), vectorized."""
    return sat[r1, c1] - sat[r0, c1] - sat[r1, c0] + sat[r0, c0]


def _fill_nearest(values: np.ndarray, valid: np.ndarray, categorical: bool):
    """Grow valid pixels into nodata ones, one 8-neighbour ring per pass."""
    filled = values.astype(np.float64).copy()
    have = valid.copy()
    dist = np.zeros(values.shape, dtype=np.int32)
    step = 0
    if not have.any():
        raise ValueError("raster has no valid pixels")
    while not have.all():
        step += 1
        acc = np.zeros_like(filled)
        cnt = np.zeros(values.shape, dtype=np.int32)
        first = np.full(values.shape, np.nan)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                src_v = np.roll(np.roll(filled, dr, 0), dc, 1)
                src_h = np.roll(np.roll(have, dr, 0), dc, 1)
                # np.roll wraps; mask the wrapped edge out.
                if dr == 1:
                    src_h[0, :] = False
                if dr == -1:
                    src_h[-1, :] = False
                if dc == 1:
                    src_h[:, 0] = False
                if dc == -1:
                    src_h[:, -1] = False
                take = src_h & ~have
                acc[take] += src_v[take]
                cnt[take] += 1
                first_unset = take & np.isnan(first)
                first[first_unset] = src_v[first_unset]
        grow = cnt > 0
        filled[grow] = first[grow] if categorical else acc[grow] / cnt[grow]
        dist[grow] = step
        have |= grow
    return filled, dist


class RasterProperty:
    """One opened, preprocessed raster ready to answer per-cube lookups."""

    def __init__(self, spec: PropertySpec, las_epsg: int | None):
        try:
            import rasterio
        except ImportError as exc:  # pragma: no cover - depends on install
            raise RuntimeError(
                "--property needs rasterio: pip install -e \".[raster]\""
            ) from exc
        from pyproj import CRS, Transformer

        self.spec = spec
        self.name = spec.name
        with rasterio.open(spec.file) as src:
            if spec.band < 1 or spec.band > src.count:
                raise ValueError(f"{spec.name}: band {spec.band} not in 1..{src.count}")
            arr = src.read(spec.band)
            self.nodata_value = src.nodatavals[spec.band - 1]
            if src.crs is None:
                raise ValueError(f"{spec.name}: raster declares no CRS")
            self.crs = CRS.from_user_input(src.crs.to_wkt())
            t = src.transform
            if abs(t.b) > 1e-12 or abs(t.d) > 1e-12:
                raise ValueError(f"{spec.name}: rotated rasters are not supported")
            self.a, self.c, self.e, self.f = t.a, t.c, t.e, t.f
            try:
                cmap = src.colormap(spec.band)
            except ValueError:
                cmap = None
            self.raster_dtype = str(arr.dtype)

        self.height, self.width = arr.shape
        self.pixel_size = (abs(self.a), abs(self.e))
        epsg = self.crs.to_epsg()
        self.crs_label = f"EPSG:{epsg}" if epsg else self.crs.name

        valid = np.ones(arr.shape, dtype=bool)
        if self.nodata_value is not None:
            valid &= arr != self.nodata_value
        if arr.dtype.kind == "f":
            valid &= np.isfinite(arr)
        self.valid = valid
        self.valid_fraction = float(valid.mean())

        self.kind = spec.kind or ("categorical" if cmap is not None else "continuous")
        self.categorical = self.kind == "categorical"
        if self.categorical and arr.dtype.kind == "f":
            raise ValueError(f"{spec.name}: categorical property on a floating-point raster")
        self.component_type, self.out_dtype = _COMPONENT[self.raster_dtype]

        filled, dist = _fill_nearest(arr, valid, self.categorical)
        self.filled = filled.astype(self.out_dtype) if self.categorical else filled
        self.fill_dist = dist

        if self.categorical:
            self.codes = np.unique(arr[valid]).astype(np.int64)
            self.class_sat = np.stack([_sat((arr == k) & valid) for k in self.codes])
        else:
            self.codes = None
            self.value_sat = _sat(np.where(valid, arr, 0).astype(np.float64))
            self.count_sat = _sat(valid.astype(np.int64))

        self.palette = {}
        if cmap is not None and self.categorical:
            for k in self.codes:
                rgba = cmap.get(int(k))
                if rgba is not None:
                    self.palette[int(k)] = [int(rgba[0]), int(rgba[1]), int(rgba[2])]

        self._to_raster = None
        if las_epsg is not None:
            las_crs = CRS.from_epsg(las_epsg)
            if not las_crs.equals(self.crs):
                self._to_raster = Transformer.from_crs(las_crs, self.crs, always_xy=True)

        self.stats_levels: dict[int, dict[int, int]] = {}
        self.cubes_filled = 0
        self.cubes_outside = 0
        self.max_fill_used = 0
        self.vmin = None
        self.vmax = None
        self.vsum = 0.0
        self.vcount = 0

    # ------------------------------------------------------------------
    def coverage(self, east_range, north_range) -> float:
        """Fraction of an (easting, northing) box, in LAS CRS, the raster covers."""
        e0, e1 = east_range
        n0, n1 = north_range
        xs = np.array([e0, e1, e0, e1])
        ys = np.array([n0, n0, n1, n1])
        if self._to_raster is not None:
            xs, ys = self._to_raster.transform(xs, ys)
        rx0, rx1 = sorted((self.c, self.c + self.a * self.width))
        ry0, ry1 = sorted((self.f, self.f + self.e * self.height))
        bx0, bx1, by0, by1 = xs.min(), xs.max(), ys.min(), ys.max()
        ix = max(0.0, min(bx1, rx1) - max(bx0, rx0))
        iy = max(0.0, min(by1, ry1) - max(by0, ry0))
        area = max((bx1 - bx0) * (by1 - by0), 1e-12)
        return float(ix * iy / area)

    def sample(self, east: np.ndarray, north: np.ndarray, cell: float, level: int) -> np.ndarray:
        """One value per cube. `east`/`north` are cube centres in the LAS CRS."""
        east = np.asarray(east, dtype=np.float64)
        north = np.asarray(north, dtype=np.float64)
        half = cell / 2.0
        if self._to_raster is None:
            x, y = east, north
            hx = np.full(len(east), half)
            hy = np.full(len(east), half)
        else:
            # Footprint half-widths measured in raster units, so a raster in
            # degrees (or any other unit) is handled like one in metres.
            x, y = (np.asarray(v, dtype=np.float64) for v in self._to_raster.transform(east, north))
            xe, ye = self._to_raster.transform(east + half, north)
            xn, yn = self._to_raster.transform(east, north + half)
            hx = np.hypot(np.asarray(xe) - x, np.asarray(xn) - x)
            hy = np.hypot(np.asarray(ye) - y, np.asarray(yn) - y)
        colf = (x - self.c) / self.a
        rowf = (y - self.f) / self.e
        # Footprint half-widths in pixels.
        hc = hx / abs(self.a)
        hr = hy / abs(self.e)

        ci = np.floor(colf).astype(np.int64)
        ri = np.floor(rowf).astype(np.int64)
        outside = (ci < 0) | (ci >= self.width) | (ri < 0) | (ri >= self.height)
        ci = np.clip(ci, 0, self.width - 1)
        ri = np.clip(ri, 0, self.height - 1)
        centre = self.filled[ri, ci]
        centre_dist = self.fill_dist[ri, ci]

        self.cubes_outside += int(outside.sum())
        filled_mask = (centre_dist > 0) | outside
        self.cubes_filled += int(filled_mask.sum())
        if len(centre_dist):
            self.max_fill_used = max(self.max_fill_used, int(centre_dist.max()))

        wide = (2 * hc >= 1.0) & (2 * hr >= 1.0)
        if not wide.any():
            values = centre
        else:
            # Pixels whose centres (j + 0.5) fall in the cube's footprint.
            c0 = np.clip(np.ceil(colf - hc - 0.5), 0, self.width).astype(np.int64)
            c1 = np.clip(np.ceil(colf + hc - 0.5), 0, self.width).astype(np.int64)
            r0 = np.clip(np.ceil(rowf - hr - 0.5), 0, self.height).astype(np.int64)
            r1 = np.clip(np.ceil(rowf + hr - 0.5), 0, self.height).astype(np.int64)
            c1 = np.maximum(c1, c0)
            r1 = np.maximum(r1, r0)
            if self.categorical:
                counts = np.stack([_rect(s, r0, r1, c0, c1) for s in self.class_sat], axis=1)
                has = counts.sum(axis=1) > 0
                # argmax takes the first maximum: ties go to the lowest code.
                majority = self.codes[np.argmax(counts, axis=1)].astype(self.out_dtype)
                values = np.where(has & wide, majority, centre)
            else:
                total = _rect(self.value_sat, r0, r1, c0, c1)
                n = _rect(self.count_sat, r0, r1, c0, c1)
                values = np.where((n > 0) & wide, total / np.maximum(n, 1), centre)

        if not self.categorical:
            if np.dtype(self.out_dtype).kind in "iu":
                info = np.iinfo(self.out_dtype)
                values = np.clip(np.rint(values), info.min, info.max)
            values = values.astype(self.out_dtype)
        else:
            values = values.astype(self.out_dtype)

        if len(values):
            vmin, vmax = values.min().item(), values.max().item()
            self.vmin = vmin if self.vmin is None else min(self.vmin, vmin)
            self.vmax = vmax if self.vmax is None else max(self.vmax, vmax)
            self.vsum += float(values.astype(np.float64).sum())
            self.vcount += len(values)
            if self.categorical:
                per = self.stats_levels.setdefault(level, {})
                k, n = np.unique(values, return_counts=True)
                for code, cnt in zip(k.tolist(), n.tolist()):
                    per[int(code)] = per.get(int(code), 0) + int(cnt)
        return values

    # ------------------------------------------------------------------
    def describe_source(self) -> dict:
        return {
            "file": str(self.spec.file),
            "band": self.spec.band,
            "crs": self.crs_label,
            "pixel_size": [round(self.pixel_size[0], 6), round(self.pixel_size[1], 6)],
            "size": [self.width, self.height],
            "nodata": self.nodata_value,
            "valid_fraction": round(self.valid_fraction, 4),
            "dtype": self.raster_dtype,
        }

    def summary_line(self, coverage: float) -> str:
        extra = f"codes {','.join(str(int(k)) for k in self.codes)}" if self.categorical else "continuous"
        return (
            f"  {self.name}: {self.spec.file.name} band {self.spec.band}, {self.crs_label}, "
            f"{self.pixel_size[0]:.6g} x {self.pixel_size[1]:.6g} pixels, covers {100 * coverage:.1f}% of LAS bounds, "
            f"{100 * self.valid_fraction:.1f}% valid, {self.kind}, {extra}"
        )

    def spec_entry(self) -> dict:
        entry = {
            "source": self.describe_source(),
            "kind": self.kind,
            "component_type": self.component_type,
            "aggregation": (
                f"cubes narrower than a pixel ({self.pixel_size[0]:.6g} x {self.pixel_size[1]:.6g} raster units) read the pixel under their "
                f"centre; larger cubes take the {'majority' if self.categorical else 'mean'} of the valid "
                "pixels whose centres lie in their footprint (ties: lowest code)"
            ),
            "nodata_fill": {
                "policy": "nearest valid pixel (8-neighbour growth, distance in pixels)",
                "cubes_on_filled_pixels": self.cubes_filled,
                "cubes_outside_raster": self.cubes_outside,
                "max_distance_px": self.max_fill_used,
            },
            "range": {"min": self.vmin, "max": self.vmax,
                      "mean": (self.vsum / self.vcount) if self.vcount else None},
        }
        if self.categorical:
            totals: dict[int, int] = {}
            for per in self.stats_levels.values():
                for k, n in per.items():
                    totals[k] = totals.get(k, 0) + n
            classes = {}
            for k in sorted(totals):
                rgb = self.palette.get(k)
                override = self.spec.colors.get(str(k))
                if override:
                    h = override.lstrip("#")
                    rgb = [int(h[i : i + 2], 16) for i in (0, 2, 4)]
                classes[str(k)] = {
                    "label": self.spec.labels.get(str(k)),
                    "rgb": rgb,
                    "hex": ("#%02x%02x%02x" % tuple(rgb)) if rgb else None,
                    "color_source": "config" if override else ("geotiff colour table" if rgb else None),
                    "cubes": totals[k],
                }
            entry["classes"] = classes
        return entry

    def report_entry(self) -> dict:
        return {
            "cubes_per_class_per_level": {str(lvl): {str(k): v for k, v in sorted(per.items())}
                                          for lvl, per in sorted(self.stats_levels.items())}
            if self.categorical else None,
            "cubes_on_filled_pixels": self.cubes_filled,
            "cubes_outside_raster": self.cubes_outside,
        }
