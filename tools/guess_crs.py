"""Infer the CRS of a LAS/LAZ file that declares none.

    python tools/guess_crs.py cloud.las
    python tools/guess_crs.py cloud.las --near -89.2,17.3

What is and is not recoverable
-----------------------------
**Latitude is recoverable.** For a transverse-Mercator grid like UTM, northing
maps to latitude almost independently of which zone you assume, so a northing
pins the site to a narrow band of latitude straight away.

**Longitude is not.** Easting is measured from each zone's *own* central
meridian, so E=269,000 means "231 km west of my central meridian" -- a
perfectly valid position in all 60 zones. Every zone yields a self-consistent
longitude inside its own area of use, so no test on the coordinates alone can
choose between them. Anyone claiming otherwise has smuggled in an assumption.

So this script works in two modes:

`blind` (no --near)
    Classifies the coordinate family, recovers latitude, and lists the UTM
    candidates it cannot distinguish between. Ends with a short list to check.

`located` (--near LON,LAT)
    Inverts the problem. Instead of guessing the CRS from the numbers, it
    projects your approximate location through **every projected CRS in the
    PROJ database** and keeps the ones that reproduce the file's actual
    coordinates. A hint good to +/-50 km is plenty, and this usually returns a
    single answer.

Before running any of this, check for the boring answers: a `.prj` sidecar, the
header's `generating_software`, or the folder name. This script reports those
first because they are right more often than any inference.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import laspy
import numpy as np

WGS84_A = 6378137.0
SIDECARS = (".prj", ".wkt", ".xml", ".aux.xml", ".qmd", ".cpg", ".txt")


# ---------------------------------------------------------------------------
# 1. The boring answers, checked first
# ---------------------------------------------------------------------------

def report_metadata(path: Path, header) -> None:
    print("METADATA — check these before trusting any inference")
    print("-" * 62)

    found = False
    for suffix in SIDECARS:
        for candidate in (path.with_suffix(suffix), Path(str(path) + suffix)):
            if candidate.exists():
                text = candidate.read_text(errors="replace").strip()
                print(f"  sidecar {candidate.name}:")
                print("    " + text[:400].replace("\n", "\n    "))
                found = True
    if not found:
        print(f"  no sidecar found ({', '.join(SIDECARS)})")

    for field in ("system_identifier", "generating_software"):
        value = str(getattr(header, field, "") or "").strip()
        print(f"  {field:22s} {value or '(empty)'}")
    print(f"  {'global_encoding.wkt':22s} {bool(getattr(header.global_encoding, 'wkt', False))}")
    print(f"  {'version':22s} {header.version}")
    vlrs = [f"{v.user_id}/{v.record_id}" for v in header.vlrs]
    print(f"  {'vlrs':22s} {', '.join(vlrs) if vlrs else '(none)'}")
    print(f"  {'folder':22s} {path.parent.name}")
    print()


# ---------------------------------------------------------------------------
# 2. Sampling
# ---------------------------------------------------------------------------

def sample_points(path: Path, count: int, stops: int = 40) -> np.ndarray:
    """A few points spread across the file, without reading the whole thing.

    Reading only the head of a LAS is misleading -- files are usually written in
    acquisition or tile order, so the first million points may cover a corner of
    the extent. But striding through every chunk means reading the entire file,
    which is hopeless at hundreds of millions of points.

    So this seeks to `stops` evenly spaced positions and reads a short block at
    each: total I/O is proportional to `count`, not to the file size. Falls back
    to a strided head read if the backend cannot seek (some LAZ layouts).
    """
    per_stop = max(1, count // stops)
    with laspy.open(str(path)) as reader:
        total = reader.header.point_count
        if total == 0:
            return np.zeros((0, 3))
        try:
            rows = []
            for i in range(stops):
                offset = min(int(i * total / stops), max(total - per_stop, 0))
                reader.seek(offset)
                block = reader.read_points(min(per_stop, total - offset))
                if len(block) == 0:
                    continue
                rows.append(
                    np.stack(
                        [np.asarray(block.x), np.asarray(block.y), np.asarray(block.z)],
                        axis=1,
                    )
                )
            if rows:
                return np.concatenate(rows)[:count]
        except Exception:
            pass  # fall through to the sequential path

    with laspy.open(str(path)) as reader:
        chunk = next(reader.chunk_iterator(min(count * 50, 2_000_000)))
        xyz = np.stack(
            [np.asarray(chunk.x), np.asarray(chunk.y), np.asarray(chunk.z)], axis=1
        )
        step = max(1, len(xyz) // count)
        print("  (note: seek unavailable — sampled the head of the file only)")
        return xyz[::step][:count]


# ---------------------------------------------------------------------------
# 3. Classify the coordinate family
# ---------------------------------------------------------------------------

def classify(mins, maxs) -> list[str]:
    x0, y0, z0 = mins
    x1, y1, z1 = maxs
    notes = []

    if -180 <= x0 and x1 <= 180 and -90 <= y0 and y1 <= 90:
        notes.append("Geographic degrees (lon/lat) — likely EPSG:4326 or similar.")
        return notes

    if abs(x0) <= 20_037_509 and abs(y0) <= 20_048_967 and max(abs(x1), abs(y1)) > 1_000_000:
        notes.append(
            "Could be Web Mercator (EPSG:3857): its bounds are +/-20,037,508 m."
        )

    if 100_000 <= x0 and x1 <= 1_000_000:
        notes.append(
            "Easting in [100k, 1M] — consistent with a transverse-Mercator grid "
            "(UTM has a 500,000 m false easting)."
        )
        offset_km = (0.5 * (x0 + x1) - 500_000) / 1000.0
        notes.append(f"  centre sits {offset_km:+.1f} km from the central meridian.")
    if 0 <= y0 and y1 <= 10_000_000:
        notes.append("Northing in [0, 10M] — consistent with UTM (north or south).")

    span_x, span_y, span_z = x1 - x0, y1 - y0, z1 - z0
    notes.append(
        f"Extent {span_x:,.0f} x {span_y:,.0f} x {span_z:,.0f} units — "
        f"{'plausible as metres' if span_z < 9000 else 'check units'}."
    )
    if span_z > 0 and 2.5 < (span_z / max(span_x, 1e-9)) * 1000 < 4.0:
        notes.append("  (a ~3.28 ratio anywhere here would hint at feet, not metres)")
    return notes


# ---------------------------------------------------------------------------
# 4. Blind mode: recover latitude, enumerate the zones we cannot separate
# ---------------------------------------------------------------------------

def latitude_from_northing(northing: float, southern: bool) -> float:
    """Exact latitude from a UTM northing, independent of the zone.

    Latitude is what a northing genuinely determines, so it is worth getting
    right rather than approximating. Any zone will do: inverse-project the
    northing on that zone's own central meridian (easting 500,000, where the
    grid meets the meridian) and keep the latitude. The answer is the same to
    within rounding for every zone -- which is precisely why northing carries no
    information about *which* zone.
    """
    from pyproj import Transformer

    epsg = 32731 if southern else 32631  # zone 31, arbitrary
    lat = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform(
        500_000.0, northing
    )[1]
    return lat


def blind(mins, maxs) -> None:
    centre_e = 0.5 * (mins[0] + maxs[0])
    centre_n = 0.5 * (mins[1] + maxs[1])

    print("BLIND INFERENCE — what the numbers alone can tell you")
    print("-" * 62)
    lat_n = latitude_from_northing(centre_n, southern=False)
    lat_s = latitude_from_northing(centre_n, southern=True)
    print(f"  If a NORTHERN UTM zone: latitude ~ {lat_n:+.2f}")
    print(f"  If a SOUTHERN UTM zone: latitude ~ {lat_s:+.2f}")
    print()

    from pyproj import Transformer

    print("  Longitude depends on the zone, and every zone is self-consistent:")
    print(f"  {'zone':>6}  {'EPSG':>6}  {'lon':>9}  {'lat':>8}   in area of use")
    rows = []
    for zone in range(1, 61):
        for southern in (False, True):
            if southern and centre_n > 9_500_000:
                continue
            epsg = (32700 if southern else 32600) + zone
            try:
                t = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
                lon, lat = t.transform(centre_e, centre_n)
            except Exception:
                continue
            if not (math.isfinite(lon) and math.isfinite(lat)):
                continue
            band_lo, band_hi = -180 + 6 * (zone - 1), -174 + 6 * (zone - 1)
            inside = band_lo <= lon <= band_hi and (lat < 0) == southern
            rows.append((zone, epsg, lon, lat, inside))
    plausible = [r for r in rows if r[4]]
    for zone, epsg, lon, lat, inside in plausible[:12]:
        print(f"  {zone:>6}  {epsg:>6}  {lon:>9.4f}  {lat:>8.4f}   yes")
    print(f"\n  {len(plausible)} of {len(rows)} zone hypotheses are internally consistent.")
    print("  That is the whole point: the coordinates cannot choose between them.")
    print("  Latitude above is reliable. To pin longitude, re-run with --near.")
    print()


# ---------------------------------------------------------------------------
# 5. Located mode: search the PROJ database for CRSs that fit
# ---------------------------------------------------------------------------

def located(mins, maxs, hint_lon: float, hint_lat: float, tolerance_km: float, limit: int) -> None:
    from pyproj import CRS, Transformer
    from pyproj.database import query_crs_info

    centre_e = 0.5 * (mins[0] + maxs[0])
    centre_n = 0.5 * (mins[1] + maxs[1])

    print(f"LOCATED SEARCH — projecting ({hint_lon}, {hint_lat}) through every projected CRS")
    print("-" * 62)

    candidates = query_crs_info(pj_types="PROJECTED_CRS", allow_deprecated=False)
    print(f"  scanning {len(candidates):,} projected CRS definitions...")

    hits = []
    for info in candidates:
        area = info.area_of_use
        if area and not (
            area.west <= hint_lon <= area.east and area.south <= hint_lat <= area.north
        ):
            continue  # the hint isn't in this CRS's declared area of use
        code = f"{info.auth_name}:{info.code}"
        try:
            t = Transformer.from_crs("EPSG:4326", code, always_xy=True)
            east, north = t.transform(hint_lon, hint_lat)
        except Exception:
            continue
        if not (math.isfinite(east) and math.isfinite(north)):
            continue
        error_km = math.hypot(east - centre_e, north - centre_n) / 1000.0
        if error_km <= tolerance_km:
            hits.append((error_km, code, info.name))

    hits.sort()
    if not hits:
        print(f"\n  No CRS reproduces the file's coordinates within {tolerance_km:g} km.")
        print("  Either the hint is wrong, the units are not metres (try feet), or")
        print("  the grid is a local/engineering system not in the EPSG registry.")
        return

    print(f"\n  {len(hits)} CRS(s) place your hint within {tolerance_km:g} km of the file's centre:\n")
    print(f"  {'off by':>9}  {'code':<14} name")
    for error_km, code, name in hits[:limit]:
        print(f"  {error_km:>7.2f}km  {code:<14} {name}")

    best_km, best_code, best_name = hits[0]
    print(f"\n  Best fit: {best_code} — {best_name}")

    t = Transformer.from_crs(best_code, "EPSG:4326", always_xy=True)
    lon0, lat0 = t.transform(mins[0], mins[1])
    lon1, lat1 = t.transform(maxs[0], maxs[1])
    print(f"  Under it, the file covers lon {min(lon0,lon1):.5f}..{max(lon0,lon1):.5f}, "
          f"lat {min(lat0,lat1):.5f}..{max(lat0,lat1):.5f}")
    print(f"  Elevation range in the file: {mins[2]:.1f}..{maxs[2]:.1f}")
    print()
    print("  CONFIRM THIS VISUALLY before committing — it takes five seconds:")
    print(f"    https://www.google.com/maps/@{(min(lat0,lat1)+max(lat0,lat1))/2:.5f},"
          f"{(min(lon0,lon1)+max(lon0,lon1))/2:.5f},14z")
    print("  Does the terrain match the point cloud, and is the elevation")
    print("  plausible for that spot? If not, the hint or the units are wrong.")
    if len(hits) > 1 and hits[1][0] - best_km < 1.0:
        print("\n  NOTE: the runner-up fits nearly as well. These are usually the same")
        print("  projection on different datums (WGS84 / NAD83 / ITRF / a local realisation).")
        print("  They differ by metres, not kilometres, and coordinates cannot tell them")
        print("  apart -- that has to come from the survey record.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Infer the CRS of a LAS/LAZ file that declares none.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Latitude is recoverable from the coordinates; longitude is not.\n"
               "Use --near with an approximate location to resolve it.",
    )
    p.add_argument("las_path", type=Path)
    p.add_argument("--near", metavar="LON,LAT",
                   help="Approximate location, e.g. --near=-89.2,17.3. "
                        "Accurate to +/-50 km is plenty")
    p.add_argument("--tolerance", type=float, default=60.0, metavar="KM",
                   help="How far a candidate may place the hint from the file centre (default: 60)")
    p.add_argument("--samples", type=int, default=2000, help="Points to sample (default: 2000)")
    p.add_argument("--limit", type=int, default=15, help="Candidates to list (default: 15)")
    # `--near -89.2,17.3` looks like an option to argparse because of the minus
    # sign, and negative longitude covers the whole of the Americas. Accept the
    # separated form by rewriting it to `--near=...` before parsing.
    raw = list(sys.argv[1:] if argv is None else argv)
    for i, token in enumerate(raw[:-1]):
        if token == "--near":
            raw[i : i + 2] = [f"--near={raw[i + 1]}"]
            break
    args = p.parse_args(raw)

    if not args.las_path.exists():
        print(f"error: not found: {args.las_path}", file=sys.stderr)
        return 1

    with laspy.open(str(args.las_path)) as reader:
        header = reader.header
    mins = np.asarray(header.mins, dtype=np.float64)
    maxs = np.asarray(header.maxs, dtype=np.float64)

    print(f"\n{args.las_path.name} — {header.point_count:,} points\n")
    report_metadata(args.las_path, header)

    # The header may claim a CRS even when a tool reported none.
    try:
        declared = header.parse_crs()
    except Exception:
        declared = None
    if declared is not None:
        print(f"NOTE: this file DOES declare a CRS: {declared.name} "
              f"(EPSG:{declared.to_epsg()}). Inference below is a cross-check only.\n")

    pts = sample_points(args.las_path, args.samples)
    smins, smaxs = pts.min(axis=0), pts.max(axis=0)
    print("SAMPLED BOUNDS")
    print("-" * 62)
    print(f"  from {len(pts):,} sampled points")
    for axis, lo, hi in zip("XYZ", smins, smaxs):
        print(f"  {axis}  {lo:>16,.3f}  ..  {hi:>16,.3f}")
    print()

    print("COORDINATE FAMILY")
    print("-" * 62)
    for note in classify(smins, smaxs):
        print(f"  {note}")
    print()

    if args.near:
        try:
            lon_s, lat_s = args.near.split(",")
            located(smins, smaxs, float(lon_s), float(lat_s), args.tolerance, args.limit)
        except ValueError:
            print("error: --near must look like -89.2,17.3", file=sys.stderr)
            return 1
    else:
        blind(smins, smaxs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
