"""Fast, header-only LAS/LAZ inspection for `--info`.

Everything here comes from the LAS header and its VLRs -- no point record is
ever read -- so this stays fast regardless of file size, unlike the full
`vox3dt` conversion.
"""

from __future__ import annotations

from pathlib import Path

import laspy

from .voxelize import _read_crs


def describe(las_path: Path) -> dict:
    """Header-only metadata for `las_path`. Never opens the point-record iterator."""
    size_bytes = las_path.stat().st_size
    with laspy.open(str(las_path)) as reader:
        header = reader.header
        epsg, crs_name = _read_crs(header)
        dims_present = [d.name for d in header.point_format.dimensions]
        mins = [float(v) for v in header.mins]
        maxs = [float(v) for v in header.maxs]
        vlrs = [
            {
                "user_id": getattr(v, "user_id", None),
                "record_id": getattr(v, "record_id", None),
                "description": getattr(v, "description", None),
            }
            for v in header.vlrs
        ]

        info = {
            "file": str(las_path),
            "size_bytes": size_bytes,
            "compressed": las_path.suffix.lower() == ".laz",
            "las_version": f"{header.version.major}.{header.version.minor}",
            "point_format_id": header.point_format.id,
            "point_count": int(header.point_count),
            "dimensions": dims_present,
            "has_rgb": any(name in ("red", "green", "blue") for name in dims_present),
            "scales": [float(v) for v in header.scales],
            "offsets": [float(v) for v in header.offsets],
            "mins": mins,
            "maxs": maxs,
            "epsg": epsg,
            "crs_name": crs_name,
            "guid": str(header.uuid) if header.uuid else None,
            "creation_date": str(header.creation_date) if header.creation_date else None,
            "system_identifier": header.system_identifier,
            "generating_software": header.generating_software,
            "vlr_count": len(header.vlrs),
            "evlr_count": len(header.evlrs) if header.evlrs else 0,
            "vlrs": vlrs,
        }

    # LAS's own axes: x/y are horizontal, z is elevation (unlike the internal
    # voxel grid, which remaps to y-up for glTF -- see voxelize.py).
    extent = [maxs[i] - mins[i] for i in range(3)]
    footprint_area = extent[0] * extent[1]
    info["extent_xyz"] = extent
    info["footprint_area_m2"] = footprint_area
    info["point_density_per_m2"] = (info["point_count"] / footprint_area) if footprint_area > 0 else None
    return info


def format_text(info: dict) -> str:
    lines = [
        f"File: {info['file']}",
        f"  size: {info['size_bytes'] / 1e6:,.2f} MB ({info['size_bytes']:,} bytes)",
        f"  compressed (LAZ): {info['compressed']}",
        "",
        "LAS format",
        f"  version: {info['las_version']}",
        f"  point format: {info['point_format_id']}",
        f"  dimensions: {', '.join(info['dimensions'])}",
        f"  has RGB: {info['has_rgb']}",
        f"  scales: {info['scales']}",
        f"  offsets: {info['offsets']}",
        "",
        "Coordinate reference",
    ]
    if info["epsg"] is not None:
        lines.append(f"  EPSG:{info['epsg']} ({info['crs_name']})")
    else:
        lines.append("  none declared -- pass --epsg to assume one, or convert without georeferencing")

    ex, ey, ez = info["extent_xyz"]
    lines += [
        "",
        "Point data",
        f"  point count: {info['point_count']:,}",
        f"  bounds (LAS x/y/z): "
        f"[{info['mins'][0]:.2f}, {info['maxs'][0]:.2f}] x "
        f"[{info['mins'][1]:.2f}, {info['maxs'][1]:.2f}] x "
        f"[{info['mins'][2]:.2f}, {info['maxs'][2]:.2f}]",
        f"  extent: {ex:.1f} x {ey:.1f} m footprint, {ez:.1f} m vertical",
    ]
    if info["point_density_per_m2"] is not None:
        lines.append(f"  density: ~{info['point_density_per_m2']:.1f} points/m^2")

    lines += [
        "",
        "Metadata",
        f"  GUID: {info['guid']}",
        f"  created: {info['creation_date']}",
        f"  system identifier: {info['system_identifier']!r}",
        f"  generating software: {info['generating_software']!r}",
        f"  VLRs: {info['vlr_count']}  EVLRs: {info['evlr_count']}",
    ]
    for v in info["vlrs"]:
        desc = f"  {v['description']}" if v["description"] else ""
        lines.append(f"    - user_id={v['user_id']!r} record_id={v['record_id']}{desc}")
    return "\n".join(lines)
