"""Command-line interface: one LAS/LAZ in, one OGC 3D Tiles tileset out."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from . import __version__
from .gltfwriter import write_baked, write_instanced
from .info import describe, format_text
from .rasterprops import RasterProperty, load_config, parse_property_arg, validate_specs
from .pyramid import pyramid_stream
from .tileset import TileMeta, build_tree, grid_to_ecef_matrix, origin_lonlat, pyramid_depth
from .voxelize import header_epsg, read_grid_spec, stream_level0_runs

DEFAULT_BRICK = 64
DEFAULT_VOXEL_SIZE = 1.0
DEFAULT_MEMORY_BUDGET_MB = 256

_DESCRIPTION = """\
Turn a LAS/LAZ point cloud into a georeferenced OGC 3D Tiles pyramid of cubes.

Far from the camera you see coarse blocks; as you approach, the viewer refines
to exact 1 m cubes. Levels are chosen by the client's screen-space error, so no
hand-written culling is involved. The source CRS and grid origin are read from
the LAS header and written into the tileset, so it lands in the right place on
the globe.
"""

_EPILOG = """\
examples
--------
  # Simplest useful run: Draco-compressed, gzip sidecars for a CDN.
  vox3dt -i site.las -o tiles/site --draco --gzip

  # Only gzipped files on disk (tiles and tileset.json); serve them under
  # their plain names with Content-Encoding: gzip.
  vox3dt -i site.las -o tiles/site --draco --gzip-only

  # Uncompressed, for a viewer without Draco support.
  vox3dt -i site.las -o tiles/site

  # Smaller payload, trading a little colour fidelity (32768 colours).
  vox3dt -i site.las -o tiles/site --draco --color-bits 5

  # Coarser voxels and fewer, heavier tiles -- faster to build, less detail.
  vox3dt -i site.las -o tiles/site --draco --voxel-size 2 --brick 128

  # GPU-instanced cubes instead of triangles: far smaller, fewer draw calls,
  # but needs EXT_mesh_gpu_instancing support in the viewer.
  vox3dt -i site.las -o tiles/site --mode instanced

  # Fast header-only inspection, no conversion (works on any file size).
  vox3dt -i site.las --info

  # Per-cube properties from GeoTIFFs (EXT_structural_metadata in each glb).
  vox3dt -i site.las -o tiles/site --draco --property Classification=landcover.tif
  vox3dt -i site.las -o tiles/site --draco --properties props.json

  # LAS with no declared CRS: assume one, or write a non-georeferenced tileset.
  vox3dt -i site.las -o tiles/site --epsg 32616
  vox3dt -i site.las -o tiles/site               # no --epsg -> local metres only

output
------
  <output>/tileset.json        the tileset, with the fitted grid->ECEF root transform
                                (omitted if the tileset is non-georeferenced)
  <output>/content/*.glb       one binary glTF per tile; with --property, each
                                cube is a feature with its own property row
  <output>/specs.json          developer reference: georeference, grid,
                                encoding, FeatureID, and per property its
                                source and the GeoTIFF colour of each class
  <output>/report.json         per-level tile/cube/byte counts for this run
  <output>/log/conversion.log  a minimal, milestone-only log of this run
  <output>/temp/               working files, deleted when the run finishes
                                (unless --keep-temp)
  *.gz alongside each file when --gzip is given; with --gzip-only the tiles
  and tileset.json exist only as .gz (specs.json, report.json stay plain)

verifying
---------
  python verify.py <output> <input.las>

  Checks GLB structure, accessor alignment, Draco attribute mapping, the
  georeference against the LAS header, and that every tile's geometry lies
  inside its declared bounding volume.

notes
-----
  Colour is always FLOAT32 COLOR_0. LuciadRIA mistypes integer vertex colours
  out of a Draco payload and ignores material baseColorFactor, so float vertex
  colour is the only channel that renders there. --color-bits 8 (the default)
  is exact: bit-identical to the uncompressed output.

  --draco affects --mode baked only. In instanced mode the mesh is a single
  cube and all the volume is in the instance buffers, which Draco does not
  compress; use --gzip there.

  Peak memory is bounded by --memory-budget (plus --chunk-points), not by the
  input file's size -- see --memory-budget below for large inputs.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vox3dt",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"vox3dt {__version__}")

    required = parser.add_argument_group("required")
    required.add_argument(
        "-i", "--input", type=Path, required=True, metavar="LAS",
        help="Source point cloud (.las or .laz)",
    )
    required.add_argument(
        "-o", "--output", type=Path, metavar="DIR",
        help="Directory to write tileset.json and content/ into (created if absent). "
             "Not required with --info",
    )

    inspect = parser.add_argument_group("inspect")
    inspect.add_argument(
        "--info", action="store_true",
        help="Print header-only metadata (CRS, bounds, point count, format...) and "
             "exit. Never reads point data, so it stays fast on any file size",
    )

    geo = parser.add_argument_group("georeference")
    geo.add_argument(
        "--epsg", type=int, default=None, metavar="CODE",
        help="Assume this EPSG code when the source LAS declares no CRS. Ignored "
             "(with a notice) if the LAS already declares one. If omitted and the "
             "LAS has no CRS, a non-georeferenced tileset is written instead",
    )

    encoding = parser.add_argument_group("encoding")
    encoding.add_argument(
        "--draco", action="store_true",
        help="Compress tile geometry with KHR_draco_mesh_compression "
             "(baked mode only; ~4x smaller gzipped, colour unchanged)",
    )
    encoding.add_argument(
        "--mode", choices=("baked", "instanced"), default="baked",
        help="baked: face-culled triangles, works everywhere (default). "
             "instanced: EXT_mesh_gpu_instancing, much smaller and far fewer "
             "draw calls, but needs viewer support",
    )
    encoding.add_argument(
        "--color-bits", type=int, default=8, metavar="N",
        help="Bits per colour channel. 8 = exact, identical to uncompressed "
             "(default). Lower quantizes onto an endpoint-preserving grid to "
             "shrink Draco payloads: 5 gives 32768 colours",
    )
    encoding.add_argument(
        "--draco-bits", type=int, default=14, metavar="N",
        help="Draco position quantization bits (default: 14). The step is about "
             "the tile's range / 2^bits: <= 5 mm on 1 m leaf tiles at 14 bits, "
             "more on coarse tiles -- small, but not lossless",
    )
    encoding.add_argument(
        "--palette-bits", type=int, default=2, metavar="N",
        help="Instanced mode only: bits per channel for the colour palette "
             "(default: 2, up to 64 entries). One draw call per entry per tile",
    )

    props = parser.add_argument_group("properties")
    props.add_argument(
        "--property", action="append", default=[], metavar="NAME=PATH",
        help="Sample a GeoTIFF into a per-cube property NAME "
             "(EXT_structural_metadata). Repeatable. Adds FeatureID, unique per cube. "
             "Kind (categorical if the raster has a colour table, else "
             "continuous) and type are detected from the raster",
    )
    props.add_argument(
        "--properties", type=Path, default=None, metavar="JSON",
        help="Config file listing properties: {\"properties\": [{\"name\", "
             "\"file\", \"band\", \"kind\", \"labels\", \"colors\"}, ...]}. "
             "Relative paths resolve against the config's folder",
    )

    pyramid = parser.add_argument_group("pyramid")
    pyramid.add_argument(
        "--voxel-size", type=float, default=DEFAULT_VOXEL_SIZE, metavar="M",
        help=f"Finest voxel size in metres (default: {DEFAULT_VOXEL_SIZE})",
    )
    pyramid.add_argument(
        "--brick", type=int, default=DEFAULT_BRICK, metavar="N",
        help=f"Cells per tile edge in X/Z (default: {DEFAULT_BRICK}). Larger "
             "means fewer, heavier tiles",
    )
    pyramid.add_argument(
        "--levels", type=int, default=None, metavar="N",
        help="Number of LOD levels. Default: derived from the site's extent, so "
             "the root covers it in a single tile",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--gzip", action="store_true",
        help="Also write .gz beside each file, for CDN upload with "
             "Content-Encoding: gzip",
    )
    output.add_argument(
        "--gzip-only", action="store_true",
        help="Like --gzip, but write the tiles and tileset.json only as .gz, "
             "with no uncompressed copy. The server must answer requests for "
             "tileset.json and .glb with the .gz and Content-Encoding: gzip",
    )
    output.add_argument(
        "--chunk-points", type=int, default=5_000_000, metavar="N",
        help="Points per LAS read pass (default: 5,000,000). Lower to cut peak "
             "memory on very large inputs",
    )
    output.add_argument(
        "--memory-budget", type=int, default=DEFAULT_MEMORY_BUDGET_MB, metavar="MB",
        help=f"Approx. RAM per pyramid level before spilling to <output>/temp/ "
             f"(default: {DEFAULT_MEMORY_BUDGET_MB}). Lower to cut peak memory "
             "further on very large inputs, at the cost of more disk I/O",
    )
    output.add_argument(
        "--keep-temp", action="store_true",
        help="Don't delete <output>/temp/ when the run finishes (debugging)",
    )
    output.add_argument("-q", "--quiet", action="store_true", help="Only report errors")
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.input.exists():
        parser.error(f"input not found: {args.input}")
    if args.info:
        return args
    if args.output is None:
        parser.error("--output is required unless --info is given")
    if args.voxel_size <= 0:
        parser.error("--voxel-size must be positive")
    if args.brick < 2:
        parser.error("--brick must be at least 2")
    if args.levels is not None and args.levels < 1:
        parser.error("--levels must be at least 1")
    if args.memory_budget < 1:
        parser.error("--memory-budget must be at least 1 (MB)")
    if args.gzip_only:
        args.gzip = True
    if args.draco and args.mode == "instanced":
        parser.error(
            "--draco has no effect in instanced mode (the mesh is one cube; the "
            "volume is in instance buffers). Use --gzip instead, or --mode baked."
        )
    args.property_specs = []
    try:
        if args.properties is not None:
            if not args.properties.exists():
                parser.error(f"--properties file not found: {args.properties}")
            args.property_specs += load_config(args.properties)
        args.property_specs += [parse_property_arg(t) for t in args.property]
        validate_specs(args.property_specs)
    except ValueError as exc:
        parser.error(str(exc))
    if args.property_specs and args.mode == "instanced":
        parser.error("--property/--properties need --mode baked (instanced would need EXT_instance_features)")
    if args.epsg is not None:
        try:
            from pyproj import CRS

            CRS.from_epsg(args.epsg)
        except Exception as exc:
            parser.error(f"--epsg {args.epsg} is not a recognized EPSG code: {exc}")
    return args


def _write(path: Path, data: bytes, gzip_out: bool, raw: bool = True) -> int:
    """Write ``data`` and/or its ``.gz`` twin. Returns the uncompressed size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw:
        path.write_bytes(data)
    if gzip_out:
        path.with_suffix(path.suffix + ".gz").write_bytes(gzip.compress(data, compresslevel=9))
    return len(data)


def run_info(args: argparse.Namespace) -> dict:
    """`--info`: header-only inspection, no conversion."""
    info = describe(args.input)
    if not args.quiet:
        print(format_text(info))
    return info


def run(args: argparse.Namespace) -> dict:
    """Build one tileset. Returns the same summary written to report.json."""
    started = time.perf_counter()
    output: Path = args.output
    temp_dir = output / "temp"
    log_dir = output / "log"

    (output / "content").mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    log_file = open(log_dir / "conversion.log", "w", encoding="utf-8")

    def say(msg: str = "") -> None:
        if not args.quiet:
            print(msg)
        log_file.write(msg + "\n")

    try:
        report = _run(args, temp_dir, say, started)
    finally:
        log_file.close()
        if not args.keep_temp and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    return report


def _run(args: argparse.Namespace, temp_dir: Path, say, started: float) -> dict:
    output: Path = args.output
    budget_bytes = args.memory_budget * 1024 * 1024

    declared_epsg = header_epsg(args.input)
    if args.epsg is not None and declared_epsg is not None:
        say(f"Note: LAS already declares EPSG:{declared_epsg}; ignoring --epsg {args.epsg}.")

    spec = read_grid_spec(args.input, args.voxel_size, epsg_override=args.epsg)
    geo = spec.georeference
    epsg_assumed = declared_epsg is None and geo.epsg is not None

    # Rasters are opened and checked before any point is read, so a bad
    # property config fails in seconds rather than an hour into a large run.
    rasters: list[RasterProperty] = []
    if args.property_specs:
        if geo.epsg is None:
            raise ValueError("--property needs a georeferenced LAS: a declared CRS or --epsg")
        say("Properties:")
        east_range = (float(spec.g_min[0]), float(spec.g_max[0]))
        north_range = (float(spec.g_min[2]), float(spec.g_max[2]))
        for prop_spec in args.property_specs:
            raster = RasterProperty(prop_spec, geo.epsg)
            coverage = raster.coverage(east_range, north_range)
            say(raster.summary_line(coverage))
            if coverage <= 0:
                raise ValueError(f"{prop_spec.name}: raster does not overlap the LAS bounds")
            rasters.append(raster)
    next_feature_id = [0]
    float_feature_tiles: list[str] = []

    depth = (args.levels - 1) if args.levels else pyramid_depth(spec.dims, args.brick)
    levels = depth + 1
    say(f"Pyramid: {levels} levels, {args.brick}-cell tiles")

    tiles_by_level: dict[int, list[TileMeta]] = {}
    per_level_counts: dict[int, dict] = {}

    def on_tile(block) -> None:
        idx, rgb = block.visible()
        if len(idx) == 0:
            return
        faces_visible = block.faces[block.surface]
        centers = (idx.astype(np.float64) + 0.5) * block.cell_size

        n_cubes = len(idx)
        tlevel = depth - block.depth_from_finest
        columns = None
        if rasters:
            first = next_feature_id[0]
            if first + n_cubes > 2**31 - 1:
                raise ValueError("more than 2**31-1 cubes: FeatureID no longer fits int32")
            ox, oy, _oz = geo.origin
            east = ox + centers[:, 0]
            north = oy - centers[:, 2]
            columns = [("FeatureID", np.arange(first, first + n_cubes, dtype=np.int32), "INT32")]
            for raster in rasters:
                columns.append(
                    (raster.name, raster.sample(east, north, block.cell_size, tlevel), raster.component_type)
                )
        if args.mode == "instanced":
            glb, stats = write_instanced(centers, rgb, block.cell_size, args.palette_bits)
        else:
            glb, stats = write_baked(
                centers, rgb, faces_visible, block.cell_size,
                draco=args.draco, draco_bits=args.draco_bits, color_bits=args.color_bits,
                properties=columns,
            )
        if not glb:
            return
        if rasters:
            next_feature_id[0] += n_cubes
        # Tight bounds, in whole cells of this level: every drawn cube lies
        # fully inside, and a parent's box contains its children's because a
        # coarse cell covers the finer cells it was built from.
        meta = TileMeta(
            level=tlevel, tx=block.tx, tz=block.tz, cell_size=block.cell_size,
            x0=int(idx[:, 0].min()), x1=int(idx[:, 0].max()) + 1,
            z0=int(idx[:, 2].min()), z1=int(idx[:, 2].max()) + 1,
            y_min=int(idx[:, 1].min()), y_max=int(idx[:, 1].max()),
        )

        if stats.get("feature_id_type") == "FLOAT":
            float_feature_tiles.append(meta.key)
        nbytes = _write(output / "content" / f"{meta.key}.glb", glb, args.gzip, raw=not args.gzip_only)
        tiles_by_level.setdefault(tlevel, []).append(meta)

        counts = per_level_counts.setdefault(tlevel, {"tiles": 0, "cubes": 0, "triangles": 0, "bytes": 0})
        counts["tiles"] += 1
        counts["cubes"] += stats.get("cubes", 0)
        counts["triangles"] += stats.get("triangles", 0)
        counts["bytes"] += nbytes

    spilled0 = stream_level0_runs(
        args.input, spec, args.brick, budget_bytes, temp_dir / "L0",
        chunk_points=args.chunk_points, verbose=not args.quiet, say=say,
    )

    say(f"\nWriting {args.mode}{' + draco' if args.draco else ''} -> {output}")
    pyramid_stats = pyramid_stream(
        spilled0, spec.dims, args.voxel_size, args.brick, levels,
        temp_dir, budget_bytes, on_tile, say=say,
    )

    non_georeferenced = geo.epsg is None
    if non_georeferenced:
        lon = lat = height = None
        transform = None
        placement_residual = None
        say(
            "\nNo CRS declared and no --epsg given -- writing a non-georeferenced "
            "tileset (local metres, no root transform)."
        )
    else:
        lon, lat, height = origin_lonlat(geo)
        extent = tuple(float(d) * args.voxel_size for d in spec.dims)
        transform, placement_residual = grid_to_ecef_matrix(geo, extent)
        say(
            f"\nGeoreference: EPSG:{geo.epsg} ({geo.crs_name})"
            f"\n  grid origin -> lon {lon:.6f}, lat {lat:.6f}, h {height:.2f} m"
            f"\n  root transform: least-squares fit, worst placement error "
            f"{placement_residual['max_horizontal_m']:.3f} m horizontal, "
            f"{placement_residual['max_vertical_m']:.3f} m vertical"
        )

    per_level, totals = [], {"tiles": 0, "cubes": 0, "triangles": 0, "bytes": 0}
    for lvl in range(levels):
        counts = per_level_counts.get(lvl, {"tiles": 0, "cubes": 0, "triangles": 0, "bytes": 0})
        cell = args.voxel_size * (2 ** (depth - lvl))
        per_level.append({"level": lvl, "cell_size": cell, **counts})
        for key in totals:
            totals[key] += counts[key]
        say(
            f"  L{lvl}  cell {cell:6.1f} m  tiles {counts['tiles']:5d}  "
            f"cubes {counts['cubes']:9,}  tris {counts['triangles']:10,}  "
            f"{counts['bytes'] / 1e6:7.2f} MB"
        )

    root = build_tree(tiles_by_level, depth, lambda tile: f"content/{tile.key}.glb")
    if transform is not None:
        root["transform"] = transform

    voxel_count = pyramid_stats[0]["cells"] if pyramid_stats else 0
    # tileset.json holds only what the 3D Tiles spec defines; everything a
    # developer might want to look up goes to specs.json instead.
    tileset: dict = {"asset": {"version": "1.1", "tilesetVersion": f"vox3dt-{__version__}", "gltfUpAxis": "Y"}}
    if rasters:
        # Value ranges, which LuciadRIA needs before it will style on a property.
        tileset["properties"] = {
            r.name: {"minimum": r.vmin, "maximum": r.vmax} for r in rasters if r.vmin is not None
        }
    tileset["geometricError"] = args.voxel_size * (2 ** (depth + 1))
    tileset["root"] = root
    _write(output / "tileset.json", json.dumps(tileset, indent=2).encode(), args.gzip, raw=not args.gzip_only)

    georef = geo.as_dict()
    georef["epsg_assumed"] = epsg_assumed
    georef["origin_lonlat"] = [lon, lat, height] if lon is not None else None
    georef["placement"] = (
        "root.transform is a least-squares affine fit mapping tile-space projected metres "
        "(E = ox + u, N = oy + v, H = oz + w) to ECEF (EPSG:4978), absorbing the projection's "
        "grid convergence and scale; glTF content is y-up: (u, v, w) = (x, -z, y)"
        if transform is not None else "non-georeferenced: local metres, no root transform"
    )
    georef["placement_residual"] = placement_residual
    specs = {
        "generator": f"vox3dt {__version__}",
        "source": {"file": args.input.name, "points": spec.n_points},
        "georeference": georef,
        "grid": {
            "voxel_size": args.voxel_size,
            "dims": list(spec.dims),
            "brick": args.brick,
            "voxel_count": voxel_count,
            "levels": [
                {"level": lvl, "cell_size": args.voxel_size * (2 ** (depth - lvl))} for lvl in range(levels)
            ],
        },
        "encoding": {
            "content": "glb",
            "mode": args.mode,
            "draco": bool(args.draco),
            "color": "COLOR_0 FLOAT32 VEC3",
            "color_bits": args.color_bits,
            "exact_color": args.color_bits >= 8,
        },
    }
    if rasters:
        specs["encoding"]["feature_id"] = (
            "_FEATURE_ID_0 per vertex (EXT_mesh_features), constant over each cube; row index into "
            "the tile's EXT_structural_metadata property table (class 'cube'). "
            + ("UNSIGNED_SHORT inside Draco, FLOAT in tiles with more than 65536 cubes"
               if args.draco else "FLOAT (uncompressed vertex attributes must be 4-byte aligned)")
        )
        specs["feature_id"] = {
            "property": "FeatureID",
            "component_type": "INT32",
            "scope": "unique per cube across the whole tileset (a cube at each level is its own feature)",
            "min": 0 if next_feature_id[0] else None,
            "max": next_feature_id[0] - 1 if next_feature_id[0] else None,
        }
        specs["properties"] = {r.name: r.spec_entry() for r in rasters}
    (output / "specs.json").write_text(json.dumps(specs, indent=2))
    if args.properties is not None:
        # The config that built this tileset, kept beside it.
        shutil.copyfile(args.properties, output / "properties.config.json")

    report = {
        "generator": f"vox3dt {__version__}",
        "input": str(args.input),
        "output": str(output),
        "mode": args.mode,
        "draco": bool(args.draco),
        # "only": tiles exist solely as .glb.gz; true: .gz beside each raw file.
        "gzip": "only" if args.gzip_only else bool(args.gzip),
        "color_bits": args.color_bits,
        "exact_color": args.color_bits >= 8,
        "voxel_size": args.voxel_size,
        "brick": args.brick,
        "levels": levels,
        "epsg": geo.epsg,
        "crs_name": geo.crs_name,
        "origin_lonlat": [lon, lat, height] if lon is not None else None,
        "grid_dims": list(spec.dims),
        "voxel_count": voxel_count,
        "content": "glb",
        "totals": totals,
        "per_level": per_level,
        "seconds": round(time.perf_counter() - started, 2),
    }
    if rasters:
        report["feature_ids"] = next_feature_id[0]
        report["feature_id_float_tiles"] = float_feature_tiles
        report["properties"] = {r.name: r.report_entry() for r in rasters}
    (output / "report.json").write_text(json.dumps(report, indent=2))

    say(
        f"\n{totals['tiles']:,} tiles, {totals['bytes'] / 1e6:.2f} MB"
        f"{' (.gz only)' if args.gzip_only else ' (+ .gz)' if args.gzip else ''} in {report['seconds']:.1f}s"
        f"\n  -> {output / ('tileset.json.gz' if args.gzip_only else 'tileset.json')}"
        f"\n  verify with: python verify.py {output} {args.input}"
    )
    return report


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        if args.info:
            run_info(args)
        else:
            run(args)
    except Exception as exc:  # noqa: BLE001 -- a CLI should not show a traceback
        print(f"vox3dt: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
