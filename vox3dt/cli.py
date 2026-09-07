"""Command-line interface: one LAS/LAZ in, one OGC 3D Tiles tileset out."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

from . import __version__
from .gltfwriter import write_baked, write_instanced
from .pyramid import build_pyramid, exposed_faces
from .tileset import (
    build_tree,
    enu_to_ecef_matrix,
    origin_lonlat,
    pyramid_depth,
    tiles_for_level,
)
from .voxelize import voxelize

DEFAULT_BRICK = 64
DEFAULT_VOXEL_SIZE = 1.0

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

  # Uncompressed, for a viewer without Draco support.
  vox3dt -i site.las -o tiles/site

  # Smaller payload, trading a little colour fidelity (32768 colours).
  vox3dt -i site.las -o tiles/site --draco --color-bits 5

  # Coarser voxels and fewer, heavier tiles -- faster to build, less detail.
  vox3dt -i site.las -o tiles/site --draco --voxel-size 2 --brick 128

  # GPU-instanced cubes instead of triangles: far smaller, fewer draw calls,
  # but needs EXT_mesh_gpu_instancing support in the viewer.
  vox3dt -i site.las -o tiles/site --mode instanced

output
------
  <output>/tileset.json        the tileset, with the ENU->ECEF root transform
  <output>/content/*.glb       one binary glTF per tile
  <output>/report.json         per-level tile/cube/byte counts for this run
  *.gz alongside each file when --gzip is given

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
        "-o", "--output", type=Path, required=True, metavar="DIR",
        help="Directory to write tileset.json and content/ into (created if absent)",
    )

    encoding = parser.add_argument_group("encoding")
    encoding.add_argument(
        "--draco", action="store_true",
        help="Compress tile geometry with KHR_draco_mesh_compression "
             "(baked mode only; roughly 3.5x smaller gzipped, colour unchanged)",
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
        help="Draco position quantization bits (default: 14). Voxels sit on an "
             "integer lattice, so this is lossless well below 14 for most tiles",
    )
    encoding.add_argument(
        "--palette-bits", type=int, default=2, metavar="N",
        help="Instanced mode only: bits per channel for the colour palette "
             "(default: 2, up to 64 entries). One draw call per entry per tile",
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
        "--chunk-points", type=int, default=5_000_000, metavar="N",
        help="Points per LAS read pass (default: 5,000,000). Lower to cut peak "
             "memory on very large inputs",
    )
    output.add_argument("-q", "--quiet", action="store_true", help="Only report errors")
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.input.exists():
        parser.error(f"input not found: {args.input}")
    if args.voxel_size <= 0:
        parser.error("--voxel-size must be positive")
    if args.brick < 2:
        parser.error("--brick must be at least 2")
    if args.levels is not None and args.levels < 1:
        parser.error("--levels must be at least 1")
    if args.draco and args.mode == "instanced":
        parser.error(
            "--draco has no effect in instanced mode (the mesh is one cube; the "
            "volume is in instance buffers). Use --gzip instead, or --mode baked."
        )
    return args


def _write(path: Path, data: bytes, gzip_out: bool) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if gzip_out:
        path.with_suffix(path.suffix + ".gz").write_bytes(gzip.compress(data, compresslevel=9))
    return len(data)


def run(args: argparse.Namespace) -> dict:
    """Build one tileset. Returns the same summary written to report.json."""
    started = time.perf_counter()
    say = (lambda *a, **k: None) if args.quiet else print

    grid = voxelize(args.input, args.voxel_size, args.chunk_points, verbose=not args.quiet)
    geo = grid.georeference

    depth = (args.levels - 1) if args.levels else pyramid_depth(grid.dims, args.brick)
    say(f"\nPyramid: {depth + 1} levels, {args.brick}-cell tiles")
    levels = build_pyramid(
        grid.index, grid.rgb, grid.dims, args.voxel_size, depth + 1, verbose=not args.quiet
    )

    # Surface-filtered cells plus their per-face exposure, per pyramid level.
    visible = []
    for level in levels:
        index, rgb = level.visible()
        visible.append((index, rgb, exposed_faces(level.index, level.dims)[level.surface]))

    lon, lat, height = origin_lonlat(geo)
    transform = enu_to_ecef_matrix(geo)
    say(
        f"\nGeoreference: EPSG:{geo.epsg} ({geo.crs_name})"
        f"\n  grid origin -> lon {lon:.6f}, lat {lat:.6f}, h {height:.2f} m"
    )

    say(f"\nWriting {args.mode}{' + draco' if args.draco else ''} -> {args.output}")
    tiles_by_level = {
        level: tiles_for_level(level, depth, args.brick, levels, visible)
        for level in range(depth + 1)
    }

    per_level, totals = [], {"tiles": 0, "cubes": 0, "triangles": 0, "bytes": 0}
    for level in range(depth + 1):
        tiles = tiles_by_level[level]
        counts = {"tiles": len(tiles), "cubes": 0, "triangles": 0, "bytes": 0}
        for tile in tiles:
            centers = tile.centers()
            if args.mode == "instanced":
                glb, stats = write_instanced(centers, tile.rgb, tile.cell_size, args.palette_bits)
            else:
                glb, stats = write_baked(
                    centers, tile.rgb, tile.faces, tile.cell_size,
                    draco=args.draco, draco_bits=args.draco_bits, color_bits=args.color_bits,
                )
            if not glb:
                continue
            counts["bytes"] += _write(args.output / "content" / f"{tile.key}.glb", glb, args.gzip)
            counts["cubes"] += stats["cubes"]
            counts["triangles"] += stats["triangles"]
        cell = levels[depth - level].cell_size
        per_level.append({"level": level, "cell_size": cell, **counts})
        for key in totals:
            totals[key] += counts[key]
        say(
            f"  L{level}  cell {cell:6.1f} m  tiles {counts['tiles']:5d}  "
            f"cubes {counts['cubes']:9,}  tris {counts['triangles']:10,}  "
            f"{counts['bytes'] / 1e6:7.2f} MB"
        )

    root = build_tree(tiles_by_level, depth, lambda tile: f"content/{tile.key}.glb")
    root["transform"] = transform
    tileset = {
        "asset": {"version": "1.1", "tilesetVersion": f"vox3dt-{__version__}", "gltfUpAxis": "Y"},
        "geometricError": levels[-1].cell_size * 2,
        "root": root,
        "extras": {
            "generator": f"vox3dt {__version__}",
            "source": args.input.name,
            "georeference": geo.as_dict(),
            "grid_dims": list(grid.dims),
            "voxel_count": len(grid),
            "encoding": {
                "mode": args.mode,
                "draco": bool(args.draco),
                "color_bits": args.color_bits,
                "exact_color": args.color_bits >= 8,
            },
        },
    }
    _write(args.output / "tileset.json", json.dumps(tileset, indent=2).encode(), args.gzip)

    report = {
        "generator": f"vox3dt {__version__}",
        "input": str(args.input),
        "output": str(args.output),
        "mode": args.mode,
        "draco": bool(args.draco),
        "color_bits": args.color_bits,
        "exact_color": args.color_bits >= 8,
        "voxel_size": args.voxel_size,
        "brick": args.brick,
        "levels": depth + 1,
        "epsg": geo.epsg,
        "crs_name": geo.crs_name,
        "origin_lonlat": [lon, lat, height],
        "grid_dims": list(grid.dims),
        "voxel_count": len(grid),
        "totals": totals,
        "per_level": per_level,
        "seconds": round(time.perf_counter() - started, 2),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))

    say(
        f"\n{totals['tiles']:,} tiles, {totals['bytes'] / 1e6:.2f} MB"
        f"{' (+ .gz)' if args.gzip else ''} in {report['seconds']:.1f}s"
        f"\n  -> {args.output / 'tileset.json'}"
        f"\n  verify with: python verify.py {args.output} {args.input}"
    )
    return report


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        run(args)
    except Exception as exc:  # noqa: BLE001 -- a CLI should not show a traceback
        print(f"vox3dt: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
