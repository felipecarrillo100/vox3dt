"""Write a CRS into a LAS 1.4 header in place, without rewriting the points.

    python tools/set_crs.py cloud.las --epsg 31983            # dry run
    python tools/set_crs.py cloud.las --epsg 31983 --apply
    python tools/set_crs.py cloud.las --restore               # undo

Why not PDAL
------------
`pdal translate in.las out.las --writers.las.a_srs=EPSG:31983` is the correct,
conventional way and you should prefer it for normal files. But it rewrites
every point: on a 113 GB cloud that is 113 GB read plus 113 GB written, hours of
I/O and 226 GB of churn, to set one metadata field. (GDAL is not an option at
all -- its LAS support is read-oriented and it has no LAS writer.)

Why this works
--------------
A normal VLR lives between the header and the point data, so adding one shifts
`offset_to_point_data` and therefore every point in the file. When
`offset_to_point_data == header_size` there is no slack, so that means a full
rewrite.

LAS 1.4 also has **Extended VLRs**, which live at the *end* of the file and are
found through `start_of_first_evlr` in the header. The spec (§ "OGC Coordinate
System WKT Record") permits the WKT CRS as either a VLR or an EVLR. So the CRS
can be appended to the end of the file, and only three header fields need to
change:

    offset   6  global_encoding      u16  -- set bit 4, the "WKT" bit
    offset 235  start_of_first_evlr  u64  -- where our EVLR begins
    offset 243  number_of_evlrs      u32  -- 1

That is a few kilobytes written and 14 bytes patched, regardless of file size.

A note on conformance: LAS 1.4 *requires* the WKT bit for point formats 6-10,
so a format-8 file with the bit clear is already out of spec. Setting it is a
fix, not a liberty.

Safety
------
`--apply` first copies the original 375-byte header to `<file>.header.bak`, so
`--restore` puts things back exactly (the appended EVLR bytes become
unreferenced trailing data that every reader ignores). Nothing else in the file
is touched, and no point is read or moved.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

LAS14_HEADER_SIZE = 375
OFF_GLOBAL_ENCODING = 6
OFF_VERSION_MAJOR = 24
OFF_HEADER_SIZE = 94
OFF_OFFSET_TO_POINT_DATA = 96
OFF_NUMBER_OF_VLRS = 100
OFF_POINT_FORMAT = 104
OFF_START_OF_FIRST_EVLR = 235
OFF_NUMBER_OF_EVLRS = 243
WKT_BIT = 0x10

EVLR_HEADER_SIZE = 60
PROJECTION_USER_ID = b"LASF_Projection"
WKT_RECORD_ID = 2112


def read_header(path: Path) -> bytes:
    with path.open("rb") as fh:
        return fh.read(LAS14_HEADER_SIZE)


def describe(header: bytes) -> dict:
    if header[:4] != b"LASF":
        raise ValueError("not a LAS file (missing LASF signature)")
    return {
        "version": f"{header[OFF_VERSION_MAJOR]}.{header[OFF_VERSION_MAJOR + 1]}",
        "global_encoding": struct.unpack_from("<H", header, OFF_GLOBAL_ENCODING)[0],
        "header_size": struct.unpack_from("<H", header, OFF_HEADER_SIZE)[0],
        "offset_to_point_data": struct.unpack_from("<I", header, OFF_OFFSET_TO_POINT_DATA)[0],
        "number_of_vlrs": struct.unpack_from("<I", header, OFF_NUMBER_OF_VLRS)[0],
        "point_format": header[OFF_POINT_FORMAT] & 0x3F,
        "start_of_first_evlr": struct.unpack_from("<Q", header, OFF_START_OF_FIRST_EVLR)[0],
        "number_of_evlrs": struct.unpack_from("<I", header, OFF_NUMBER_OF_EVLRS)[0],
    }


def build_wkt_evlr(wkt: str, description: str = "vox3dt set_crs") -> bytes:
    """A LASF_Projection / 2112 OGC WKT record, as an Extended VLR."""
    payload = wkt.encode("utf-8") + b"\x00"  # spec recommends null termination
    return (
        struct.pack("<H", 0)
        + PROJECTION_USER_ID.ljust(16, b"\x00")
        + struct.pack("<H", WKT_RECORD_ID)
        + struct.pack("<Q", len(payload))
        + description.encode("utf-8")[:32].ljust(32, b"\x00")
        + payload
    )


def wkt_for_epsg(epsg: int, flavour: str) -> tuple[str, str]:
    from pyproj import CRS

    crs = CRS.from_epsg(epsg)
    return crs.to_wkt(flavour), crs.name


def apply_crs(path: Path, epsg: int, flavour: str, force: bool) -> None:
    header = read_header(path)
    info = describe(header)

    if info["version"] != "1.4":
        raise SystemExit(
            f"error: this trick needs LAS 1.4 (EVLRs); file is {info['version']}.\n"
            f"       Use: pdal translate in.las out.las --writers.las.a_srs=EPSG:{epsg}"
        )
    if (info["number_of_vlrs"] or info["number_of_evlrs"]) and not force:
        raise SystemExit(
            f"error: file already has {info['number_of_vlrs']} VLR(s) and "
            f"{info['number_of_evlrs']} EVLR(s); it may already declare a CRS.\n"
            "       Inspect it first, then re-run with --force to append anyway."
        )

    wkt, name = wkt_for_epsg(epsg, flavour)
    evlr = build_wkt_evlr(wkt)

    backup = path.with_suffix(path.suffix + ".header.bak")
    if not backup.exists():
        backup.write_bytes(header)
        print(f"  saved original header -> {backup.name}")

    size = os.path.getsize(path)
    with path.open("r+b") as fh:
        fh.seek(0, os.SEEK_END)
        evlr_offset = fh.tell()
        fh.write(evlr)

        encoding = struct.unpack_from("<H", header, OFF_GLOBAL_ENCODING)[0] | WKT_BIT
        fh.seek(OFF_GLOBAL_ENCODING)
        fh.write(struct.pack("<H", encoding))
        fh.seek(OFF_START_OF_FIRST_EVLR)
        fh.write(struct.pack("<Q", evlr_offset))
        fh.seek(OFF_NUMBER_OF_EVLRS)
        fh.write(struct.pack("<I", info["number_of_evlrs"] + 1))
        fh.flush()
        os.fsync(fh.fileno())

    print(f"  appended {len(evlr):,}-byte EVLR at offset {evlr_offset:,}")
    print(f"  set global_encoding WKT bit, start_of_first_evlr, number_of_evlrs")
    print(f"  file grew by {len(evlr):,} bytes (was {size:,})")
    print(f"\n  CRS written: EPSG:{epsg} — {name}")


def restore(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".header.bak")
    if not backup.exists():
        raise SystemExit(f"error: no backup at {backup.name}")
    original = backup.read_bytes()
    if len(original) != LAS14_HEADER_SIZE:
        raise SystemExit(f"error: backup is {len(original)} bytes, expected {LAS14_HEADER_SIZE}")
    with path.open("r+b") as fh:
        fh.seek(0)
        fh.write(original)
        fh.flush()
        os.fsync(fh.fileno())
    print(f"  header restored from {backup.name}")
    print("  the appended EVLR bytes remain as unreferenced trailing data;")
    print("  every reader ignores them, and re-applying will overwrite nothing.")


def verify(path: Path, expect_crs: bool = True) -> None:
    try:
        import laspy
    except ImportError:
        print("  (laspy not available — skipping read-back check)")
        return
    with laspy.open(str(path)) as reader:
        crs = reader.header.parse_crs()
    if crs is None:
        if expect_crs:
            print("  READ-BACK FAILED: laspy still reports no CRS")
        else:
            print("  read-back OK: laspy reports no CRS, as expected after --restore")
        return
    if not expect_crs:
        print(f"  NOTE: laspy still reports {crs.name} — the restore did not take effect")
        return
    print(f"  read-back OK: laspy reports {crs.name} (EPSG:{crs.to_epsg()})")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Write a CRS into a LAS 1.4 header in place, without rewriting points.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Runs as a dry run unless --apply is given.",
    )
    p.add_argument("las_path", type=Path)
    p.add_argument("--epsg", type=int, help="EPSG code to write, e.g. 31983")
    p.add_argument(
        "--wkt-flavour", default="WKT1_GDAL", choices=("WKT1_GDAL", "WKT2_2019"),
        help="WKT dialect (default: WKT1_GDAL, the most widely readable)",
    )
    p.add_argument("--apply", action="store_true", help="Actually modify the file")
    p.add_argument("--restore", action="store_true", help="Put the original header back")
    p.add_argument("--force", action="store_true", help="Append even if VLRs already exist")
    args = p.parse_args(argv)

    if not args.las_path.exists():
        print(f"error: not found: {args.las_path}", file=sys.stderr)
        return 1

    print(f"\n{args.las_path.name}")
    info = describe(read_header(args.las_path))
    for key, value in info.items():
        print(f"  {key:22s} {value}")
    print()

    if args.restore:
        restore(args.las_path)
        verify(args.las_path, expect_crs=False)
        return 0

    if args.epsg is None:
        p.error("--epsg is required (or use --restore)")

    if not args.apply:
        wkt, name = wkt_for_epsg(args.epsg, args.wkt_flavour)
        evlr = build_wkt_evlr(wkt)
        print("DRY RUN — nothing written. --apply would:")
        print(f"  append a {len(evlr):,}-byte EVLR at offset {os.path.getsize(args.las_path):,}")
        print(f"  set global_encoding bit 4 (WKT): {info['global_encoding']} -> "
              f"{info['global_encoding'] | WKT_BIT}")
        print(f"  set start_of_first_evlr: {info['start_of_first_evlr']} -> "
              f"{os.path.getsize(args.las_path):,}")
        print(f"  set number_of_evlrs: {info['number_of_evlrs']} -> {info['number_of_evlrs'] + 1}")
        print(f"\n  CRS: EPSG:{args.epsg} — {name}")
        print(f"  Points read or moved: 0")
        return 0

    apply_crs(args.las_path, args.epsg, args.wkt_flavour, args.force)
    print()
    verify(args.las_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
