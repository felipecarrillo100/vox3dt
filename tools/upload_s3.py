"""Upload a gzipped 3D Tiles tileset to S3 under its *uncompressed* key names.

    python tools/upload_s3.py production/vale_gzip \
        s3://revolutiondata/vale/MAC/slice_2/product/cubifiedchunkswith3DTiles

Why this exists
---------------
`tileset.json` references tiles as `content/L3_0_0.glb`. On disk we only keep
the gzipped twins (`content/L3_0_0.glb.gz`) to save disk. Apache can bridge
that gap with a rewrite rule; S3 cannot -- it serves keys literally, so a key
ending in `.glb.gz` 404s for every tile the viewer asks for.

The fix is not to rename anything. An S3 key is an argument to PutObject, so
this reads `<name>.glb.gz` and stores it at `<name>.glb`, with:

    Content-Encoding: gzip     the body really is gzip; the browser inflates it
    Content-Type: model/gltf-binary    what the bytes are *after* inflating

`aws s3 cp --recursive` can't do this (the CLI derives each key from the local
path), and a per-file `aws s3 cp` loop means ~6,700 process launches. One
process with a thread pool is minutes instead of hours.

The tileset is uploaded last, so a viewer never sees an entry point whose
tiles have not all landed yet.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# One year, immutable: a tile's content never changes once published -- a new
# build writes new tile names. The tileset itself is the mutable entry point.
TILE_CACHE = "public, max-age=31536000, immutable"
TILESET_CACHE = "public, max-age=300, must-revalidate"


def plan(src: Path) -> list[tuple[Path, str, str, str]]:
    """(local file, s3 key suffix, content type, cache control), tileset last."""
    jobs = []
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(src).as_posix()
        if rel.endswith(".gz"):
            rel = rel[:-3]          # the .gz is an encoding, not part of the name
        elif path.name.startswith("."):
            continue                # .htaccess and friends: Apache-only, skip
        if rel.endswith(".glb"):
            jobs.append((path, rel, "model/gltf-binary", TILE_CACHE))
        elif rel.endswith(".json"):
            jobs.append((path, rel, "application/json", TILESET_CACHE))
    jobs.sort(key=lambda j: j[1].endswith(".json"))
    return jobs


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Upload a gzipped 3D Tiles directory to S3 under uncompressed key names.",
        epilog="Runs as a dry run unless --apply is given.",
    )
    p.add_argument("source", type=Path, help="Local directory of .gz files")
    p.add_argument("dest", help="s3://bucket/prefix")
    p.add_argument("--apply", action="store_true", help="Actually upload")
    p.add_argument("--workers", type=int, default=16, help="Parallel uploads (default: 16)")
    args = p.parse_args(argv)

    if not args.dest.startswith("s3://"):
        p.error("dest must look like s3://bucket/prefix")
    bucket, _, prefix = args.dest[5:].partition("/")
    prefix = prefix.strip("/")

    if not args.source.is_dir():
        print(f"error: not a directory: {args.source}", file=sys.stderr)
        return 1

    jobs = plan(args.source)
    tiles = sum(1 for j in jobs if j[2] == "model/gltf-binary")
    total = sum(j[0].stat().st_size for j in jobs)
    print(f"\n{args.source}  ->  s3://{bucket}/{prefix}/")
    print(f"  {len(jobs):,} objects ({tiles:,} tiles), {total / 1e6:,.1f} MB gzipped")
    for path, key, ctype, _cache in jobs[:2] + (["..."] if len(jobs) > 3 else []) + jobs[-1:]:  # type: ignore[operator]
        if path == "...":
            print("    ...")
            continue
        print(f"    {path.name:36s} -> {prefix}/{key:34s} {ctype}")

    if not args.apply:
        print("\nDRY RUN -- nothing uploaded. Re-run with --apply.")
        return 0

    import boto3
    from boto3.s3.transfer import TransferConfig

    s3 = boto3.client("s3")
    cfg = TransferConfig(use_threads=False)   # we parallelize across files, not within
    done = 0
    failed: list[tuple[str, str]] = []

    def put(job) -> None:
        path, key, ctype, cache = job
        s3.upload_file(
            str(path), bucket, f"{prefix}/{key}" if prefix else key,
            ExtraArgs={
                "ContentEncoding": "gzip",
                "ContentType": ctype,
                "CacheControl": cache,
            },
            Config=cfg,
        )

    # Tiles first (all but the trailing tileset), in parallel.
    tile_jobs, tileset_jobs = jobs[:-1], jobs[-1:]
    if tileset_jobs and not tileset_jobs[0][1].endswith(".json"):
        tile_jobs, tileset_jobs = jobs, []

    print()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(put, j): j for j in tile_jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:                      # noqa: BLE001
                failed.append((job[1], str(exc)))
            done += 1
            if done % 250 == 0 or done == len(tile_jobs):
                print(f"  {done:,}/{len(tile_jobs):,} tiles", flush=True)

    if failed:
        print(f"\n  {len(failed)} FAILED -- tileset not uploaded, so nothing is half-live:")
        for key, err in failed[:5]:
            print(f"    {key}: {err}")
        print("  Re-run to retry (uploads are idempotent).")
        return 1

    for job in tileset_jobs:
        put(job)
        print(f"  uploaded entry point: {prefix}/{job[1]}")

    host = f"{bucket}.s3.amazonaws.com"
    print(f"\n  endpoint: https://{host}/{prefix}/tileset.json")
    print("  remaining: bucket CORS, and disable CloudFront auto-compression on this path")
    return 0


if __name__ == "__main__":
    sys.exit(main())
