"""The vox3dt runs the suite checks, and how their output is summarised and compared.

A run's *manifest* is what the golden files store: the list of files written,
a SHA-256 of every tile (decompressed when only the .gz exists), and the full
tileset.json, specs.json and report.json minus fields that change from machine
to machine (run time, absolute paths). Tiles are stored as hashes, not bytes,
so the golden files stay a few KB per case.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from synth import Site

GOLDEN_DIR = Path(__file__).parent / "golden"


@dataclass
class Case:
    name: str
    args: list[str]
    no_crs: bool = False
    #: Count RunSpiller run files written, to prove the external sort was used.
    count_spills: bool = False


CASES = [
    Case("plain", []),
    Case("draco", ["--draco"]),
    Case("gzip", ["--draco", "--gzip"]),
    Case("gzip_only", ["--draco", "--gzip-only"]),
    Case("props", ["--draco", "--properties", "{config}"]),
    Case("props_raw", ["--properties", "{config}"]),
    Case("props_cli", ["--draco", "--property", "Classification={landcover}"]),
    # One 260 x 260 tile holds more than 65,536 cubes: FLOAT feature ids under Draco.
    Case("float_ids", ["--draco", "--properties", "{config}", "--brick", "260"]),
    Case("spill", ["--draco", "--memory-budget", "1", "--chunk-points", "5000"], count_spills=True),
    Case("instanced", ["--mode", "instanced"]),
    Case("color5", ["--draco", "--color-bits", "5"]),
    Case("no_crs", ["--draco"], no_crs=True),
    Case("epsg", ["--draco", "--epsg", "31983"], no_crs=True),
]
BY_NAME = {c.name: c for c in CASES}


@dataclass
class Result:
    case: Case
    out: Path
    las: Path
    spills: int = 0
    manifest: dict = field(default_factory=dict)


def argv(case: Case, site: Site, out: Path) -> list[str]:
    las = site.las_no_crs if case.no_crs else site.las
    fill = {"config": str(site.config), "landcover": str(site.landcover)}
    return ["-i", str(las), "-o", str(out), "-q", *(a.format(**fill) for a in case.args)]


@contextlib.contextmanager
def _spill_counter():
    from vox3dt import extsort

    original = extsort.RunSpiller._write_run
    counter = {"runs": 0}

    def counting(self, arr):
        if len(arr):
            counter["runs"] += 1
        return original(self, arr)

    extsort.RunSpiller._write_run = counting
    try:
        yield counter
    finally:
        extsort.RunSpiller._write_run = original


def run(case: Case, site: Site, out: Path) -> Result:
    from vox3dt.cli import main

    with _spill_counter() as counter:
        rc = main(argv(case, site, out))
    assert rc == 0, f"vox3dt exited with {rc} for case {case.name}"
    result = Result(case=case, out=out, las=site.las_no_crs if case.no_crs else site.las, spills=counter["runs"])
    result.manifest = manifest(out)
    return result


# ---------------------------------------------------------------------------
# Manifest


def _read(path: Path) -> bytes:
    """A file's bytes, or its .gz twin's decompressed bytes when only that exists."""
    if path.exists():
        return path.read_bytes()
    return gzip.decompress(path.with_name(path.name + ".gz").read_bytes())


def manifest(out: Path) -> dict:
    files = sorted(
        p.relative_to(out).as_posix() for p in out.rglob("*")
        if p.is_file() and p.relative_to(out).parts[0] not in ("log", ".vox3dt-temp")
    )
    tiles = {}
    for name in files:
        if name.startswith("content/"):
            stem = name.removesuffix(".gz")
            if stem not in tiles:
                tiles[stem] = hashlib.sha256(_read(out / stem)).hexdigest()

    specs = json.loads((out / "specs.json").read_text())
    for prop in specs.get("properties", {}).values():
        prop["source"]["file"] = Path(prop["source"]["file"]).name
    report = json.loads((out / "report.json").read_text())
    for key in ("seconds", "input", "output"):
        report.pop(key, None)
    return {
        "files": files,
        "tiles": tiles,
        "tileset": json.loads(_read(out / "tileset.json")),
        "specs": specs,
        "report": report,
    }


def golden_path(case: Case) -> Path:
    return GOLDEN_DIR / f"{case.name}.json"


def write_golden(case: Case, data: dict) -> None:
    GOLDEN_DIR.mkdir(exist_ok=True)
    golden_path(case).write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


def load_golden(case: Case) -> dict:
    return json.loads(golden_path(case).read_text())


# ---------------------------------------------------------------------------
# Comparison

#: Floats come out of a least-squares fit and PROJ; allow the last bits to differ.
REL_TOL = 1e-9
ABS_TOL = 1e-9


def diff(expected, actual, path: str = "$", limit: int = 20) -> list[str]:
    """Differences between two JSON values; floats compared within REL_TOL/ABS_TOL."""
    out: list[str] = []

    def walk(a, b, p):
        if len(out) >= limit:
            return
        if isinstance(a, bool) or isinstance(b, bool):
            if a is not b:
                out.append(f"{p}: expected {a!r}, got {b!r}")
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if isinstance(a, int) and isinstance(b, int):
                if a != b:
                    out.append(f"{p}: expected {a}, got {b}")
            elif not math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL):
                out.append(f"{p}: expected {a!r}, got {b!r}")
        elif isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                if k not in b:
                    out.append(f"{p}.{k}: missing")
                elif k not in a:
                    out.append(f"{p}.{k}: unexpected (value {b[k]!r:.80})")
                else:
                    walk(a[k], b[k], f"{p}.{k}")
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                out.append(f"{p}: expected {len(a)} items, got {len(b)}")
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{p}[{i}]")
        elif a != b:
            out.append(f"{p}: expected {a!r:.80}, got {b!r:.80}")

    walk(expected, actual, path)
    return out
