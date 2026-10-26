"""Output-folder safety: what an existing folder holds, and what vox3dt may delete.

Before a run writes anything, the output folder is classified:

- **empty** -- missing, or holding nothing but ``.DS_Store``: go ahead.
- **vox3dt** -- an earlier vox3dt build, recognised by its ``report.json``
  generator, its ``tileset.json`` tilesetVersion, or the ``.vox3dt`` marker a
  run writes before its first tile (so even a crashed run is recognised). Its
  vox3dt files are replaced after a confirmation (``--overwrite`` from a
  script); everything else in the folder is left alone.
- **foreign** -- anything else: refused, nothing deleted or written.

Only these are ever deleted, matched by exact name -- never a whole folder
except vox3dt's own working folder:

    tileset.json  tileset.json.gz  specs.json  report.json
    properties.config.json  .vox3dt
    content/L<level>_<x>_<z>.glb[.gz]     (directly in content/, no deeper)
    log/conversion.log
    .vox3dt-temp/                         (vox3dt's working folder)

then ``content/`` and ``log/`` if that left them empty.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

MARKER = ".vox3dt"
TEMP_DIR = ".vox3dt-temp"
TOP_FILES = ("tileset.json", "tileset.json.gz", "specs.json", "report.json", "properties.config.json", MARKER)
TILE_RE = re.compile(r"^L\d+_\d+_\d+\.glb(\.gz)?$")
IGNORED = {".DS_Store"}


class OutputRefused(Exception):
    """The output folder cannot be used. Nothing was deleted or written."""


class Cancelled(Exception):
    """The user declined the replacement. Nothing was deleted."""


@dataclass
class OutputInfo:
    path: Path
    kind: str  # empty | vox3dt | foreign
    delete: list[Path] = field(default_factory=list)  # files, in deletion order
    keep: list[str] = field(default_factory=list)  # relative paths left alone
    tiles: int = 0
    nbytes: int = 0
    generator: str | None = None
    built: str | None = None


def _read_json(path: Path):
    try:
        raw = path.read_bytes()
        return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)
    except (OSError, ValueError, EOFError):
        return None


def _generator(out: Path) -> str | None:
    """'vox3dt x.y.z' if any of vox3dt's signatures is present, else None."""
    report = _read_json(out / "report.json")
    if isinstance(report, dict) and str(report.get("generator", "")).startswith("vox3dt"):
        return str(report["generator"])
    for name in ("tileset.json", "tileset.json.gz"):
        tileset = _read_json(out / name)
        version = (tileset or {}).get("asset", {}).get("tilesetVersion", "") if isinstance(tileset, dict) else ""
        if str(version).startswith("vox3dt-"):
            return "vox3dt " + str(version).removeprefix("vox3dt-")
    marker = out / MARKER
    if marker.is_file() and not marker.is_symlink():
        data = _read_json(marker)
        return str(data.get("generator", "vox3dt")) if isinstance(data, dict) else "vox3dt"
    return None


def _all_entries(out: Path) -> list[str]:
    """Every file (and link) under `out`, relative, without following links."""
    found = []
    for dirpath, dirnames, filenames in os.walk(out, followlinks=False):
        rel = Path(dirpath).relative_to(out)
        links = [d for d in dirnames if (Path(dirpath) / d).is_symlink()]
        for name in filenames + links:
            if name not in IGNORED:
                found.append((rel / name).as_posix())
    return sorted(found)


def inspect(out: Path) -> OutputInfo:
    """Classify `out` and, for a vox3dt folder, list exactly what would be deleted."""
    if out.is_symlink() and not out.exists():
        raise OutputRefused(f"{out} is a broken link")
    if out.exists() and not out.is_dir():
        raise OutputRefused(f"{out} exists and is not a folder")
    if not out.exists() or not any(p.name not in IGNORED for p in out.iterdir()):
        return OutputInfo(path=out, kind="empty")

    generator = _generator(out)
    if generator is None:
        return OutputInfo(path=out, kind="foreign", keep=sorted(p.name for p in out.iterdir() if p.name not in IGNORED))

    for sub in ("content", "log", TEMP_DIR):
        if (out / sub).is_symlink():
            raise OutputRefused(f"{out / sub} is a link to another folder; vox3dt will not delete through it")

    info = OutputInfo(path=out, kind="vox3dt", generator=generator)
    for name in TOP_FILES:
        p = out / name
        if p.is_file() or p.is_symlink():
            info.delete.append(p)
    content = out / "content"
    if content.is_dir():
        for p in sorted(content.iterdir()):
            if TILE_RE.match(p.name) and (p.is_file() or p.is_symlink()):
                info.delete.append(p)
                info.tiles += 1
    log = out / "log" / "conversion.log"
    if log.is_file() or log.is_symlink():
        info.delete.append(log)
    temp = out / TEMP_DIR
    temp_files = []
    if temp.is_dir():
        temp_files = [temp / f for f in _all_entries(temp)]

    deleted = {p.relative_to(out).as_posix() for p in info.delete}
    info.keep = [e for e in _all_entries(out)
                 if e not in deleted and e not in ("content", "log") and not (e == TEMP_DIR or e.startswith(TEMP_DIR + "/"))]
    info.nbytes = sum(p.lstat().st_size for p in info.delete + temp_files if not p.is_dir())
    report = out / "report.json"
    stamp = report if report.is_file() else (out / MARKER if (out / MARKER).is_file() else None)
    if stamp is not None:
        info.built = datetime.fromtimestamp(stamp.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return info


def _size(n: int) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}"
    return f"{n} bytes"


def _names(items: list[str], limit: int = 5) -> str:
    shown = ", ".join(items[:limit])
    return shown + (f", and {len(items) - limit} more" if len(items) > limit else "")


def _count(info: OutputInfo) -> int:
    return len(info.delete) + (1 if (info.path / TEMP_DIR).is_dir() else 0)


def describe(info: OutputInfo) -> str:
    """The text shown before the confirmation question."""
    built = f"built {info.built} by {info.generator}" if info.built else f"written by {info.generator}"
    tops = [p.relative_to(info.path).as_posix() for p in info.delete if p.parent != info.path / "content"]
    lines = [
        f"{info.path} already holds a vox3dt tileset:",
        f"  {built}, {info.tiles:,} tiles, {_size(info.nbytes)}",
        "",
        "To rebuild, these vox3dt files will be deleted:",
    ]
    if info.tiles:
        lines.append(f"  content/   {info.tiles:,} tiles")
    if tops:
        lines.append("  " + ", ".join(tops))
    if (info.path / TEMP_DIR).is_dir():
        lines.append(f"  {TEMP_DIR}/ (working files of an earlier run)")
    if info.keep:
        lines += ["", "These files are not vox3dt's and will be kept:", "  " + _names(info.keep)]
    return "\n".join(lines)


def is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stderr.isatty()


def confirm(info: OutputInfo, overwrite: bool, interactive: bool | None = None, ask=None) -> None:
    """Return if the run may go ahead; raise OutputRefused or Cancelled otherwise."""
    ask = ask or input
    if info.kind == "empty":
        return
    if info.kind == "foreign":
        raise OutputRefused(
            f"{info.path} is not empty and is not a vox3dt output (it contains {_names(info.keep, 3)}).\n"
            "Choose a new or empty folder. Nothing was deleted or written."
        )
    if overwrite:
        return
    if interactive is None:
        interactive = is_interactive()
    if not interactive:
        raise OutputRefused(
            f"{info.path} already holds a vox3dt tileset ({info.tiles:,} tiles).\n"
            "Re-run with --overwrite to replace it, or choose a new folder. Nothing was deleted."
        )
    print(describe(info), file=sys.stderr)
    try:
        answer = ask(f"\nDelete {_count(info):,} files and rebuild? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        answer = ""
    if answer.strip().lower() not in ("y", "yes"):
        raise Cancelled("Cancelled. Nothing was deleted.")


def clear(info: OutputInfo) -> int:
    """Delete exactly what `inspect` listed. Returns the number of entries removed."""
    if info.kind != "vox3dt":
        return 0
    removed = 0
    for p in info.delete:
        p.unlink(missing_ok=True)
        removed += 1
    temp = info.path / TEMP_DIR
    if temp.is_dir() and not temp.is_symlink():
        shutil.rmtree(temp)
        removed += 1
    for sub in ("content", "log"):
        d = info.path / sub
        if d.is_dir() and not d.is_symlink() and not any(d.iterdir()):
            d.rmdir()
    return removed


def write_marker(out: Path, version: str) -> None:
    (out / MARKER).write_text(json.dumps({
        "generator": f"vox3dt {version}",
        "started": datetime.now().astimezone().isoformat(timespec="seconds"),
    }) + "\n")
