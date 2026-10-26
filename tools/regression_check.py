"""Offline, on-demand regression check on a large real dataset.

The test suite (tests/) runs on synthetic data only. This script is the
manual double-check on a real site: build the tileset with one version of the
code, record a baseline, rebuild later with another version and compare. It is
never run by pytest, and nothing it reads or writes belongs in git -- the
dataset, the output and the baseline stay in --work, which must be outside
the repo or gitignored (e.g. production/).

    # 1. Baseline from a known-good version, checked out on its own:
    git worktree add ../vox3dt-v0.1.0 v0.1.0
    python tools/regression_check.py --work production/check --code ../vox3dt-v0.1.0 \\
        baseline -- -i <site.las> --epsg <code> --draco --gzip-only --properties <props.json>

    # 2. Later, the same vox3dt arguments with the current code:
    python tools/regression_check.py --work production/check \\
        compare [--verify] -- -i <site.las> --epsg <code> --draco --gzip-only --properties <props.json>

    # Or record a baseline from a tileset that already exists, without a rebuild:
    python tools/regression_check.py --work production/check baseline --existing <tileset-dir>

The comparison is the one tests/ uses: every tile byte-identical (decompressed),
tileset.json, specs.json and report.json identical apart from run time and
paths. --verify also runs verify.py on the new tileset (decodes every tile, slow).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests"))

from cases import diff, manifest  # noqa: E402


def _refuse_tracked(work: Path) -> None:
    """--work inside the repo must be gitignored, so no output can be committed."""
    try:
        work.resolve().relative_to(REPO)
    except ValueError:
        return
    probe = work / "probe.glb"
    if subprocess.run(["git", "check-ignore", "-q", str(probe)], cwd=REPO).returncode != 0:
        sys.exit(f"--work {work} is inside the repo but not gitignored; use production/ or a folder outside the repo")


def _build(code: Path, out: Path, vox_args: list[str]) -> float:
    if "-o" in vox_args or "--output" in vox_args:
        sys.exit("leave -o out of the vox3dt arguments; the output goes to --work")
    if out.exists():
        shutil.rmtree(out)  # always one of our own *_run folders under --work
    env = {**os.environ, "PYTHONPATH": str(code)}
    started = time.perf_counter()
    cmd = [sys.executable, "-m", "vox3dt.cli", *vox_args, "-o", str(out)]
    print("$", " ".join(cmd), f"  (code: {code})", flush=True)
    subprocess.run(cmd, env=env, check=True)
    return time.perf_counter() - started


def _manifest(out: Path) -> dict:
    print(f"Hashing tiles in {out} ...", flush=True)
    return manifest(out)


def cmd_baseline(args) -> int:
    if args.existing:
        out = args.existing
    else:
        out = args.work / "baseline_run"
        _build(args.code, out, args.vox_args)
    data = _manifest(out)
    data["source"] = {"tileset": str(out), "code": str(args.code), "vox3dt_args": args.vox_args}
    path = args.work / "baseline.json"
    path.write_text(json.dumps(data, indent=1, sort_keys=True))
    print(f"Baseline: {len(data['tiles'])} tiles -> {path}")
    return 0


def cmd_compare(args) -> int:
    baseline_path = args.work / "baseline.json"
    if not baseline_path.exists():
        sys.exit(f"no baseline at {baseline_path}; run the 'baseline' command first")
    want = json.loads(baseline_path.read_text())
    out = args.work / "compare_run"
    seconds = _build(args.code, out, args.vox_args)
    got = _manifest(out)

    problems = []
    if want["files"] != got["files"]:
        missing = sorted(set(want["files"]) - set(got["files"]))
        extra = sorted(set(got["files"]) - set(want["files"]))
        problems.append(f"files: {len(missing)} missing {missing[:5]}, {len(extra)} unexpected {extra[:5]}")
    changed = sorted(t for t in set(want["tiles"]) & set(got["tiles"]) if want["tiles"][t] != got["tiles"][t])
    if changed:
        problems.append(f"{len(changed)} of {len(want['tiles'])} tiles changed: {changed[:10]}")
    for key in ("tileset", "specs", "report"):
        problems += [f"{key} {d}" for d in diff(want[key], got[key], "$", limit=50)]

    print(f"\nBuilt in {seconds / 60:.1f} min; {len(got['tiles'])} tiles compared.")
    if args.verify:
        las = args.vox_args[args.vox_args.index("-i") + 1] if "-i" in args.vox_args else None
        if las is None:
            problems.append("--verify needs -i in the vox3dt arguments")
        else:
            rc = subprocess.run([sys.executable, str(args.code / "verify.py"), str(out), las]).returncode
            if rc:
                problems.append("verify.py reported failures (see above)")
    if problems:
        print("DIFFERENT from baseline:\n  " + "\n  ".join(problems))
        return 1
    print("IDENTICAL to baseline.")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    vox_args = []
    if "--" in argv:
        i = argv.index("--")
        argv, vox_args = argv[:i], argv[i + 1:]

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, required=True, help="Folder for builds and baseline.json (gitignored or outside the repo)")
    parser.add_argument("--code", type=Path, default=REPO, help="Checkout whose vox3dt/ and verify.py to run (default: this repo)")
    sub = parser.add_subparsers(dest="command", required=True)
    base = sub.add_parser("baseline", help="Build (or read --existing) and record the baseline")
    base.add_argument("--existing", type=Path, help="Record this already-built tileset instead of building")
    comp = sub.add_parser("compare", help="Build with --code and compare against the baseline")
    comp.add_argument("--verify", action="store_true", help="Also run verify.py on the new tileset (slow)")
    args = parser.parse_args(argv)
    args.vox_args = vox_args
    args.code = args.code.resolve()

    if not (args.code / "vox3dt" / "cli.py").exists():
        parser.error(f"--code {args.code} has no vox3dt/cli.py")
    if not vox_args and not (args.command == "baseline" and args.existing):
        parser.error("give the vox3dt arguments after --, e.g. -- -i site.las --draco")
    _refuse_tracked(args.work)
    args.work.mkdir(parents=True, exist_ok=True)
    return cmd_baseline(args) if args.command == "baseline" else cmd_compare(args)


if __name__ == "__main__":
    sys.exit(main())
