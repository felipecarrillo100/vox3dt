"""Every case passes all of verify.py's checks, and verify.py catches a bad box.

One verify.py limitation is worked around here rather than changed:
``check_placement`` reads corners from each glb's POSITION bounds, which in
instanced mode describe a single unit cube, not the instances. It is skipped
for the instanced case; test_golden.py checks the instanced root transform is
the draco case's, and test_georef.py checks that one against known answers.
"""

from __future__ import annotations

import json
import shutil

import pytest

import cases
import verify


def _verify(out, las) -> tuple[int, list[str]]:
    verify.FAIL.clear()
    verify.OK.clear()
    rc = verify.main(out, las)
    return rc, list(dict.fromkeys(verify.FAIL))


@pytest.mark.parametrize("name", [c.name for c in cases.CASES])
def test_verify_passes(name, results, monkeypatch):
    r = results(name)
    if name == "instanced":
        monkeypatch.setattr(verify, "check_placement", lambda *args: None)
    rc, failed = _verify(r.out, r.las)
    assert rc == 0 and not failed, "verify.py failures:\n  " + "\n  ".join(failed[:20])
    assert len(verify.OK) > 100


@pytest.mark.parametrize("name, shrink", [("plain", 0.01), ("draco", 0.01), ("instanced", 0.01)])
def test_verify_catches_boxes_smaller_than_the_geometry(name, shrink, results, tmp_path, monkeypatch):
    """Leaf boxes 1 cm too small on every side must fail (they once passed at 0.9 m)."""
    out = tmp_path / name
    shutil.copytree(results(name).out, out)
    tileset = json.loads((out / "tileset.json").read_text())

    def walk(node):
        if not node.get("children"):
            for i in (3, 7, 11):
                node["boundingVolume"]["box"][i] -= shrink
        for child in node.get("children", []):
            walk(child)

    walk(tileset["root"])
    (out / "tileset.json").write_text(json.dumps(tileset))
    monkeypatch.setattr(verify, "check_placement", lambda *args: None)
    _rc, failed = _verify(out, results(name).las)
    overflow = [f for f in failed if "geometry inside its bounding box" in f]
    assert overflow, "verify.py let geometry outside its box pass"
