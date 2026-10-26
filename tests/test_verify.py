"""Every georeferenced case passes all of verify.py's checks.

verify.py reads the root transform unconditionally, so the non-georeferenced
case is left out; its tiles are byte-identical to the draco case's anyway.

Two verify.py limitations are worked around here rather than changed:

- ``_cell_size_of`` caches cell sizes by tile file name for the life of the
  process, which is right for one tileset per run but stale across cases. The
  cache is dropped before each case.
- ``check_placement`` reads corners from each glb's POSITION bounds, which in
  instanced mode describe a single unit cube, not the instances. It is skipped
  for the instanced case; test_georef.py covers placement through the root
  transform for every mode, since the transform does not depend on the mode.
"""

from __future__ import annotations

import pytest

import cases
import verify


@pytest.mark.parametrize("name", [c.name for c in cases.CASES if c.name != "no_crs"])
def test_verify_passes(name, results, monkeypatch):
    r = results(name)
    verify.FAIL.clear()
    verify.OK.clear()
    if hasattr(verify._cell_size_of, "cache"):
        del verify._cell_size_of.cache
    if name == "instanced":
        monkeypatch.setattr(verify, "check_placement", lambda *args: None)
    rc = verify.main(r.out, r.las)
    failed = list(dict.fromkeys(verify.FAIL))
    assert rc == 0 and not failed, "verify.py failures:\n  " + "\n  ".join(failed[:20])
    assert len(verify.OK) > 100
