"""The output folder: what vox3dt replaces, what it keeps, and what it refuses.

Every refusal and every cancellation must leave the folder exactly as it was;
`snapshot` records names, contents and links to prove it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from vox3dt import outdir
from vox3dt.cli import main


def snapshot(root: Path) -> dict:
    """Every entry under root: file hash, link target, or 'dir'. Links are not followed."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames + filenames:
            p = Path(dirpath) / name
            rel = p.relative_to(root).as_posix()
            if p.is_symlink():
                out[rel] = "link -> " + os.readlink(p)
            elif p.is_dir():
                out[rel] = "dir"
            else:
                out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def build(site, out: Path, *extra: str) -> int:
    return main(["-i", str(site.las), "-o", str(out), "-q", "--draco", *extra])


@pytest.fixture(scope="module")
def previous_build(site, tmp_path_factory) -> Path:
    """A finished vox3dt build at the default brick: 39 tiles."""
    out = tmp_path_factory.mktemp("previous") / "tiles"
    assert build(site, out) == 0
    return out


@pytest.fixture
def old(previous_build, tmp_path) -> Path:
    """A private copy of the previous build to run over."""
    dst = tmp_path / "tiles"
    shutil.copytree(previous_build, dst, symlinks=True)
    return dst


@pytest.fixture(autouse=True)
def not_interactive(monkeypatch):
    """pytest never has a terminal, but make it explicit; tests that need one override it."""
    monkeypatch.setattr(outdir, "is_interactive", lambda: False)


def tiles_in(out: Path) -> set[str]:
    return {p.name for p in (out / "content").iterdir() if outdir.TILE_RE.match(p.name)}


def reported_tiles(out: Path) -> int:
    return json.loads((out / "report.json").read_text())["totals"]["tiles"]


# ---------------------------------------------------------------------------
# Folders that need no confirmation


def test_new_folder(site, tmp_path):
    out = tmp_path / "new" / "tiles"
    assert build(site, out) == 0
    assert (out / outdir.MARKER).is_file()
    assert json.loads((out / outdir.MARKER).read_text())["generator"].startswith("vox3dt ")


@pytest.mark.parametrize("contents", [[], [".DS_Store"]])
def test_empty_folder(site, tmp_path, contents):
    out = tmp_path / "tiles"
    out.mkdir()
    for name in contents:
        (out / name).write_text("x")
    assert build(site, out) == 0
    assert len(tiles_in(out)) == reported_tiles(out)


# ---------------------------------------------------------------------------
# Replacing a previous build


def test_replaces_only_vox3dt_files(site, old):
    # The user's own files, some of them in vox3dt's folders or named like tiles.
    (old / "serve.json").write_text('{"port": 8080}')
    (old / "content" / "notes.txt").write_text("mine")
    (old / "content" / "sub").mkdir()
    (old / "content" / "sub" / "L9_0_0.glb").write_text("not a tile of this build")
    (old / "content" / "L1_0_0.glb.bak").write_text("backup")
    (old / "temp").mkdir()
    (old / "temp" / "precious.txt").write_text("never delete me")
    mine = {k: v for k, v in snapshot(old).items()
            if k in ("serve.json", "content/notes.txt", "content/sub", "content/sub/L9_0_0.glb",
                     "content/L1_0_0.glb.bak", "temp", "temp/precious.txt")}
    assert len(tiles_in(old)) == 39

    assert build(site, old, "--overwrite", "--brick", "260") == 0

    after = snapshot(old)
    assert {k: after.get(k) for k in mine} == mine, "the user's files must survive unchanged"
    assert len(tiles_in(old)) == reported_tiles(old) < 39, "no tile of the previous build may remain"
    assert "deleted" in (old / "log" / "conversion.log").read_text().lower()


def test_gzip_only_over_plain_leaves_no_plain_files(site, old):
    assert build(site, old, "--overwrite", "--gzip-only") == 0
    assert not list((old / "content").glob("*.glb"))
    assert not (old / "tileset.json").exists()
    assert len(list((old / "content").glob("*.glb.gz"))) == reported_tiles(old)


def test_properties_config_of_a_previous_build_is_removed(site, tmp_path):
    out = tmp_path / "tiles"
    assert build(site, out, "--properties", str(site.config)) == 0
    assert (out / "properties.config.json").exists()
    assert build(site, out, "--overwrite") == 0
    assert not (out / "properties.config.json").exists()


@pytest.mark.parametrize("remove", [
    ["report.json", "tileset.json", "specs.json"],  # a run that crashed after its first tiles
    [outdir.MARKER],  # a build by vox3dt 0.1.0, before the marker existed
    [outdir.MARKER, "report.json"],  # recognised by tileset.json alone
])
def test_recognises_partial_and_older_builds(site, old, remove):
    for name in remove:
        (old / name).unlink()
    assert outdir.inspect(old).kind == "vox3dt"
    assert build(site, old, "--overwrite", "--brick", "260") == 0
    assert len(tiles_in(old)) == reported_tiles(old)


# ---------------------------------------------------------------------------
# Refusals: nothing deleted, nothing written


def _refused(capsys, site, out, *extra) -> str:
    before = snapshot(out) if out.exists() else None
    rc = build(site, out, *extra)
    err = capsys.readouterr().err
    assert rc == 1, err
    assert (snapshot(out) if out.exists() else None) == before, "a refusal must not change the folder"
    return err


def test_refuses_a_folder_of_other_files(capsys, site, tmp_path):
    out = tmp_path / "project"
    (out / "src").mkdir(parents=True)
    (out / "src" / "main.py").write_text("print('hi')")
    (out / "README.md").write_text("# mine")
    err = _refused(capsys, site, out, "--overwrite")
    assert "is not empty and is not a vox3dt output" in err
    assert "README.md" in err and "Nothing was deleted or written" in err


def test_refuses_lookalike_files_from_another_tool(capsys, site, tmp_path):
    out = tmp_path / "site"
    (out / "content").mkdir(parents=True)
    (out / "content" / "L0_0_0.glb").write_bytes(b"someone else's model")
    (out / "content" / "index.md").write_text("a website page")
    (out / "report.json").write_text('{"generator": "othertool 2.0"}')
    (out / "tileset.json").write_text('{"asset": {"version": "1.1", "tilesetVersion": "2024-06"}}')
    _refused(capsys, site, out, "--overwrite")


def test_refuses_content_linked_elsewhere(capsys, site, old, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    shutil.move(old / "content", elsewhere)
    (old / "content").symlink_to(elsewhere, target_is_directory=True)
    tiles_before = snapshot(elsewhere)
    err = _refused(capsys, site, old, "--overwrite")
    assert "is a link to another folder" in err
    assert snapshot(elsewhere) == tiles_before


def test_refuses_a_file_as_output(capsys, site, tmp_path):
    out = tmp_path / "tiles"
    out.write_text("a file")
    assert "is not a folder" in _refused(capsys, site, out)


def test_script_without_overwrite_is_refused(capsys, site, old):
    err = _refused(capsys, site, old)
    assert "--overwrite" in err and "Nothing was deleted" in err


def test_bad_input_fails_before_touching_the_folder(capsys, site, old, monkeypatch):
    monkeypatch.setattr(outdir, "is_interactive", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("must not ask about a run that will fail"))
    err = _refused(capsys, site, old, "--property", f"Classification={site.far_raster}")
    assert "does not overlap" in err


# ---------------------------------------------------------------------------
# The question, when someone is at the terminal


@pytest.fixture
def terminal(monkeypatch):
    monkeypatch.setattr(outdir, "is_interactive", lambda: True)

    def answer(value):
        def ask(prompt):
            if isinstance(value, type) and issubclass(value, BaseException):
                raise value
            return value
        monkeypatch.setattr("builtins.input", ask)

    return answer


@pytest.mark.parametrize("reply", ["y", "Y", "yes", " YES "])
def test_yes_replaces(capsys, site, old, terminal, reply):
    (old / "serve.json").write_text("{}")
    terminal(reply)
    assert build(site, old, "--brick", "260") == 0
    err = capsys.readouterr().err
    assert "already holds a vox3dt tileset" in err
    assert "39 tiles" in err
    assert "These files are not vox3dt's and will be kept:\n  serve.json" in err
    assert (old / "serve.json").read_text() == "{}"


@pytest.mark.parametrize("reply", ["n", "no", "", "sure", EOFError, KeyboardInterrupt])
def test_anything_but_yes_cancels(capsys, site, old, terminal, reply):
    terminal(reply)
    err = _refused(capsys, site, old)
    assert "Cancelled. Nothing was deleted." in err


def test_overwrite_never_asks(site, old, terminal, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("--overwrite must not ask"))
    assert build(site, old, "--overwrite") == 0
