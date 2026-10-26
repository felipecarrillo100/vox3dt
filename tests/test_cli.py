"""Command-line behaviour: errors stay clear, --info and --version keep working."""

from __future__ import annotations

import json

import pytest

from vox3dt import __version__
from vox3dt.cli import main


def _usage_error(capsys, argv) -> str:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
    return capsys.readouterr().err


@pytest.mark.parametrize("extra, message", [
    (["--draco", "--mode", "instanced"], "no effect in instanced mode"),
    (["--epsg", "999999"], "not a recognized EPSG code"),
    (["--property", "FeatureID={landcover}"], "is reserved"),
    (["--property", "Classification"], "expects NAME=PATH"),
    (["--property", "Classification={landcover}", "--property", "Classification={landcover}"], "given twice"),
    (["--property", "Classification={missing}"], "raster not found"),
    (["--properties", "{missing}"], "--properties file not found"),
    (["--mode", "instanced", "--property", "Classification={landcover}"], "need --mode baked"),
    (["--voxel-size", "0"], "--voxel-size must be positive"),
    (["--brick", "1"], "--brick must be at least 2"),
])
def test_usage_errors(capsys, site, tmp_path, extra, message):
    fill = {"landcover": site.landcover, "missing": tmp_path / "missing.tif"}
    argv = ["-i", str(site.las), "-o", str(tmp_path / "out"), *(a.format(**fill) for a in extra)]
    assert message in _usage_error(capsys, argv)


def test_output_required_without_info(capsys, site):
    assert "--output is required" in _usage_error(capsys, ["-i", str(site.las)])


def test_missing_input(capsys, tmp_path):
    assert "input not found" in _usage_error(capsys, ["-i", str(tmp_path / "nope.las"), "-o", str(tmp_path)])


@pytest.mark.parametrize("las, raster, message", [
    ("las", "far_raster", "does not overlap the LAS bounds"),
    ("las_no_crs", "landcover", "needs a georeferenced LAS"),
])
def test_property_errors_fail_before_reading_points(capsys, site, tmp_path, las, raster, message):
    out = tmp_path / "out"
    rc = main(["-i", str(getattr(site, las)), "-o", str(out), "-q",
               "--property", f"Classification={getattr(site, raster)}"])
    assert rc == 1
    assert message in capsys.readouterr().err
    assert not list((out / "content").glob("*")), "no tile may be written"


def test_epsg_ignored_when_las_declares_a_crs(site, tmp_path):
    out = tmp_path / "out"
    assert main(["-i", str(site.las), "-o", str(out), "-q", "--epsg", "32723"]) == 0
    assert "ignoring --epsg 32723" in (out / "log" / "conversion.log").read_text()
    assert json.loads((out / "report.json").read_text())["epsg"] == 31983


def test_info(capsys, site):
    assert main(["-i", str(site.las), "--info"]) == 0
    out = capsys.readouterr().out
    assert "31983" in out
    assert "94,117" in out or "94117" in out


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"vox3dt {__version__}"
