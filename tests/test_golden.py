"""Output must match tests/golden/ exactly, and related cases must agree with each other."""

from __future__ import annotations

import gzip

import pytest

import cases


@pytest.mark.parametrize("name", [c.name for c in cases.CASES])
def test_matches_golden(name, results, update_golden):
    case = cases.BY_NAME[name]
    got = results(name).manifest
    if update_golden:
        cases.write_golden(case, got)
        return
    if not cases.golden_path(case).exists():
        pytest.fail(f"no golden file for {name}; run `pytest --update-golden` and review the diff")
    want = cases.load_golden(case)

    problems = []
    if want["files"] != got["files"]:
        missing = sorted(set(want["files"]) - set(got["files"]))
        extra = sorted(set(got["files"]) - set(want["files"]))
        problems.append(f"files: missing {missing[:10]}, unexpected {extra[:10]}")
    changed = sorted(t for t in set(want["tiles"]) & set(got["tiles"]) if want["tiles"][t] != got["tiles"][t])
    if changed:
        problems.append(f"{len(changed)} tile(s) changed: {changed[:10]}")
    for key in ("tileset", "specs", "report"):
        problems += [f"{key} {d}" for d in cases.diff(want[key], got[key], "$")]
    assert not problems, "output differs from golden:\n  " + "\n  ".join(problems)


def test_gzip_twins_match_plain_files(results):
    out = results("gzip").out
    gz = sorted(out.rglob("*.gz"))
    assert len(gz) == len(results("gzip").manifest["tiles"]) + 1  # every tile + tileset.json
    for path in gz:
        assert gzip.decompress(path.read_bytes()) == path.with_suffix("").read_bytes(), path.name


def test_gzip_only_writes_no_plain_copies(results):
    r = results("gzip_only")
    assert not list(r.out.rglob("*.glb"))
    assert not (r.out / "tileset.json").exists()
    assert (r.out / "tileset.json.gz").exists()
    assert (r.out / "specs.json").exists() and (r.out / "report.json").exists()
    assert r.manifest["report"]["gzip"] == "only"


@pytest.mark.parametrize("name", ["gzip", "gzip_only", "spill", "no_crs", "epsg"])
def test_same_tiles_as_draco(name, results):
    """Gzip, memory budget and CRS handling must not change a single tile byte."""
    assert results(name).manifest["tiles"] == results("draco").manifest["tiles"]


@pytest.mark.parametrize("name", ["plain", "instanced", "color5", "props", "epsg"])
def test_same_root_transform_as_draco(name, results):
    """Placement depends only on the LAS and CRS, never on the encoding."""
    want = results("draco").manifest["tileset"]["root"]["transform"]
    assert results(name).manifest["tileset"]["root"]["transform"] == want


def test_spill_really_uses_the_external_sort(results):
    assert results("draco").spills == 0
    assert results("spill").spills > 1
    assert results("spill").manifest["tileset"] == results("draco").manifest["tileset"]


def test_epsg_flag_matches_declared_crs(results):
    epsg, draco = results("epsg").manifest, results("draco").manifest
    assert epsg["tileset"] == draco["tileset"]
    assert epsg["specs"]["georeference"]["epsg_assumed"] is True
    assert draco["specs"]["georeference"]["epsg_assumed"] is False


def test_no_crs_is_not_georeferenced(results):
    m = results("no_crs").manifest
    assert "transform" not in m["tileset"]["root"]
    assert m["report"]["epsg"] is None and m["report"]["origin_lonlat"] is None


def test_properties_cover_fill_and_outside(results, site):
    r = results("props")
    classification = r.manifest["report"]["properties"]["Classification"]
    assert classification["cubes_on_filled_pixels"] > classification["cubes_outside_raster"] > 0
    classes = r.manifest["specs"]["properties"]["Classification"]["classes"]
    assert sorted(classes) == ["0", "1", "3", "4"]
    assert classes["4"]["hex"] == "#88cc44" and classes["4"]["color_source"] == "config"
    assert classes["0"]["color_source"] == "geotiff colour table"
    assert r.manifest["specs"]["properties"]["Elevation"]["kind"] == "continuous"
    assert (r.out / "properties.config.json").read_bytes() == site.config.read_bytes()


def test_property_kind_detected_from_colour_table(results):
    spec = results("props_cli").manifest["specs"]["properties"]["Classification"]
    assert spec["kind"] == "categorical"


def test_float_feature_ids_past_65536_cubes(results):
    assert results("float_ids").manifest["report"]["feature_id_float_tiles"]
    assert results("props").manifest["report"]["feature_id_float_tiles"] == []
