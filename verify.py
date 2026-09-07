"""Structural verification for a generated tileset.

    python verify.py <output-dir> <input.las>

Checks the things that would otherwise fail silently in a viewer:

1. Every .glb parses as valid GLB -- header, chunk layout, 4-byte alignment.
2. Accessor offsets are aligned to their component size and fit their
   bufferView, which fits the BIN chunk.
3. Draco payloads decode, and -- the important one -- the semantic to
   unique_id mapping and each accessor's component count and data type agree
   with the descriptors inside the payload. A decode round-trip alone cannot
   catch a mismatch here, because DracoPy resolves attributes by type; two
   real bugs reached a browser this way.
4. Draco normals survive quantization as unit-length, axis-aligned vectors.
5. The root transform matches EPSG->ECEF of the declared grid origin, that
   origin matches the LAS header's bounding box, the rotation is orthonormal
   and right-handed, and the site lands inside its declared UTM zone.
6. The tile tree is a well-formed quadtree: at most 4 children, child
   geometricError strictly below parent, leaves at 0.
7. Every tile's geometry lies inside its declared bounding volume.
"""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

FAIL: list[str] = []
OK: list[str] = []


def check(cond: bool, label: str) -> bool:
    (OK if cond else FAIL).append(label)
    return cond


def parse_glb(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    magic, version, length = struct.unpack_from("<4sII", raw, 0)
    assert magic == b"glTF", f"{path.name}: bad magic {magic!r}"
    assert version == 2, f"{path.name}: version {version}"
    assert length == len(raw), f"{path.name}: header length {length} != file {len(raw)}"
    offset, js, binary = 12, None, b""
    while offset < len(raw):
        clen, ctype = struct.unpack_from("<I4s", raw, offset)
        data = raw[offset + 8 : offset + 8 + clen]
        if ctype == b"JSON":
            js = json.loads(data)
        elif ctype == b"BIN\x00":
            binary = data
        offset += 8 + clen
        assert offset % 4 == 0, f"{path.name}: chunk not 4-byte aligned"
    assert js is not None, f"{path.name}: no JSON chunk"
    return js, binary


def check_accessors(name: str, gltf: dict, binary: bytes) -> None:
    sizes = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
    counts = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
    for i, acc in enumerate(gltf.get("accessors", [])):
        if "bufferView" not in acc:
            continue  # Draco-supplied; metadata only.
        view = gltf["bufferViews"][acc["bufferView"]]
        stride = sizes[acc["componentType"]] * counts[acc["type"]]
        need = acc["count"] * stride
        start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        check(
            start % sizes[acc["componentType"]] == 0,
            f"{name}: accessor {i} aligned to component size",
        )
        check(need <= view["byteLength"], f"{name}: accessor {i} fits its bufferView")
        check(
            view.get("byteOffset", 0) + view["byteLength"] <= len(binary),
            f"{name}: bufferView {acc['bufferView']} inside BIN chunk",
        )


#: glTF semantic -> Draco GeometryAttribute::Type
_SEMANTIC_ATT = {"POSITION": 0, "NORMAL": 1, "COLOR_0": 2, "TEXCOORD_0": 3}
#: glTF componentType -> Draco DataType
_COMPONENT_DT = {5121: 2, 5123: 4, 5125: 6, 5126: 9}
_TYPE_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
_DT_NAME = {2: "UINT8", 4: "UINT16", 6: "UINT32", 9: "FLOAT32", 10: "FLOAT64"}


def check_draco_attribute_mapping(name: str, gltf: dict, prim: dict, ext: dict, payload: bytes) -> None:
    """The checks a decode round-trip can never make.

    `DracoPy.decode` resolves attributes by *type* (`mesh.points`,
    `mesh.normals`, `mesh.colors`), so it returns correct data no matter what
    unique_ids they carry or what the accessors claim. Two whole classes of
    bug therefore round-trip perfectly and still fail to render:

    1. a wrong semantic -> unique_id mapping (POSITION resolving to the colour
       attribute, which produced `vec4f(vec3<u32>, 1.0)` in LuciadRIA's WGSL);
    2. an accessor whose `type`/`componentType` disagrees with the Draco
       attribute it describes (a VEC4 COLOR_0 accessor over a 3-component
       Draco attribute, which produced `vec4f(vec4<u32>, 1.0)`).

    Both are caught by reading the payload's own descriptors and comparing them
    against the accessors, attribute by attribute.
    """
    from vox3dt.gltfwriter import payload_attributes

    declared = ext["attributes"]
    semantics = sorted(declared, key=lambda s: declared[s])
    for sem in semantics:
        if not check(sem in _SEMANTIC_ATT, f"{name}: known semantic {sem}"):
            return
        if not check(
            sem in prim["attributes"],
            f"{name}: {sem} declared in the draco extension also has an accessor",
        ):
            return

    found = payload_attributes(payload)
    if not check(
        bool(found),
        f"{name}: draco attribute descriptors readable for {semantics}",
    ):
        return

    by_uid = {d["unique_id"]: d for d in found}
    for sem in semantics:
        uid = declared[sem]
        if not check(uid in by_uid, f"{name}: declared unique_id {uid} for {sem} exists"):
            continue
        att = by_uid[uid]
        att_type, data_type, components = att["attribute_type"], att["data_type"], att["num_components"]
        acc = gltf["accessors"][prim["attributes"][sem]]

        # GENERIC (4) is legal for COLOR_0: the extension maps semantics by
        # unique_id, not by Draco attribute type.
        check(
            att_type == _SEMANTIC_ATT[sem] or (sem == "COLOR_0" and att_type == 4),
            f"{name}: {sem} -> draco id {uid} has attribute_type {att_type}",
        )
        check(
            components == _TYPE_COMPONENTS[acc["type"]],
            f"{name}: {sem} accessor {acc['type']} "
            f"({_TYPE_COMPONENTS[acc['type']]} components) matches draco "
            f"({components} components)",
        )
        check(
            data_type == _COMPONENT_DT.get(acc["componentType"]),
            f"{name}: {sem} accessor componentType {acc['componentType']} matches "
            f"draco {_DT_NAME.get(data_type, data_type)}",
        )
    # POSITION must not alias any other attribute.
    others = [declared[s] for s in semantics if s != "POSITION"]
    if "POSITION" in declared:
        check(
            declared["POSITION"] not in others,
            f"{name}: POSITION does not alias another attribute",
        )


def check_draco(name: str, gltf: dict, binary: bytes) -> None:
    for i, prim in enumerate(gltf["meshes"][0]["primitives"]):
        ext = prim.get("extensions", {}).get("KHR_draco_mesh_compression")
        if ext:
            label = name if len(gltf["meshes"][0]["primitives"]) == 1 else f"{name}#{i}"
            _check_draco_primitive(label, gltf, prim, ext, binary)


def _check_draco_primitive(name: str, gltf: dict, prim: dict, ext: dict, binary: bytes) -> None:
    import DracoPy

    view = gltf["bufferViews"][ext["bufferView"]]
    start = view.get("byteOffset", 0)
    payload = binary[start : start + view["byteLength"]]

    check_draco_attribute_mapping(name, gltf, prim, ext, payload)
    mesh = DracoPy.decode(payload)

    declared = gltf["accessors"][prim["attributes"]["POSITION"]]["count"]
    # Draco deduplicates vertices, so the decoded count can be lower than the
    # count declared in the accessor. Both Cesium and the spec accept this; what
    # must hold is that the geometry survives.
    check(len(mesh.points) > 0, f"{name}: draco POSITION decodes ({len(mesh.points)} pts)")
    check(len(mesh.faces) > 0, f"{name}: draco indices decode ({len(mesh.faces)} tris)")
    check(
        mesh.normals is not None and len(mesh.normals) == len(mesh.points),
        f"{name}: draco NORMAL present, matches vertex count",
    )
    check(
        len(mesh.points) <= declared,
        f"{name}: decoded vertex count {len(mesh.points)} <= declared {declared}",
    )
    # Normals must still be axis-aligned unit vectors after quantization.
    if mesh.normals is not None and len(mesh.normals):
        n = np.asarray(mesh.normals, dtype=np.float64)
        norms = np.linalg.norm(n, axis=1)
        check(
            bool(np.all(np.abs(norms - 1.0) < 0.05)),
            f"{name}: draco normals stay unit length (max dev {np.abs(norms - 1).max():.4f})",
        )
        axis_aligned = np.abs(np.abs(n).max(axis=1) - 1.0) < 0.05
        check(
            bool(axis_aligned.mean() > 0.99),
            f"{name}: draco normals stay axis-aligned ({100 * axis_aligned.mean():.1f}%)",
        )


def check_georeference(tileset: dict, las_path: Path) -> None:
    from pyproj import Transformer

    import laspy

    t = tileset["root"]["transform"]
    # Column-major: the 4th column is the ECEF origin.
    ecef_origin = np.array(t[12:15])

    geo = tileset["extras"]["georeference"]
    epsg = geo["horizontal_epsg"]
    o = geo["grid_origin_world"]

    to_ecef = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4978", always_xy=True)
    expect = np.array(to_ecef.transform(o["easting"], o["northing"], o["height"]))
    check(
        float(np.linalg.norm(ecef_origin - expect)) < 1e-3,
        f"tileset transform origin matches EPSG:{epsg} -> ECEF "
        f"(delta {np.linalg.norm(ecef_origin - expect):.6f} m)",
    )

    # And that origin should sit at the LAS bounding box corner the encoder used.
    header = laspy.open(str(las_path)).header
    mins, maxs = np.asarray(header.mins), np.asarray(header.maxs)
    check(
        abs(o["easting"] - mins[0]) < 1e-6,
        f"grid origin easting == LAS min X ({o['easting']:.3f})",
    )
    check(
        abs(o["height"] - mins[2]) < 1e-6,
        f"grid origin height == LAS min Z ({o['height']:.3f})",
    )
    # Z is mirrored, so the origin northing is at the TOP of the box.
    check(
        mins[1] < o["northing"] <= maxs[1] + 1.0,
        f"grid origin northing sits at the box top ({o['northing']:.3f}, "
        f"LAS max Y {maxs[1]:.3f}) -- consistent with the mirrored z axis",
    )

    # Sanity: does it land where the CRS says? UTM 16N covers -90..-84 lon.
    to_geog = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4979", always_xy=True)
    lon, lat, _ = to_geog.transform(o["easting"], o["northing"], o["height"])
    check(-90 <= lon <= -84, f"longitude {lon:.4f} inside UTM zone 16N")
    check(0 < lat < 84, f"latitude {lat:.4f} in the northern hemisphere")
    print(f"    -> site at lon {lon:.6f}, lat {lat:.6f}")

    # The ENU basis columns must be orthonormal, or nothing will sit level.
    m = np.array(t).reshape(4, 4).T  # column-major -> rows
    basis = m[:3, :3]
    check(
        float(np.abs(basis @ basis.T - np.eye(3)).max()) < 1e-9,
        "root transform rotation is orthonormal",
    )
    check(
        abs(float(np.linalg.det(basis)) - 1.0) < 1e-9,
        f"root transform is right-handed (det {np.linalg.det(basis):.9f})",
    )


def check_bounds(root_dir: Path, tileset: dict) -> None:
    """Every tile's geometry must lie inside its declared bounding box."""

    def walk(node):
        yield node
        for c in node.get("children", []):
            yield from walk(c)

    checked = 0
    for node in walk(tileset["root"]):
        uri = node.get("content", {}).get("uri")
        if not uri:
            continue
        gltf, binary = parse_glb(root_dir / uri)
        box = node["boundingVolume"]["box"]
        centre = np.array(box[0:3])
        half = np.array([box[3], box[7], box[11]])

        pts = []
        for node_j in gltf.get("nodes", []):
            inst = node_j.get("extensions", {}).get("EXT_mesh_gpu_instancing")
            if inst:
                acc = gltf["accessors"][inst["attributes"]["TRANSLATION"]]
                v = gltf["bufferViews"][acc["bufferView"]]
                off = v.get("byteOffset", 0) + acc.get("byteOffset", 0)
                pts.append(
                    np.frombuffer(binary, "<f4", acc["count"] * 3, off).reshape(-1, 3)
                )
        if not pts:
            # baked: use each primitive's POSITION accessor min/max instead
            for prim in gltf["meshes"][0]["primitives"]:
                acc = gltf["accessors"][prim["attributes"]["POSITION"]]
                if "min" in acc:
                    pts.append(np.array([acc["min"], acc["max"]], dtype=np.float64))
            if not pts:
                continue

        p = np.concatenate(pts)
        # glTF y-up -> 3D Tiles z-up ENU: (x, y, z) -> (x, -z, y)
        enu = np.stack([p[:, 0], -p[:, 2], p[:, 1]], axis=1)
        # Allow half a cell of slack: instance translations are cube centres.
        slack = float(node["geometricError"]) if node["geometricError"] else 1.0
        inside = np.all(np.abs(enu - centre) <= half + slack + 1e-6, axis=1)
        if not check(bool(inside.all()), f"{uri}: geometry inside its bounding box"):
            worst = np.abs(enu - centre).max(axis=0) - half
            print(f"      overflow by {worst}")
        checked += 1
    print(f"    -> {checked} tiles bounds-checked")


def check_tree(tileset: dict) -> None:
    def walk(node, depth=0):
        yield node, depth
        for c in node.get("children", []):
            yield from walk(c, depth + 1)

    nodes = list(walk(tileset["root"]))
    check(len(nodes) > 0, "tileset has tiles")
    for node, _ in nodes:
        kids = node.get("children", [])
        check(len(kids) <= 4, "no node has more than 4 children (quadtree)")
        if kids:
            check(
                all(k["geometricError"] < node["geometricError"] for k in kids),
                "child geometricError strictly less than parent",
            )
        check(node.get("refine") == "REPLACE", "refine is REPLACE")
    leaves = [n for n, _ in nodes if not n.get("children")]
    check(
        all(n["geometricError"] == 0 for n in leaves),
        "leaf tiles have geometricError 0",
    )
    check(
        tileset["geometricError"] > tileset["root"]["geometricError"],
        "tileset geometricError exceeds the root tile's",
    )
    print(f"    -> {len(nodes)} tiles, {len(leaves)} leaves, depth {max(d for _, d in nodes)}")


def main(root: Path, las: Path) -> int:
    print(f"Verifying {root}")
    tileset = json.loads((root / "tileset.json").read_text())

    print("  tree structure")
    check_tree(tileset)
    print("  georeference")
    check_georeference(tileset, las)
    print("  glb integrity")
    n_draco = 0
    for glb in sorted((root / "content").glob("*.glb")):
        gltf, binary = parse_glb(glb)
        check_accessors(glb.name, gltf, binary)
        before = len(OK) + len(FAIL)
        check_draco(glb.name, gltf, binary)
        if len(OK) + len(FAIL) > before:
            n_draco += 1
    print(f"    -> {len(list((root / 'content').glob('*.glb')))} glb files, {n_draco} draco-encoded")
    print("  bounding volumes")
    check_bounds(root, tileset)

    print(f"\n{len(OK)} checks passed, {len(FAIL)} failed")
    for f in dict.fromkeys(FAIL):
        print(f"  FAIL  {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
