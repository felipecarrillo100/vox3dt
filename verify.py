"""Structural verification for a generated tileset.

    python verify.py <output-dir> <input.las> [tileset-file-name]

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
5. The grid origin matches the LAS header's bounding box, the root transform's
   axes are near-orthogonal, near unit scale and right-handed, and the site
   lands inside its CRS's area of use.
6. The tile tree is a well-formed quadtree: at most 4 children, child
   geometricError strictly below parent, leaves at 0, and every child's box
   inside its parent's. The root box is within one root cell of the LAS bounds.
7. Every tile's geometry lies inside its declared bounding volume.
8. Placement: the root transform puts a lattice of grid points spanning the
   site where the projection says, within half a voxel horizontally and one
   vertically; and the outermost cube edges, pushed through it to ECEF and back
   into the LAS CRS, land on the LAS bounds -- west and south on them, east and
   north overhanging by less than one voxel -- at all four corners.
9. Property tiles (--property): EXT_mesh_features points _FEATURE_ID_0 at an
   EXT_structural_metadata property table whose columns are 8-byte aligned and
   hold one value per feature; _FEATURE_ID_0 is an integer per vertex, constant
   over each cube, mapping cubes to rows one-to-one; FeatureID is unique across
   the tileset, and property values lie inside tileset.json's declared ranges.
"""

from __future__ import annotations

import gzip
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
    """Parse a tile by its .glb name, reading the .glb.gz twin when only that exists (--gzip-only)."""
    if path.exists():
        return parse_glb_bytes(path.name, path.read_bytes())
    return parse_glb_bytes(path.name, gzip.decompress(path.with_name(path.name + ".gz").read_bytes()))


def tile_paths(content: Path) -> list[Path]:
    """Every tile, by its .glb name, whether it is on disk raw or only gzipped."""
    names = {p.name for p in content.glob("*.glb")} | {p.name[:-3] for p in content.glob("*.glb.gz")}
    return [content / n for n in sorted(names)]


def parse_glb_bytes(name: str, raw: bytes) -> tuple[dict, bytes]:
    magic, version, length = struct.unpack_from("<4sII", raw, 0)
    assert magic == b"glTF", f"{name}: bad magic {magic!r}"
    assert version == 2, f"{name}: version {version}"
    assert length == len(raw), f"{name}: header length {length} != content {len(raw)}"
    offset, js, binary = 12, None, b""
    while offset < len(raw):
        clen, ctype = struct.unpack_from("<I4s", raw, offset)
        data = raw[offset + 8 : offset + 8 + clen]
        if ctype == b"JSON":
            js = json.loads(data)
        elif ctype == b"BIN\x00":
            binary = data
        offset += 8 + clen
        assert offset % 4 == 0, f"{name}: chunk not 4-byte aligned"
    assert js is not None, f"{name}: no JSON chunk"
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
_SEMANTIC_ATT = {"POSITION": 0, "NORMAL": 1, "COLOR_0": 2, "TEXCOORD_0": 3, "_FEATURE_ID_0": 4}
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


def load_georeference(root: Path, tileset: dict) -> dict:
    """specs.json (current builds) or tileset.json extras (older ones)."""
    specs = root / "specs.json"
    if specs.exists():
        return json.loads(specs.read_text())["georeference"]
    return tileset["extras"]["georeference"]


def check_georeference(tileset: dict, las_path: Path, geo: dict) -> None:
    from pyproj import CRS, Transformer

    import laspy

    t = tileset["root"]["transform"]
    epsg = geo["horizontal_epsg"]
    o = geo["grid_origin_world"]
    vs = geo["voxel_size"]

    # The grid origin must sit at the LAS bounding-box corner the encoder used.
    header = laspy.open(str(las_path)).header
    mins, maxs = np.asarray(header.mins), np.asarray(header.maxs)
    nx, ny, nz = (int(math.floor((maxs[i] - mins[i]) / vs)) + 1 for i in (0, 2, 1))
    check(
        abs(o["easting"] - mins[0]) < 1e-6,
        f"grid origin easting == LAS min X ({o['easting']:.3f})",
    )
    check(
        abs(o["height"] - mins[2]) < 1e-6,
        f"grid origin height == LAS min Z ({o['height']:.3f})",
    )
    # Z is mirrored, so the origin northing is the north edge of the
    # northernmost row: exactly minN + nz * voxel_size.
    check(
        abs(o["northing"] - (mins[1] + nz * vs)) < 1e-6,
        f"grid origin northing == LAS min Y + nz*voxel ({o['northing']:.3f}) -- "
        "north edge of the northernmost row",
    )

    # The root transform must put tile-space metres where the projection says:
    # E = ox + u, N = oy + v, H = oz + w. Checked on a lattice spanning the
    # whole grid, back in the source CRS. A single affine frame cannot follow
    # the Earth's curvature, so allow half a voxel horizontally, one vertically.
    m = np.array(t).reshape(4, 4).T  # column-major -> rows
    u, v, w = (a.ravel() for a in np.meshgrid(
        np.linspace(0, nx * vs, 11), np.linspace(-nz * vs, 0, 11), np.linspace(0, ny * vs, 3)))
    placed = (m @ np.stack([u, v, w, np.ones_like(u)]))[:3]
    back = Transformer.from_crs("EPSG:4978", f"EPSG:{epsg}", always_xy=True)
    e, n, h = back.transform(*placed)
    horiz = float(np.hypot(e - (o["easting"] + u), n - (o["northing"] + v)).max())
    vert = float(np.abs(h - (o["height"] + w)).max())
    check(horiz < 0.5 * vs, f"root transform places the grid within half a voxel horizontally (worst {horiz:.3f} m)")
    check(vert < vs, f"root transform places the grid within one voxel vertically (worst {vert:.3f} m)")

    # Sanity: does it land inside the CRS's own area of use?
    to_geog = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4979", always_xy=True)
    lon, lat, _ = to_geog.transform(o["easting"], o["northing"], o["height"])
    area = CRS.from_epsg(epsg).area_of_use
    if area is not None:
        check(
            area.west - 1e-9 <= lon <= area.east + 1e-9 and area.south - 1e-9 <= lat <= area.north + 1e-9,
            f"origin lon {lon:.4f}, lat {lat:.4f} inside EPSG:{epsg}'s area of use",
        )
    print(f"    -> site at lon {lon:.6f}, lat {lat:.6f}; worst placement {horiz:.3f} m horizontal, {vert:.3f} m vertical")

    # The basis is a fitted affine, not a pure rotation: it carries the
    # projection's scale factor (well under 0.1% off unity for UTM) and its
    # convergence. It must still be near-orthogonal, near unit and right-handed.
    basis = m[:3, :3]
    gram = basis.T @ basis
    scale = np.sqrt(np.diag(gram))
    check(bool(np.all(np.abs(scale - 1.0) < 1e-3)), f"root transform axis scales near 1 ({scale.round(6).tolist()})")
    off = np.abs(gram / np.outer(scale, scale) - np.eye(3)).max()
    check(off < 1e-3, f"root transform axes near-orthogonal (worst cosine {off:.2e})")
    check(float(np.linalg.det(basis)) > 0, f"root transform is right-handed (det {np.linalg.det(basis):.9f})")


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


def check_root_extent(tileset: dict, geo: dict, las_path: Path) -> None:
    """The root box should hug the LAS bounds: overhang under one root cell."""
    import laspy

    header = laspy.open(str(las_path)).header
    mins, maxs = np.asarray(header.mins), np.asarray(header.maxs)
    o = geo["grid_origin_world"]
    b = tileset["root"]["boundingVolume"]["box"]
    c, h = np.array(b[0:3]), np.array([b[3], b[7], b[11]])
    lo, hi = c - h, c + h  # ENU metres relative to the grid origin
    las_lo = np.array([mins[0] - o["easting"], mins[1] - o["northing"], mins[2] - o["height"]])
    las_hi = np.array([maxs[0] - o["easting"], maxs[1] - o["northing"], maxs[2] - o["height"]])
    root_cell = float(tileset["root"]["geometricError"]) or geo["voxel_size"]
    over = np.maximum(las_lo - lo, 0) + np.maximum(hi - las_hi, 0)
    check(bool(np.all(lo <= las_lo + 1e-6) and np.all(hi >= las_hi - 1e-6)),
          "root box contains the LAS bounds")
    check(bool(np.all(over <= root_cell + 1e-6)),
          f"root box overhangs the LAS bounds by under one root cell ({root_cell:g} m): "
          f"overhang E/N/U {over.round(2).tolist()} m")


def check_placement(root_dir: Path, tileset: dict, geo: dict, las_path: Path) -> None:
    """Outermost cube edges, through the root transform, against the LAS bounds."""
    from pyproj import Transformer

    import laspy

    header = laspy.open(str(las_path)).header
    mins, maxs = np.asarray(header.mins), np.asarray(header.maxs)
    vs = geo["voxel_size"]
    lo = np.full(3, np.inf)
    hi = -lo
    leaves = []

    def walk(node):
        if node.get("children"):
            for c in node["children"]:
                walk(c)
        elif node.get("content"):
            leaves.append(node["content"]["uri"])

    walk(tileset["root"])
    for uri in leaves:
        gltf, _binary = parse_glb(root_dir / uri)
        for prim in gltf["meshes"][0]["primitives"]:
            acc = gltf["accessors"][prim["attributes"]["POSITION"]]
            lo = np.minimum(lo, acc["min"])
            hi = np.maximum(hi, acc["max"])
    M = np.array(tileset["root"]["transform"]).reshape(4, 4).T
    to_crs = Transformer.from_crs("EPSG:4978", f"EPSG:{geo['horizontal_epsg']}", always_xy=True)

    # The fitted root transform is exact to well under half a voxel horizontally
    # (see check_georeference), so edges are compared to that tolerance.
    tol = 0.5 * vs

    def world(x, y, z):  # glTF y-up -> tile space (x, -z, y) -> ECEF -> LAS CRS
        return np.array(to_crs.transform(*(M @ np.array([x, -z, y, 1.0]))[:3]))

    corners = {"NW": (lo[0], lo[2]), "NE": (hi[0], lo[2]), "SW": (lo[0], hi[2]), "SE": (hi[0], hi[2])}
    for name, (x, z) in corners.items():
        e, n, hgt = world(x, lo[1], z)
        want_e = mins[0] if name.endswith("W") else None
        if want_e is not None:
            check(abs(e - want_e) < tol, f"{name} corner: west edge on LAS min X (off {e - want_e:+.3f} m)")
        else:
            check(-tol <= e - maxs[0] < vs + tol, f"{name} corner: east edge overhangs LAS max X by < 1 voxel ({e - maxs[0]:+.3f} m)")
        if name.startswith("S"):
            check(abs(n - mins[1]) < tol, f"{name} corner: south edge on LAS min Y (off {n - mins[1]:+.3f} m)")
        else:
            check(-tol <= n - maxs[1] < vs + tol, f"{name} corner: north edge overhangs LAS max Y by < 1 voxel ({n - maxs[1]:+.3f} m)")
    _e, _n, bottom = world(lo[0], lo[1], lo[2])
    check(abs(bottom - mins[2]) < vs, f"bottom on LAS min Z within one voxel of vertical tolerance (off {bottom - mins[2]:+.3f} m)")
    _e, _n, top = world(lo[0], hi[1], lo[2])
    check(-vs <= top - maxs[2] < 2 * vs, f"top overhangs LAS max Z by < 1 voxel, give or take one voxel of vertical tolerance ({top - maxs[2]:+.3f} m)")


def check_metadata(root_dir: Path, tileset: dict) -> None:
    """Property tables: schema, alignment, _FEATURE_ID_0 per cube, FeatureID uniqueness, ranges."""
    import DracoPy

    declared = tileset.get("properties", {})
    tiles = tile_paths(root_dir / "content")
    dtypes = {"INT8": "<i1", "UINT8": "<u1", "INT16": "<i2", "UINT16": "<u2", "INT32": "<i4",
              "UINT32": "<u4", "INT64": "<i8", "UINT64": "<u8", "FLOAT32": "<f4", "FLOAT64": "<f8"}
    feature_ids = []
    for path in tiles:
        gltf, binary = parse_glb(path)
        prim = gltf["meshes"][0]["primitives"][0]
        meta = gltf.get("extensions", {}).get("EXT_structural_metadata")
        mf = prim.get("extensions", {}).get("EXT_mesh_features")
        if not check(meta is not None and mf is not None, f"{path.name}: EXT_mesh_features + EXT_structural_metadata present"):
            continue
        used = gltf.get("extensionsUsed", [])
        check("EXT_mesh_features" in used and "EXT_structural_metadata" in used,
              f"{path.name}: metadata extensions listed in extensionsUsed")
        fids = mf["featureIds"]
        if not check(len(fids) == 1 and fids[0].get("attribute") == 0 and fids[0].get("propertyTable") == 0,
                     f"{path.name}: one feature id set, _FEATURE_ID_0 -> property table 0"):
            continue
        n = fids[0]["featureCount"]
        table = meta["propertyTables"][0]
        cls = meta["schema"]["classes"][table["class"]]["properties"]
        check(table["count"] == n, f"{path.name}: property table count equals featureCount")
        check("FeatureID" in table["properties"], f"{path.name}: property table has FeatureID")
        cols = {}
        for name, desc in table["properties"].items():
            comp = cls[name]["componentType"]
            view = gltf["bufferViews"][desc["values"]]
            off = view.get("byteOffset", 0)
            check(off % 8 == 0, f"{path.name}: {name} bufferView 8-byte aligned")
            check(view["byteLength"] == n * np.dtype(dtypes[comp]).itemsize,
                  f"{path.name}: {name} holds one {comp} per feature")
            cols[name] = np.frombuffer(binary, dtypes[comp], n, off)
        feature_ids.append(cols["FeatureID"])
        for name, vals in cols.items():
            if name in declared:
                d = declared[name]
                check(bool(vals.min() >= d["minimum"] and vals.max() <= d["maximum"]),
                      f"{name} values inside tileset.json's declared range")

        ext = prim.get("extensions", {}).get("KHR_draco_mesh_compression")
        if ext:
            view = gltf["bufferViews"][ext["bufferView"]]
            mesh = DracoPy.decode(binary[view.get("byteOffset", 0) : view.get("byteOffset", 0) + view["byteLength"]])
            pos = mesh.get_attribute_by_unique_id(ext["attributes"]["POSITION"])["data"]
            nrm = mesh.get_attribute_by_unique_id(ext["attributes"]["NORMAL"])["data"]
            fid = np.asarray(mesh.get_attribute_by_unique_id(ext["attributes"]["_FEATURE_ID_0"])["data"]).ravel()
            faces = mesh.faces
        else:
            def read(acc_i, comps):
                acc = gltf["accessors"][acc_i]
                v = gltf["bufferViews"][acc["bufferView"]]
                dt = {5126: "<f4", 5123: "<u2", 5125: "<u4"}[acc["componentType"]]
                return np.frombuffer(binary, dt, acc["count"] * comps, v.get("byteOffset", 0) + acc.get("byteOffset", 0))
            pos = read(prim["attributes"]["POSITION"], 3).reshape(-1, 3)
            nrm = read(prim["attributes"]["NORMAL"], 3).reshape(-1, 3)
            fid = read(prim["attributes"]["_FEATURE_ID_0"], 1)
            faces = read(prim["indices"], 1).reshape(-1, 3)
        check(bool(np.all(fid == np.rint(fid))), f"{path.name}: _FEATURE_ID_0 values are integers")
        fid = np.rint(fid).astype(np.int64)
        check(bool(fid.min() >= 0 and fid.max() < n), f"{path.name}: _FEATURE_ID_0 inside [0, featureCount)")
        fb = fid[faces]
        check(bool(np.all(fb[:, 0] == fb[:, 1]) and np.all(fb[:, 0] == fb[:, 2])),
              f"{path.name}: _FEATURE_ID_0 constant over each triangle")
        # Map each triangle to its cube: step a quarter cell inward from the face.
        cs = _cell_size_of(tileset, path.name)
        cen = pos[faces].mean(axis=1) - nrm[faces[:, 0]] * (cs * 0.25)
        cube = np.floor(cen / cs).astype(np.int64)
        key = (cube[:, 0] * 1_000_003 + cube[:, 1]) * 1_000_003 + cube[:, 2]
        pairs = np.unique(np.stack([key, fb[:, 0]], axis=1), axis=0)
        n_cubes = len(np.unique(key))
        check(len(pairs) == n_cubes == len(np.unique(fb[:, 0])) == n,
              f"{path.name}: cubes map one-to-one onto property rows ({n_cubes} cubes, featureCount {n})")
    if feature_ids:
        allf = np.concatenate(feature_ids)
        check(len(np.unique(allf)) == len(allf), f"FeatureID unique across the tileset ({len(allf):,} cubes)")
        check(int(allf.min()) == 0 and int(allf.max()) == len(allf) - 1, "FeatureID covers 0..N-1 with no gaps")
    print(f"    -> {len(tiles)} tiles, {sum(len(f) for f in feature_ids):,} features")


def _cell_size_of(tileset: dict, file_name: str) -> float:
    """A tile's cell size: its geometricError, or the voxel size at the leaves."""
    if not hasattr(_cell_size_of, "cache"):
        cache = {}

        def walk(node, parent_err=None):
            uri = node.get("content", {}).get("uri")
            err = node["geometricError"]
            if uri:
                cache[uri.split("/")[-1]] = err if err else parent_err / 2
            for c in node.get("children", []):
                walk(c, err)

        # The tileset's own geometricError stands in as the root's parent, for a single-tile tileset.
        walk(tileset["root"], tileset["geometricError"])
        _cell_size_of.cache = cache
    return _cell_size_of.cache[file_name]


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
    # Each child's box inside its parent's (all boxes share the root frame).
    def extent(node):
        b = node["boundingVolume"]["box"]
        c, h = np.array(b[0:3]), np.array([b[3], b[7], b[11]])
        return c - h, c + h

    for node, _ in nodes:
        lo, hi = extent(node)
        for kid in node.get("children", []):
            klo, khi = extent(kid)
            check(
                bool(np.all(klo >= lo - 1e-6) and np.all(khi <= hi + 1e-6)),
                "every child's bounding box lies inside its parent's",
            )
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


def main(root: Path, las: Path, tileset_name: str = "tileset.json") -> int:
    print(f"Verifying {root / tileset_name}")
    path = root / tileset_name
    # --gzip-only leaves only tileset.json.gz.
    raw = path.read_bytes() if path.exists() else gzip.decompress(path.with_name(path.name + ".gz").read_bytes())
    tileset = json.loads(raw)
    geo = load_georeference(root, tileset)

    print("  tree structure")
    check_tree(tileset)
    check_root_extent(tileset, geo, las)
    print("  georeference")
    check_georeference(tileset, las, geo)
    print("  placement")
    check_placement(root, tileset, geo, las)
    print("  content integrity")
    n_draco = 0
    has_metadata = False
    files = tile_paths(root / "content")
    for path in files:
        gltf, binary = parse_glb(path)
        has_metadata |= "EXT_structural_metadata" in gltf.get("extensionsUsed", [])
        check_accessors(path.name, gltf, binary)
        before = len(OK) + len(FAIL)
        check_draco(path.name, gltf, binary)
        if len(OK) + len(FAIL) > before:
            n_draco += 1
    print(f"    -> {len(files)} tiles, {n_draco} draco-encoded")
    if has_metadata:
        print("  property tables")
        check_metadata(root, tileset)
    print("  bounding volumes")
    check_bounds(root, tileset)

    print(f"\n{len(OK)} checks passed, {len(FAIL)} failed")
    for f in dict.fromkeys(FAIL):
        print(f"  FAIL  {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2]), *sys.argv[3:4]))
