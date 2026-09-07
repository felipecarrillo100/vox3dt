"""Write one tile's cubes as a self-contained ``.glb``.

Two encodings, both of which colour geometry through ``COLOR_0``:

``baked``
    Ordinary indexed triangles, **face-culled**: only faces touching empty
    space are emitted, so an interior cube contributes nothing and a typical
    surface cube contributes 1-3 quads instead of 6. Optionally compressed with
    ``KHR_draco_mesh_compression``. This is the proven path.

``instanced``
    One 24-vertex cube reused by every cube in the tile via
    ``EXT_mesh_gpu_instancing`` -- roughly 12 bytes per cube, and the same
    draw-call economics as ``THREE.InstancedMesh``. Colours are quantized to a
    small palette and split into one node per palette entry.

What LuciadRIA accepts (measured, not assumed)
----------------------------------------------
Three things were established the hard way while getting Draco to load:

1. **Integer ``COLOR_0`` does not work.** A normalized ``UNSIGNED_BYTE``
   colour out of a Draco payload is spec-legal and renders fine *without*
   Draco, but LuciadRIA carries the integer type into its generated WGSL and
   emits ``vec4f(vec4<u32>, 1.0)``, which has no matching constructor, so the
   shader will not compile and the layer never loads.
2. **Material ``baseColorFactor`` is ignored.** A tile split into one primitive
   per palette entry, each with a solid material, loads without error and
   renders entirely grey.
3. **Float ``COLOR_0`` works.** So colour is always FLOAT32 here. Under Draco
   it rides a **GENERIC** attribute, which is legal because the extension maps
   glTF semantics by ``unique_id``, not by Draco attribute type.

Point 2 is why the instanced encoder bakes each palette entry's colour into its
own 24-vertex ``COLOR_0`` block rather than relying on a material: about 288
bytes per palette entry per tile, and colour arrives through the only channel
that actually renders. A material is written alongside anyway, for viewers that
do honour it.

On Draco
--------
``KHR_draco_mesh_compression`` compresses *mesh primitives*, so it applies to
``baked`` only -- in ``instanced`` mode the mesh is a single cube and every byte
of volume lives in the instance buffers, which Draco does not touch. For those,
gzip at the CDN (which this pipeline emits with ``--gzip``) or
``EXT_meshopt_compression`` are the relevant tools.

Where Draco does apply it does well: voxel faces sit on an integer lattice, so
position quantization is lossless at enough bits, and axis-aligned normals take
only 6 distinct values.
"""

from __future__ import annotations

import functools
import json
import struct

import numpy as np

# glTF component types and buffer targets.
_FLOAT = 5126
_UBYTE = 5121
_USHORT = 5123
_UINT = 5125
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963

# Draco GeometryAttribute::Type and DataType enum values.
_D_POSITION, _D_NORMAL, _D_COLOR, _D_GENERIC = 0, 1, 2, 4

# Face order must match pyramid._NEIGHBOURS: +X, -X, +Y, -Y, +Z, -Z.
# For each face: (normal, tangent, bitangent) with tangent x bitangent = normal,
# so the quad (-t-b, t-b, t+b, -t+b) winds counter-clockwise seen from outside
# (glTF's default front face).
_FACES = [
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
    ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    ((0, 0, -1), (0, 1, 0), (1, 0, 0)),
]


# ---------------------------------------------------------------------------
# Colour handling
# ---------------------------------------------------------------------------

def quantize_palette(rgb: np.ndarray, bits: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Reduce colours to at most ``(2**bits)**3`` entries. Instanced mode only.

    Each channel is bucketed to ``2**bits`` levels; a bucket's palette colour is
    the *mean* of the colours that fell in it, so the result tracks the real
    data rather than snapping to a fixed grid. Returns ``(palette, index)``.
    """
    shift = 8 - bits
    buckets = (rgb.astype(np.uint16) >> shift).astype(np.int64)
    keys = (buckets[:, 0] << (2 * bits)) | (buckets[:, 1] << bits) | buckets[:, 2]
    uniq, index = np.unique(keys, return_inverse=True)
    palette = np.zeros((len(uniq), 3), dtype=np.uint8)
    for c in range(3):
        sums = np.bincount(index, weights=rgb[:, c].astype(np.float64), minlength=len(uniq))
        counts = np.bincount(index, minlength=len(uniq))
        palette[:, c] = np.rint(sums / np.maximum(counts, 1)).clip(0, 255).astype(np.uint8)
    return palette, index.astype(np.int64)


def normalize_colors(rgb: np.ndarray, bits: int = 8) -> np.ndarray:
    """0-255 colours -> 0..1 FLOAT32, optionally on a reduced level grid.

    At ``bits >= 8`` (the default) this is a plain ``/255``, which reproduces an
    uncompressed normalized ``UNSIGNED_BYTE`` colour **exactly**: that is what
    such an attribute becomes on the GPU. Verified end to end -- max and mean
    per-channel error against the uncompressed variant are both 0.

    Below 8 bits the grid is **endpoint-preserving**::

        q = round(v / 255 * (2**bits - 1)) ;  out = q / (2**bits - 1)

    so 0 stays 0 and 255 stays 1.0. An earlier version snapped to bucket
    *centres*, which minimizes average error but cannot reach either endpoint --
    at 4 bits it turned black into 8/255 and white into 247/255, visibly lifting
    shadows and dimming highlights for no size saving. Only banding remains as a
    quantization artifact now.

    Reducing bits shrinks a Draco payload because repeated values are what its
    entropy coder and gzip both exploit; DracoPy does not quantize generic
    attributes itself, so this is the only lever on their size.
    """
    if bits <= 0 or bits >= 8:
        return rgb.astype(np.float32) / 255.0
    levels = (1 << bits) - 1
    q = np.rint(rgb.astype(np.float64) / 255.0 * levels)
    return (q / levels).astype(np.float32)


# ---------------------------------------------------------------------------
# Draco attribute-id discovery
# ---------------------------------------------------------------------------

def payload_attributes(payload: bytes) -> list[dict]:
    """Authoritative attribute descriptors for a Draco payload.

    ``DracoPy.decode(...).attributes`` reports each attribute's ``unique_id``,
    ``attribute_type``, ``data_type`` and ``num_components`` directly.
    """
    import DracoPy

    return [
        {
            "unique_id": int(a["unique_id"]),
            "attribute_type": int(a["attribute_type"]),
            "data_type": int(a["data_type"]),
            "num_components": int(a["num_components"]),
            "name": a.get("name"),
        }
        for a in DracoPy.decode(payload).attributes
    ]


@functools.lru_cache(maxsize=8)
def draco_attribute_ids(with_colors: bool = True) -> dict[str, int]:
    """Probe DracoPy with a tiny mesh to learn how it numbers attributes.

    ``KHR_draco_mesh_compression`` maps glTF semantics to Draco ``unique_id``s,
    which are assigned in attribute *creation* order -- not glTF's semantic
    order, and not stably across attribute sets. DracoPy adds POSITION **last**,
    so with normals and colours present this returns
    ``{COLOR_0: 0, NORMAL: 1, POSITION: 2}``.

    Hardcoding the intuitive ``{POSITION: 0, NORMAL: 1, COLOR_0: 2}`` pointed
    POSITION at the colour attribute and broke the LuciadRIA layer, which is
    why it is probed rather than assumed. Ids depend on which attributes exist,
    not on vertex count, so a 4-vertex probe is representative. Cached.
    """
    import DracoPy

    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    kwargs: dict = {"normals": np.tile(np.array([0.0, 0.0, 1.0]), (4, 1))}
    if with_colors:
        kwargs["generic_attributes"] = {
            "COLOR": np.array(
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [0.2, 0.4, 0.6]],
                dtype=np.float32,
            )
        }

    payload = DracoPy.encode(points, faces=faces, quantization_bits=14, compression_level=7, **kwargs)
    semantic_of = {_D_POSITION: "POSITION", _D_NORMAL: "NORMAL", _D_GENERIC: "COLOR_0"}
    ids: dict[str, int] = {}
    for att in payload_attributes(payload):
        semantic = semantic_of.get(att["attribute_type"])
        if semantic and semantic not in ids:
            ids[semantic] = att["unique_id"]

    expected = {"POSITION", "NORMAL"} | ({"COLOR_0"} if with_colors else set())
    if set(ids) != expected:
        raise RuntimeError(
            f"Draco probe returned attributes {sorted(ids)}, expected {sorted(expected)}. "
            "Refusing to write a KHR_draco_mesh_compression tileset with a guessed "
            "mapping -- a wrong mapping loads as corrupt geometry, not as an error."
        )
    return ids


# ---------------------------------------------------------------------------
# GLB assembly
# ---------------------------------------------------------------------------

class _Buffer:
    """Accumulates binary blobs plus the accessor/bufferView JSON for them."""

    def __init__(self) -> None:
        self.blob = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []

    def _view(self, data: bytes, target: int | None) -> int:
        while len(self.blob) % 4:
            self.blob.append(0)
        offset = len(self.blob)
        self.blob.extend(data)
        view: dict = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        self.views.append(view)
        return len(self.views) - 1

    def _accessor(self, array, component_type, type_, normalized, with_bounds) -> dict:
        acc: dict = {
            "componentType": component_type,
            "count": int(array.shape[0]) if array.ndim > 1 else int(array.size),
            "type": type_,
        }
        if normalized:
            acc["normalized"] = True
        if with_bounds:
            a = array.reshape(array.shape[0], -1) if array.ndim > 1 else array.reshape(-1, 1)
            acc["min"] = [float(v) for v in a.min(axis=0)]
            acc["max"] = [float(v) for v in a.max(axis=0)]
        return acc

    def add(
        self,
        array: np.ndarray,
        component_type: int,
        type_: str,
        target: int | None,
        normalized: bool = False,
        with_bounds: bool = False,
    ) -> int:
        """An accessor backed by real bytes in the BIN chunk."""
        acc = self._accessor(array, component_type, type_, normalized, with_bounds)
        acc["bufferView"] = self._view(np.ascontiguousarray(array).tobytes(), target)
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def add_meta(
        self,
        array: np.ndarray,
        component_type: int,
        type_: str,
        normalized: bool = False,
        with_bounds: bool = False,
    ) -> int:
        """An accessor with **no** ``bufferView``.

        ``KHR_draco_mesh_compression`` allows this: the accessor keeps the
        metadata a client needs (count, type, min/max) while the data itself
        comes from the Draco payload. Writing a bufferView too would store the
        geometry twice -- the bug that once made a Draco run *larger* than the
        uncompressed one (114 MB vs 92 MB).
        """
        self.accessors.append(
            self._accessor(array, component_type, type_, normalized, with_bounds)
        )
        return len(self.accessors) - 1

    def add_raw(self, data: bytes) -> int:
        """A bufferView with no accessor -- used for the Draco payload."""
        return self._view(data, None)


def _material(rgb) -> dict:
    """A solid PBR material. Note LuciadRIA ignores this; see module docstring."""
    r, g, b = (float(c) / 255.0 for c in rgb)
    return {
        "pbrMetallicRoughness": {
            "baseColorFactor": [r, g, b, 1.0],
            "metallicFactor": 0.0,
            "roughnessFactor": 1.0,
        },
        "doubleSided": False,
    }


def _pack_glb(gltf: dict, binary: bytes) -> bytes:
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * (-len(js) % 4)
    bin_ = bytes(binary) + b"\x00" * (-len(binary) % 4)
    total = 12 + 8 + len(js) + (8 + len(bin_) if bin_ else 0)
    out = bytearray()
    out += b"glTF" + struct.pack("<II", 2, total)
    out += struct.pack("<I", len(js)) + b"JSON" + js
    if bin_:
        out += struct.pack("<I", len(bin_)) + b"BIN\x00" + bin_
    return bytes(out)


# ---------------------------------------------------------------------------
# instanced
# ---------------------------------------------------------------------------

def _cube(size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """24-vertex cube centred on the origin, with per-face normals."""
    h = size / 2.0
    pos, nrm, idx = [], [], []
    for f, (n, t, b) in enumerate(_FACES):
        n_, t_, b_ = np.array(n, float), np.array(t, float), np.array(b, float)
        c = n_ * h
        for sign_t, sign_b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            pos.append(c + t_ * (sign_t * h) + b_ * (sign_b * h))
            nrm.append(n_)
        base = f * 4
        idx += [base, base + 1, base + 2, base, base + 2, base + 3]
    return (
        np.array(pos, dtype=np.float32),
        np.array(nrm, dtype=np.float32),
        np.array(idx, dtype=np.uint16),
    )


def write_instanced(
    centers: np.ndarray,
    rgb: np.ndarray,
    cell_size: float,
    palette_bits: int = 2,
) -> tuple[bytes, dict]:
    """One node per palette entry, each instancing a shared cube.

    ``centers`` are cube centres in tile-local glTF coordinates (metres, Y-up).

    POSITION, NORMAL and the indices are shared by every node. Each node gets
    its own flat ``COLOR_0`` block (24 identical vertex colours, ~288 bytes) so
    colour arrives through the attribute LuciadRIA actually reads rather than
    through a material it ignores.
    """
    palette, colour_index = quantize_palette(rgb, palette_bits)
    buf = _Buffer()

    pos, nrm, idx = _cube(cell_size)
    a_pos = buf.add(pos, _FLOAT, "VEC3", _ARRAY_BUFFER, with_bounds=True)
    a_nrm = buf.add(nrm, _FLOAT, "VEC3", _ARRAY_BUFFER)
    a_idx = buf.add(idx, _USHORT, "SCALAR", _ELEMENT_ARRAY_BUFFER)

    meshes, nodes, materials = [], [], []
    for k in range(len(palette)):
        selected = colour_index == k
        if not selected.any():
            continue
        flat_colour = np.tile(normalize_colors(palette[k : k + 1]), (24, 1))
        a_col = buf.add(flat_colour, _FLOAT, "VEC3", _ARRAY_BUFFER)
        # EXT_mesh_gpu_instancing: instance-attribute bufferViews must not
        # declare a target.
        a_tr = buf.add(centers[selected].astype(np.float32), _FLOAT, "VEC3", None, with_bounds=True)

        materials.append(_material(palette[k]))
        meshes.append(
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": a_pos, "NORMAL": a_nrm, "COLOR_0": a_col},
                        "indices": a_idx,
                        "material": len(materials) - 1,
                    }
                ]
            }
        )
        nodes.append(
            {
                "mesh": len(meshes) - 1,
                "extensions": {"EXT_mesh_gpu_instancing": {"attributes": {"TRANSLATION": a_tr}}},
            }
        )

    gltf = {
        "asset": {"version": "2.0", "generator": "vox3dt"},
        "extensionsUsed": ["EXT_mesh_gpu_instancing"],
        "extensionsRequired": ["EXT_mesh_gpu_instancing"],
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": buf.accessors,
        "bufferViews": buf.views,
        "buffers": [{"byteLength": len(buf.blob)}],
    }
    glb = _pack_glb(gltf, buf.blob)
    return glb, {
        "cubes": int(len(centers)),
        "triangles": 12 * int(len(centers)),
        "palette_entries": len(nodes),
        "bytes": len(glb),
    }


# ---------------------------------------------------------------------------
# baked
# ---------------------------------------------------------------------------

def build_faces(
    centers: np.ndarray, rgb: np.ndarray, faces: np.ndarray, cell_size: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expand exposed faces into vertex arrays. ``faces`` is the (M, 6) mask."""
    h = cell_size / 2.0
    corners = ((-1, -1), (1, -1), (1, 1), (-1, 1))
    pos_parts, nrm_parts, col_parts, idx_parts = [], [], [], []
    vertex_base = 0

    for f, (n, t, b) in enumerate(_FACES):
        selected = faces[:, f]
        count = int(selected.sum())
        if not count:
            continue
        n_ = np.array(n, dtype=np.float64)
        t_ = np.array(t, dtype=np.float64)
        b_ = np.array(b, dtype=np.float64)
        c = centers[selected] + n_ * h

        quad = np.empty((count, 4, 3), dtype=np.float32)
        for k, (st, sb) in enumerate(corners):
            quad[:, k, :] = c + t_ * (st * h) + b_ * (sb * h)
        pos_parts.append(quad.reshape(-1, 3))
        nrm_parts.append(np.tile(n_.astype(np.float32), (count * 4, 1)))
        col_parts.append(np.repeat(rgb[selected], 4, axis=0))

        base = vertex_base + np.arange(count, dtype=np.uint32) * 4
        idx_parts.append(
            np.stack([base, base + 1, base + 2, base, base + 2, base + 3], axis=1).reshape(-1)
        )
        vertex_base += count * 4

    if not pos_parts:
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, empty, np.zeros((0, 3), dtype=np.uint8), np.zeros(0, dtype=np.uint32)

    return (
        np.concatenate(pos_parts),
        np.concatenate(nrm_parts),
        np.concatenate(col_parts),
        np.concatenate(idx_parts),
    )


def write_baked(
    centers: np.ndarray,
    rgb: np.ndarray,
    faces: np.ndarray,
    cell_size: float,
    draco: bool = False,
    draco_bits: int = 14,
    color_bits: int = 8,
) -> tuple[bytes, dict]:
    """Face-culled triangles, optionally Draco-compressed.

    ``COLOR_0`` is FLOAT32 in both cases -- see the module docstring for why an
    integer colour cannot be used with Draco in LuciadRIA. At the default
    ``color_bits=8`` the values are exact.
    """
    position, normal, colour_rgb, indices = build_faces(centers, rgb, faces, cell_size)
    stats = {
        "cubes": int(len(centers)),
        "vertices": int(len(position)),
        "triangles": int(len(indices) // 3),
        "quads": int(len(indices) // 6),
        "draco": False,
    }
    if len(position) == 0:
        return b"", stats

    colour = normalize_colors(colour_rgb, color_bits)
    buf = _Buffer()
    primitive: dict = {"material": 0}
    extensions: list[str] = []

    if draco:
        import DracoPy

        payload = DracoPy.encode(
            position.astype(np.float32),
            faces=indices.reshape(-1, 3).astype(np.uint32),
            quantization_bits=draco_bits,
            compression_level=7,
            # DracoPy asserts float64 for normals specifically, not float32.
            normals=normal.astype(np.float64),
            # A GENERIC FLOAT32 attribute, not Draco's COLOR attribute: the
            # extension maps semantics by unique_id, so COLOR_0 can legally
            # point at it -- and being float, no `normalized` handling is
            # required of the viewer.
            generic_attributes={"COLOR": colour},
            preserve_order=False,
        )
        primitive["extensions"] = {
            "KHR_draco_mesh_compression": {
                "bufferView": buf.add_raw(payload),
                "attributes": dict(draco_attribute_ids(with_colors=True)),
            }
        }
        extensions.append("KHR_draco_mesh_compression")
        stats.update(draco=True, draco_payload_bytes=len(payload))
        # Metadata-only accessors: the geometry is stored once, in the payload.
        a_pos = buf.add_meta(position, _FLOAT, "VEC3", with_bounds=True)
        a_nrm = buf.add_meta(normal, _FLOAT, "VEC3")
        a_col = buf.add_meta(colour, _FLOAT, "VEC3")
        a_idx = buf.add_meta(indices.astype(np.uint32), _UINT, "SCALAR")
    else:
        a_pos = buf.add(position, _FLOAT, "VEC3", _ARRAY_BUFFER, with_bounds=True)
        a_nrm = buf.add(normal, _FLOAT, "VEC3", _ARRAY_BUFFER)
        a_col = buf.add(colour, _FLOAT, "VEC3", _ARRAY_BUFFER)
        a_idx = buf.add(indices.astype(np.uint32), _UINT, "SCALAR", _ELEMENT_ARRAY_BUFFER)

    primitive["attributes"] = {"POSITION": a_pos, "NORMAL": a_nrm, "COLOR_0": a_col}
    primitive["indices"] = a_idx

    gltf = {
        "asset": {"version": "2.0", "generator": "vox3dt"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [primitive]}],
        "materials": [_material((255, 255, 255))],
        "accessors": buf.accessors,
        "bufferViews": buf.views,
        "buffers": [{"byteLength": len(buf.blob)}],
    }
    if extensions:
        gltf["extensionsUsed"] = extensions
        gltf["extensionsRequired"] = extensions

    glb = _pack_glb(gltf, buf.blob)
    stats["bytes"] = len(glb)
    return glb, stats
