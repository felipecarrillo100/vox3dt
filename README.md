# vox3dt — voxel point clouds to OGC 3D Tiles

Turns a LAS/LAZ point cloud into a **georeferenced OGC 3D Tiles pyramid of
cubes**: coarse blocks when the camera is far away, exact 1 m cubes when it's
close. The viewer picks levels by screen-space error, so there is no
hand-written culling pass, and the source CRS is carried through so the tileset
lands in the right place on the globe.

## Install

Python 3.10 or newer. A virtual environment is the least troublesome route —
see [Troubleshooting](#troubleshooting) if anything goes wrong.

```bash
cd Voxelization3dtiles
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[draco]"
```

The `[draco]` extra pulls in `laspy`, `lazrs`, `numpy`, `pyproj` **and**
`DracoPy` in one step. Quote it — zsh treats bare square brackets as a glob and
will error without the quotes. Drop the extra (`pip install -e .`) if you never
intend to use `--draco`.

Re-run `source .venv/bin/activate` in each new terminal session.

## Use

```bash
# The usual run: Draco-compressed, with gzip sidecars for a CDN.
vox3dt -i site.las -o tiles/site --draco --gzip

# Check it
python3 verify.py tiles/site site.las
```

### A worked example

Building the Yaloch slice into `production/yaloch`, run from the repo root:

```bash
python3 -m vox3dt.cli \
  -i ../VoxelizationCodeBase/inputdata/MDS_Yaloch_satellite_pointcloud_clipped.las \
  -o production/yaloch \
  --draco --gzip
```

Step by step, this:

1. **Streams the LAS** in 5 M-point batches, snapping each point onto a 1 m grid
   and averaging the colour of every point that lands in a cell. Memory scales
   with the number of *occupied voxels*, not the point count, so large inputs
   are fine. Reads `EPSG:32616` and the grid origin out of the header.
2. **Builds the LOD pyramid** — here 4 levels at 8 m, 4 m, 2 m and 1 m, the
   count derived from the site's 357 m extent. Each level halves all three axes,
   and each is surface-filtered independently so buried cells cost nothing.
3. **Writes one `.glb` per tile** into `production/yaloch/content/` — 47 tiles,
   Draco-compressed, colour exact.
4. **Writes `tileset.json`** with the quadtree, per-level `geometricError`, box
   bounding volumes and the ENU→ECEF root transform that places the site at
   lon −89.174263, lat 17.314134.
5. **Writes `.gz` beside every file** for CDN upload, and a `report.json` with
   per-level tile, cube and byte counts.

About 11 seconds, 44.6 MB of tiles plus 3.5 MB of gzip sidecars. Verify with:

```bash
python3 verify.py production/yaloch \
  ../VoxelizationCodeBase/inputdata/MDS_Yaloch_satellite_pointcloud_clipped.las
```

`python3 -m vox3dt.cli` and `vox3dt` are interchangeable; the module form works
without the console script being on `PATH`.

To publish, upload the `.gz` files under the original key names with
`Content-Encoding: gzip`, and `Content-Type: model/gltf-binary` for the `.glb`s,
`application/json` for `tileset.json`.

`vox3dt --help` lists every option with examples. The two you need are
`-i/--input` and `-o/--output`; everything else has a sensible default.

| Option | Default | What it does |
|---|---|---|
| `--draco` | off | Compress geometry with `KHR_draco_mesh_compression`. ~3.2× smaller gzipped, colour unchanged |
| `--mode` | `baked` | `baked` = face-culled triangles, works everywhere. `instanced` = `EXT_mesh_gpu_instancing`, ~15× smaller but needs viewer support |
| `--gzip` | off | Write `.gz` beside each file, for `Content-Encoding: gzip` |
| `--voxel-size` | `1.0` | Finest voxel size, metres |
| `--brick` | `64` | Cells per tile edge in X/Z. Larger = fewer, heavier tiles |
| `--levels` | derived | LOD level count. Default sizes the root to cover the whole site |
| `--color-bits` | `8` | 8 = exact. Lower quantizes colour to shrink Draco payloads |
| `--draco-bits` | `14` | Draco position quantization bits |
| `--palette-bits` | `2` | Instanced mode: colour palette size (2 → ≤64 entries) |
| `--chunk-points` | `5,000,000` | Points per LAS read pass; lower to cut peak memory |

Output:

```
<output>/tileset.json     the tileset, with the ENU->ECEF root transform
<output>/content/*.glb    one binary glTF per tile
<output>/report.json      per-level tile/cube/byte counts for the run
```

## Results on the Yaloch sample

9,912,680 points → 264,209 voxels at 1 m, EPSG:32616 (WGS 84 / UTM 16N),
landing at lon −89.174263, lat 17.314134 — Petén, Guatemala. About 11 s
end to end including Draco.

**Pyramid** — 4 levels, derived from the 357 m extent at 64-cell tiles:

| Level | Cell | Tiles | Cubes drawn | Kept by surface filter |
|---|---|---|---|---|
| 0 (root) | 8 m | 1 | 3,241 | 88.8% |
| 1 | 4 m | 4 | 14,172 | 89.2% |
| 2 | 2 m | 9 | 61,087 | 92.0% |
| 3 (leaf) | 1 m | 33 | 251,710 | 95.3% |

**Encodings:**

| Command | Raw | Gzipped | Colour | LuciadRIA |
|---|---|---|---|---|
| `--draco --gzip` | 44.56 MB | **3.49 MB** | exact | **works** |
| `--gzip` | 114.27 MB | 13.89 MB | exact | works |
| `--mode instanced --gzip` | 4.22 MB | **0.92 MB** | ≤64 palette | untested |

Draco colour is **bit-identical** to the uncompressed output — measured, not
assumed: max per-channel error across all 47 tiles is 0.000. Draco does not
quantize generic float attributes, so the values survive encoding exactly.

Face culling earns its keep on its own: at the finest level the baked variant
emits 1,052,432 triangles against 3,020,520 for six full faces per cube, so only
about 2.1 of 6 faces per surface cube are actually exposed.

## What LuciadRIA accepts

Three findings, each of which cost a debugging round trip, and all three now
encoded in the defaults:

1. **Integer `COLOR_0` does not work with Draco.** A normalized
   `UNSIGNED_BYTE` colour is spec-legal and renders fine *without* Draco, but
   LuciadRIA carries the integer type into its generated WGSL and emits
   `vec4f(vec4<u32>, 1.0)` — no matching constructor, shader won't compile,
   layer never loads. Note it reports *four* components against a VEC3
   accessor: colours are widened to RGBA internally. There is no encoder-side
   spelling of normalized-integer colour that avoids this.
2. **Material `baseColorFactor` is ignored.** A tile split into one primitive
   per palette entry with solid materials loads without error and renders
   entirely grey.
3. **Float `COLOR_0` works.** So colour here is always FLOAT32. Under Draco it
   rides a **GENERIC** attribute, which is legal because the extension maps
   glTF semantics by `unique_id`, not by Draco attribute type.

Point 1 looks like a LuciadRIA defect — the glTF spec requires `normalized` to
be honoured — and is worth reporting upstream. Point 2 is why the instanced
encoder bakes each palette entry's colour into its own 24-vertex `COLOR_0`
block (~288 bytes per entry per tile) instead of relying on a material.

The cost of using float colour everywhere is that the *uncompressed* output is
about 13% larger gzipped than it would be with `UNSIGNED_BYTE` colour. That
buys a single colour path across both encodings, and Draco is the recommended
output anyway.

## Design

```
vox3dt/
  voxelize.py    LAS -> voxel grid + Georeference (CRS, origin, axis signs)
  pyramid.py     isotropic 2x downsample per level; per-level face exposure
  gltfwriter.py  GLB writer: baked triangles (optionally Draco) or instanced
  tileset.py     quadtree, box bounding volumes, ENU->ECEF root transform
  cli.py         argument handling and orchestration
verify.py        structural verification of a generated tileset
```

### Quadtree, not octree

Measured on the sample: **3.14%** of the bounding volume is occupied, and the
median vertical thickness of an occupied footprint cell is **2 voxels** (mean
2.7, p95 6). The data is a thick skin, not a volume. An octree would spend
nearly every node on empty air and vertical splits would buy almost nothing, so
each tile covers a square of ground and carries its full Y range. 3D Tiles
permits arbitrary tree shapes.

Not coincidentally, this is also why the reference encoder in
`VoxelizationCodeBase` chunks into 1000 × 1 × 1000 slabs — the same property
was already discovered there.

### Isotropic levels

The reference encoder's medium tiers coarsen X and Z only (`scale: [4,1,4]`),
keeping Y at full resolution, because its viewer draws flat slabs. A 3D Tiles
pyramid needs *cubic* cells at every level, otherwise `geometricError` means
something different horizontally than vertically and the client's screen-space
error test picks the wrong level. Each level here halves all three axes.

The surface filter runs **per level**, not once. Coarsening makes the model more
solid, so the fraction culled grows going up — coarse tiles cover more ground
but hold proportionally fewer cubes.

### Coordinate systems

1. **Grid** — integer voxel indices, y-up, z mirrored (matching the reference
   encoder, whose axis swap flips handedness and mirrors z to restore it).
2. **Tile content (glTF)** — metres, y-up: `(gx, gy, gz) * cell_size`.
   *No remap is needed* — the encoder's z-mirror already produces exactly
   glTF's right-handed y-up convention. glTF x is east, y is up, z is south.
3. **Tile space** — the runtime rotates glTF y-up to z-up, giving ENU:
   `(east, north, up) = (x, -z, y)`. Bounding volumes are declared here.
4. **ECEF** — the root tile's `transform` is ENU→ECEF at the dataset origin.

Only the root carries a `transform`, so all content and bounds are in
dataset-local ENU metres. At a 7 km site that costs ~0.4 mm of float32
precision, irrelevant against a 1 m voxel.

### Draco attribute ids are probed, never assumed

`KHR_draco_mesh_compression` maps glTF semantics to Draco `unique_id`s, which
are assigned in attribute *creation* order — not glTF's semantic order, and not
stably across attribute sets. DracoPy adds POSITION **last**:

| Attributes encoded | Draco `unique_id`s |
|---|---|
| position only | `POSITION=0` |
| position + normals | `NORMAL=0`, `POSITION=1` |
| position + normals + colour | `COLOR_0=0`, `NORMAL=1`, `POSITION=2` |

Hardcoding the intuitive `{POSITION: 0, NORMAL: 1, COLOR_0: 2}` pointed POSITION
at the colour attribute and broke the layer. `draco_attribute_ids()` now probes
a 4-vertex mesh and reads the ids back from `DracoPy.decode(...).attributes`,
and refuses to write Draco output if the probe is unclear — a wrong mapping
loads as corrupt geometry, not as an error.

### Colour quantization

`--color-bits 8` (the default) is a plain `/255`, exact. Below 8 the grid is
**endpoint-preserving**, `round(v / 255 × (2**bits − 1)) / (2**bits − 1)`, so 0
stays 0 and 255 stays 1.0.

An earlier version snapped to bucket *centres*. That minimizes average error but
cannot reach either endpoint — at 4 bits black became 8/255 and white 247/255,
visibly lifting shadows and dimming highlights for no size saving. Only banding
remains as an artifact now.

| `--color-bits` | Colours | Raw | Gzipped | Max error |
|---|---|---|---|---|
| **8** (default) | exact | 44.56 MB | 3.49 MB | **0** |
| 5 | 32,768 | 39.29 MB | 1.91 MB | 4/255 |
| 4 | 4,096 | 36.17 MB | 1.57 MB | 8/255 |
| 3 | 512 | 33.17 MB | 1.38 MB | 16/255 |

Quantizing shrinks Draco payloads because repeated values are what its entropy
coder and gzip both exploit; DracoPy does not quantize generic attributes
itself, so this is the only lever on their size.

## Verification

`verify.py` checks what would otherwise fail silently in a viewer: GLB
structure and alignment, accessor bounds, the Draco attribute mapping and
per-attribute component count and data type, normals staying unit-length and
axis-aligned through quantization, the root transform against the LAS header's
CRS and bounding box, quadtree well-formedness, and that every tile's geometry
lies inside its declared bounding volume.

Current: 1,388 checks for `--draco`, 730 uncompressed, 1,855 for `instanced` —
0 failures.

The Draco mapping check exists because a decode round-trip **cannot** catch a
mapping error: `DracoPy.decode` resolves attributes by type, so wrong ids and
mismatched accessors both round-trip perfectly. 495 checks once passed on a
tileset that could not render.

## Troubleshooting

### `ModuleNotFoundError: No module named 'numpy'`

Dependencies aren't installed, or the virtual environment isn't active. The
command itself is fine.

```bash
cd Voxelization3dtiles
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[draco]"
```

If you already made the venv, you likely just need `source .venv/bin/activate`
again — it does not persist across terminal sessions.

### `zsh: no matches found: .[draco]`

zsh globs bare square brackets. Quote the argument:
`pip install -e ".[draco]"`.

### `error: externally-managed-environment`

Homebrew and system Python refuse installs into their own site-packages
(PEP 668). Use a venv as above, or install into your user site-packages:

```bash
python3 -m pip install --user laspy lazrs numpy pyproj DracoPy
```

Append `--break-system-packages` if that is still refused. With a `--user`
install, `python3 -m vox3dt.cli` works but the bare `vox3dt` command will not —
see the next item.

### `command not found: vox3dt`

The console script isn't on `PATH`. Either activate the venv
(`source .venv/bin/activate`), or use the module form, which never needs
`PATH`:

```bash
python3 -m vox3dt.cli -i site.las -o tiles/site --draco --gzip
```

A `pip install --user` puts the script in `~/Library/Python/3.x/bin` on macOS,
which is not on `PATH` by default.

### `command not found: python`

macOS ships no `python` command — only `python3`. Every example here uses
`python3`.

### `--draco has no effect in instanced mode`

Deliberate. In instanced mode the mesh is a single 24-vertex cube and all the
volume lives in the instance buffers, which `KHR_draco_mesh_compression` does
not touch. Use `--gzip` there, or `--mode baked` if you want Draco.

### `error: input not found: ...`

The path is resolved relative to your current directory. Either `cd` to the repo
root first, or use an absolute path (`~/git/...`), which is robust to where you
run from.

### The layer loads in LuciadRIA but renders grey

Colour is arriving through a channel LuciadRIA ignores. See
[What LuciadRIA accepts](#what-luciadria-accepts) — `baseColorFactor` is not
honoured, and integer `COLOR_0` out of a Draco payload fails to compile. The
defaults here avoid both; this should only appear if you have modified the
writer.

### Apple Silicon

`laspy`, `lazrs`, `pyproj` and `DracoPy` all ship arm64 macOS wheels, so nothing
compiles from source. If pip does start building, your Python is likely x86 under
Rosetta.

## Known gaps

- **Vertical datum is assumed ellipsoidal.** If the source LAS uses orthometric
  height — common for surveyed data — the tileset sits off by the local geoid
  separation, tens of metres in Central America. The LAS declares only a
  horizontal CRS, so this needs answering from the survey metadata.
- **`instanced` is untested in LuciadRIA.** It is the most promising encoding
  by a wide margin (0.92 MB gzipped, one draw call per palette entry per tile),
  and now carries colour through `COLOR_0` rather than a material, so the
  grey-render problem should not apply. Needs a real load to confirm.
- **Not run at production scale.** The 158 M-voxel slice would give roughly 8
  levels and ~8,500 tiles. The surface filter builds whole-level arrays, which
  is the most likely thing to need reworking into per-region passes.
- **No greedy face merging.** Merging coplanar adjacent faces into larger quads
  should cut baked triangle counts several-fold on flat terrain. Only worth
  doing if `baked` stays the primary encoding.
