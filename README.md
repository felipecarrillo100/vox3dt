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
cd vox3dt
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
   bounding volumes fitted tightly to the drawn cubes, and a root transform
   fitted to the projection that places the grid origin at lon −89.174263,
   lat 17.314143, every cube within ~1 mm of where the CRS puts it.
5. **Writes `.gz` beside every file** for CDN upload, `specs.json` with the
   georeference and grid details, and a `report.json` with per-level tile,
   cube and byte counts.

About 11 seconds, 44.6 MB of tiles plus 3.5 MB of gzip sidecars. Verify with:

```bash
python3 verify.py production/yaloch \
  ../VoxelizationCodeBase/inputdata/MDS_Yaloch_satellite_pointcloud_clipped.las
```

`python3 -m vox3dt.cli` and `vox3dt` are interchangeable; the module form works
without the console script being on `PATH`.

To publish, upload the `.gz` files under the original key names with
`Content-Encoding: gzip`, and `Content-Type: model/gltf-binary` for the `.glb`s,
`application/json` for `tileset.json`. Every tile is a `.glb`, with or without
properties.

`--gzip-only` keeps only the gzipped files on disk, `tileset.json.gz` and
`content/*.glb.gz` — on Yaloch with Draco, 4.7 MB of content instead of 50 MB.
The tileset still names its tiles `.glb`, so whatever serves them must answer a
request for `tileset.json` or `content/X.glb` with the `.gz` file and
`Content-Encoding: gzip` (an Apache rewrite, or the upload storing each `.gz`
under its plain key). A plain static server such as
`python -m http.server` will 404 every tile. `verify.py` reads either layout.
Only `tileset.json` and `content/` are needed by a viewer; `specs.json`,
`report.json` and `log/` are for you.

### Building Vale with its classification rasters

Run from the repo root, with the venv active. The LAS declares no CRS, so
`--epsg` supplies it; the config lists the six yearly GeoTIFFs plus
`Classification` (= 2025). Roughly 40-60 minutes; with `--gzip-only` the
output is 1.9 GB (19.4 GB of tiles uncompressed).

```bash
source .venv/bin/activate
vox3dt -i input/Mac_complete_pointcloud_shaped_and_gps_2026.las \
  -o production/vale_props \
  --epsg 31983 --draco --gzip-only \
  --properties production/vale_props.config.json

# Optional, ~40 minutes: decodes every tile.
python3 verify.py production/vale_props input/Mac_complete_pointcloud_shaped_and_gps_2026.las
```

`vox3dt --help` lists every option with examples. The two you need are
`-i/--input` and `-o/--output`; everything else has a sensible default.

| Option | Default | What it does |
|---|---|---|
| `--draco` | off | Compress geometry with `KHR_draco_mesh_compression`. ~4× smaller gzipped, colour unchanged |
| `--mode` | `baked` | `baked` = face-culled triangles, works everywhere. `instanced` = `EXT_mesh_gpu_instancing`, ~15× smaller but needs viewer support |
| `--gzip` | off | Write `.gz` beside each file, for `Content-Encoding: gzip` |
| `--gzip-only` | off | Like `--gzip`, but tiles and `tileset.json` are written only as `.gz`, with no uncompressed copy |
| `--voxel-size` | `1.0` | Finest voxel size, metres |
| `--brick` | `64` | Cells per tile edge in X/Z. Larger = fewer, heavier tiles |
| `--levels` | derived | LOD level count. Default sizes the root to cover the whole site |
| `--color-bits` | `8` | 8 = exact. Lower quantizes colour to shrink Draco payloads |
| `--draco-bits` | `14` | Draco position quantization bits |
| `--palette-bits` | `2` | Instanced mode: colour palette size (2 → ≤64 entries) |
| `--chunk-points` | `5,000,000` | Points per LAS read pass; lower to cut peak memory |
| `--memory-budget` | `256` MB | Approx. RAM per pyramid level before spilling to `<output>/temp/`; lower to cut peak memory further, at the cost of more disk I/O |
| `--keep-temp` | off | Don't delete `<output>/temp/` when the run finishes (debugging) |
| `--epsg` | none | Assume this EPSG when the LAS declares no CRS (ignored, with a notice, if it already has one) |
| `--info` | off | Print header-only metadata and exit — no point data is read, so it's fast on any file size. `--output` is not required |
| `--property` | none | `NAME=PATH`: sample a GeoTIFF into a per-cube property (`EXT_structural_metadata`). Repeatable. Adds `FeatureID` |
| `--properties` | none | JSON config listing properties, for full control — see [Per-cube properties from GeoTIFFs](#per-cube-properties-from-geotiffs) |

Output:

```
<output>/tileset.json        the tileset -- standard 3D Tiles fields only, with
                              the fitted grid->ECEF root transform (omitted entirely
                              for a non-georeferenced tileset)
<output>/content/*.glb       one binary glTF per tile, with or without properties
<output>/specs.json          developer reference: georeference, grid, encoding,
                              FeatureID, and per property its source raster and
                              the GeoTIFF colour of each class code
<output>/report.json         per-level tile/cube/byte counts for the run
<output>/properties.config.json  copy of --properties, when given
<output>/log/conversion.log  a minimal, milestone-only log of the run
<output>/temp/               working files; deleted when the run finishes
                              unless --keep-temp is given
```

### Inspecting a file before converting it

```bash
vox3dt -i site.las --info
```

Reads only the LAS header and VLRs — point count, bounds, CRS, point format,
compression, density — and returns instantly regardless of file size. Useful
before committing to a run that may take a while on a large input.

### Files with no declared CRS

If the source LAS declares a CRS, nothing changes. If it doesn't:

- Pass `--epsg <code>` to assume that CRS, as if the header had declared it.
- Omit it, and vox3dt writes a **non-georeferenced** tileset instead: no
  root `transform`, coordinates are plain local metres, and the
  `epsg`/`crs_name`/origin fields in `specs.json` and `report.json` are
  `null`. This is a deliberate, printed choice, not a silent fallback.

### Per-cube properties from GeoTIFFs

```bash
# One raster per property, everything else detected from the raster.
vox3dt -i site.las -o tiles/site --draco --property Classification=landcover.tif

# Or a config, e.g. one classification per year.
vox3dt -i vale.las -o tiles/vale --epsg 31983 --draco --properties props.json
```

```json
{
  "properties": [
    { "name": "Classification_2020", "file": "classified_2020.tif", "kind": "categorical" },
    { "name": "Classification_2025", "file": "classified_2025.tif", "kind": "categorical",
      "labels": { "3": "vegetation" } },
    { "name": "Intensity", "file": "intensity.tif", "kind": "continuous" }
  ]
}
```

Keys: `name` and `file` (required; relative paths resolve against the config's
folder), `band` (default 1), `kind` (`categorical` / `continuous`, default
detected: categorical when the raster has a colour table), `labels` and
`colors` (code → text / `#rrggbb`, used in `specs.json` only), and an optional
`defaults` object applied to every entry. Needs `pip install -e ".[raster]"`.

With any property the tiles stay `.glb` and carry it as **3D Tiles 1.1
metadata**, so every cube is a feature:

- an **`EXT_structural_metadata`** property table (class `cube`, schema embedded
  in each tile) with **`FeatureID`** (`INT32`, unique per cube across the whole
  tileset — what a click returns) plus one column per property, the column type
  following the raster's (`UINT8` for a byte classification, `FLOAT32` for a
  continuous one);
- a per-vertex **`_FEATURE_ID_0`** under **`EXT_mesh_features`**, constant over
  all faces of a cube: its row in the table. Inside Draco it is
  `UNSIGNED_SHORT`, or `FLOAT` in a tile with more than 65,536 cubes (listed in
  `report.json`); uncompressed it is always `FLOAT`;
- `tileset.json`'s standard `properties` block declaring each column's range,
  which LuciadRIA needs before it will bind a mesh attribute.

LuciadRIA 2025.0 and 2026.1 style (`attributes` + `meshStyle`) and pick on these
exactly as they did on the earlier b3dm batch tables, Draco included. In 2026.x
use `map.pickAtAsync`: the synchronous `pickAt` no longer returns 3D Tiles hits.

Values: a cube narrower than a raster pixel reads the pixel under its centre;
a wider cube takes the **majority** (categorical) or **mean** (continuous) of
the valid pixels whose centres lie in its footprint. Nodata pixels are filled
from the nearest valid pixel; `specs.json` records how many cubes that
affected. Raster and LAS CRS may differ — cube centres are reprojected.

Every raster is opened and checked — CRS, band, coverage of the LAS bounds —
before any point is read, so a bad config fails in seconds.

## Results on the Yaloch sample

9,912,680 points → 264,209 voxels at 1 m, EPSG:32616 (WGS 84 / UTM 16N),
landing at lon −89.174263, lat 17.314143 — Petén, Guatemala. About 11 s
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

Draco is **4.0× smaller gzipped** than the uncompressed output, and its colour is **bit-identical** to it — measured, not
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
  voxelize.py    LAS header/streaming -> brick-major voxel runs + Georeference
  extsort.py     bounded-memory external sort/merge for voxel aggregates
  pyramid.py     isotropic 2x downsample per level; per-tile face exposure
  gltfwriter.py  GLB writer: baked triangles (optionally Draco) or instanced
  tileset.py     quadtree, tight box bounding volumes, fitted grid->ECEF root transform
  info.py        header-only LAS inspection for --info
  rasterprops.py GeoTIFF sampling for --property: nodata fill, footprint majority/mean
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

### Bounded memory for large inputs

Face exposure needs to know, for every occupied cell, whether each of its 6
neighbours is occupied — which used to mean sorting a whole pyramid level's
occupied-cell set in RAM. That doesn't scale to a 100 GB+ input.

The fix is the sort key voxels are streamed under: instead of plain
`(x, y, z)` order, cells are keyed **brick-major** — grouped into the same
`--brick`-sized tile buckets the tileset already uses, with a 1-cell margin
in the key layout carrying a thin "apron" of each tile's boundary-adjacent
neighbour cells. That apron is exactly the context a tile needs to answer
its own face-exposure queries correctly, including at tile edges — so the
whole pipeline processes one tile at a time instead of one whole level.

Cell aggregates are accumulated up to `--memory-budget` (default 256 MB) per
pyramid level; past that, `extsort.py` spills sorted, unique-by-key runs to
`<output>/temp/` and reads them back through a bounded k-way merge. Because
this data is a thin surface, not a solid volume, occupancy drops sharply per
level, so coarser levels usually never cross the budget and stay resident in
RAM with zero disk I/O — the same code path handles both cases with no
branching. Peak memory ends up bounded by `--chunk-points` and
`--memory-budget`, not by the input file's size.

### Coordinate systems

1. **Grid** — integer voxel indices, y-up, z mirrored (matching the reference
   encoder, whose axis swap flips handedness and mirrors z to restore it).
2. **Tile content (glTF)** — metres, y-up: `(gx, gy, gz) * cell_size`.
   *No remap is needed* — the encoder's z-mirror already produces exactly
   glTF's right-handed y-up convention. glTF x is east, y is up, z is south.
3. **Tile space** — the runtime rotates glTF y-up to z-up, giving ENU:
   `(east, north, up) = (x, -z, y)`. Bounding volumes are declared here.
4. **ECEF** — the root tile's `transform` maps tile space to ECEF. It is a
   least-squares affine fit, over the whole site, of *projected* metres
   (`E = ox + u, N = oy + v, H = oz + w`) to ECEF.

Why a fit and not the ENU frame at the origin: tile space holds projected
(UTM) metres, not local east/north metres. The UTM grid is rotated from true
north by the meridian convergence (0.38° at Vale, 0.65° at Yaloch) and scaled
by the projection's scale factor, so an ENU frame at the origin is exact there
and drifts with distance — measured 5.7 m across Yaloch and **54 m** at Vale's
far corner. The fitted affine absorbs both; what remains is the Earth's
curvature under one flat frame:

| Site | ENU at origin, worst | Fitted transform, worst |
|---|---|---|
| Yaloch (357 m) | 5.7 m | 0.001 m horizontal, 0.003 m vertical |
| Vale (6.9 × 4.9 km) | 54 m horizontal, 5.6 m vertical | 0.23 m horizontal, 0.82 m vertical |

The run prints the residual and `specs.json` records it
(`georeference.placement_residual`). Per-tile transforms would bring Vale to
millimetres, at the cost of a chained transform per tile; not needed at 1 m
voxels.

Only the root carries a `transform`, so all content and bounds are in
dataset-local metres. At a 7 km site that costs ~0.4 mm of float32 precision,
irrelevant against a 1 m voxel.

### Draco attribute ids are probed, never assumed

`KHR_draco_mesh_compression` maps glTF semantics to Draco `unique_id`s, which
are assigned in attribute *creation* order — not glTF's semantic order, and not
stably across attribute sets — nor across DracoPy versions. DracoPy 2.0.0, read
back from the payloads it writes, adds POSITION **last**:

| Attributes encoded | Draco `unique_id`s (DracoPy 2.0.0) |
|---|---|
| position + normals + colour | `NORMAL=0`, `COLOR_0=1`, `POSITION=2` |
| … + `_FEATURE_ID_0` (properties) | `NORMAL=0`, `COLOR_0=1`, `_FEATURE_ID_0=2`, `POSITION=3` |

An earlier DracoPy produced `COLOR_0=0, NORMAL=1, POSITION=2` for the same
input, which is exactly why nothing here may depend on the order. Hardcoding
the intuitive `{POSITION: 0, NORMAL: 1, COLOR_0: 2}` pointed POSITION at the
colour attribute and broke the layer. `draco_attribute_ids()` probes a 4-vertex
mesh and reads the ids back from `DracoPy.decode(...).attributes` — colour and
`_FEATURE_ID_0` are both GENERIC, so they are told apart by component count (3 vs 1)
— and refuses to write Draco output if the probe is unclear: a wrong mapping
loads as corrupt geometry, not as an error.

### Draco position quantization is not lossless

`--draco-bits 14` quantizes each tile's positions over the tile's own range, so
the step is roughly `range / 2^14` and the error up to half that. Measured on
Vale: **≤ 5 mm** on 1 m leaf tiles, 3 cm on 16 m cubes, 21 cm on the 128 m root
cubes — always far below the cell size of its level, but not zero.

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
axis-aligned through quantization, quadtree well-formedness with every child
box inside its parent's, and that every tile's geometry lies inside its
declared bounding volume. Placement is checked end to end: the grid origin
against the LAS header, the root transform against the CRS on a lattice
spanning the site, and the outermost cube edges at all four corners against
the LAS bounds. For tiles with properties it also checks that
`EXT_mesh_features` points `_FEATURE_ID_0` at a property table holding one
8-byte-aligned value per feature in every column, that `_FEATURE_ID_0` maps
cubes to rows one-to-one, that `FeatureID` is unique across the tileset, and
that property values sit inside `tileset.json`'s declared ranges.

Current, 0 failures: Yaloch 1,447 checks plain `.glb` + Draco; 2,530 with two
properties (a 4-class `UINT8` and a `FLOAT32` raster) + Draco, 1,731 without
Draco. Vale has not been rebuilt since properties moved from b3dm to glb.

`python3 verify.py <dir> <las> [tileset-file]` checks another tileset file in
the same folder, e.g. an edited copy of `tileset.json`.

The Draco mapping check exists because a decode round-trip **cannot** catch a
mapping error: `DracoPy.decode` resolves attributes by type, so wrong ids and
mismatched accessors both round-trip perfectly. 495 checks once passed on a
tileset that could not render.

## Tests

```bash
pip install -e ".[test]"
pytest                    # about 40 s
```

The suite builds a synthetic site at run time (`tests/synth.py`: a 300 x 280 m
LAS with terrain, a lake, a road, a building and 55 multi-return trees, plus a
categorical and a continuous GeoTIFF) and runs `vox3dt` on it in 13
configurations: plain, Draco, `--gzip`, `--gzip-only`, properties with and
without Draco, a tile past 65,536 cubes, a 1 MB memory budget that forces the
external sort, instanced, 5-bit colour, no CRS and `--epsg`. No real data is
involved, and nothing is downloaded.

Each run must match `tests/golden/` exactly: every tile's bytes (as a
SHA-256), and `tileset.json`, `specs.json` and `report.json` apart from run
time and paths. On top of that every run passes `verify.py`, the georeference
is checked against known answers (the grid corners land on the LAS bounds,
not a voxel south), runs that must agree do (gzip, memory budget and CRS
handling never change a tile), and the command-line errors stay clear.

When an output change is intended, regenerate and review the diff:

```bash
pytest --update-golden
git diff tests/golden/
```

`requirements-lock.txt` records the exact versions the golden files were made
with. A golden mismatch right after upgrading one of them is the upgrade, not
the code.

`tools/regression_check.py` is the manual double-check on a large real
dataset: record a baseline with one version of the code, rebuild with another,
compare every tile. It never runs under pytest, and its output and baseline
must stay outside git (`production/` is ignored). See its docstring for usage.

## Troubleshooting

### `ModuleNotFoundError: No module named 'numpy'`

Dependencies aren't installed, or the virtual environment isn't active. The
command itself is fine.

```bash
cd vox3dt 
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

- **LuciadRIA's selection highlight is wrong for `FeatureID` above 2²⁴
  (16,777,216).** Picking returns the exact `FeatureID` and properties, but
  the orange highlight lands on several neighbouring cubes, or on none —
  consistent with RIA comparing ids as float32 on the GPU. Vale's ids reach
  128 M, so only its westernmost ~2 km of leaf cubes highlight correctly. Being
  reported to Luciad with a 25 KB synthetic repro. Seen on the b3dm builds;
  not yet rechecked on the glb ones.
- **Tilesets built before the northing fix sit one voxel south**, and before
  the fitted transform drift up to 54 m (Vale). Rebuild them.
- **Vertical datum is assumed ellipsoidal.** If the source LAS uses orthometric
  height — common for surveyed data — the tileset sits off by the local geoid
  separation, tens of metres in Central America. The LAS declares only a
  horizontal CRS, so this needs answering from the survey metadata.
- **`instanced` is untested in LuciadRIA.** It is the most promising encoding
  by a wide margin (0.92 MB gzipped, one draw call per palette entry per tile),
  and now carries colour through `COLOR_0` rather than a material, so the
  grey-render problem should not apply. Needs a real load to confirm.
- **The 100 GB+ pipeline is validated on synthetic and small real inputs, not
  yet on a full production-scale file.** The bounded-memory redesign (see
  [Bounded memory for large inputs](#bounded-memory-for-large-inputs)) has
  been checked for correctness — including a forced-spill run reproducing the
  Yaloch sample's output byte-for-byte — but its peak-memory behaviour on an
  actual 100 GB+ file is still to be confirmed.
- **No greedy face merging.** Merging coplanar adjacent faces into larger quads
  should cut baked triangle counts several-fold on flat terrain. Only worth
  doing if `baked` stays the primary encoding.
