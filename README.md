# block-mesh-bgm

[![Crates.io](https://img.shields.io/crates/v/block-mesh-bgm.svg)](https://crates.io/crates/block-mesh-bgm)
[![Docs.rs](https://docs.rs/block-mesh-bgm/badge.svg)](https://docs.rs/block-mesh-bgm)

`block-mesh-bgm` is a companion crate for [`block-mesh`](https://github.com/bonsairobo/block-mesh-rs).
It exposes a `block_mesh::greedy_quads`-style API backed by a binary-mask greedy meshing implementation.

The default entry point keeps the current maximum-merge behavior. A second
configurable entry point can optionally:

- stop merges that would cross differing ambient-occlusion corner signatures on opaque faces

## Goals

- Match `block_mesh::greedy_quads` visible-face geometry for supported inputs.
- Stay close to `block_mesh::greedy_quads` quad counts while prioritizing speed.
- Reuse `block_mesh` public types (`QuadBuffer`, `UnorientedQuad`, `OrientedBlockFace`).
- Avoid the voxel remapping and intermediate buffer conversions required by standalone binary mesher crates.

## How It Works

The implementation follows the same public contract as `block_mesh::greedy_quads`,
but it arrives there with a different internal representation.

1. It treats the queried extent as a padded box.
   The outer one-voxel shell is only used to decide whether an interior face is visible.
2. It builds compact occupancy columns for all three axes.
   Each column is a `u64` whose bits represent voxels along one axis.
3. It derives visible-face rows by comparing a source column with its neighbour column.
   This turns face visibility into cheap bitwise operations.
4. It greedily merges visible cells back into `UnorientedQuad`s, checking
   `MergeVoxel::merge_value()` only when the visibility mask says a merge is possible.

The 62-voxel interior limit exists because each queried axis needs two padding
voxels and the full padded run must fit inside one `u64`.

## Reading The Source

The crate is split by stage:

- `src/lib.rs`: public API, crate docs, and the top-level pipeline
- `src/context.rs`: query validation and precomputed indexing/layout facts
- `src/prep.rs`: occupancy columns and visible-face row construction
- `src/merge.rs`: unit-quad emission and the carry-based greedy merger
- `src/merge_modes.rs`: AO-safe alternate merge policy built on binary exterior-plane occupancy masks
- `src/face.rs`: translation between `block-mesh` face orientation and the mesher's internal axis naming

## Limitations

- Interior query extents are limited to at most `62` voxels per axis.
- Transparency semantics intentionally match `block_mesh::VoxelVisibility`.
- The crate operates directly on the caller's voxel slice; it does not resample or repack chunk data.

## Example

```rust
use block_mesh::ndshape::{ConstShape, ConstShape3u32};
use block_mesh::{
    MergeVoxel, Voxel, VoxelVisibility, RIGHT_HANDED_Y_UP_CONFIG,
};
use block_mesh_bgm::{
    binary_greedy_quads, binary_greedy_quads_with_config, BinaryGreedyQuadsBuffer,
    BinaryGreedyQuadsConfig,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BoolVoxel(bool);

const EMPTY: BoolVoxel = BoolVoxel(false);
const FULL: BoolVoxel = BoolVoxel(true);

impl Voxel for BoolVoxel {
    fn get_visibility(&self) -> VoxelVisibility {
        if *self == EMPTY {
            VoxelVisibility::Empty
        } else {
            VoxelVisibility::Opaque
        }
    }
}

impl MergeVoxel for BoolVoxel {
    type MergeValue = Self;

    fn merge_value(&self) -> Self::MergeValue {
        *self
    }
}

type ChunkShape = ConstShape3u32<18, 18, 18>;

let mut voxels = [EMPTY; ChunkShape::SIZE as usize];
for i in 0..ChunkShape::SIZE {
    let [x, y, z] = ChunkShape::delinearize(i);
    voxels[i as usize] = if ((x * x + y * y + z * z) as f32).sqrt() < 15.0 {
        FULL
    } else {
        EMPTY
    };
}

let mut buffer = BinaryGreedyQuadsBuffer::new();
binary_greedy_quads(
    &voxels,
    &ChunkShape {},
    [0; 3],
    [17; 3],
    &RIGHT_HANDED_Y_UP_CONFIG.faces,
    &mut buffer,
);

assert!(buffer.quads.num_quads() > 0);

let config = BinaryGreedyQuadsConfig {
    ambient_occlusion_safe: true,
};

binary_greedy_quads_with_config(
    &voxels,
    &ChunkShape {},
    [0; 3],
    [17; 3],
    &RIGHT_HANDED_Y_UP_CONFIG.faces,
    &config,
    &mut buffer,
);
```

## Merge Modes

`binary_greedy_quads` is still the zero-extra-work path.

`binary_greedy_quads_with_config` currently adds one optional merge restriction:

- `ambient_occlusion_safe`: only merges opaque faces when their four ambient-occlusion corner values match. It does not compute or emit lighting data for you; it only keeps the mesh compatible with per-vertex AO shading.

The current AO-safe implementation is binary-mask based.

Instead of attaching a full AO signature to every visible cell and comparing
those signatures throughout the merge loop, it reuses the packed `opaque_cols`
built during prep and looks at the opaque occupancy in the plane just outside
the current face.

From those exterior-plane occupancy rows, the mesher uses shifts,
intersections, and masks to identify cells that are:

- forced to stay unit quads
- allowed to merge only within their current row
- allowed to merge only across rows at width `1`
- or still free to use the normal bidirectional greedy carry merge

That means most AO splitting work happens as whole-row bitmask logic before the
hot merge loop starts. In practice, this keeps AO-safe meshing much closer to
the vanilla fast path than a naive per-cell AO-signature approach, while still
preserving AO-friendly quad boundaries.

## Development

Release-quality changes should be validated with:

```text
cargo test
cargo bench
cargo doc --no-deps
cargo check -p block-mesh-bgm-examples --examples
cargo package --allow-dirty --list
```

The benchmark suite always includes the main reference points:

- `visible_block_faces`: the fast "one quad per visible face" baseline from `block-mesh`
- `greedy_quads`: the upstream greedy implementation
- `binary_greedy_quads`: this crate
- `binary_greedy_quads_ao_safe`: the AO-safe merge mode from this crate

That makes it easier to reason about where time is going:
`visible_block_faces` is the speed target, while `greedy_quads` is the output-shape baseline.

## Benchmark Snapshot

Recent local `cargo bench --bench bench` Criterion medians on my machine:

| Case | `visible_block_faces` | `greedy_quads` | `binary_greedy_quads` |
| --- | ---: | ---: | ---: |
| `dense-sphere` | `46.47 µs` | `292.20 µs` | `38.15 µs` |
| `translucent-sphere` | `45.20 µs` | `304.32 µs` | `40.03 µs` |
| `translucent-shell-sphere` | `43.71 µs` | `331.60 µs` | `46.14 µs` |
| `layered-caves` | `83.93 µs` | `856.14 µs` | `86.64 µs` |
| `checkerboard` | `84.74 µs` | `780.25 µs` | `77.45 µs` |
| `partial-extent` | `41.15 µs` | `290.83 µs` | `50.84 µs` |
| `translucent-mix` | `97.57 µs` | `1.0524 ms` | `140.54 µs` |
| `layered-caves-2x2x2` | `795.17 µs` | `4.9070 ms` | `675.18 µs` |

Useful takeaways from that run:

- `binary_greedy_quads` is already faster than `visible_block_faces` on `dense-sphere`, `translucent-sphere`, `checkerboard`, and `layered-caves-2x2x2`.
- It is essentially at parity on `layered-caves`.
- The main remaining misses are `partial-extent`, `translucent-shell-sphere`, and especially the deliberately hostile `translucent-mix` case.
- It remains much faster than upstream `greedy_quads` across every benchmark in the suite.

## Visual Examples

The workspace includes an `examples_crate` for visual inspection.

Build-check the examples with:

```text
cargo check -p block-mesh-bgm-examples --examples
```

Run the side-by-side renderer with:

```text
cargo run -p block-mesh-bgm-examples --example render
```

That example places `visible_block_faces`, `greedy_quads`, and
`binary_greedy_quads` side-by-side and logs their quad counts.
Press `Space` to toggle wireframe so you can switch between surface shading and quad layout.
Press `T` to switch between the original striped opaque sphere and a solid-core sphere wrapped in translucent voxels.
Press `O` to switch the translucent demo between `AlphaToCoverage` and camera-level OIT.

Run the merge-mode viewer with:

```text
cargo run -p block-mesh-bgm-examples --example merge_modes
```

That example renders one opaque sphere with wireframe always enabled so you can
inspect how the configurable merge policies change the quad layout.
Press `U` to toggle between binary greedy output and unit quads, `S` to toggle
between an opaque sphere and an opaque torus, and `A` to toggle AO-safe
merging together with AO vertex shading.

Run the `bevy_voxel_world` integration example with:

```text
cargo run -p block-mesh-bgm-examples --example custom_meshing
```

That example is based on the `bevy_voxel_world` custom meshing demo, but swaps in
this crate's binary greedy mesher.

## License

This crate follows the same dual-license model as `block-mesh`:

- Apache License, Version 2.0, in `LICENSE.Apache-2.0`
- MIT license, in `LICENSE.MIT`

You may choose either license.
