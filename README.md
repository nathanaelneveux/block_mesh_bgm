# block-mesh-bgm

[![Crates.io](https://img.shields.io/crates/v/block-mesh-bgm.svg)](https://crates.io/crates/block-mesh-bgm)
[![Docs.rs](https://docs.rs/block-mesh-bgm/badge.svg)](https://docs.rs/block-mesh-bgm)

`block-mesh-bgm` is a companion crate for [`block-mesh`](https://github.com/bonsairobo/block-mesh-rs).

It provides a `block_mesh::greedy_quads`-style API backed by a binary-mask greedy meshing implementation designed for high performance and low overhead.

## API At A Glance

- `binary_greedy_quads` for the maximum-merge fast path
- `binary_greedy_quads_ao_safe` when quad boundaries must stay compatible with per-vertex ambient occlusion

---

## Goals

- Match `block_mesh::greedy_quads` visible-face geometry for supported inputs
- Stay close to `block_mesh::greedy_quads` quad counts while prioritizing speed
- Reuse `block_mesh` public types (`QuadBuffer`, `UnorientedQuad`, `OrientedBlockFace`)
- Avoid voxel remapping or intermediate buffer conversions
- Keep AO-safe meshing close to the performance of the non-AO path

---

## How It Works

The public API mirrors `block_mesh::greedy_quads`, but the internal representation is different.

1. Treat the queried extent as a padded box  
   The outer one-voxel shell is only used to determine face visibility
2. Build compact occupancy columns  
   Each column is stored as a `u64` bitmask representing voxels along one axis
3. Derive visible-face rows  
   Face visibility becomes simple bitwise comparisons between adjacent columns
4. Greedily merge visible cells  
   Merging only consults `MergeVoxel::merge_value()` when masks indicate a merge is possible

The 62-voxel interior limit exists because each padded axis must fit inside a single `u64`.

---

## AO-Safe Meshing

`binary_greedy_quads` is the zero-extra-work fast path.

`binary_greedy_quads_ao_safe` enforces merge boundaries compatible with ambient occlusion shading. It does not compute AO values for you; it only preserves the boundaries that AO shading depends on.

### Key Idea

Instead of computing AO first and attaching signatures to each face:

- This implementation derives **AO-compatible merge constraints directly from binary occupancy**
- Specifically, it examines the **exterior plane of opaque voxels adjacent to each face**
- Using bitwise shifts and masks, it determines where merges would violate AO consistency

### Merge Classification

From exterior-plane occupancy, each visible cell is classified into one of:

- **unit** → must remain a single quad  
- **horizontal** → may merge only within its row  
- **vertical** → may merge only across rows (width = 1)  
- **bidirectional** → can use full greedy merging  

This classification happens using **whole-row bit operations before the hot merge loop**.

Only the remaining bidirectional cells go through the full greedy carry merge.

### What This Means

- No AO signatures are computed or stored during meshing  
- No per-cell AO comparisons inside the merge loop  
- AO-safe constraints are enforced **purely from topology**

After meshing, AO values can still be computed per vertex in the usual way.

---

## Why This Is Fast

Traditional AO-safe greedy meshing:

- Computes AO per vertex or per face
- Stores AO signatures
- Compares signatures during merging

This implementation:

- Uses **bitwise row operations instead of per-cell AO computation**
- Removes AO as a data dependency during merging
- Pushes most AO work **outside the hot loop**

In practice, this keeps AO-safe meshing much closer to the baseline fast path, and in some cases can outperform naive approaches even without AO.

---

## Reading the Source

The crate is split by stage:

- `src/lib.rs` — public API and pipeline
- `src/context.rs` — query validation and layout
- `src/prep.rs` — occupancy columns and visibility masks
- `src/merge.rs` — greedy merging
- `src/ao.rs` — AO-safe merge logic derived from occupancy masks
- `src/face.rs` — face orientation mapping

---

## Limitations

- Interior query extents are limited to `62` voxels per axis
- Transparency semantics match `block_mesh::VoxelVisibility`
- Operates directly on the caller’s voxel slice (no repacking)

---

## Example

```rust
use block_mesh::ndshape::{ConstShape, ConstShape3u32};
use block_mesh::{
    MergeVoxel, Voxel, VoxelVisibility, RIGHT_HANDED_Y_UP_CONFIG,
};
use block_mesh_bgm::{
    binary_greedy_quads, binary_greedy_quads_ao_safe, BinaryGreedyQuadsBuffer,
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

binary_greedy_quads_ao_safe(
    &voxels,
    &ChunkShape {},
    [0; 3],
    [17; 3],
    &RIGHT_HANDED_Y_UP_CONFIG.faces,
    &mut buffer,
);
```

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
- `binary_greedy_quads_ao_safe`: the AO-safe path from this crate

That makes it easier to reason about where time is going:
`visible_block_faces` is the speed target, while `greedy_quads` is the output-shape baseline.

## Benchmark Snapshot

Recent local `cargo bench --bench bench` Criterion medians on my machine.
Treat these as relative comparisons between meshers, not as universal absolute timings.

### Core Cases

| Case | `visible_block_faces` | `greedy_quads` | `binary_greedy_quads` |
| --- | ---: | ---: | ---: |
| `dense-sphere` | `40.262 µs` | `252.25 µs` | `33.284 µs` |
| `translucent-sphere` | `40.621 µs` | `260.87 µs` | `34.820 µs` |
| `translucent-shell-sphere` | `38.191 µs` | `259.11 µs` | `35.486 µs` |
| `layered-caves` | `69.536 µs` | `742.53 µs` | `73.930 µs` |
| `checkerboard` | `74.162 µs` | `656.34 µs` | `67.541 µs` |
| `partial-extent` | `36.081 µs` | `231.45 µs` | `42.878 µs` |
| `translucent-mix` | `81.532 µs` | `830.92 µs` | `119.75 µs` |
| `layered-caves-2x2x2` | `612.30 µs` | `4.1293 ms` | `558.68 µs` |

### AO-Safe Cases

| Case | `visible_block_faces` | `binary_greedy_quads` | `binary_greedy_quads_ao_safe` |
| --- | ---: | ---: | ---: |
| `dense-sphere-ao` | `40.569 µs` | `33.737 µs` | `44.427 µs` |
| `translucent-shell-sphere-ao` | `38.530 µs` | `36.091 µs` | `48.287 µs` |
| `layered-caves-2x2x2` | `612.30 µs` | `558.68 µs` | `432.73 µs` |
| `ao-boundary-stress` | `18.419 µs` | `19.301 µs` | `29.456 µs` |
| `ao-unit-patterns` | `18.234 µs` | `19.128 µs` | `29.429 µs` |

Useful takeaways from that run:

- `binary_greedy_quads` is faster than `visible_block_faces` on `dense-sphere`, `translucent-sphere`, `translucent-shell-sphere`, `checkerboard`, and `layered-caves-2x2x2`.
- It is still very close on `layered-caves`, and the main remaining misses are `partial-extent` and the deliberately hostile `translucent-mix` case.
- `binary_greedy_quads` remains much faster than upstream `greedy_quads` across every benchmark in the suite.
- `binary_greedy_quads_ao_safe` is slower than the vanilla path on the small AO-sensitive microbenchmarks, but on the real multi-chunk `layered-caves-2x2x2` case it is currently faster than both `visible_block_faces` and vanilla binary.

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

Run the ambient-occlusion viewer with:

```text
cargo run -p block-mesh-bgm-examples --example ambient_occlusion
```

That example renders one opaque sphere with wireframe always enabled so you can
inspect how AO-safe meshing changes the quad layout.
Press `U` to toggle between binary greedy output and unit quads, `S` to toggle
between an opaque sphere and an opaque torus, and `A` to toggle AO-safe
merging together with AO vertex shading.

Run the `bevy_voxel_world` integration example with:

```text
cargo run -p block-mesh-bgm-examples --example custom_meshing
```

That example is based on the `bevy_voxel_world` custom meshing demo, but swaps in
this crate's binary greedy mesher.
Press `A` to toggle ambient occlusion for the visible-faces path and AO-safe
binary greedy meshing, so you can compare both shaded output and chunk timing
inside the same world.

## License

This crate follows the same dual-license model as `block-mesh`:

- Apache License, Version 2.0, in `LICENSE.Apache-2.0`
- MIT license, in `LICENSE.MIT`

You may choose either license.
