# block-mesh-bgm

[![Crates.io](https://img.shields.io/crates/v/block-mesh-bgm.svg)](https://crates.io/crates/block-mesh-bgm)
[![Docs.rs](https://docs.rs/block-mesh-bgm/badge.svg)](https://docs.rs/block-mesh-bgm)

`block-mesh-bgm` is a `block-mesh`-compatible binary greedy mesher for voxel chunks.

Binary greedy meshing is attractive for the obvious reasons: fewer quads, less wasted work, and better chunk throughput when meshing is on the hot path. `block_mesh::greedy_quads` is useful, but it is also painfully slow if throughput is the thing you care about.

That is the gap this crate is trying to close. It keeps the `block-mesh` calling style and public quad types, then replaces the expensive part with a binary-mask pipeline that is much happier doing this work at speed.

If you need a refresher on shared concepts like padding, `Voxel`, `MergeVoxel`, or the original meshing API, the [`block-mesh` README](https://github.com/bonsairobo/block-mesh-rs/blob/main/README.md) already covers that well.

There are two entry points:

- `binary_greedy_quads` for the fastest path
- `binary_greedy_quads_ao_safe` when you want greedy meshing that still respects AO-relevant boundaries (and is still fast!)

## What Changes Here

Nothing here is trying to replace the rest of `block-mesh`. You can keep using the same surrounding types and swap out the mesher.

Internally, the mesher packs occupancy into `u64` columns, derives visible faces with bitwise comparisons, and only consults `MergeVoxel` when the masks say a merge is possible.

That comes with one deliberate constraint: query extents are limited to `62` interior voxels per axis, or `64` including the required one-voxel border. For the usual chunk sizes, that is generally the shape you wanted anyway.

Transparency semantics still follow `block_mesh::VoxelVisibility`, and the mesher works directly on the caller's voxel slice without repacking it into a different layout first.

## Quick Example

```rust
use block_mesh::ndshape::{ConstShape, ConstShape3u32};
use block_mesh::{
    MergeVoxel, Voxel, VoxelVisibility, RIGHT_HANDED_Y_UP_CONFIG,
};
use block_mesh_bgm::{binary_greedy_quads, BinaryGreedyQuadsBuffer};

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
```

`binary_greedy_quads_ao_safe` uses the same buffer and the same general call shape.

## AO-Safe, Not AO-Packed

This crate does not pack AO signatures into the meshing path. In AO-safe mode it just preserves the merge boundaries that vertex AO depends on, then leaves the AO values themselves to whatever shading step you already have.

Full AO packing means doing neighborhood checks during meshing and dragging AO data through the merge process itself.

`binary_greedy_quads_ao_safe` goes the other direction. It looks at opaque occupancy in the exterior plane next to a face, derives AO constraints a whole row at a time with shifts and masks, peels off the obvious unit-only / one-direction-only cases in bulk, and only sends the remaining cells through the full bidirectional merge logic. The AO rule stays binary the whole time.

On real geometry it can actually be a better mesher. In the `layered-caves-2x2x2` benchmark, `binary_greedy_quads_ao_safe` comes in at `432.73 µs`, faster than vanilla `binary_greedy_quads` at `558.68 µs`, because a lot of the AO-constrained work becomes simpler to merge once those cases are classified up front.

When you do go compute AO signatures later, you are doing it over the final quads, not per voxel. That means fewer things to scan and a faster mesh to begin with. The `custom_meshing` example in this workspace shows exactly that flow: it meshes first, computes AO from the resulting quads during mesh construction, and shows the average chunk meshing time in the on-screen overlay.

## Performance

`block_mesh::greedy_quads` is dramatically slower than this crate in every benchmark here, and usually by enough that it stops looking like a serious option unless you specifically need its exact behavior. In several cases, this crate is even competitive with or faster than `visible_block_faces`, which is the obvious speed baseline.

Recent local `cargo bench --bench bench` Criterion medians on my machine:

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

It is slower than vanilla `binary_greedy_quads` on the small AO-sensitive microbenches, but in cases closer to real-world geometry like `layered-caves-2x2x2` it can actually be faster.

If you want to run the same benchmark suite yourself:

```text
cargo bench --bench bench
```

If you want to see the tradeoffs play out in an actual world instead of a benchmark table, the `custom_meshing` example below is the best place to start.

## Examples

```text
cargo run -p block-mesh-bgm-examples --example render
```

`render`: `visible_block_faces`, `greedy_quads`, and `binary_greedy_quads` side by side with quad counts and translucent rendering modes.

```text
cargo run -p block-mesh-bgm-examples --example ambient_occlusion
```

`ambient_occlusion`: a focused AO viewer for inspecting the quad-layout effect of AO-safe meshing on a sphere and a torus.

```text
cargo run -p block-mesh-bgm-examples --example custom_meshing
```

`custom_meshing`: all three meshers head-to-head in the same `bevy_voxel_world` scene, with average chunk meshing time in the overlay.

## License

Same licensing model as `block-mesh`: MIT or Apache-2.0, at your option.
