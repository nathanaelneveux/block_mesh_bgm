# block-mesh-bgm

[![Crates.io](https://img.shields.io/crates/v/block-mesh-bgm.svg)](https://crates.io/crates/block-mesh-bgm)
[![Docs.rs](https://docs.rs/block-mesh-bgm/badge.svg)](https://docs.rs/block-mesh-bgm)

`block-mesh-bgm` is a `block-mesh`-compatible binary greedy mesher for voxel chunks.

Binary greedy meshing is attractive for the obvious reasons: fewer quads, less wasted work, and better chunk throughput when meshing is on the hot path. `block_mesh::greedy_quads` is useful, but it is also painfully slow if throughput is the thing you care about.

That is the gap this crate is trying to close. It keeps the `block-mesh` calling style and public quad types, then replaces the expensive part with a binary-mask pipeline that is much happier doing this work at speed.

There are three entry points, each a drop-in for its `block-mesh` counterpart:

| `block-mesh` | `block-mesh-bgm` | buffer |
| --- | --- | --- |
| `greedy_quads` | `binary_greedy_quads` | `BinaryGreedyQuadsBuffer` |
| | `binary_greedy_quads_ao_safe` | `BinaryGreedyQuadsBuffer` |
| `visible_block_faces` | `binary_visible_block_faces` | `BinaryUnitQuadBuffer` |

`binary_greedy_quads_ao_safe` is greedy meshing that respects the boundaries vertex AO depends on (and is still fast!), and `FaceAoSampler` computes that AO for the final quads.

## Switching From `block-mesh`

Nothing here is trying to replace the rest of `block-mesh`. The voxel traits, shapes, extents, face configurations, and output quad types are all the same, so switching is a change of function and buffer type:

```rust,ignore
// before
let mut buffer = GreedyQuadsBuffer::new(voxels.len());
greedy_quads(&voxels, &shape, min, max, &faces, &mut buffer);

// after
let mut buffer = BinaryGreedyQuadsBuffer::new();
binary_greedy_quads(&voxels, &shape, min, max, &faces, &mut buffer);

// `buffer.quads` is the same `QuadBuffer` as before.
```

A few differences worth knowing:

- Every query size works. Interiors up to `62` voxels per axis (a `64³` padded chunk) are meshed in one pass; larger queries, like whole `.vox` models, are meshed in tiles of at most `62`. Visibility and AO are exact across tile seams, but quads do not merge across them.
- The greedy output covers exactly the same visible faces as `greedy_quads`, usually with a very similar quad count, but not necessarily the same partition.
- These functions clear their output buffer first. (`block_mesh::visible_block_faces` appends, so callers reset it themselves.)
- Transparency semantics follow `block_mesh::VoxelVisibility` exactly, and the mesher works directly on your voxel slice without repacking it.

If you need a refresher on shared concepts like padding, `Voxel`, `MergeVoxel`, or the original meshing API, the [`block-mesh` README](https://github.com/bonsairobo/block-mesh-rs/blob/main/README.md) covers them well.

## How It Works

Internally, the mesher packs occupancy into `u64` columns (eight voxels per step, then a bit-matrix transpose for the second axis), derives visible faces with bitwise comparisons a whole row at a time, and only consults `MergeVoxel` when the masks say a merge is possible. The [crate docs](https://docs.rs/block-mesh-bgm) walk through the pipeline and the AO classification.

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

On real geometry it can actually be a better mesher. In the `layered-caves-2x2x2` benchmark, `binary_greedy_quads_ao_safe` comes in at `218.70 µs`, faster than vanilla `binary_greedy_quads` at `307.46 µs`, because a lot of the AO-constrained work becomes simpler to merge once those cases are classified up front.

When you do go compute AO signatures later, you are doing it over the final quads, not per voxel. That means fewer things to scan and a faster mesh to begin with. The `custom_meshing` example in this workspace shows exactly that flow: it meshes first, computes AO from the resulting quads during mesh construction, and shows the average chunk meshing time in the on-screen overlay.

`FaceAoSampler` is the piece that computes it. Construct one per face group, then sample only the final quads:

```rust,ignore
use block_mesh_bgm::FaceAoSampler;

for (group, face) in buffer.quads.groups.iter().zip(faces) {
    let sampler = FaceAoSampler::for_shape(face, &ChunkShape {});
    for quad in group {
        // Four 0..=3 values in `quad_corners` vertex order.
        let vertex_ao = sampler.sample(&voxels, quad.minimum, |voxel| voxel.is_opaque());
    }
}
```

The sampler precomputes the eight neighbour offsets for its face orientation, packs the eight occupancy tests into one byte, and uses a lookup table for the four standard voxel-AO values. Run its focused benchmark with `cargo bench -p block-mesh-bgm-examples --bench ao`.

## Performance

`block_mesh::greedy_quads` is dramatically slower than this crate in every benchmark here, usually by an order of magnitude. `binary_greedy_quads` beats even `visible_block_faces`, the obvious speed baseline, on most cases while emitting far fewer quads, and `binary_visible_block_faces` is faster than `visible_block_faces` everywhere.

Recent local `cargo bench --bench bench` Criterion medians on an Apple M4. Every case is a `34³` padded chunk unless named otherwise; `layered-caves-2x2x2` meshes eight chunks, and `large-model-130x66x98` is one query spanning several tiles.

### Greedy Meshing

| Case | `greedy_quads` | `binary_greedy_quads` | `binary_greedy_quads_ao_safe` |
| --- | ---: | ---: | ---: |
| `empty` | `142.58 µs` | `6.7751 µs` | `6.5790 µs` |
| `uniform-opaque` | `208.41 µs` | `8.7534 µs` | `8.7599 µs` |
| `dense-sphere` | `263.40 µs` | `21.005 µs` | `25.067 µs` |
| `translucent-sphere` | `272.40 µs` | `22.850 µs` | `25.788 µs` |
| `translucent-shell-sphere` | `266.75 µs` | `25.660 µs` | `29.873 µs` |
| `layered-caves` | `748.59 µs` | `61.785 µs` | `54.398 µs` |
| `checkerboard` | `668.57 µs` | `59.612 µs` | `59.861 µs` |
| `partial-extent` | `248.05 µs` | `37.646 µs` | `32.812 µs` |
| `translucent-mix` | `842.69 µs` | `111.68 µs` | `97.227 µs` |
| `layered-caves-2x2x2` | `4.1295 ms` | `307.46 µs` | `218.70 µs` |
| `large-model-130x66x98` | `9.7650 ms` | `1.0241 ms` | `1.0309 ms` |

### Unit Faces

| Case | `visible_block_faces` | `binary_visible_block_faces` |
| --- | ---: | ---: |
| `empty` | `17.771 µs` | `6.5473 µs` |
| `uniform-opaque` | `45.862 µs` | `8.7762 µs` |
| `dense-sphere` | `40.774 µs` | `15.178 µs` |
| `translucent-sphere` | `40.034 µs` | `17.084 µs` |
| `translucent-shell-sphere` | `38.424 µs` | `19.168 µs` |
| `layered-caves` | `77.080 µs` | `26.138 µs` |
| `checkerboard` | `72.494 µs` | `49.611 µs` |
| `partial-extent` | `35.326 µs` | `14.392 µs` |
| `translucent-mix` | `79.057 µs` | `38.860 µs` |
| `layered-caves-2x2x2` | `659.23 µs` | `124.67 µs` |
| `large-model-130x66x98` | `2.1078 ms` | `528.96 µs` |

### AO-Safe Cases

| Case | `visible_block_faces` | `binary_greedy_quads` | `binary_greedy_quads_ao_safe` |
| --- | ---: | ---: | ---: |
| `dense-sphere-ao` | `40.787 µs` | `20.899 µs` | `24.903 µs` |
| `translucent-shell-sphere-ao` | `38.311 µs` | `24.862 µs` | `29.563 µs` |
| `layered-caves-2x2x2` | `659.23 µs` | `307.46 µs` | `218.70 µs` |
| `ao-boundary-stress` | `18.564 µs` | `11.133 µs` | `11.635 µs` |
| `ao-unit-patterns` | `18.297 µs` | `10.773 µs` | `11.450 µs` |

AO-safe meshing costs a little extra on small AO-heavy microbenches, but on terrain-like geometry it is *faster* than vanilla (`layered-caves-2x2x2`, `translucent-mix`), because classifying AO constraints up front leaves most cells on cheaper one-direction merge paths.

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
