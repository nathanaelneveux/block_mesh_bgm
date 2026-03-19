# block-mesh-bgm

`block-mesh-bgm` is a companion crate for [`block-mesh`](https://github.com/bonsairobo/block-mesh-rs).
It exposes a `block_mesh::greedy_quads`-style API backed by a binary-mask greedy meshing implementation.

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

assert!(buffer.quads.num_quads() > 0);
```

## Development

Release-quality changes should be validated with:

```text
cargo test
cargo bench
cargo doc --no-deps
```

The benchmark suite compares three paths on the same voxel datasets:

- `visible_block_faces`: the fast "one quad per visible face" baseline from `block-mesh`
- `greedy_quads`: the upstream greedy implementation
- `binary_greedy_quads`: this crate

That makes it easier to reason about where time is going:
`visible_block_faces` is the speed target, while `greedy_quads` is the output-shape baseline.

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

Run the `bevy_voxel_world` integration example with:

```text
cargo run -p block-mesh-bgm-examples --example custom_meshing
```

That example is based on the `bevy_voxel_world` custom meshing demo, but swaps in
this crate's binary greedy mesher.
