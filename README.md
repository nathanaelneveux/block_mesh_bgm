# block-mesh-bgm

[![Crates.io](https://img.shields.io/crates/v/block-mesh-bgm.svg)](https://crates.io/crates/block-mesh-bgm)
[![Docs.rs](https://docs.rs/block-mesh-bgm/badge.svg)](https://docs.rs/block-mesh-bgm)

**Binary greedy meshing for [`block-mesh`](https://github.com/bonsairobo/block-mesh-rs).** Same voxels, same shapes, same quads — about ten times faster than `greedy_quads`, and usually faster than even `visible_block_faces` while emitting far fewer quads.

- **Drop-in.** Swap one function and one buffer type. Everything else in `block-mesh` stays.
- **Any size.** From `64³` chunks in one pass to whole `.vox` models, tiled for you.
- **AO-safe greedy meshing.** Greedy quads with exact per-vertex ambient occlusion, worked out from bitmasks instead of per-face AO signatures. On cave-heavy terrain it even beats plain greedy meshing.

## Getting Started

```toml
[dependencies]
block-mesh = "0.2"
block-mesh-bgm = "0.2"
```

Mesh a padded chunk, then turn each quad into vertices exactly as you would with `block-mesh`:

```rust
use block_mesh::ndshape::{ConstShape, ConstShape3u32};
use block_mesh::{MergeVoxel, Voxel, VoxelVisibility, RIGHT_HANDED_Y_UP_CONFIG};
use block_mesh_bgm::{binary_greedy_quads, BinaryGreedyQuadsBuffer};

#[derive(Clone, Copy, PartialEq, Eq)]
struct Block(bool);

impl Voxel for Block {
    fn get_visibility(&self) -> VoxelVisibility {
        if self.0 { VoxelVisibility::Opaque } else { VoxelVisibility::Empty }
    }
}

impl MergeVoxel for Block {
    type MergeValue = bool;
    fn merge_value(&self) -> bool { self.0 }
}

// A 32³ chunk plus one voxel of padding on every side.
type Chunk = ConstShape3u32<34, 34, 34>;

let mut voxels = [Block(false); Chunk::SIZE as usize];
for i in 0..Chunk::SIZE {
    let [x, y, z] = Chunk::delinearize(i);
    voxels[i as usize] = Block(x * x + y * y + z * z < 30 * 30);
}

let faces = RIGHT_HANDED_Y_UP_CONFIG.faces;
let mut buffer = BinaryGreedyQuadsBuffer::new();
binary_greedy_quads(&voxels, &Chunk {}, [0; 3], [33; 3], &faces, &mut buffer);

for (group, face) in buffer.quads.groups.iter().zip(&faces) {
    for quad in group {
        let positions = face.quad_mesh_positions(quad, 1.0);
        let normals = face.quad_mesh_normals();
        // ...push into your mesh
    }
}
```

Reuse one buffer per thread; it keeps its allocations between calls. The [API docs](https://docs.rs/block-mesh-bgm) cover choosing a mesher, how it works, and common questions.

## Switching From `block-mesh`

| `block-mesh` | `block-mesh-bgm` | buffer |
| --- | --- | --- |
| `greedy_quads` | `binary_greedy_quads` | `BinaryGreedyQuadsBuffer` |
| | `binary_greedy_quads_ao_safe` | `BinaryGreedyQuadsBuffer` |
| `visible_block_faces` | `binary_visible_block_faces` | `BinaryUnitQuadBuffer` |

The arguments are identical, and `buffer.quads` is the same `QuadBuffer` (or `UnitQuadBuffer`) you already consume. Three things differ:

- **Quad partition.** You get exactly the same visible faces as `greedy_quads`, and usually a similar quad count, but not necessarily the same rectangles.
- **Buffer clearing.** These functions clear their output first. Upstream `visible_block_faces` appends.
- **Tile seams.** Interiors wider than `62` voxels are meshed in tiles. Visibility and AO stay exact across seams, but quads don't merge across them.

## AO-Safe Greedy Meshing

Greedy meshing and per-vertex ambient occlusion are a famously awkward pair. Merge two faces whose corners should darken differently and the quad smears one shading across both. The usual fix is to compute an AO signature for every visible face and merge only matching signatures, which puts an eight-voxel lookup per face on the hot path.

`binary_greedy_quads_ao_safe` never computes a signature. It reads the plane just outside each face as bitsets the mesher already has, and sorts a whole row of faces at once into four classes: free to merge, row-only, column-only, or single. Single and one-direction faces are peeled off in bulk on cheap paths, and only the rest goes through the full two-direction merge.

Every quad comes out with uniform AO, and faces still merge wherever AO allows. The speed depends on the scene: on noisy, cave-heavy terrain most faces land in the cheap classes, so it beats plain greedy meshing; on smooth surfaces it costs 15–30% more.

| Case | `binary_greedy_quads` | `binary_greedy_quads_ao_safe` |
| --- | ---: | ---: |
| `layered-caves-2x2x2` | `307.46 µs` | `218.70 µs` |
| `translucent-mix` | `111.68 µs` | `97.227 µs` |
| `layered-caves` | `61.785 µs` | `54.398 µs` |
| `dense-sphere` | `21.005 µs` | `25.067 µs` |

Because AO is uniform per quad, you compute it once per final quad with `FaceAoSampler`, not once per voxel face:

```rust,ignore
use block_mesh_bgm::{binary_greedy_quads_ao_safe, FaceAoSampler};

binary_greedy_quads_ao_safe(&voxels, &Chunk {}, [0; 3], [33; 3], &faces, &mut buffer);

for (group, face) in buffer.quads.groups.iter().zip(faces) {
    let sampler = FaceAoSampler::for_shape(face, &Chunk {});
    for quad in group {
        // 0 (dark) ..= 3 (open) per corner, in `quad_corners` order.
        let vertex_ao = sampler.sample(&voxels, quad.minimum, |block| block.0);
    }
}
```

The [API docs](https://docs.rs/block-mesh-bgm/latest/block_mesh_bgm/#how-ao-safe-merging-works) walk through the four classes, bit by bit.

## Performance

Criterion medians on an Apple M4 (`cargo bench --bench bench`). Cases are `34³` padded chunks unless named otherwise; `layered-caves-2x2x2` is eight chunks, and the large model is a single query spanning several tiles.

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

<details>
<summary>Unit faces: <code>visible_block_faces</code> vs <code>binary_visible_block_faces</code></summary>

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

</details>

## Examples

The workspace's `examples_crate` has three Bevy demos:

```text
cargo run -p block-mesh-bgm-examples --example render           # all meshers side by side, with quad counts
cargo run -p block-mesh-bgm-examples --example ambient_occlusion  # AO-safe quad layouts on a sphere and torus
cargo run -p block-mesh-bgm-examples --example custom_meshing     # head-to-head in a bevy_voxel_world scene
```

## License

MIT or Apache-2.0, at your option — the same as `block-mesh`.
