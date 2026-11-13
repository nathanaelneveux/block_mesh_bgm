# block-mesh-bgm

[![Crates.io](https://img.shields.io/crates/v/block-mesh-bgm.svg)](https://crates.io/crates/block-mesh-bgm)
[![Docs.rs](https://docs.rs/block-mesh-bgm/badge.svg)](https://docs.rs/block-mesh-bgm)

**Binary greedy meshing for [`block-mesh`](https://github.com/bonsairobo/block-mesh-rs).** Same voxels, same shapes, same quads — swap one function and mesh four chunks of smooth hills in **70 µs** instead of **887 µs**.

It also does something binary greedy meshers usually give up on: it merges faces while keeping per-vertex ambient occlusion (AO) exact, and on cave-heavy terrain it is even faster than plain greedy meshing.

This crate replaces the meshers and nothing else. Padding, `Voxel`, `MergeVoxel`, shapes, and turning quads into vertices work exactly as in `block-mesh`, and its [README](https://github.com/bonsairobo/block-mesh-rs) is still the place to learn them.

[Quick start](#mesh-your-first-chunk) · [From block-mesh](#switch-from-block-mesh) · [From other BGM crates](#coming-from-binary-greedy-meshing) · [AO](#ao-safe-not-ao-baked) · [Benchmarks](#benchmarks)

## Mesh your first chunk

In a Rust binary project, run `cargo add block-mesh@0.2 block-mesh-bgm@0.2`. Save this as `src/main.rs` and run `cargo run`. Expect `6 quads`.

```rust
use block_mesh::ndshape::{ConstShape, ConstShape3u32};
use block_mesh::{MergeVoxel, Voxel, VoxelVisibility, RIGHT_HANDED_Y_UP_CONFIG};
use block_mesh_bgm::{binary_greedy_quads, BinaryGreedyQuadsBuffer};

#[derive(Clone, Copy)]
struct Block(u8);

impl Voxel for Block {
    fn get_visibility(&self) -> VoxelVisibility {
        if self.0 == 0 {
            VoxelVisibility::Empty
        } else {
            VoxelVisibility::Opaque
        }
    }
}

impl MergeVoxel for Block {
    type MergeValue = u8;

    fn merge_value(&self) -> u8 {
        self.0
    }
}

fn main() {
    // 0 is air. The two blocks share material 1.
    type Chunk = ConstShape3u32<4, 4, 4>;
    let mut voxels = [Block(0); Chunk::SIZE as usize];
    voxels[Chunk::linearize([1, 1, 1]) as usize] = Block(1);
    voxels[Chunk::linearize([2, 1, 1]) as usize] = Block(1);

    let faces = RIGHT_HANDED_Y_UP_CONFIG.faces;
    let mut buffer = BinaryGreedyQuadsBuffer::new();
    binary_greedy_quads(&voxels, &Chunk {}, [0; 3], [3; 3], &faces, &mut buffer);

    println!("{} quads", buffer.quads.num_quads());
}
```

The touching faces disappear, and the four pairs along the sides merge. Change the second block to `Block(2)` to see the same shape take **10 quads** across two materials.

## Switch from `block-mesh`

| What you need | Mesher | Output buffer |
| --- | --- | --- |
| Replace `greedy_quads` | `binary_greedy_quads` | `BinaryGreedyQuadsBuffer` |
| Merge faces and keep vertex AO on opaque surfaces | `binary_greedy_quads_ao_safe` | `BinaryGreedyQuadsBuffer` |
| Replace `visible_block_faces` for one quad per face | `binary_visible_block_faces` | `BinaryUnitQuadBuffer` |

Keep your voxels, shape, bounds, and face config. Call the new buffer’s `new()` without arguments and read `buffer.quads`.

- **Face coverage stays the same.** Quad layout, count, and order can differ.
- **Every call clears the output.** If you relied on upstream `visible_block_faces` appending, collect the quads after each call.
- **Queries tile automatically beyond 62 interior voxels on any axis.** Tiles read their neighbors for visibility and AO. Quads stop at tile seams.

The mesher reads your voxel slice in place, with no repacking into a fixed layout, and translucency follows `VoxelVisibility` exactly. Reuse one buffer per worker to retain its allocations.

## Coming from binary-greedy-meshing

This crate builds on [cgerikj's binary greedy meshing](https://github.com/cgerikj/binary-greedy-meshing) and its Rust port, [`binary-greedy-meshing`](https://github.com/Inspirateur/binary-greedy-meshing). Like them, it packs occupancy into `u64` masks, culls 64 faces at a time, and only looks up voxel types where the masks say a merge is possible. What changes:

| | cgerikj v2 | `binary-greedy-meshing` | `block-mesh-bgm` |
| --- | --- | --- | --- |
| Input | Padded `64³` voxels plus an occupancy mask | Padded `64³` `u16` array in ZXY order | Your voxel type, any `block-mesh` shape |
| Chunk size | `64³` padded, cubic | `62³` interior | Any; larger queries are tiled |
| Transparency | Not built in | Yes | Yes, via `VoxelVisibility` |
| Ambient occlusion | Baked in v1, dropped in v2 | Not supported | AO-safe merging + per-quad sampler |
| Output | Packed quads for vertex pulling | Packed `u64` quads | `block-mesh` quads |

If you render with vertex pulling, pack each `UnorientedQuad` into your own format after meshing.

## AO-safe, not AO-baked

Binary greedy meshers usually treat AO as a speed tax. cgerikj's v1 baked it into the mesh, doubling meshing time on its sphere scene; v2 dropped it, and the Rust port's AO issue is still open. The underlying problem: a big merged quad can only shade its four corners, so merging across an AO change smears or erases the darkening.

`binary_greedy_quads_ao_safe` doesn't bake AO or compute per-face AO signatures. It keeps only the merge boundaries AO depends on. It reads the plane just outside each face as bitsets the mesher already has, and sorts a whole row of faces at once into four classes: free to merge, row-only, column-only, or single. Single and one-direction faces are peeled off in bulk on cheap paths; only the rest takes the full two-direction merge.

![Top view of the same floor. Binary greedy uses four floor quads. AO-safe greedy uses twelve and shades the ring around the raised voxel.](docs/ao-quad-layout.png)

Every opaque quad comes out with uniform AO, so you sample it once per quad with [`FaceAoSampler`](https://docs.rs/block-mesh-bgm/latest/block_mesh_bgm/struct.FaceAoSampler.html):

```rust,ignore
for (group, face) in buffer.quads.groups.iter().zip(faces) {
    let sampler = FaceAoSampler::for_shape(face, &Chunk {});
    for quad in group {
        // Four values in vertex order, from 0 (dark) to 3 (open).
        let ao = sampler.sample(&voxels, quad.minimum, |v| v.get_visibility() == VoxelVisibility::Opaque);
    }
}
```

On cave-heavy terrain most faces land in the cheap classes, so AO-safe meshing beats plain greedy meshing; on smooth hills it costs more (see [Benchmarks](#benchmarks)). From a checkout, run the demo and press **A** to watch the shading and quad count change:

```sh
cargo run --release -p block-mesh-bgm-examples --example ambient_occlusion
```

## Benchmarks

Criterion medians on an Apple M4 with reused buffers. Timings cover meshing only.

| Scene | `greedy_quads` | Binary greedy | AO-safe greedy |
| --- | ---: | ---: | ---: |
| Dense sphere · one `34³` padded chunk | 250 µs | 21 µs | 24 µs |
| Smooth hills · four chunks | 887 µs | 70 µs | 92 µs |
| Layered caves · eight chunks | 3.77 ms | 304 µs | 216 µs |
| Large model · one `130×66×98` padded query | 9.39 ms | 1.03 ms | 1.03 ms |

Binary greedy is competitive with or faster than even `visible_block_faces`, the usual speed baseline, while emitting far fewer quads; for unit faces on the hills, `binary_visible_block_faces` took 60 µs versus 139 µs. AO-safe merging adds work on the smooth hills, where slopes split many quads for AO; in the caves most faces take its cheaper merge paths.

Measure in release mode. Debug builds of bitwise code are many times slower.

Run `cargo bench --bench bench` for the full suite. Add `-- smooth-hills` to focus on the hills. The table used 50 samples, a 1-second warm-up, and a 2-second measurement target per case.

## More examples

From a checkout, run `cargo run --release -p block-mesh-bgm-examples --example <name>`.

| Demo | What to try |
| --- | --- |
| [`render`](examples_crate/render/main.rs) | Compare quad counts across meshers. Press **Space** to toggle wireframes and **T** to change scenes. |
| [`ambient_occlusion`](examples_crate/ambient_occlusion/main.rs) | Press **U** to compare with unit faces and **S** to swap the sphere for a torus. |
| [`custom_meshing`](examples_crate/custom_meshing/main.rs) | Watch chunk rebuild timings as the camera flies through a `bevy_voxel_world` scene. Press **1**, **2**, or **3** to switch meshers. |

## License

[MIT](LICENSE.MIT) or [Apache-2.0](LICENSE.Apache-2.0), at your option.
