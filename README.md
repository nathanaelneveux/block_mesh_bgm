# block-mesh-bgm

[![Crates.io](https://img.shields.io/crates/v/block-mesh-bgm.svg)](https://crates.io/crates/block-mesh-bgm)
[![Docs.rs](https://docs.rs/block-mesh-bgm/badge.svg)](https://docs.rs/block-mesh-bgm)

**Binary greedy meshing for `block-mesh`, with AO-safe merging.** It replaces the meshers and nothing else: keep your voxel data and quad types. On the [smooth-hills benchmark](#benchmarks), binary greedy meshing was about **13× as fast** as `block_mesh::greedy_quads`, and twice as fast as even `visible_block_faces`.

The AO-safe mesher merges faces without breaking per-vertex ambient occlusion (AO). It doesn't pack AO into the meshing path; it keeps only the merge boundaries AO depends on, so you sample AO once per final quad. On the eight-chunk layered-caves benchmark, AO-safe meshing took **26% less time** than plain binary greedy meshing.

[From block-mesh](#switch-from-block-mesh) · [Quick start](#mesh-your-first-chunk) · [AO-safe merging](#ao-safe-not-ao-packed) · [Benchmarks](#benchmarks)

## Switch from `block-mesh`

| What you need | Mesher | Output buffer |
| --- | --- | --- |
| Replace `greedy_quads` | `binary_greedy_quads` | `BinaryGreedyQuadsBuffer` |
| Preserve vertex AO on opaque surfaces | `binary_greedy_quads_ao_safe` | `BinaryGreedyQuadsBuffer` |
| Replace `visible_block_faces` | `binary_visible_block_faces` | `BinaryUnitQuadBuffer` |

Pass the same voxels, shape, bounds, and face config. The mesher builds its masks directly from your voxel slice. Create the new buffer with `new()` and reuse one per worker.

- **Face coverage stays the same.** Quad layout, count, and order can differ.
- **Every call clears the output.** If you relied on upstream `visible_block_faces` appending, collect the quads after each call.
- **Queries tile automatically beyond 62 interior voxels on any axis.** Tiles read neighboring voxels for visibility and AO. Quads stop at tile seams.

Pair each group in `buffer.quads` with its entry in `faces` to use the existing [`OrientedBlockFace`](https://docs.rs/block-mesh/latest/block_mesh/struct.OrientedBlockFace.html) vertex helpers.

## Mesh your first chunk

In a Rust binary project, run `cargo add block-mesh@0.2 block-mesh-bgm@0.2`. Save this as `src/main.rs` and run `cargo run`. Expect `6 quads`. For voxel traits, shapes, and padding, see the [`block-mesh` docs](https://docs.rs/block-mesh/latest/block_mesh/).

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

Change the second block to `Block(2)` to see the same shape take **10 quads** across two materials.

## AO-Safe, Not AO-Packed

A merged quad shades only its four corners, so merging across a change in corner darkness smears the AO. Full AO packing solves that by doing neighborhood checks during meshing and dragging AO data for every face through the merge process itself.

`binary_greedy_quads_ao_safe` goes the other direction. It does not pack AO into the meshing path; it just preserves the merge boundaries that vertex AO depends on:

- It looks at opaque occupancy in the plane just outside each face, which the mesher already has as bitmasks.
- It derives AO constraints a whole row at a time with shifts and masks, and peels off the unit-only and one-direction-only cases in bulk.
- Only the remaining cells go through the full two-direction merge. The AO rule stays binary the whole time.

![Top view of the same floor. Binary greedy uses four floor quads. AO-safe greedy uses twelve and shades the ring around the raised voxel.](docs/ao-quad-layout.png)

Every opaque quad comes out with uniform AO, so you compute AO over the final quads, not per voxel: fewer things to scan, from a mesh that was faster to build. [`FaceAoSampler`](https://docs.rs/block-mesh-bgm/latest/block_mesh_bgm/struct.FaceAoSampler.html) samples each quad once at `quad.minimum`, counting only opaque voxels as occluders; it precomputes its face's eight neighbor offsets, packs the eight occupancy tests into one byte, and looks the corner values up in a table. Translucent faces follow the ordinary greedy merge rules.

Paste this inside `main` after the quick start's `println!` to check for shading around a raised block.

```rust
use block_mesh_bgm::{binary_greedy_quads_ao_safe, FaceAoSampler};

voxels[Chunk::linearize([2, 2, 1]) as usize] = Block(1);
binary_greedy_quads_ao_safe(&voxels, &Chunk {}, [0; 3], [3; 3], &faces, &mut buffer);

let mut shaded = false;
for (group, face) in buffer.quads.groups.iter().zip(faces) {
    let sampler = FaceAoSampler::for_shape(face, &Chunk {});
    for quad in group {
        let ao = sampler.sample(&voxels, quad.minimum, |voxel| {
            voxel.get_visibility() == VoxelVisibility::Opaque
        });
        // One value per vertex, from 0 (dark) to 3 (open).
        shaded |= ao.iter().any(|&value| value < 3);
    }
}
assert!(shaded);
```

On real geometry it can actually be the faster mesher: in the layered caves many faces land on the cheaper paths once they are classified up front (see [Benchmarks](#benchmarks)). Smooth hills, where slopes split many quads, cost about a third more. From a checkout, run the demo and press **A** to watch the shading and quad count change:

```sh
cargo run --release -p block-mesh-bgm-examples --example ambient_occlusion
```

## Benchmarks

`block_mesh::greedy_quads` is dramatically slower than this crate in every benchmark here. More interesting is `visible_block_faces`, the obvious speed baseline: binary greedy meshing is competitive with or faster than it on most scenes, while merging quads as it goes.

| Scene | `visible_block_faces` | `greedy_quads` | Binary greedy | AO-safe greedy |
| --- | ---: | ---: | ---: | ---: |
| Dense sphere · one `34³` chunk | 40 µs | 264 µs | 21 µs | 25 µs |
| Layered caves · one `34³` chunk | 74 µs | 741 µs | 61 µs | 54 µs |
| Translucent mix · one `34³` chunk | 78 µs | 837 µs | 112 µs | 98 µs |
| Smooth hills · four chunks | 145 µs | 931 µs | 72 µs | 95 µs |
| Layered caves · eight chunks | 646 µs | 4.05 ms | 296 µs | 220 µs |
| Large model · one `130×66×98` query | 2.20 ms | 9.94 ms | 1.06 ms | 1.05 ms |

The translucent mix is the exception: a per-voxel jumble of opaque, translucent, and empty blocks where little can merge, so `visible_block_faces` wins. When you want one quad per face anyway, `binary_visible_block_faces` is 2–5× faster than `visible_block_faces` across these scenes.

Criterion medians on an Apple M4 (50 samples, 1 s warm-up, 2 s measurement), meshing only, with reused buffers. Run `cargo bench --bench bench` for the full suite. To see the trade-offs play out in an actual world instead of a table, start with the `custom_meshing` example.

## More examples

From a checkout, run `cargo run --release -p block-mesh-bgm-examples --example <name>`. The demos use Bevy 0.20, which needs Rust 1.97.1 or newer.

| Demo | What to try |
| --- | --- |
| [`render`](examples_crate/render/main.rs) | Compare quad counts across meshers. Press **Space** to toggle wireframes and **T** to change scenes. |
| [`ambient_occlusion`](examples_crate/ambient_occlusion/main.rs) | Press **U** to compare with unit faces and **S** to swap the sphere for a torus. |
| [`custom_meshing`](examples_crate/custom_meshing/main.rs) | Watch chunk rebuild timings as the camera flies through a `bevy_voxel_world` scene. Press **1**–**4** to switch meshers, including `bevy_voxel_world`'s built-in one, and **A** to toggle AO. |

## Acknowledgements

This crate builds directly on [cgerikj's binary greedy meshing](https://github.com/cgerikj/binary-greedy-meshing): the occupancy-mask culling and bitwise merging at its core come from there. I used Inspirateur's [`binary-greedy-meshing`](https://github.com/Inspirateur/binary-greedy-meshing) Rust port before deciding to build a mesher that plugs straight into `block-mesh`. If you'd rather work with packed quads and fixed-size chunks than with `block-mesh` types, both are excellent.

## License

[MIT](LICENSE.MIT) or [Apache-2.0](LICENSE.Apache-2.0), at your option.
