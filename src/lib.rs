//! Binary greedy meshing for [`block-mesh`](block_mesh), with AO-safe merging.
//!
//! It replaces the meshers and nothing else: keep your voxel data and quad
//! types. In the smooth-hills [benchmark],
//! [`binary_greedy_quads`] was about 13× as fast as
//! [`block_mesh::greedy_quads`].
//!
//! [`binary_greedy_quads_ao_safe`] uses occupancy masks to find where opaque
//! faces can merge while preserving per-vertex ambient occlusion (AO).
//! Sample AO once per opaque quad. AO-safe meshing took 29% less time than
//! plain binary greedy meshing on the layered-caves benchmark, and 31% more on
//! the smooth hills.
//!
//! [benchmark]: https://github.com/nathanaelneveux/block_mesh_bgm#benchmarks
//!
//! [Quick start](#mesh-your-first-chunk) · [Choose a mesher](#choose-a-mesher)
//! · [Migration](#switch-from-block-mesh) · [AO-safe merging](#ao-safe-not-ao-baked)
//! · [Troubleshooting](#troubleshooting)
//!
//! # Mesh your first chunk
//!
//! In a Rust binary project, run `cargo add block-mesh@0.2 block-mesh-bgm@0.2`.
//! Paste the example into `src/main.rs` and run `cargo run`. It prints `6 quads`.
//! For voxel traits, shapes, and padding, see the [`block_mesh`] docs.
//!
//! ```
//! use block_mesh::ndshape::{ConstShape, ConstShape3u32};
//! use block_mesh::{MergeVoxel, Voxel, VoxelVisibility, RIGHT_HANDED_Y_UP_CONFIG};
//! use block_mesh_bgm::{binary_greedy_quads, BinaryGreedyQuadsBuffer};
//!
//! #[derive(Clone, Copy)]
//! struct Block(u8);
//!
//! impl Voxel for Block {
//!     fn get_visibility(&self) -> VoxelVisibility {
//!         if self.0 == 0 {
//!             VoxelVisibility::Empty
//!         } else {
//!             VoxelVisibility::Opaque
//!         }
//!     }
//! }
//!
//! impl MergeVoxel for Block {
//!     type MergeValue = u8;
//!
//!     fn merge_value(&self) -> u8 {
//!         self.0
//!     }
//! }
//!
//! fn main() {
//!     // 0 is air. The two blocks share material 1.
//!     type Chunk = ConstShape3u32<4, 4, 4>;
//!     let mut voxels = [Block(0); Chunk::SIZE as usize];
//!     voxels[Chunk::linearize([1, 1, 1]) as usize] = Block(1);
//!     voxels[Chunk::linearize([2, 1, 1]) as usize] = Block(1);
//!
//!     let faces = RIGHT_HANDED_Y_UP_CONFIG.faces;
//!     let mut buffer = BinaryGreedyQuadsBuffer::new();
//!     binary_greedy_quads(&voxels, &Chunk {}, [0; 3], [3; 3], &faces, &mut buffer);
//!
//!     # assert_eq!(buffer.quads.num_quads(), 6);
//!     println!("{} quads", buffer.quads.num_quads());
//! }
//! ```
//!
//! Change the second block to `Block(2)` to see 10 quads across two materials.
//!
//! # Choose a mesher
//!
//! | What you need | Mesher | Output buffer |
//! | --- | --- | --- |
//! | Merge faces with matching materials | [`binary_greedy_quads`] | [`BinaryGreedyQuadsBuffer`] |
//! | Keep per-vertex AO on opaque surfaces | [`binary_greedy_quads_ao_safe`] | [`BinaryGreedyQuadsBuffer`] |
//! | Keep one quad per visible face, for example to give each block its own UVs | [`binary_visible_block_faces`] | [`BinaryUnitQuadBuffer`] |
//!
//! # Switch from `block-mesh`
//!
//! Replace [`block_mesh::greedy_quads`] or [`block_mesh::visible_block_faces`]
//! using the table above. Keep the same inputs and call the new buffer's
//! `new()` without arguments. Read the upstream [`QuadBuffer`] or
//! [`UnitQuadBuffer`] from `buffer.quads`.
//!
//! - **Quad layout.** The same faces are covered. Layout, count, and order can differ.
//! - **Clearing.** Each call clears the buffer. Collect the output after each
//!   call if you relied on upstream `visible_block_faces` appending.
//! - **Tiling.** The mesher tiles any axis with more than 62 interior voxels.
//!   Tiles read neighboring voxels for visibility and AO. Quads stop at seams.
//!
//! The mesher builds masks directly from your voxel slice. Reuse one buffer
//! per worker to retain its allocations. Pair each quad group with its
//! entry in `faces` to use the existing [`OrientedBlockFace`] vertex helpers.
//!
//! # AO-safe, not AO-baked
//!
//! A merged quad shades only its four corners, so merging across an AO change
//! smears the darkening. Rather than bake AO into the mesh,
//! [`binary_greedy_quads_ao_safe`] reuses the mesher's occupancy masks to
//! find legal merge directions for a whole row of opaque faces at a time.
//! This skips per-face AO signatures and lets the mesher use simpler paths
//! for unit faces and faces that can merge along one axis.
//!
//! Each unit face in a merged opaque quad has the same four corner values in
//! the same order. Call [`FaceAoSampler`] once per opaque quad at
//! `quad.minimum` and count only [`block_mesh::VoxelVisibility::Opaque`]
//! voxels as occluders.
//!
//! Translucent faces follow the ordinary greedy merge rules. If opaque and
//! translucent blocks can share a material ID, return both the ID and
//! visibility from [`MergeVoxel::merge_value`]. Try the [`FaceAoSampler`]
//! example or the [Bevy demos].
//!
//! [Bevy demos]: https://github.com/nathanaelneveux/block_mesh_bgm/tree/main/examples_crate
//!
//! ## How AO-safe merging works
//!
//! The mesher looks at the eight neighbors around the face in the plane just
//! outside it. Each corner reads two side neighbors and their shared diagonal.
//! The upper-left corner uses `d`, `b`, and `a`.
//!
//! ```text
//!        previous row    a  b  c
//!         current row    d  ·  e
//!            next row    f  g  h
//! ```
//!
//! Shifts and masks classify a whole row of visible opaque faces at once.
//! Each class limits the directions in which faces with equal merge values
//! can join. In the table, *solid* means opaque and *open* includes both
//! empty and translucent voxels.
//!
//! | Class | What the surrounding voxels look like | Allowed merge |
//! | --- | --- | --- |
//! | Flat | All eight neighbors are open | Both axes |
//! | Column only | One or both side columns form a solid wall. Each side column is uniform, and `b` and `g` are open | Across rows |
//! | Row only | The previous and next rows are each uniform, at least one is solid, and `d` and `e` are open | Along the row |
//! | Unit | Any other arrangement | A single face per quad |
//!
//! The mesher emits unit faces directly and uses simpler merge paths for faces
//! that can join along one axis. The full greedy merge handles the remaining
//! faces, including translucent ones. The [benchmarks][benchmark] report
//! the meshing cost on hills and caves.
//!
//! # How the meshers work
//!
//! 1. **Pack occupancy.** The mesher packs each padded X row into opaque and
//!    translucent bitsets. A bit-matrix transpose gives it the Y columns.
//! 2. **Find visible faces.** Compare each occupancy row to its neighbor in
//!    the face's normal direction. Opaque faces show next to empty or
//!    translucent voxels. Translucent faces show next to empty voxels.
//! 3. **Emit quads.** The unit mesher emits each set bit as a quad. The greedy
//!    meshers grow runs across rows and join them side by side when the run
//!    lengths and merge values match. [`MergeVoxel`] is only consulted where
//!    the bitmasks say a merge is possible.
//!
//! # Troubleshooting
//!
//! ## Why is meshing slower than the benchmarks?
//!
//! Build in release mode, or raise `opt-level` for dependencies in dev
//! builds. Bitwise code is many times slower unoptimized.
//!
//! ## Why does my mesh look different after switching?
//!
//! The faces are the same, but quad edges may fall in different places than
//! with [`block_mesh::greedy_quads`]. For shading changes near corners, see
//! [AO-safe, not AO-baked](#ao-safe-not-ao-baked).
#![warn(missing_docs)]

mod ao;
mod context;
mod face;
mod merge;
mod prep;
mod sampler;

use ao::AoSafe;
use block_mesh::{MergeVoxel, OrientedBlockFace, QuadBuffer, UnitQuadBuffer, Voxel};
use context::{MeshingContext, Query};
use merge::{mesh_face_with, FaceJob, FacePolicy, Greedy, UnitFaces};
use ndshape::Shape;
use prep::{build_axis_columns, build_visible_row_pair, reset_columns, reset_visible_rows};
pub use sampler::FaceAoSampler;

/// Output and reusable scratch for [`binary_greedy_quads`] and
/// [`binary_greedy_quads_ao_safe`].
///
/// Reuse it across calls. Scratch and output storage retain their capacity
/// and grow as needed.
#[derive(Default)]
pub struct BinaryGreedyQuadsBuffer {
    /// The quads, one group per face in the order of the `faces` you passed.
    pub quads: QuadBuffer,
    scratch: Scratch,
}

impl BinaryGreedyQuadsBuffer {
    /// Creates an empty buffer.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Output and reusable scratch for [`binary_visible_block_faces`].
///
/// Reuse it across calls to retain its allocations.
#[derive(Default)]
pub struct BinaryUnitQuadBuffer {
    /// The quads, one group per face in the order of the `faces` you passed.
    pub quads: UnitQuadBuffer,
    scratch: Scratch,
}

impl BinaryUnitQuadBuffer {
    /// Creates an empty buffer.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Scratch shared by every entry point. Sized for one tile; never shrinks.
#[derive(Default)]
struct Scratch {
    // Stage 1: packed opaque occupancy columns for the two bit axes used by
    // the fixed scan plans.
    opaque_cols: [Vec<u64>; 2],
    // Stage 1: packed translucent occupancy columns with the same layout.
    trans_cols: [Vec<u64>; 2],
    // Stage 2: visibility rows for the negative face of one normal axis.
    neg_rows: Vec<u64>,
    // Stage 2: visibility rows for the positive face of the same axis.
    pos_rows: Vec<u64>,
}

/// Greedily merges visible faces in a padded voxel query.
///
/// As with [`block_mesh::greedy_quads`], adjacent coplanar faces can merge
/// when their voxels have the same [`MergeVoxel::merge_value`]. The binary
/// mesher covers the same faces and may partition them differently.
///
/// `min` and `max` are inclusive and include one padding voxel on each side.
/// Only the interior can emit faces. Use a `faces` array with one entry per
/// signed axis, such as [`block_mesh::RIGHT_HANDED_Y_UP_CONFIG`]. Output groups
/// follow that array's order.
///
/// Each call clears `output`, including for a valid query with no interior.
/// Interiors longer than 62 voxels on any axis are tiled. Quads stop at tile
/// seams. See [Mesh your first chunk](crate#mesh-your-first-chunk) for an example.
///
/// # Panics
///
/// Panics if `voxels` has fewer elements than `voxels_shape.size()`, the shape
/// has a zero-length axis, or the bounds are inverted or outside the shape.
pub fn binary_greedy_quads<T, S>(
    voxels: &[T],
    voxels_shape: &S,
    min: [u32; 3],
    max: [u32; 3],
    faces: &[OrientedBlockFace; 6],
    output: &mut BinaryGreedyQuadsBuffer,
) where
    T: MergeVoxel,
    S: Shape<3, Coord = u32>,
{
    let BinaryGreedyQuadsBuffer { quads, scratch } = output;
    mesh_query::<T, S, Greedy>(
        voxels,
        voxels_shape,
        min,
        max,
        faces,
        &mut quads.groups,
        scratch,
    );
}

/// Greedily merges visible faces while preserving opaque-surface voxel AO.
///
/// Each unit face in a merged opaque quad has the same four corner AO values
/// in the same order. Use [`FaceAoSampler`] once per quad with only opaque
/// voxels as occluders. Translucent faces use the ordinary greedy rules.
///
/// See [`binary_greedy_quads`] for padding, tiling, and buffer behavior.
///
/// # Panics
///
/// Same as [`binary_greedy_quads`].
pub fn binary_greedy_quads_ao_safe<T, S>(
    voxels: &[T],
    voxels_shape: &S,
    min: [u32; 3],
    max: [u32; 3],
    faces: &[OrientedBlockFace; 6],
    output: &mut BinaryGreedyQuadsBuffer,
) where
    T: MergeVoxel,
    S: Shape<3, Coord = u32>,
{
    let BinaryGreedyQuadsBuffer { quads, scratch } = output;
    mesh_query::<T, S, AoSafe>(
        voxels,
        voxels_shape,
        min,
        max,
        faces,
        &mut quads.groups,
        scratch,
    );
}

/// Emits one quad per visible face of the padded query's interior.
///
/// It emits the same faces as [`block_mesh::visible_block_faces`]. Each call
/// clears `output`. See [`binary_greedy_quads`] for the bounds and face-group
/// conventions shared by all three meshers.
///
/// ```
/// # use block_mesh::ndshape::{ConstShape, ConstShape3u32};
/// # use block_mesh::{Voxel, VoxelVisibility, RIGHT_HANDED_Y_UP_CONFIG};
/// # #[derive(Clone, Copy)]
/// # struct Block(bool);
/// # impl Voxel for Block {
/// #     fn get_visibility(&self) -> VoxelVisibility {
/// #         if self.0 { VoxelVisibility::Opaque } else { VoxelVisibility::Empty }
/// #     }
/// # }
/// use block_mesh_bgm::{binary_visible_block_faces, BinaryUnitQuadBuffer};
///
/// type Chunk = ConstShape3u32<18, 18, 18>;
///
/// // A single block in the middle of the chunk.
/// let mut voxels = [Block(false); Chunk::SIZE as usize];
/// voxels[Chunk::linearize([8, 8, 8]) as usize] = Block(true);
///
/// let mut buffer = BinaryUnitQuadBuffer::new();
/// let faces = RIGHT_HANDED_Y_UP_CONFIG.faces;
/// binary_visible_block_faces(&voxels, &Chunk {}, [0; 3], [17; 3], &faces, &mut buffer);
///
/// assert_eq!(buffer.quads.num_quads(), 6);
/// ```
///
/// # Panics
///
/// Same as [`binary_greedy_quads`].
pub fn binary_visible_block_faces<T, S>(
    voxels: &[T],
    voxels_shape: &S,
    min: [u32; 3],
    max: [u32; 3],
    faces: &[OrientedBlockFace; 6],
    output: &mut BinaryUnitQuadBuffer,
) where
    T: Voxel,
    S: Shape<3, Coord = u32>,
{
    let BinaryUnitQuadBuffer { quads, scratch } = output;
    mesh_query::<T, S, UnitFaces>(
        voxels,
        voxels_shape,
        min,
        max,
        faces,
        &mut quads.groups,
        scratch,
    );
}

/// Validates the query, then meshes it one tile at a time.
fn mesh_query<T, S, P>(
    voxels: &[T],
    voxels_shape: &S,
    min: [u32; 3],
    max: [u32; 3],
    faces: &[OrientedBlockFace; 6],
    groups: &mut [Vec<P::Quad>; 6],
    scratch: &mut Scratch,
) where
    T: Voxel,
    S: Shape<3, Coord = u32>,
    P: FacePolicy<T>,
{
    for group in groups.iter_mut() {
        group.clear();
    }

    let Some(query) = Query::new(voxels, voxels_shape, min, max, faces) else {
        return;
    };
    query.for_each_tile(|interior_min, interior_max| {
        let context = MeshingContext::for_tile(&query, interior_min, interior_max);
        mesh_tile::<T, P>(voxels, &query, &context, groups, scratch);
    });
}

/// Runs the full pipeline over one tile of at most 62 interior voxels per
/// axis, appending to `groups`.
fn mesh_tile<T, P>(
    voxels: &[T],
    query: &Query,
    context: &MeshingContext,
    groups: &mut [Vec<P::Quad>; 6],
    scratch: &mut Scratch,
) where
    T: Voxel,
    P: FacePolicy<T>,
{
    let Scratch {
        opaque_cols,
        trans_cols,
        neg_rows,
        pos_rows,
    } = scratch;

    // Stage 1: build the X- and Y-packed occupancy columns used by the three
    // fixed face scan plans. No plan packs Z, so building that third
    // representation would only add work.
    reset_columns(opaque_cols, trans_cols, context.query_shape);
    let occupancy = build_axis_columns(
        voxels,
        context.query_min(),
        context.query_shape,
        context.strides,
        opaque_cols,
        trans_cols,
    );
    if !occupancy.can_have_visible_faces() {
        return;
    }

    for n_axis in 0..3 {
        let slice = context.slice_plan(n_axis);

        // Stage 2: derive visible-face rows for both signs of the current normal axis.
        reset_visible_rows(neg_rows, slice.total_rows());
        reset_visible_rows(pos_rows, slice.total_rows());
        let (neg_slice_masks, pos_slice_masks) = build_visible_row_pair(
            context.query_shape,
            slice.bit_axis,
            slice.bit_len,
            slice.outer_len,
            slice.n_len,
            n_axis,
            slice.outer_axis,
            &opaque_cols[slice.bit_axis],
            &trans_cols[slice.bit_axis],
            occupancy.has_translucent,
            neg_rows,
            pos_rows,
        );

        // Stages 3 and 4: merge each face's rows into quads with the selected
        // policy, specialized for the face's orientation.
        let [neg_face, pos_face] = query.face_indices[n_axis];
        for (face_index, rows, slice_masks) in [
            (neg_face, &neg_rows[..], neg_slice_masks),
            (pos_face, &pos_rows[..], pos_slice_masks),
        ] {
            let axes = query.face_axes[face_index];
            let job = FaceJob {
                voxels,
                context,
                slice,
                rows,
                slice_masks,
                n_sign: axes.n_sign,
                opaque_cols: &opaque_cols[slice.bit_axis],
            };
            let bit_is_u = slice.bit_axis == axes.u_axis;
            mesh_face_with::<T, P>(n_axis, bit_is_u, &job, &mut groups[face_index]);
        }
    }
}

/// Returns `width` consecutive set bits starting at bit `start`.
///
/// `width` may be the full 64 bits: the padded query of a 62-voxel interior
/// is exactly 64 wide, so `(1 << width) - 1` would overflow there.
#[inline]
pub(crate) fn bit_mask(start: usize, width: usize) -> u64 {
    debug_assert!(start + width <= 64);
    if width == 0 {
        0
    } else {
        (u64::MAX >> (64 - width)) << start
    }
}
