//! Binary greedy meshing for [`block-mesh`](block_mesh).
//!
//! Same voxels, same shapes, same quads: this crate is a drop-in replacement
//! for the `block-mesh` meshers, built on bitmasks instead of per-voxel
//! loops. It is about ten times faster than [`greedy_quads`], and usually
//! faster than even [`visible_block_faces`] while emitting far fewer quads.
//!
//! It also does something greedy meshers usually shy away from: greedy
//! merging that keeps per-vertex [ambient occlusion](#ambient-occlusion)
//! exact, without ever computing a per-face AO signature.
//!
//! | `block-mesh` | this crate | buffer |
//! | --- | --- | --- |
//! | [`greedy_quads`] | [`binary_greedy_quads`] | [`BinaryGreedyQuadsBuffer`] |
//! | | [`binary_greedy_quads_ao_safe`] | [`BinaryGreedyQuadsBuffer`] |
//! | [`visible_block_faces`] | [`binary_visible_block_faces`] | [`BinaryUnitQuadBuffer`] |
//!
//! [`greedy_quads`]: block_mesh::greedy_quads
//! [`visible_block_faces`]: block_mesh::visible_block_faces
//!
//! # Getting started
//!
//! Mesh a padded chunk, then turn each quad into vertices exactly as you would
//! with `block-mesh`:
//!
//! ```
//! use block_mesh::ndshape::{ConstShape, ConstShape3u32};
//! use block_mesh::{MergeVoxel, Voxel, VoxelVisibility, RIGHT_HANDED_Y_UP_CONFIG};
//! use block_mesh_bgm::{binary_greedy_quads, BinaryGreedyQuadsBuffer};
//!
//! #[derive(Clone, Copy, PartialEq, Eq)]
//! struct Block(bool);
//!
//! impl Voxel for Block {
//!     fn get_visibility(&self) -> VoxelVisibility {
//!         if self.0 { VoxelVisibility::Opaque } else { VoxelVisibility::Empty }
//!     }
//! }
//!
//! impl MergeVoxel for Block {
//!     type MergeValue = bool;
//!     fn merge_value(&self) -> bool { self.0 }
//! }
//!
//! // A 32³ chunk plus one voxel of padding on every side.
//! type Chunk = ConstShape3u32<34, 34, 34>;
//!
//! let mut voxels = [Block(false); Chunk::SIZE as usize];
//! for i in 0..Chunk::SIZE {
//!     let [x, y, z] = Chunk::delinearize(i);
//!     voxels[i as usize] = Block(x * x + y * y + z * z < 30 * 30);
//! }
//!
//! let faces = RIGHT_HANDED_Y_UP_CONFIG.faces;
//! let mut buffer = BinaryGreedyQuadsBuffer::new();
//! binary_greedy_quads(&voxels, &Chunk {}, [0; 3], [33; 3], &faces, &mut buffer);
//!
//! for (group, face) in buffer.quads.groups.iter().zip(&faces) {
//!     for quad in group {
//!         let positions = face.quad_mesh_positions(quad, 1.0);
//!         let normals = face.quad_mesh_normals();
//!         // ...push into your mesh
//!     }
//! }
//! # assert!(buffer.quads.num_quads() > 0);
//! ```
//!
//! The query `[min, max]` includes one voxel of padding on every side. Only
//! faces of the interior `[min + 1, max - 1]` are emitted; the padding just
//! tells the mesher what each interior face is up against.
//!
//! Reuse one buffer per thread. It keeps its allocations between calls.
//!
//! # Choosing a mesher
//!
//! - [`binary_greedy_quads`] merges faces with equal
//!   [`merge_value`](block_mesh::MergeVoxel::merge_value)s into as few quads
//!   as it can. Use it unless you need one of the others.
//! - [`binary_greedy_quads_ao_safe`] merges the same way, except across
//!   ambient-occlusion boundaries. Use it with per-vertex AO.
//! - [`binary_visible_block_faces`] emits one quad per face. Use it when
//!   something per-voxel, like a texture atlas without tiling, rules out
//!   merging.
//!
//! # Ambient occlusion
//!
//! Greedy meshing and per-vertex AO are a famously awkward pair. Merge two
//! faces whose corners should darken differently and the quad smears one
//! shading across both. The usual fix is to compute an AO signature for every
//! visible face and merge only matching signatures, which puts an eight-voxel
//! lookup per face on the hot path.
//!
//! [`binary_greedy_quads_ao_safe`] gets the same guarantee, uniform AO across
//! every quad, from bitmasks the mesher already has. Compute the AO itself
//! once per final quad with [`FaceAoSampler`], not once per voxel face.
//!
//! ## How AO-safe merging works
//!
//! Voxel AO gives each corner of a face a value from `0` to `3`, from the
//! eight voxels around the face in the plane just outside it:
//!
//! ```text
//!     prev row     a  b  c        each corner looks at its two side
//!     current row  d  ·  e        neighbours and the diagonal between
//!     next row     f  g  h        them, e.g. the top-left corner: d, a, b
//! ```
//!
//! Comparing per-face AO signatures in the merge loop would be slow. Instead,
//! the mesher reads those three rows as bitsets (it already has them as
//! occupancy columns) and sorts a whole row of opaque faces at once, with
//! shifts and masks, into four classes:
//!
//! - **Flat**: no occluders nearby. Every corner is `3`, so the face merges
//!   freely.
//! - **Vertical**: a wall runs beside the face through all three rows (`a d f`
//!   or `c e h`), each side column is all solid or all empty, and `b` and `g`
//!   are empty. AO repeats from row to row, so the face may merge across rows
//!   at width 1.
//! - **Horizontal**: `a c` or `f h` are solid, each of those rows matches
//!   itself on both sides, and `d` and `e` are empty. AO repeats along the
//!   row, so the face may merge within it.
//! - **Unit**: anything else with an occluder nearby. It stays `1x1`.
//!
//! Unit faces are emitted directly, one-direction classes take cheap merge
//! paths, and only flat (and translucent) faces go through the full
//! two-direction merge. On noisy, cave-heavy terrain most faces land in the
//! cheap classes, which makes AO-safe meshing *faster* than plain greedy
//! meshing there; smooth surfaces cost 15–30% more. The tests check every
//! quad against a brute-force AO evaluation.
//!
//! # How it works
//!
//! Every mesher runs the same pipeline on tiles of at most 62 interior voxels
//! per axis, so that each padded row fits one `u64`. A `64³` padded chunk is
//! one tile; larger queries are split for you.
//!
//! 1. **Pack occupancy.** Each X row becomes an opaque and a translucent
//!    bitset, eight voxels per step with a multiply trick. Y columns are a
//!    bit-matrix transpose of the X rows.
//! 2. **Find visible faces.** A face is visible where `voxel & !neighbour`,
//!    computed for a whole row at once.
//! 3. **Merge.** Runs grow down columns, then join sideways when their merge
//!    values and heights match. `MergeVoxel` is only consulted where the
//!    bitmasks say a merge is possible.
//!
//! Tiles read their neighbours' voxels as padding, so visibility and AO are
//! exact across tile seams. Quads never merge across one.
//!
//! # FAQ
//!
//! **Why don't I get the same quads as `greedy_quads`?** You get exactly the
//! same visible faces, and usually a very similar number of quads, but the
//! rectangles may be split differently.
//!
//! **Why are the faces on my chunk's border missing?** The outermost layer of
//! the query is padding and never produces faces. Fill it with your
//! neighbouring chunks' voxels.
//!
//! **Do I have to clear the buffer?** No. Every function here clears its
//! output first (unlike upstream `visible_block_faces`, which appends).
//!
//! **Where can I see it running?** The repository's
//! [`examples_crate`](https://github.com/nathanaelneveux/block_mesh_bgm/tree/main/examples_crate)
//! has Bevy demos comparing every mesher, AO layouts included.
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
/// Keep one per thread: it holds on to its allocations, so after the first
/// call meshing never touches the heap except to grow the output.
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
/// Keep one per thread, like [`BinaryGreedyQuadsBuffer`].
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

/// Meshes the visible faces in `[min, max]` into as few quads as it can.
///
/// A drop-in for [`block_mesh::greedy_quads`]: faces merge when their voxels'
/// [`merge_value`](MergeVoxel::merge_value)s are equal. You get exactly the
/// visible faces `greedy_quads` would emit, usually in a similar number of
/// quads, though not necessarily the same rectangles. The output is cleared
/// first. See the [crate docs](crate#getting-started) for an example.
///
/// Any query size works. Interiors wider than 62 voxels are meshed in tiles,
/// and quads don't merge across tile seams.
///
/// # Panics
///
/// Panics if `voxels` is shorter than `voxels_shape`, or if `[min, max]` is
/// empty or reaches outside the shape.
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

/// Like [`binary_greedy_quads`], but never merges across an ambient-occlusion
/// boundary.
///
/// Every unit face inside an opaque quad has the same four vertex AO values,
/// so [`FaceAoSampler`] can compute AO once per quad; its docs show the two
/// together. Translucent faces merge as usual. AO stays exact across tile
/// seams.
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

/// Emits one quad per visible face in `[min, max]`.
///
/// A drop-in for [`block_mesh::visible_block_faces`] that emits exactly the
/// same faces, for any query size. Unlike upstream, it clears the output
/// first instead of appending.
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
