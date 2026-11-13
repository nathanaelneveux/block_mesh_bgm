//! `block-mesh`-compatible binary greedy meshing.
//!
//! [`binary_greedy_quads`] mirrors the call shape of
//! [`block_mesh::greedy_quads`], but it reaches the same kind of `QuadBuffer`
//! through a different internal pipeline:
//!
//! 1. Validate the padded query extent and precompute a few indexing facts.
//! 2. Build the packed occupancy columns used by the fixed face scan plans,
//!    one `u64` per orthogonal column.
//! 3. Turn those columns into visible-face rows with cheap bitwise tests.
//! 4. Merge each face slice into quads, either as unit quads or with a
//!    carry-based greedy row merger.
//!
//! The implementation is split into the same stages in source:
//!
//! - `context`: query validation and precomputed layout facts
//! - `prep`: occupancy columns and visible-face rows
//! - `merge`: unit emission and carry-based greedy merging
//! - `ao`: AO-safe alternate merge policy built on binary occupancy masks
//! - `face`: face-orientation helpers shared by the other stages
//!
//! The crate exposes two public entry points:
//!
//! - [`binary_greedy_quads`]: the current maximum-merge implementation with no
//!   extra policy checks
//! - [`binary_greedy_quads_ao_safe`]: the same visibility pipeline, but with
//!   AO-safe merge restrictions for opaque faces
//!
//! # AO-safe mode
//!
//! AO-safe mode keeps the same prep pipeline as vanilla meshing. The difference
//! is only in how face rows are merged:
//!
//! - vanilla merging only checks `MergeVoxel::merge_value()`
//! - AO-safe merging also looks at opaque occupancy in the plane just outside
//!   the visible face
//! - from that exterior plane it derives whole-row binary masks that prove
//!   which cells must stay unit, which may only merge in one direction, and
//!   which are still free to use the normal carry merge
//!
//! That keeps the AO rule in terms of the same binary data the mesher already
//! uses elsewhere, instead of carrying per-cell lighting data through the hot
//! merge loop.
//!
//! # Terminology
//!
//! The source uses three axis names repeatedly:
//!
//! - `n_axis`: the face normal axis
//! - `outer_axis`: the axis that advances from one row to the next within a
//!   face slice
//! - `bit_axis`: the axis packed into the row's `u64` bitset
//!
//! Every meshed query is also split into two nested boxes:
//!
//! - the **query extent** `[min, max]`, which includes one voxel of padding on
//!   every side
//! - the **interior extent** `[min + 1, max - 1]`, whose faces may actually
//!   produce quads
//!
//! The padding is never emitted. It only exists so the mesher can decide
//! whether an interior face is visible.
//!
//! # Why the `62`-voxel limit exists
//!
//! Each axis of the padded query is packed into one `u64`. That leaves room for
//! at most `62` interior voxels plus the two required padding voxels.
//!
//! # Example
//!
//! ```
//! use block_mesh::ndshape::{ConstShape, ConstShape3u32};
//! use block_mesh::{
//!     MergeVoxel, Voxel, VoxelVisibility, RIGHT_HANDED_Y_UP_CONFIG,
//! };
//! use block_mesh_bgm::{binary_greedy_quads, BinaryGreedyQuadsBuffer};
//!
//! #[derive(Clone, Copy, Debug, Eq, PartialEq)]
//! struct BoolVoxel(bool);
//!
//! const EMPTY: BoolVoxel = BoolVoxel(false);
//! const FULL: BoolVoxel = BoolVoxel(true);
//!
//! impl Voxel for BoolVoxel {
//!     fn get_visibility(&self) -> VoxelVisibility {
//!         if *self == EMPTY {
//!             VoxelVisibility::Empty
//!         } else {
//!             VoxelVisibility::Opaque
//!         }
//!     }
//! }
//!
//! impl MergeVoxel for BoolVoxel {
//!     type MergeValue = Self;
//!
//!     fn merge_value(&self) -> Self::MergeValue {
//!         *self
//!     }
//! }
//!
//! type ChunkShape = ConstShape3u32<18, 18, 18>;
//!
//! let mut voxels = [EMPTY; ChunkShape::SIZE as usize];
//! for i in 0..ChunkShape::SIZE {
//!     let [x, y, z] = ChunkShape::delinearize(i);
//!     voxels[i as usize] = if ((x * x + y * y + z * z) as f32).sqrt() < 15.0 {
//!         FULL
//!     } else {
//!         EMPTY
//!     };
//! }
//!
//! let mut buffer = BinaryGreedyQuadsBuffer::new();
//! binary_greedy_quads(
//!     &voxels,
//!     &ChunkShape {},
//!     [0; 3],
//!     [17; 3],
//!     &RIGHT_HANDED_Y_UP_CONFIG.faces,
//!     &mut buffer,
//! );
//!
//! assert!(buffer.quads.num_quads() > 0);
//! ```
//!
//! The AO-safe entry point uses the same buffer type:
//!
//! ```
//! # use block_mesh::ndshape::{ConstShape, ConstShape3u32};
//! # use block_mesh::{
//! #     MergeVoxel, Voxel, VoxelVisibility, RIGHT_HANDED_Y_UP_CONFIG,
//! # };
//! # use block_mesh_bgm::{binary_greedy_quads_ao_safe, BinaryGreedyQuadsBuffer};
//! # #[derive(Clone, Copy, Debug, Eq, PartialEq)]
//! # struct BoolVoxel(bool);
//! # impl Voxel for BoolVoxel {
//! #     fn get_visibility(&self) -> VoxelVisibility {
//! #         if self.0 {
//! #             VoxelVisibility::Opaque
//! #         } else {
//! #             VoxelVisibility::Empty
//! #         }
//! #     }
//! # }
//! # impl MergeVoxel for BoolVoxel {
//! #     type MergeValue = Self;
//! #     fn merge_value(&self) -> Self::MergeValue {
//! #         *self
//! #     }
//! # }
//! # type ChunkShape = ConstShape3u32<4, 4, 4>;
//! # let voxels = [BoolVoxel(false); ChunkShape::SIZE as usize];
//! let mut buffer = BinaryGreedyQuadsBuffer::new();
//! binary_greedy_quads_ao_safe(
//!     &voxels,
//!     &ChunkShape {},
//!     [0; 3],
//!     [3; 3],
//!     &RIGHT_HANDED_Y_UP_CONFIG.faces,
//!     &mut buffer,
//! );
//! ```
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

/// Reusable output and scratch storage for [`binary_greedy_quads`] and
/// [`binary_greedy_quads_ao_safe`].
///
/// Reusing one buffer per worker thread keeps the hot meshing path focused on
/// bitwise work instead of repeated heap growth.
#[derive(Default)]
pub struct BinaryGreedyQuadsBuffer {
    /// Output quads grouped by face, matching `block-mesh`'s `QuadBuffer`.
    pub quads: QuadBuffer,
    scratch: Scratch,
}

impl BinaryGreedyQuadsBuffer {
    /// Creates an empty reusable output/scratch buffer.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Reusable output and scratch storage for [`binary_visible_block_faces`].
#[derive(Default)]
pub struct BinaryUnitQuadBuffer {
    /// Output quads grouped by face, matching `block-mesh`'s `UnitQuadBuffer`.
    pub quads: UnitQuadBuffer,
    scratch: Scratch,
}

impl BinaryUnitQuadBuffer {
    /// Creates an empty reusable output/scratch buffer.
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

/// Generates greedy quads using a binary-mask-backed implementation.
///
/// The public API mirrors [`block_mesh::greedy_quads`]. The output lands in
/// [`BinaryGreedyQuadsBuffer::quads`], which uses the same public
/// `block-mesh` types as the upstream mesher. The buffer is cleared first.
///
/// # Output contract
///
/// This implementation preserves the set of visible unit faces that
/// `block_mesh::greedy_quads` would emit, and it usually stays very close to
/// the upstream quad count. It does not promise the exact same quad
/// partition.
///
/// # Extents
///
/// Any query extent is supported. Interiors up to `62` voxels per axis (`64`
/// including the one-voxel border, the usual binary-meshing chunk size) are
/// meshed in one pass. Wider queries are meshed in tiles of at most `62`
/// interior voxels per axis; visibility is exact across tile seams, but quads
/// do not merge across them.
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

/// Generates greedy quads with AO-safe merge restrictions for opaque faces.
///
/// This uses the same visibility pipeline as [`binary_greedy_quads`], but it
/// refuses merges that would cross AO boundaries on opaque faces: every unit
/// face inside an emitted opaque quad has the same four vertex AO values.
/// Translucent faces keep the same merge rule they use in the vanilla path.
///
/// This only changes the merge policy. It does not emit AO values or lighting
/// data for you; [`FaceAoSampler`] computes them per quad afterwards.
///
/// Extents work as in [`binary_greedy_quads`], and AO stays exact across tile
/// seams.
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

/// Emits one unit quad per visible face, like
/// [`block_mesh::visible_block_faces`].
///
/// Uses the same binary visibility pipeline as [`binary_greedy_quads`] and
/// emits exactly the faces `visible_block_faces` would, for any query extent.
/// Unlike upstream, the output buffer is cleared first.
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
