//! `block-mesh`-compatible binary greedy meshing.
//!
//! [`binary_greedy_quads`] mirrors the call shape of
//! [`block_mesh::greedy_quads`], but it reaches the same kind of `QuadBuffer`
//! through a different internal pipeline:
//!
//! 1. Validate the padded query extent and precompute a few indexing facts.
//! 2. Build packed occupancy columns, one `u64` per orthogonal column.
//! 3. Turn those columns into visible-face rows with cheap bitwise tests.
//! 4. Merge each face slice into quads, either as unit quads or with a
//!    carry-based greedy row merger.
//!
//! The implementation is split into the same stages in source:
//!
//! - `context`: query validation and precomputed layout facts
//! - `prep`: occupancy columns and visible-face rows
//! - `merge`: unit emission and carry-based greedy merging
//! - `merge_modes`: AO-safe and T-junction-free alternate merge policies
//! - `face`: face-orientation helpers shared by the other stages
//!
//! The crate exposes two public entry points:
//!
//! - [`binary_greedy_quads`]: the current maximum-merge implementation with no
//!   extra policy checks
//! - [`binary_greedy_quads_with_config`]: the same visibility pipeline, but
//!   with optional merge restrictions for AO-friendly quads and coplanar
//!   T-junction elimination
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
//! The configurable entry point uses the same buffer type:
//!
//! ```
//! # use block_mesh::ndshape::{ConstShape, ConstShape3u32};
//! # use block_mesh::{
//! #     MergeVoxel, Voxel, VoxelVisibility, RIGHT_HANDED_Y_UP_CONFIG,
//! # };
//! # use block_mesh_bgm::{
//! #     binary_greedy_quads_with_config, BinaryGreedyQuadsBuffer,
//! #     BinaryGreedyQuadsConfig,
//! # };
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
//! let config = BinaryGreedyQuadsConfig {
//!     ambient_occlusion_safe: true,
//!     eliminate_t_junctions: true,
//! };
//!
//! binary_greedy_quads_with_config(
//!     &voxels,
//!     &ChunkShape {},
//!     [0; 3],
//!     [3; 3],
//!     &RIGHT_HANDED_Y_UP_CONFIG.faces,
//!     &config,
//!     &mut buffer,
//! );
//! ```
#![warn(missing_docs)]

mod context;
mod face;
mod merge;
mod merge_modes;
mod prep;

use block_mesh::{MergeVoxel, OrientedBlockFace, QuadBuffer};
use context::MeshingContext;
use merge::mesh_face_rows;
use merge_modes::{mesh_face_rows_with_features, FeatureScratch};
use ndshape::Shape;
use prep::{build_axis_columns, build_visible_row_pair, reset_columns, reset_visible_rows};

/// Additional merge-policy controls for [`binary_greedy_quads_with_config`].
///
/// The default configuration keeps the current maximum-merge behavior and does
/// not change the output of [`binary_greedy_quads`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BinaryGreedyQuadsConfig {
    /// Prevents merges of opaque faces whose four ambient-occlusion corner
    /// values differ.
    ///
    /// This only changes the merge policy. It does not emit AO data.
    pub ambient_occlusion_safe: bool,
    /// Splits slice-local quads until no coplanar T-junctions remain within a
    /// face group.
    ///
    /// This mode uses a dedicated conforming merge kernel and applies a small
    /// slice-local cleanup split only when needed. It does not change
    /// visibility.
    pub eliminate_t_junctions: bool,
}

/// Reusable output and scratch storage for [`binary_greedy_quads`].
///
/// Reusing one buffer per worker thread keeps the hot meshing path focused on
/// bitwise work instead of repeated heap growth.
#[derive(Default)]
pub struct BinaryGreedyQuadsBuffer {
    /// Output quads grouped by face, matching `block-mesh`'s `QuadBuffer`.
    pub quads: QuadBuffer,
    // Stage 1 scratch: packed opaque occupancy columns for each candidate bit axis.
    opaque_cols: [Vec<u64>; 3],
    // Stage 1 scratch: packed translucent occupancy columns with the same layout.
    trans_cols: [Vec<u64>; 3],
    // Stage 2 scratch: visibility rows for the negative face of one normal axis.
    visible_rows: Vec<u64>,
    // Stage 2 scratch: visibility rows for the positive face of the same axis.
    visible_rows_alt: Vec<u64>,
    // Stage 4 scratch: carry lengths for the current face slice.
    carry_runs: Vec<u8>,
    // Alternate-mode scratch: AO keys, slice-local quads, and split workspace.
    feature_scratch: FeatureScratch,
}

impl BinaryGreedyQuadsBuffer {
    /// Creates an empty reusable output/scratch buffer.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Generates greedy quads using a binary-mask-backed implementation.
///
/// The public API mirrors [`block_mesh::greedy_quads`]. The output lands in
/// [`BinaryGreedyQuadsBuffer::quads`], which uses the same public
/// `block-mesh` types as the upstream mesher.
///
/// # Output contract
///
/// This implementation preserves the set of visible unit faces that
/// `block_mesh::greedy_quads` would emit for supported inputs, and it usually
/// stays very close to the upstream quad count. It does not promise the exact
/// same quad partition.
///
/// # Supported extents
///
/// The interior query extent may be at most `62` voxels wide on each axis.
/// Including the required one-voxel border, the full query box may therefore be
/// at most `64` voxels wide on each axis.
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
    binary_greedy_quads_impl::<T, S, false, false>(voxels, voxels_shape, min, max, faces, output);
}

/// Generates greedy quads with configurable merge restrictions.
///
/// [`BinaryGreedyQuadsConfig::default`] matches [`binary_greedy_quads`]
/// exactly. Alternate configurations may emit more quads in exchange for AO-
/// safe merging, coplanar T-junction elimination, or both.
///
/// `ambient_occlusion_safe` only affects opaque faces. Translucent faces keep
/// the same merge rule they use in the vanilla path.
///
/// `eliminate_t_junctions` switches to a dedicated conforming merge rule and
/// may apply a conservative slice-local cleanup split. It never changes the
/// set of visible unit faces.
pub fn binary_greedy_quads_with_config<T, S>(
    voxels: &[T],
    voxels_shape: &S,
    min: [u32; 3],
    max: [u32; 3],
    faces: &[OrientedBlockFace; 6],
    config: &BinaryGreedyQuadsConfig,
    output: &mut BinaryGreedyQuadsBuffer,
) where
    T: MergeVoxel,
    S: Shape<3, Coord = u32>,
{
    match (config.ambient_occlusion_safe, config.eliminate_t_junctions) {
        (false, false) => {
            binary_greedy_quads_impl::<T, S, false, false>(
                voxels,
                voxels_shape,
                min,
                max,
                faces,
                output,
            );
        }
        (true, false) => {
            binary_greedy_quads_impl::<T, S, true, false>(
                voxels,
                voxels_shape,
                min,
                max,
                faces,
                output,
            );
        }
        (false, true) => {
            binary_greedy_quads_impl::<T, S, false, true>(
                voxels,
                voxels_shape,
                min,
                max,
                faces,
                output,
            );
        }
        (true, true) => {
            binary_greedy_quads_impl::<T, S, true, true>(
                voxels,
                voxels_shape,
                min,
                max,
                faces,
                output,
            );
        }
    }
}

fn binary_greedy_quads_impl<T, S, const AO_SAFE: bool, const NO_T_JUNCTIONS: bool>(
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
    let Some(context) = MeshingContext::new(voxels, voxels_shape, min, max, faces) else {
        output.quads.reset();
        return;
    };

    output.quads.reset();

    let BinaryGreedyQuadsBuffer {
        quads,
        opaque_cols,
        trans_cols,
        visible_rows,
        visible_rows_alt,
        carry_runs,
        feature_scratch,
    } = output;

    // Stage 1: build reusable occupancy columns for all three candidate bit axes.
    reset_columns(opaque_cols, trans_cols, context.query_shape);
    let has_translucent = build_axis_columns(
        voxels,
        min,
        context.query_shape,
        context.strides,
        opaque_cols,
        trans_cols,
    );

    for n_axis in 0..3 {
        let slice = context.slice_plan(n_axis);
        let [neg_face_index, pos_face_index] = context.face_indices[n_axis];
        let neg_axes = context.face_axes[neg_face_index];
        let pos_axes = context.face_axes[pos_face_index];

        // Stage 2: derive visible-face rows for both signs of the current normal axis.
        reset_visible_rows(visible_rows, slice.total_rows());
        reset_visible_rows(visible_rows_alt, slice.total_rows());
        let (neg_unit_only, pos_unit_only) = build_visible_row_pair(
            context.query_shape,
            slice.bit_axis,
            slice.bit_len,
            slice.outer_len,
            slice.n_len,
            n_axis,
            slice.outer_axis,
            &opaque_cols[slice.bit_axis],
            &trans_cols[slice.bit_axis],
            has_translucent,
            visible_rows,
            visible_rows_alt,
        );

        // Stage 3 and 4: emit unit quads for fragmented slices, otherwise
        // merge rows with either the vanilla maximum-merge path or one of the
        // alternate merge policies.
        if AO_SAFE || NO_T_JUNCTIONS {
            mesh_face_rows_with_features::<T, AO_SAFE, NO_T_JUNCTIONS>(
                voxels,
                &context,
                slice,
                visible_rows,
                neg_unit_only,
                neg_axes,
                feature_scratch,
                &mut quads.groups[neg_face_index],
            );
            mesh_face_rows_with_features::<T, AO_SAFE, NO_T_JUNCTIONS>(
                voxels,
                &context,
                slice,
                visible_rows_alt,
                pos_unit_only,
                pos_axes,
                feature_scratch,
                &mut quads.groups[pos_face_index],
            );
        } else {
            mesh_face_rows(
                voxels,
                &context,
                slice,
                visible_rows,
                neg_unit_only,
                neg_axes,
                carry_runs,
                &mut quads.groups[neg_face_index],
            );
            mesh_face_rows(
                voxels,
                &context,
                slice,
                visible_rows_alt,
                pos_unit_only,
                pos_axes,
                carry_runs,
                &mut quads.groups[pos_face_index],
            );
        }
    }
}

#[inline]
pub(crate) fn bit_mask(start: usize, width: usize) -> u64 {
    ((1u64 << width) - 1) << start
}
