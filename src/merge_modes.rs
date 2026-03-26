//! AO-safe alternate merge policy.
//!
//! The default [`crate::binary_greedy_quads`] path stays on the existing
//! carry-based maximum-merge implementation in [`crate::merge`]. This module
//! handles the AO-safe variant. Instead of
//! materializing one AO signature byte per visible opaque cell, it derives
//! merge constraints directly from binary opaque-occupancy rows in the face's
//! exterior sample plane.
//!
//! The core idea is:
//!
//! - use `opaque_cols` to build whole-row AO constraint masks with shifts and
//!   intersections
//! - peel away cells that are provably unit-only, row-only, or column-only
//! - run the full greedy carry kernel only on the residual cells that still may
//!   merge in both directions

#[cfg(feature = "internal-profiler")]
use std::time::Instant;

use block_mesh::{MergeVoxel, UnorientedQuad};

use crate::bit_mask;
use crate::context::{MeshingContext, SlicePlan};
use crate::face::{write_quad, write_unit_quad, FaceAxes};
use crate::prep::column_row_layout;

/// Internal feature bitset for alternate merge policies.
///
/// The call shape stays stable even as new feature-specific kernels are added.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct MergeFeatures {
    bits: u8,
}

impl MergeFeatures {
    const AMBIENT_OCCLUSION_SAFE: u8 = 1 << 0;

    /// No alternate merge restrictions.
    pub(crate) const NONE: Self = Self { bits: 0 };

    /// Restrict merges to cells whose AO-compatible boundaries match.
    pub(crate) const AO_SAFE: Self = Self {
        bits: Self::AMBIENT_OCCLUSION_SAFE,
    };

    #[inline]
    const fn ambient_occlusion_safe(self) -> bool {
        self.bits & Self::AMBIENT_OCCLUSION_SAFE != 0
    }
}

/// Reusable scratch that only AO-safe meshing needs.
#[derive(Default)]
pub(crate) struct FeatureScratch {
    // Visible cells whose exterior occupancy forces a 1x1 quad.
    unit_rows: Vec<u64>,
    // Visible cells that may only merge along the packed row axis.
    horizontal_rows: Vec<u64>,
    // Visible cells that may only merge from row to row, always at width 1.
    vertical_rows: Vec<u64>,
    // Visible cells that are still free to use the normal bidirectional greedy
    // merge after the stricter AO-only cases are peeled away.
    bidir_rows: Vec<u64>,
    carry_runs: Vec<u8>,
}

/// Dispatches one face slice to the requested alternate merge policy.
#[inline(always)]
pub(crate) fn mesh_face_rows_with_features<T>(
    voxels: &[T],
    context: &MeshingContext,
    slice: SlicePlan,
    visible_rows: &[u64],
    unit_only: bool,
    axes: FaceAxes,
    opaque_cols: &[u64],
    features: MergeFeatures,
    scratch: &mut FeatureScratch,
    quads: &mut Vec<UnorientedQuad>,
) where
    T: MergeVoxel,
{
    debug_assert!(
        features != MergeFeatures::NONE,
        "alternate merge dispatch should not be used for the vanilla path"
    );

    if features.ambient_occlusion_safe() {
        mesh_face_rows_ao_safe(
            voxels,
            context,
            slice,
            visible_rows,
            unit_only,
            axes,
            opaque_cols,
            scratch,
            quads,
        );
        return;
    }

    unreachable!("unsupported alternate merge feature set: {:?}", features);
}

/// Chooses the compile-time-specialized AO-safe kernel for one face
/// orientation.
#[inline(always)]
fn mesh_face_rows_ao_safe<T>(
    voxels: &[T],
    context: &MeshingContext,
    slice: SlicePlan,
    visible_rows: &[u64],
    unit_only: bool,
    axes: FaceAxes,
    opaque_cols: &[u64],
    scratch: &mut FeatureScratch,
    quads: &mut Vec<UnorientedQuad>,
) where
    T: MergeVoxel,
{
    match (axes.n_axis, slice.bit_axis == axes.u_axis) {
        (0, false) => mesh_face_rows_ao_safe_impl::<T, 0, false>(
            voxels,
            context,
            slice,
            visible_rows,
            unit_only,
            axes,
            opaque_cols,
            scratch,
            quads,
        ),
        (0, true) => mesh_face_rows_ao_safe_impl::<T, 0, true>(
            voxels,
            context,
            slice,
            visible_rows,
            unit_only,
            axes,
            opaque_cols,
            scratch,
            quads,
        ),
        (1, false) => mesh_face_rows_ao_safe_impl::<T, 1, false>(
            voxels,
            context,
            slice,
            visible_rows,
            unit_only,
            axes,
            opaque_cols,
            scratch,
            quads,
        ),
        (1, true) => mesh_face_rows_ao_safe_impl::<T, 1, true>(
            voxels,
            context,
            slice,
            visible_rows,
            unit_only,
            axes,
            opaque_cols,
            scratch,
            quads,
        ),
        (2, false) => mesh_face_rows_ao_safe_impl::<T, 2, false>(
            voxels,
            context,
            slice,
            visible_rows,
            unit_only,
            axes,
            opaque_cols,
            scratch,
            quads,
        ),
        (2, true) => mesh_face_rows_ao_safe_impl::<T, 2, true>(
            voxels,
            context,
            slice,
            visible_rows,
            unit_only,
            axes,
            opaque_cols,
            scratch,
            quads,
        ),
        _ => unreachable!(),
    }
}

#[inline(always)]
fn mesh_face_rows_ao_safe_impl<T, const N_AXIS: usize, const BIT_IS_U: bool>(
    voxels: &[T],
    context: &MeshingContext,
    slice: SlicePlan,
    visible_rows: &[u64],
    unit_only: bool,
    axes: FaceAxes,
    opaque_cols: &[u64],
    scratch: &mut FeatureScratch,
    quads: &mut Vec<UnorientedQuad>,
) where
    T: MergeVoxel,
{
    #[cfg(feature = "internal-profiler")]
    crate::profile::record_slice(unit_only);

    if unit_only {
        for n_local in 0..slice.n_len {
            let row_start = n_local * slice.outer_len;
            #[cfg(feature = "internal-profiler")]
            let unit_emit_start = Instant::now();
            emit_unit_slice::<N_AXIS>(
                context.interior_min,
                slice.outer_axis,
                slice.bit_axis,
                context.interior_min[N_AXIS] + n_local as u32,
                &visible_rows[row_start..row_start + slice.outer_len],
                quads,
            );
            #[cfg(feature = "internal-profiler")]
            crate::profile::record_unit_emit(unit_emit_start.elapsed());
        }
        return;
    }

    let n_base = context.interior_min[N_AXIS];
    let outer_base = context.interior_min[slice.outer_axis];
    let bit_base = context.interior_min[slice.bit_axis];
    let outer_stride = context.strides[slice.outer_axis];
    let bit_stride = context.strides[slice.bit_axis];

    for n_local in 0..slice.n_len {
        let row_start = n_local * slice.outer_len;
        let slice_rows = &visible_rows[row_start..row_start + slice.outer_len];
        let n_coord = n_base + n_local as u32;
        let n_index_base = context.interior_start_index + n_local * context.strides[N_AXIS];

        #[cfg(feature = "internal-profiler")]
        let key_build_start = Instant::now();
        build_slice_ao_masks(
            context.query_shape,
            N_AXIS,
            slice,
            n_local,
            axes.n_sign,
            opaque_cols,
            slice_rows,
            &mut scratch.unit_rows,
            &mut scratch.horizontal_rows,
            &mut scratch.vertical_rows,
            &mut scratch.bidir_rows,
        );
        #[cfg(feature = "internal-profiler")]
        crate::profile::record_key_build(key_build_start.elapsed());

        #[cfg(feature = "internal-profiler")]
        let carry_start = Instant::now();
        // Emit the AO-constrained rows in increasing order of generality. The
        // first three avoid the full carry kernel entirely; only the residual
        // bidirectionally mergeable rows need the same greedy row merger shape
        // as vanilla meshing.
        emit_unit_rows::<N_AXIS>(
            &scratch.unit_rows[..slice.outer_len],
            n_coord,
            outer_base,
            bit_base,
            quads,
        );
        emit_horizontal_rows::<T, N_AXIS, BIT_IS_U>(
            voxels,
            &scratch.horizontal_rows[..slice.outer_len],
            n_index_base,
            outer_stride,
            bit_stride,
            slice.bit_len,
            n_coord,
            outer_base,
            bit_base,
            quads,
        );
        emit_vertical_rows::<T, N_AXIS, BIT_IS_U>(
            voxels,
            &scratch.vertical_rows[..slice.outer_len],
            n_index_base,
            outer_stride,
            bit_stride,
            slice.bit_len,
            n_coord,
            outer_base,
            bit_base,
            &mut scratch.carry_runs,
            quads,
        );
        mesh_bidir_rows::<T, N_AXIS, BIT_IS_U>(
            voxels,
            &scratch.bidir_rows[..slice.outer_len],
            n_index_base,
            outer_stride,
            bit_stride,
            slice.bit_len,
            n_coord,
            outer_base,
            bit_base,
            &mut scratch.carry_runs,
            quads,
        );
        #[cfg(feature = "internal-profiler")]
        crate::profile::record_carry_total(carry_start.elapsed());
    }
}

#[inline(always)]
fn build_slice_ao_masks(
    query_shape: [usize; 3],
    n_axis: usize,
    slice: SlicePlan,
    n_local: usize,
    face_sign: i32,
    opaque_cols: &[u64],
    slice_rows: &[u64],
    unit_rows: &mut Vec<u64>,
    horizontal_rows: &mut Vec<u64>,
    vertical_rows: &mut Vec<u64>,
    bidir_rows: &mut Vec<u64>,
) {
    // AO-safe meshing only cares about opaque occupancy in the plane just
    // outside the visible face. From that outside plane we can derive, a row at
    // a time, whether a visible cell:
    //
    // - must stay a unit quad
    // - may only merge within its current row
    // - may only merge across rows at width 1
    // - or is unconstrained enough to fall back to the normal carry merge
    //
    // This avoids carrying a per-cell AO signature through the merge loop.
    reset_mask_rows(unit_rows, slice.outer_len);
    reset_mask_rows(horizontal_rows, slice.outer_len);
    reset_mask_rows(vertical_rows, slice.outer_len);
    reset_mask_rows(bidir_rows, slice.outer_len);

    let interior_bit_mask = bit_mask(0, slice.bit_len);
    let (base_offset, n_stride, column_outer_stride) =
        column_row_layout(query_shape, slice.bit_axis, n_axis, slice.outer_axis);
    let source_n = n_local + 1;
    let outside_n = if face_sign < 0 {
        source_n - 1
    } else {
        source_n + 1
    };
    let source_base = base_offset + source_n * n_stride;
    let outside_base = base_offset + outside_n * n_stride;

    for (outer_local, &row_bits) in slice_rows.iter().enumerate() {
        if row_bits == 0 {
            continue;
        }

        let source_opaque =
            (opaque_cols[source_base + outer_local * column_outer_stride] >> 1) & interior_bit_mask;
        let opaque_visible = row_bits & source_opaque;

        if opaque_visible == 0 {
            // Translucent and empty-visible faces do not participate in the AO
            // rule set. They keep the vanilla merge behavior.
            bidir_rows[outer_local] = row_bits;
            #[cfg(feature = "internal-profiler")]
            crate::profile::record_key_row(0, row_bits.count_ones(), true);
            continue;
        }

        let current_row = (opaque_cols[outside_base + outer_local * column_outer_stride] >> 1)
            & interior_bit_mask;
        let prev_row = if outer_local > 0 {
            (opaque_cols[outside_base + (outer_local - 1) * column_outer_stride] >> 1)
                & interior_bit_mask
        } else {
            0
        };
        let next_row = if outer_local + 1 < slice.outer_len {
            (opaque_cols[outside_base + (outer_local + 1) * column_outer_stride] >> 1)
                & interior_bit_mask
        } else {
            0
        };

        // Classify the visible opaque bits by the merge directions that remain
        // legal once AO boundaries are respected.
        let neighbor_curr = ((current_row << 1) | (current_row >> 1)) & interior_bit_mask;
        let shared_depth = prev_row & current_row & next_row;
        let vertical = opaque_visible
            & ((shared_depth << 1) & interior_bit_mask)
            & (shared_depth >> 1)
            & (!prev_row)
            & (!next_row)
            & interior_bit_mask;

        let edge_from_next = (((next_row << 1) & interior_bit_mask) ^ (next_row >> 1))
            & interior_bit_mask;
        let edge_from_prev = (((prev_row << 1) & interior_bit_mask) ^ (prev_row >> 1))
            & interior_bit_mask;
        let unit = opaque_visible
            & !vertical
            & (neighbor_curr | edge_from_next | edge_from_prev)
            & interior_bit_mask;

        let horiz_from_next = ((next_row << 1) & interior_bit_mask) & (next_row >> 1);
        let horiz_from_prev = ((prev_row << 1) & interior_bit_mask) & (prev_row >> 1);
        let horizontal = opaque_visible
            & !vertical
            & !unit
            & (horiz_from_next | horiz_from_prev)
            & interior_bit_mask;

        let bidir = row_bits & !(unit | horizontal | vertical);

        unit_rows[outer_local] = unit;
        horizontal_rows[outer_local] = horizontal;
        vertical_rows[outer_local] = vertical;
        bidir_rows[outer_local] = bidir;

        #[cfg(feature = "internal-profiler")]
        crate::profile::record_key_row(
            opaque_visible.count_ones(),
            bidir.count_ones(),
            opaque_visible & !(unit | horizontal | vertical) == opaque_visible,
        );
    }
}

#[inline(always)]
fn emit_unit_rows<const N_AXIS: usize>(
    rows: &[u64],
    n_coord: u32,
    outer_base: u32,
    bit_base: u32,
    quads: &mut Vec<UnorientedQuad>,
) {
    #[cfg(feature = "internal-profiler")]
    crate::profile::record_unit_quads(
        rows.iter().map(|row_bits| row_bits.count_ones() as usize).sum::<usize>(),
    );

    for (outer_local, &row_bits) in rows.iter().enumerate() {
        if row_bits == 0 {
            continue;
        }

        emit_single_row_unit_quads::<N_AXIS>(
            row_bits,
            n_coord,
            outer_base + outer_local as u32,
            bit_base,
            quads,
        );
    }
}

#[inline(always)]
fn emit_horizontal_rows<T, const N_AXIS: usize, const BIT_IS_U: bool>(
    voxels: &[T],
    rows: &[u64],
    n_index_base: usize,
    outer_stride: usize,
    bit_stride: usize,
    bit_len: usize,
    n_coord: u32,
    outer_base: u32,
    bit_base: u32,
    quads: &mut Vec<UnorientedQuad>,
) where
    T: MergeVoxel,
{
    for (outer_local, &row_bits) in rows.iter().enumerate() {
        if row_bits == 0 {
            continue;
        }

        #[cfg(feature = "internal-profiler")]
        let emit_start = Instant::now();
        emit_single_row_runs_plain::<T, N_AXIS, BIT_IS_U>(
            voxels,
            row_bits,
            n_index_base + outer_local * outer_stride,
            bit_stride,
            bit_len,
            n_coord,
            outer_base + outer_local as u32,
            bit_base,
            quads,
        );
        #[cfg(feature = "internal-profiler")]
        {
            crate::profile::record_single_row();
            crate::profile::record_emit_single(emit_start.elapsed());
        }
    }
}

#[inline(always)]
fn emit_vertical_rows<T, const N_AXIS: usize, const BIT_IS_U: bool>(
    voxels: &[T],
    rows: &[u64],
    n_index_base: usize,
    outer_stride: usize,
    bit_stride: usize,
    bit_len: usize,
    n_coord: u32,
    outer_base: u32,
    bit_base: u32,
    carry_runs: &mut Vec<u8>,
    quads: &mut Vec<UnorientedQuad>,
) where
    T: MergeVoxel,
{
    // These rows are only allowed to continue from one row to the next. Since
    // width can never grow, the carry state only tracks height and this path is
    // much cheaper than the general greedy kernel.
    reset_carry_runs(carry_runs, bit_len);
    let mut has_incoming_carry = false;

    for outer_local in 0..rows.len() {
        let row_bits = rows[outer_local];
        if row_bits == 0 {
            has_incoming_carry = false;
            continue;
        }

        let next_row_bits = if outer_local + 1 < rows.len() {
            rows[outer_local + 1]
        } else {
            0
        };
        let row_base_index = n_index_base + outer_local * outer_stride;
        let overlapping_bits = row_bits & next_row_bits;

        #[cfg(feature = "internal-profiler")]
        crate::profile::record_carry_row(row_bits, overlapping_bits);
        #[cfg(feature = "internal-profiler")]
        crate::profile::record_ao_overlap_candidates(overlapping_bits);

        if overlapping_bits == 0 {
            #[cfg(feature = "internal-profiler")]
            let emit_start = Instant::now();
            if has_incoming_carry {
                emit_carried_row_unit_quads::<N_AXIS, BIT_IS_U>(
                    row_bits,
                    n_coord,
                    outer_base + outer_local as u32,
                    bit_base,
                    carry_runs,
                    quads,
                );
                #[cfg(feature = "internal-profiler")]
                crate::profile::record_terminal_row();
            } else {
                emit_single_row_unit_quads::<N_AXIS>(
                    row_bits,
                    n_coord,
                    outer_base + outer_local as u32,
                    bit_base,
                    quads,
                );
                #[cfg(feature = "internal-profiler")]
                crate::profile::record_single_row();
            }
            #[cfg(feature = "internal-profiler")]
            crate::profile::record_emit_terminal(emit_start.elapsed());
            has_incoming_carry = false;
            continue;
        }

        #[cfg(feature = "internal-profiler")]
        let continue_start = Instant::now();
        let continue_mask = build_continue_mask_plain(
            voxels,
            overlapping_bits,
            row_base_index,
            bit_stride,
            outer_stride,
            carry_runs,
        );
        #[cfg(feature = "internal-profiler")]
        {
            crate::profile::record_continue_mask(continue_start.elapsed());
            crate::profile::record_continued_bits(continue_mask);
        }

        let ended_bits = row_bits & !continue_mask;
        if ended_bits != 0 {
            #[cfg(feature = "internal-profiler")]
            let emit_start = Instant::now();
            if has_incoming_carry {
                emit_carried_row_unit_quads::<N_AXIS, BIT_IS_U>(
                    ended_bits,
                    n_coord,
                    outer_base + outer_local as u32,
                    bit_base,
                    carry_runs,
                    quads,
                );
                #[cfg(feature = "internal-profiler")]
                crate::profile::record_terminal_row();
            } else {
                emit_single_row_unit_quads::<N_AXIS>(
                    ended_bits,
                    n_coord,
                    outer_base + outer_local as u32,
                    bit_base,
                    quads,
                );
                #[cfg(feature = "internal-profiler")]
                crate::profile::record_single_row();
            }
            #[cfg(feature = "internal-profiler")]
            crate::profile::record_emit_terminal(emit_start.elapsed());
        }

        has_incoming_carry = continue_mask != 0;
    }
}

#[inline(always)]
fn mesh_bidir_rows<T, const N_AXIS: usize, const BIT_IS_U: bool>(
    voxels: &[T],
    rows: &[u64],
    n_index_base: usize,
    outer_stride: usize,
    bit_stride: usize,
    bit_len: usize,
    n_coord: u32,
    outer_base: u32,
    bit_base: u32,
    carry_runs: &mut Vec<u8>,
    quads: &mut Vec<UnorientedQuad>,
) where
    T: MergeVoxel,
{
    // These rows are the residual case: nothing in the exterior-plane AO masks
    // proves that they must stay unit-width or row-local, so they can reuse
    // the same bidirectional carry merge shape as the vanilla path.
    reset_carry_runs(carry_runs, bit_len);
    let mut has_incoming_carry = false;

    for outer_local in 0..rows.len() {
        let row_bits = rows[outer_local];
        if row_bits == 0 {
            has_incoming_carry = false;
            continue;
        }

        let next_row_bits = if outer_local + 1 < rows.len() {
            rows[outer_local + 1]
        } else {
            0
        };
        let row_base_index = n_index_base + outer_local * outer_stride;
        let overlapping_bits = row_bits & next_row_bits;

        #[cfg(feature = "internal-profiler")]
        crate::profile::record_carry_row(row_bits, overlapping_bits);
        #[cfg(feature = "internal-profiler")]
        crate::profile::record_ao_overlap_candidates(overlapping_bits);

        if overlapping_bits == 0 {
            #[cfg(feature = "internal-profiler")]
            let emit_start = Instant::now();
            if has_incoming_carry {
                emit_terminal_row_runs_plain::<T, N_AXIS, BIT_IS_U>(
                    voxels,
                    row_bits,
                    row_base_index,
                    bit_stride,
                    bit_len,
                    n_coord,
                    outer_base + outer_local as u32,
                    bit_base,
                    carry_runs,
                    quads,
                );
                #[cfg(feature = "internal-profiler")]
                crate::profile::record_terminal_row();
            } else {
                emit_single_row_runs_plain::<T, N_AXIS, BIT_IS_U>(
                    voxels,
                    row_bits,
                    row_base_index,
                    bit_stride,
                    bit_len,
                    n_coord,
                    outer_base + outer_local as u32,
                    bit_base,
                    quads,
                );
                #[cfg(feature = "internal-profiler")]
                crate::profile::record_single_row();
            }
            #[cfg(feature = "internal-profiler")]
            crate::profile::record_emit_single(emit_start.elapsed());
            has_incoming_carry = false;
            continue;
        }

        #[cfg(feature = "internal-profiler")]
        let continue_start = Instant::now();
        let continue_mask = build_continue_mask_plain(
            voxels,
            overlapping_bits,
            row_base_index,
            bit_stride,
            outer_stride,
            carry_runs,
        );
        #[cfg(feature = "internal-profiler")]
        {
            crate::profile::record_continue_mask(continue_start.elapsed());
            crate::profile::record_continued_bits(continue_mask);
        }

        let ended_bits = row_bits & !continue_mask;
        if !has_incoming_carry {
            if ended_bits != 0 {
                #[cfg(feature = "internal-profiler")]
                let emit_start = Instant::now();
                emit_single_row_runs_plain::<T, N_AXIS, BIT_IS_U>(
                    voxels,
                    ended_bits,
                    row_base_index,
                    bit_stride,
                    bit_len,
                    n_coord,
                    outer_base + outer_local as u32,
                    bit_base,
                    quads,
                );
                #[cfg(feature = "internal-profiler")]
                {
                    crate::profile::record_single_row();
                    crate::profile::record_emit_single(emit_start.elapsed());
                }
            }
            has_incoming_carry = continue_mask != 0;
            continue;
        }

        if ended_bits == 0 {
            has_incoming_carry = continue_mask != 0;
            continue;
        }

        #[cfg(feature = "internal-profiler")]
        let emit_start = Instant::now();
        emit_mixed_row_runs_plain::<T, N_AXIS, BIT_IS_U>(
            voxels,
            ended_bits,
            row_base_index,
            bit_stride,
            bit_len,
            n_coord,
            outer_base + outer_local as u32,
            bit_base,
            carry_runs,
            quads,
        );
        #[cfg(feature = "internal-profiler")]
        {
            crate::profile::record_mixed_row();
            crate::profile::record_emit_mixed(emit_start.elapsed());
        }
        has_incoming_carry = continue_mask != 0;
    }
}

#[inline(always)]
fn build_continue_mask_plain<T>(
    voxels: &[T],
    mut overlapping_bits: u64,
    row_base_index: usize,
    bit_stride: usize,
    outer_stride: usize,
    carry_runs: &mut [u8],
) -> u64
where
    T: MergeVoxel,
{
    let mut continue_mask = 0u64;

    while overlapping_bits != 0 {
        let bit_local = overlapping_bits.trailing_zeros() as usize;
        let bit = 1u64 << bit_local;
        let voxel_index = row_base_index + bit_local * bit_stride;
        let quad_value = unsafe { voxels.get_unchecked(voxel_index) }.merge_value();

        if unsafe { row_bit_continues(voxels, voxel_index, outer_stride, &quad_value) } {
            continue_mask |= bit;
            carry_runs[bit_local] += 1;
        }

        overlapping_bits &= overlapping_bits - 1;
    }

    continue_mask
}

#[inline(always)]
fn emit_mixed_row_runs_plain<T, const N_AXIS: usize, const BIT_IS_U: bool>(
    voxels: &[T],
    mut row_bits: u64,
    row_base_index: usize,
    bit_stride: usize,
    bit_len: usize,
    n_coord: u32,
    outer_coord: u32,
    bit_base: u32,
    carry_runs: &mut [u8],
    quads: &mut Vec<UnorientedQuad>,
) where
    T: MergeVoxel,
{
    let base_len = quads.len();
    quads.reserve(row_bits.count_ones() as usize);
    let spare = quads.spare_capacity_mut();
    let mut written = 0usize;

    while row_bits != 0 {
        let bit_local = row_bits.trailing_zeros() as usize;
        let voxel_index = row_base_index + bit_local * bit_stride;
        let quad_value = unsafe { voxels.get_unchecked(voxel_index) }.merge_value();
        let run_carry = carry_runs[bit_local];
        let run_length = run_carry as usize + 1;
        let mut run_width = 1usize;

        while bit_local + run_width < bit_len {
            let next_bit_local = bit_local + run_width;
            let next_bit = 1u64 << next_bit_local;
            if row_bits & next_bit == 0 || carry_runs[next_bit_local] != run_carry {
                break;
            }

            let next_index = row_base_index + next_bit_local * bit_stride;
            if unsafe { voxels.get_unchecked(next_index) }
                .merge_value()
                .ne(&quad_value)
            {
                break;
            }

            carry_runs[next_bit_local] = 0;
            run_width += 1;
        }

        row_bits &= !bit_mask(bit_local, run_width);
        write_quad::<N_AXIS, BIT_IS_U>(
            &mut spare[written],
            n_coord,
            outer_coord - run_carry as u32,
            bit_base + bit_local as u32,
            run_width as u32,
            run_length as u32,
        );
        written += 1;
        carry_runs[bit_local] = 0;
    }

    unsafe {
        quads.set_len(base_len + written);
    }
}

#[inline(always)]
fn emit_single_row_runs_plain<T, const N_AXIS: usize, const BIT_IS_U: bool>(
    voxels: &[T],
    row_bits: u64,
    row_base_index: usize,
    bit_stride: usize,
    bit_len: usize,
    n_coord: u32,
    outer_coord: u32,
    bit_base: u32,
    quads: &mut Vec<UnorientedQuad>,
) where
    T: MergeVoxel,
{
    let base_len = quads.len();
    quads.reserve(row_bits.count_ones() as usize);
    let spare = quads.spare_capacity_mut();
    let mut written = 0usize;
    let mut bits_here = row_bits;

    while bits_here != 0 {
        let bit_local = bits_here.trailing_zeros() as usize;
        let voxel_index = row_base_index + bit_local * bit_stride;
        let quad_value = unsafe { voxels.get_unchecked(voxel_index) }.merge_value();
        let mut run_width = 1usize;

        while bit_local + run_width < bit_len {
            let next_bit_local = bit_local + run_width;
            let next_bit = 1u64 << next_bit_local;
            if bits_here & next_bit == 0 {
                break;
            }

            let next_index = row_base_index + next_bit_local * bit_stride;
            if unsafe { voxels.get_unchecked(next_index) }
                .merge_value()
                .ne(&quad_value)
            {
                break;
            }

            run_width += 1;
        }

        bits_here &= !bit_mask(bit_local, run_width);
        write_quad::<N_AXIS, BIT_IS_U>(
            &mut spare[written],
            n_coord,
            outer_coord,
            bit_base + bit_local as u32,
            run_width as u32,
            1,
        );
        written += 1;
    }

    unsafe {
        quads.set_len(base_len + written);
    }
}

#[inline(always)]
fn emit_terminal_row_runs_plain<T, const N_AXIS: usize, const BIT_IS_U: bool>(
    voxels: &[T],
    row_bits: u64,
    row_base_index: usize,
    bit_stride: usize,
    bit_len: usize,
    n_coord: u32,
    outer_coord: u32,
    bit_base: u32,
    carry_runs: &mut [u8],
    quads: &mut Vec<UnorientedQuad>,
) where
    T: MergeVoxel,
{
    let base_len = quads.len();
    quads.reserve(row_bits.count_ones() as usize);
    let spare = quads.spare_capacity_mut();
    let mut written = 0usize;
    let mut bits_here = row_bits;

    while bits_here != 0 {
        let bit_local = bits_here.trailing_zeros() as usize;
        let voxel_index = row_base_index + bit_local * bit_stride;
        let quad_value = unsafe { voxels.get_unchecked(voxel_index) }.merge_value();
        let run_carry = carry_runs[bit_local];
        let run_length = run_carry as usize + 1;
        let mut run_width = 1usize;

        while bit_local + run_width < bit_len {
            let next_bit_local = bit_local + run_width;
            let next_bit = 1u64 << next_bit_local;
            if bits_here & next_bit == 0 || carry_runs[next_bit_local] != run_carry {
                break;
            }

            let next_index = row_base_index + next_bit_local * bit_stride;
            if unsafe { voxels.get_unchecked(next_index) }
                .merge_value()
                .ne(&quad_value)
            {
                break;
            }

            carry_runs[next_bit_local] = 0;
            run_width += 1;
        }

        bits_here &= !bit_mask(bit_local, run_width);
        write_quad::<N_AXIS, BIT_IS_U>(
            &mut spare[written],
            n_coord,
            outer_coord - run_carry as u32,
            bit_base + bit_local as u32,
            run_width as u32,
            run_length as u32,
        );
        written += 1;
        carry_runs[bit_local] = 0;
    }

    unsafe {
        quads.set_len(base_len + written);
    }
}

#[inline(always)]
unsafe fn row_bit_continues<T>(
    voxels: &[T],
    voxel_index: usize,
    outer_stride: usize,
    quad_value: &T::MergeValue,
) -> bool
where
    T: MergeVoxel,
{
    voxels
        .get_unchecked(voxel_index + outer_stride)
        .merge_value()
        .eq(quad_value)
}

#[inline(always)]
fn emit_single_row_unit_quads<const N_AXIS: usize>(
    mut row_bits: u64,
    n_coord: u32,
    outer_coord: u32,
    bit_base: u32,
    quads: &mut Vec<UnorientedQuad>,
) {
    let quad_count = row_bits.count_ones() as usize;
    let base_len = quads.len();
    quads.reserve(quad_count);
    let spare = quads.spare_capacity_mut();
    let mut written = 0usize;

    while row_bits != 0 {
        let bit_local = row_bits.trailing_zeros() as usize;
        row_bits &= row_bits - 1;
        write_unit_quad::<N_AXIS>(
            &mut spare[written],
            n_coord,
            outer_coord,
            bit_base + bit_local as u32,
        );
        written += 1;
    }

    unsafe {
        quads.set_len(base_len + written);
    }
}

#[inline(always)]
fn emit_carried_row_unit_quads<const N_AXIS: usize, const BIT_IS_U: bool>(
    mut row_bits: u64,
    n_coord: u32,
    outer_coord: u32,
    bit_base: u32,
    carry_runs: &mut [u8],
    quads: &mut Vec<UnorientedQuad>,
) {
    let quad_count = row_bits.count_ones() as usize;
    let base_len = quads.len();
    quads.reserve(quad_count);
    let spare = quads.spare_capacity_mut();
    let mut written = 0usize;

    while row_bits != 0 {
        let bit_local = row_bits.trailing_zeros() as usize;
        row_bits &= row_bits - 1;
        let run_carry = carry_runs[bit_local] as u32;
        write_quad::<N_AXIS, BIT_IS_U>(
            &mut spare[written],
            n_coord,
            outer_coord - run_carry,
            bit_base + bit_local as u32,
            1,
            run_carry + 1,
        );
        carry_runs[bit_local] = 0;
        written += 1;
    }

    unsafe {
        quads.set_len(base_len + written);
    }
}

#[inline(always)]
fn reset_carry_runs(carry_runs: &mut Vec<u8>, len: usize) {
    carry_runs.resize(len, 0);
    carry_runs.fill(0);
}

#[inline(always)]
fn reset_mask_rows(rows: &mut Vec<u64>, len: usize) {
    rows.resize(len, 0);
    rows.fill(0);
}

#[inline(always)]
fn emit_unit_slice<const N_AXIS: usize>(
    interior_min: [u32; 3],
    outer_axis: usize,
    bit_axis: usize,
    n_coord: u32,
    visible_rows: &[u64],
    quads: &mut Vec<UnorientedQuad>,
) {
    let outer_base = interior_min[outer_axis];
    let bit_base = interior_min[bit_axis];
    let additional_quads = visible_rows
        .iter()
        .map(|row_bits| row_bits.count_ones() as usize)
        .sum::<usize>();
    #[cfg(feature = "internal-profiler")]
    crate::profile::record_unit_quads(additional_quads);
    let base_len = quads.len();
    quads.reserve(additional_quads);
    let spare = quads.spare_capacity_mut();
    let mut written = 0usize;

    for (outer_local, &row_bits) in visible_rows.iter().enumerate() {
        let mut bits = row_bits;
        let outer_coord = outer_base + outer_local as u32;

        while bits != 0 {
            let bit_local = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            write_unit_quad::<N_AXIS>(
                &mut spare[written],
                n_coord,
                outer_coord,
                bit_base + bit_local as u32,
            );
            written += 1;
        }
    }

    unsafe {
        quads.set_len(base_len + written);
    }
}
