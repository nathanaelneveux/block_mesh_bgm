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

use block_mesh::{MergeVoxel, UnorientedQuad};

use crate::bit_mask;
use crate::context::{MeshingContext, SlicePlan};
use crate::face::{write_quad, write_unit_quad, FaceAxes};
use crate::merge::{
    build_continue_mask, emit_mixed_row_runs, emit_single_row_runs, emit_terminal_row_runs,
    emit_unit_slice, reset_carry_runs,
};
use crate::prep::column_row_layout;

/// Reusable scratch that only AO-safe meshing needs.
#[derive(Default)]
pub(crate) struct AoScratch {
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

/// Merges one face slice with AO-safe merge restrictions.
#[inline(always)]
pub(crate) fn mesh_face_rows_ao_safe<T>(
    voxels: &[T],
    context: &MeshingContext,
    slice: SlicePlan,
    visible_rows: &[u64],
    unit_only: bool,
    axes: FaceAxes,
    opaque_cols: &[u64],
    scratch: &mut AoScratch,
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
    scratch: &mut AoScratch,
    quads: &mut Vec<UnorientedQuad>,
) where
    T: MergeVoxel,
{
    if unit_only {
        for n_local in 0..slice.n_len {
            let row_start = n_local * slice.outer_len;
            emit_unit_slice::<N_AXIS>(
                context.interior_min,
                slice.outer_axis,
                slice.bit_axis,
                context.interior_min[N_AXIS] + n_local as u32,
                &visible_rows[row_start..row_start + slice.outer_len],
                quads,
            );
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
    let padded_bit_mask = bit_mask(0, slice.bit_len + 2);
    let interior_padded_mask = bit_mask(1, slice.bit_len);
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

        let visible_padded = (row_bits << 1) & interior_padded_mask;
        let source_opaque =
            opaque_cols[source_base + outer_local * column_outer_stride] & interior_padded_mask;
        let opaque_visible = visible_padded & source_opaque;

        if opaque_visible == 0 {
            // Translucent and empty-visible faces do not participate in the AO
            // rule set. They keep the vanilla merge behavior.
            bidir_rows[outer_local] = row_bits;
            continue;
        }

        let current_row =
            opaque_cols[outside_base + outer_local * column_outer_stride] & padded_bit_mask;
        let prev_row = opaque_cols
            [outside_base + outer_local * column_outer_stride - column_outer_stride]
            & padded_bit_mask;
        let next_row = opaque_cols
            [outside_base + outer_local * column_outer_stride + column_outer_stride]
            & padded_bit_mask;

        // Classify the visible opaque bits by the merge directions that remain
        // legal once AO boundaries are respected.
        let (unit, horizontal, vertical) = classify_ao_opaque_row(
            opaque_visible,
            prev_row,
            current_row,
            next_row,
            interior_padded_mask,
        );

        let unit = (unit >> 1) & interior_bit_mask;
        let horizontal = (horizontal >> 1) & interior_bit_mask;
        let vertical = (vertical >> 1) & interior_bit_mask;
        let bidir = row_bits & !(unit | horizontal | vertical);

        unit_rows[outer_local] = unit;
        horizontal_rows[outer_local] = horizontal;
        vertical_rows[outer_local] = vertical;
        bidir_rows[outer_local] = bidir;
    }
}

#[inline(always)]
fn classify_ao_opaque_row(
    opaque_visible: u64,
    prev_row: u64,
    current_row: u64,
    next_row: u64,
    interior_bit_mask: u64,
) -> (u64, u64, u64) {
    let neighbor_curr = ((current_row << 1) | (current_row >> 1)) & interior_bit_mask;
    let shared_depth = prev_row & current_row & next_row;
    let vertical_seed =
        (((shared_depth << 1) & interior_bit_mask) | (shared_depth >> 1)) & interior_bit_mask;
    let vertical = opaque_visible & vertical_seed & (!prev_row) & (!next_row) & interior_bit_mask;

    let edge_from_next =
        (((next_row << 1) & interior_bit_mask) ^ (next_row >> 1)) & interior_bit_mask;
    let edge_from_prev =
        (((prev_row << 1) & interior_bit_mask) ^ (prev_row >> 1)) & interior_bit_mask;
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

    (unit, horizontal, vertical)
}

#[inline(always)]
fn emit_unit_rows<const N_AXIS: usize>(
    rows: &[u64],
    n_coord: u32,
    outer_base: u32,
    bit_base: u32,
    quads: &mut Vec<UnorientedQuad>,
) {
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

        emit_single_row_runs::<T, N_AXIS, BIT_IS_U>(
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

        if overlapping_bits == 0 {
            if has_incoming_carry {
                emit_carried_row_unit_quads::<N_AXIS, BIT_IS_U>(
                    row_bits,
                    n_coord,
                    outer_base + outer_local as u32,
                    bit_base,
                    carry_runs,
                    quads,
                );
            } else {
                emit_single_row_unit_quads::<N_AXIS>(
                    row_bits,
                    n_coord,
                    outer_base + outer_local as u32,
                    bit_base,
                    quads,
                );
            }
            has_incoming_carry = false;
            continue;
        }

        let continue_mask = build_continue_mask(
            voxels,
            overlapping_bits,
            row_base_index,
            bit_stride,
            outer_stride,
            carry_runs,
        );

        let ended_bits = row_bits & !continue_mask;
        if ended_bits != 0 {
            if has_incoming_carry {
                emit_carried_row_unit_quads::<N_AXIS, BIT_IS_U>(
                    ended_bits,
                    n_coord,
                    outer_base + outer_local as u32,
                    bit_base,
                    carry_runs,
                    quads,
                );
            } else {
                emit_single_row_unit_quads::<N_AXIS>(
                    ended_bits,
                    n_coord,
                    outer_base + outer_local as u32,
                    bit_base,
                    quads,
                );
            }
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

        if overlapping_bits == 0 {
            if has_incoming_carry {
                emit_terminal_row_runs::<T, N_AXIS, BIT_IS_U>(
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
            } else {
                emit_single_row_runs::<T, N_AXIS, BIT_IS_U>(
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
            }
            has_incoming_carry = false;
            continue;
        }

        let continue_mask = build_continue_mask(
            voxels,
            overlapping_bits,
            row_base_index,
            bit_stride,
            outer_stride,
            carry_runs,
        );

        let ended_bits = row_bits & !continue_mask;
        if !has_incoming_carry {
            if ended_bits != 0 {
                emit_single_row_runs::<T, N_AXIS, BIT_IS_U>(
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
            }
            has_incoming_carry = continue_mask != 0;
            continue;
        }

        if ended_bits == 0 {
            has_incoming_carry = continue_mask != 0;
            continue;
        }

        emit_mixed_row_runs::<T, N_AXIS, BIT_IS_U>(
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
        has_incoming_carry = continue_mask != 0;
    }
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
fn reset_mask_rows(rows: &mut Vec<u64>, len: usize) {
    rows.resize(len, 0);
    rows.fill(0);
}

#[cfg(test)]
mod tests {
    use super::classify_ao_opaque_row;

    #[test]
    fn vertical_rule_matches_updated_binary_proof() {
        let mask = (1u64 << 7) - 1;
        let opaque_visible = 0b011_0110;
        let prev_row = 0b100_0000;
        let current_row = 0b100_1001;
        let next_row = 0b100_0001;

        let (unit, horizontal, vertical) =
            classify_ao_opaque_row(opaque_visible, prev_row, current_row, next_row, mask);

        assert_eq!(vertical, 0b010_0000);
        assert_eq!(horizontal, 0);
        assert_eq!(unit & vertical, 0);
    }

    #[test]
    fn internal_corner_becomes_unit_not_vertical() {
        let mask = (1u64 << 6) - 1;
        let opaque_visible = 0b011_101;
        let prev_row = 0b111_111;
        let current_row = 0b100_000;
        let next_row = 0b100_000;

        let (unit, horizontal, vertical) =
            classify_ao_opaque_row(opaque_visible, prev_row, current_row, next_row, mask);

        assert_eq!(vertical, 0);
        assert_eq!(unit & 0b010_000, 0b010_000);
        assert_eq!(horizontal & 0b010_000, 0);
    }
}
