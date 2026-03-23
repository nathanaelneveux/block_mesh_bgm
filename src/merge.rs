//! Stage 3 and 4 of the pipeline.
//!
//! Once a face slice has been reduced to visibility rows, there are two useful
//! regimes:
//!
//! - **unit-only slices**: no visible cell touches another visible cell, so the
//!   mesher can emit one quad per bit without any merge bookkeeping
//! - **carry slices**: visible rows may continue across multiple outer rows, so
//!   the mesher tracks vertical carry lengths in `carry_runs`
//!
//! The carry path uses a few small fast paths:
//!
//! - rows with no outgoing overlap
//! - rows with outgoing overlap but no incoming carry
//! - the general case where some runs both continue and end on the current row

use block_mesh::{MergeVoxel, UnorientedQuad};

use crate::bit_mask;
use crate::context::{MeshingContext, SlicePlan};
use crate::face::{write_quad, write_unit_quad, FaceAxes};

/// Chooses the compile-time-specialized merge kernel for one face orientation.
#[inline(always)]
pub(crate) fn mesh_face_rows<T>(
    voxels: &[T],
    context: &MeshingContext,
    slice: SlicePlan,
    visible_rows: &[u64],
    unit_only: bool,
    axes: FaceAxes,
    carry_runs: &mut Vec<u8>,
    quads: &mut Vec<UnorientedQuad>,
) where
    T: MergeVoxel,
{
    match (axes.n_axis, slice.bit_axis == axes.u_axis) {
        (0, false) => mesh_face_rows_impl::<T, 0, false>(
            voxels,
            context,
            slice,
            visible_rows,
            unit_only,
            carry_runs,
            quads,
        ),
        (0, true) => mesh_face_rows_impl::<T, 0, true>(
            voxels,
            context,
            slice,
            visible_rows,
            unit_only,
            carry_runs,
            quads,
        ),
        (1, false) => mesh_face_rows_impl::<T, 1, false>(
            voxels,
            context,
            slice,
            visible_rows,
            unit_only,
            carry_runs,
            quads,
        ),
        (1, true) => mesh_face_rows_impl::<T, 1, true>(
            voxels,
            context,
            slice,
            visible_rows,
            unit_only,
            carry_runs,
            quads,
        ),
        (2, false) => mesh_face_rows_impl::<T, 2, false>(
            voxels,
            context,
            slice,
            visible_rows,
            unit_only,
            carry_runs,
            quads,
        ),
        (2, true) => mesh_face_rows_impl::<T, 2, true>(
            voxels,
            context,
            slice,
            visible_rows,
            unit_only,
            carry_runs,
            quads,
        ),
        _ => unreachable!(),
    }
}

#[inline(always)]
fn mesh_face_rows_impl<T, const N_AXIS: usize, const BIT_IS_U: bool>(
    voxels: &[T],
    context: &MeshingContext,
    slice: SlicePlan,
    visible_rows: &[u64],
    unit_only: bool,
    carry_runs: &mut Vec<u8>,
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

    mesh_face_carry::<T, N_AXIS, BIT_IS_U>(voxels, context, slice, visible_rows, carry_runs, quads);
}

#[inline(always)]
fn mesh_face_carry<T, const N_AXIS: usize, const BIT_IS_U: bool>(
    voxels: &[T],
    context: &MeshingContext,
    slice: SlicePlan,
    rows: &[u64],
    carry_runs: &mut Vec<u8>,
    quads: &mut Vec<UnorientedQuad>,
) where
    T: MergeVoxel,
{
    reset_carry_runs(carry_runs, slice.bit_len);
    let n_base = context.interior_min[N_AXIS];
    let bit_base = context.interior_min[slice.bit_axis];
    let outer_base = context.interior_min[slice.outer_axis];
    let outer_stride = context.strides[slice.outer_axis];
    let bit_stride = context.strides[slice.bit_axis];

    for n_local in 0..slice.n_len {
        carry_runs.fill(0);
        let mut has_incoming_carry = false;
        let n_index_base = context.interior_start_index + n_local * context.strides[N_AXIS];
        let n_coord = n_base + n_local as u32;
        let slice_start = n_local * slice.outer_len;
        let slice_rows = &rows[slice_start..slice_start + slice.outer_len];

        for outer_local in 0..slice.outer_len {
            let row_bits = slice_rows[outer_local];
            if row_bits == 0 {
                continue;
            }

            let next_row_bits = if outer_local + 1 < slice.outer_len {
                slice_rows[outer_local + 1]
            } else {
                0
            };
            let row_base_index = n_index_base + outer_local * outer_stride;
            let overlapping_bits = row_bits & next_row_bits;

            // Every run ends on this row, so the general continuation logic is unnecessary.
            if overlapping_bits == 0 {
                if has_incoming_carry {
                    emit_terminal_row_runs::<T, N_AXIS, BIT_IS_U>(
                        voxels,
                        row_bits,
                        row_base_index,
                        bit_stride,
                        slice.bit_len,
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
                        slice.bit_len,
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

            // These runs are starting fresh on this row, so they can use the
            // cheap single-row emitter instead of the full carry-aware path.
            if !has_incoming_carry {
                if ended_bits != 0 {
                    emit_single_row_runs::<T, N_AXIS, BIT_IS_U>(
                        voxels,
                        ended_bits,
                        row_base_index,
                        bit_stride,
                        slice.bit_len,
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
                slice.bit_len,
                n_coord,
                outer_base + outer_local as u32,
                bit_base,
                carry_runs,
                quads,
            );
            has_incoming_carry = continue_mask != 0;
        }
    }
}

/// Builds the mask of bits whose runs continue into the next row.
#[inline(always)]
fn build_continue_mask<T>(
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

/// Emits ended runs for a row that also contains some continuing carry.
#[inline(always)]
fn emit_mixed_row_runs<T, const N_AXIS: usize, const BIT_IS_U: bool>(
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
        let run_length = carry_runs[bit_local] as usize + 1;
        let mut run_width = 1usize;

        while bit_local + run_width < bit_len {
            let next_bit_local = bit_local + run_width;
            let next_bit = 1u64 << next_bit_local;
            if row_bits & next_bit == 0 || carry_runs[next_bit_local] != carry_runs[bit_local] {
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
        let start_outer = outer_coord - carry_runs[bit_local] as u32;
        write_quad::<N_AXIS, BIT_IS_U>(
            &mut spare[written],
            n_coord,
            start_outer,
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

/// Emits horizontal runs for a row with no incoming carry.
#[inline(always)]
fn emit_single_row_runs<T, const N_AXIS: usize, const BIT_IS_U: bool>(
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

/// Emits runs for a row where every visible bit ends on the current row.
#[inline(always)]
fn emit_terminal_row_runs<T, const N_AXIS: usize, const BIT_IS_U: bool>(
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

/// Emits every visible cell in a fragmented slice as a `1x1` quad.
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

#[inline(always)]
fn reset_carry_runs(carry_runs: &mut Vec<u8>, len: usize) {
    carry_runs.resize(len, 0);
    carry_runs.fill(0);
}

/// Checks whether the same face can continue one row farther along the outer axis.
#[inline]
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
