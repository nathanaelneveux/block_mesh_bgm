//! AO-safe alternate merge policy.
//!
//! The default [`crate::binary_greedy_quads`] path stays on the existing
//! carry-based maximum-merge implementation in [`crate::merge`]. This module
//! only handles the AO-safe variant, which keeps the same carry structure but
//! refuses to merge opaque faces whose four corner AO values differ.

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

    /// Restrict merges to cells with matching ambient-occlusion signatures.
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
    ao_keys: Vec<u8>,
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
    let outer_stride = context.strides[slice.outer_axis];
    let bit_stride = context.strides[slice.bit_axis];

    for n_local in 0..slice.n_len {
        let row_start = n_local * slice.outer_len;
        let slice_rows = &visible_rows[row_start..row_start + slice.outer_len];
        let n_coord = n_base + n_local as u32;
        let n_index_base = context.interior_start_index + n_local * context.strides[N_AXIS];

        #[cfg(feature = "internal-profiler")]
        let key_build_start = Instant::now();
        build_slice_ao_keys(
            context.query_shape,
            N_AXIS,
            slice,
            n_local,
            axes.n_sign,
            opaque_cols,
            slice_rows,
            &mut scratch.ao_keys,
        );
        #[cfg(feature = "internal-profiler")]
        crate::profile::record_key_build(key_build_start.elapsed());

        #[cfg(feature = "internal-profiler")]
        let carry_start = Instant::now();
        mesh_face_carry_ao::<T, N_AXIS, BIT_IS_U>(
            voxels,
            n_index_base,
            outer_stride,
            bit_stride,
            slice,
            slice_rows,
            &scratch.ao_keys,
            &mut scratch.carry_runs,
            n_coord,
            context.interior_min[slice.outer_axis],
            context.interior_min[slice.bit_axis],
            quads,
        );
        #[cfg(feature = "internal-profiler")]
        crate::profile::record_carry_total(carry_start.elapsed());
    }
}

#[inline(always)]
fn mesh_face_carry_ao<T, const N_AXIS: usize, const BIT_IS_U: bool>(
    voxels: &[T],
    n_index_base: usize,
    outer_stride: usize,
    bit_stride: usize,
    slice: SlicePlan,
    slice_rows: &[u64],
    ao_keys: &[u8],
    carry_runs: &mut Vec<u8>,
    n_coord: u32,
    outer_base: u32,
    bit_base: u32,
    quads: &mut Vec<UnorientedQuad>,
) where
    T: MergeVoxel,
{
    reset_carry_runs(carry_runs, slice.bit_len);
    let mut has_incoming_carry = false;

    for outer_local in 0..slice.outer_len {
        let row_bits = slice_rows[outer_local];
        if row_bits == 0 {
            has_incoming_carry = false;
        } else {
            let next_row_bits = if outer_local + 1 < slice.outer_len {
                slice_rows[outer_local + 1]
            } else {
                0
            };
            let row_base_index = n_index_base + outer_local * outer_stride;
            let row_keys = ao_key_row(ao_keys, outer_local, slice.bit_len);
            let overlapping_bits = row_bits & next_row_bits;
            #[cfg(feature = "internal-profiler")]
            crate::profile::record_carry_row(row_bits, overlapping_bits);

            if overlapping_bits == 0 {
                if has_incoming_carry {
                    #[cfg(feature = "internal-profiler")]
                    let emit_start = Instant::now();
                    emit_terminal_row_runs_ao::<T, N_AXIS, BIT_IS_U>(
                        voxels,
                        row_bits,
                        row_base_index,
                        bit_stride,
                        slice.bit_len,
                        n_coord,
                        outer_base + outer_local as u32,
                        bit_base,
                        row_keys,
                        carry_runs,
                        quads,
                    );
                    #[cfg(feature = "internal-profiler")]
                    {
                        crate::profile::record_terminal_row();
                        crate::profile::record_emit_terminal(emit_start.elapsed());
                    }
                } else {
                    #[cfg(feature = "internal-profiler")]
                    let emit_start = Instant::now();
                    emit_single_row_runs_ao::<T, N_AXIS, BIT_IS_U>(
                        voxels,
                        row_bits,
                        row_base_index,
                        bit_stride,
                        slice.bit_len,
                        n_coord,
                        outer_base + outer_local as u32,
                        bit_base,
                        row_keys,
                        quads,
                    );
                    #[cfg(feature = "internal-profiler")]
                    {
                        crate::profile::record_single_row();
                        crate::profile::record_emit_single(emit_start.elapsed());
                    }
                }
                has_incoming_carry = false;
            } else {
                let next_row_keys = ao_key_row(ao_keys, outer_local + 1, slice.bit_len);
                let ao_overlap_bits =
                    match_ao_overlap_bits(overlapping_bits, row_keys, next_row_keys);
                #[cfg(feature = "internal-profiler")]
                crate::profile::record_ao_overlap_candidates(ao_overlap_bits);

                if ao_overlap_bits == 0 {
                    if has_incoming_carry {
                        #[cfg(feature = "internal-profiler")]
                        let emit_start = Instant::now();
                        emit_terminal_row_runs_ao::<T, N_AXIS, BIT_IS_U>(
                            voxels,
                            row_bits,
                            row_base_index,
                            bit_stride,
                            slice.bit_len,
                            n_coord,
                            outer_base + outer_local as u32,
                            bit_base,
                            row_keys,
                            carry_runs,
                            quads,
                        );
                        #[cfg(feature = "internal-profiler")]
                        {
                            crate::profile::record_terminal_row();
                            crate::profile::record_emit_terminal(emit_start.elapsed());
                        }
                    } else {
                        #[cfg(feature = "internal-profiler")]
                        let emit_start = Instant::now();
                        emit_single_row_runs_ao::<T, N_AXIS, BIT_IS_U>(
                            voxels,
                            row_bits,
                            row_base_index,
                            bit_stride,
                            slice.bit_len,
                            n_coord,
                            outer_base + outer_local as u32,
                            bit_base,
                            row_keys,
                            quads,
                        );
                        #[cfg(feature = "internal-profiler")]
                        {
                            crate::profile::record_single_row();
                            crate::profile::record_emit_single(emit_start.elapsed());
                        }
                    }
                    has_incoming_carry = false;
                    continue;
                }

                #[cfg(feature = "internal-profiler")]
                let continue_start = Instant::now();
                let continue_mask = build_continue_mask(
                    voxels,
                    ao_overlap_bits,
                    row_base_index,
                    bit_stride,
                    outer_stride,
                    row_keys,
                    next_row_keys,
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
                        emit_single_row_runs_ao::<T, N_AXIS, BIT_IS_U>(
                            voxels,
                            ended_bits,
                            row_base_index,
                            bit_stride,
                            slice.bit_len,
                            n_coord,
                            outer_base + outer_local as u32,
                            bit_base,
                            row_keys,
                            quads,
                        );
                        #[cfg(feature = "internal-profiler")]
                        {
                            crate::profile::record_single_row();
                            crate::profile::record_emit_single(emit_start.elapsed());
                        }
                    }
                    has_incoming_carry = continue_mask != 0;
                } else if ended_bits == 0 {
                    has_incoming_carry = continue_mask != 0;
                } else {
                    #[cfg(feature = "internal-profiler")]
                    let emit_start = Instant::now();
                    emit_mixed_row_runs_ao::<T, N_AXIS, BIT_IS_U>(
                        voxels,
                        ended_bits,
                        row_base_index,
                        bit_stride,
                        slice.bit_len,
                        n_coord,
                        outer_base + outer_local as u32,
                        bit_base,
                        row_keys,
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
        }
    }
}

#[inline(always)]
fn build_continue_mask<T>(
    voxels: &[T],
    mut overlapping_bits: u64,
    row_base_index: usize,
    bit_stride: usize,
    outer_stride: usize,
    row_ao_keys: &[u8],
    next_row_ao_keys: &[u8],
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
        let quad_ao_key = row_ao_keys[bit_local];
        let next_ao_key = next_row_ao_keys[bit_local];

        if cell_matches(
            voxels,
            voxel_index + outer_stride,
            &quad_value,
            quad_ao_key,
            next_ao_key,
        ) {
            continue_mask |= bit;
            carry_runs[bit_local] += 1;
        }

        overlapping_bits &= overlapping_bits - 1;
    }

    continue_mask
}

#[inline(always)]
fn match_ao_overlap_bits(
    overlapping_bits: u64,
    row_ao_keys: &[u8],
    next_row_ao_keys: &[u8],
) -> u64 {
    let mut matching_bits = 0u64;
    let mut bits = overlapping_bits;

    while bits != 0 {
        let bit_local = bits.trailing_zeros() as usize;
        let bit = 1u64 << bit_local;
        if row_ao_keys[bit_local] == next_row_ao_keys[bit_local] {
            matching_bits |= bit;
        }
        bits &= bits - 1;
    }

    matching_bits
}

#[inline(always)]
fn emit_mixed_row_runs_ao<T, const N_AXIS: usize, const BIT_IS_U: bool>(
    voxels: &[T],
    mut row_bits: u64,
    row_base_index: usize,
    bit_stride: usize,
    bit_len: usize,
    n_coord: u32,
    outer_coord: u32,
    bit_base: u32,
    row_ao_keys: &[u8],
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
        let quad_ao_key = row_ao_keys[bit_local];
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
            let next_ao_key = row_ao_keys[next_bit_local];
            if !cell_matches(voxels, next_index, &quad_value, quad_ao_key, next_ao_key) {
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

    #[cfg(feature = "internal-profiler")]
    crate::profile::record_mixed_quads(written);
}

#[inline(always)]
fn emit_single_row_runs_ao<T, const N_AXIS: usize, const BIT_IS_U: bool>(
    voxels: &[T],
    row_bits: u64,
    row_base_index: usize,
    bit_stride: usize,
    bit_len: usize,
    n_coord: u32,
    outer_coord: u32,
    bit_base: u32,
    row_ao_keys: &[u8],
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
        let quad_ao_key = row_ao_keys[bit_local];
        let mut run_width = 1usize;

        while bit_local + run_width < bit_len {
            let next_bit_local = bit_local + run_width;
            let next_bit = 1u64 << next_bit_local;
            if bits_here & next_bit == 0 {
                break;
            }

            let next_index = row_base_index + next_bit_local * bit_stride;
            let next_ao_key = row_ao_keys[next_bit_local];
            if !cell_matches(voxels, next_index, &quad_value, quad_ao_key, next_ao_key) {
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

    #[cfg(feature = "internal-profiler")]
    crate::profile::record_single_quads(written);
}

#[inline(always)]
fn emit_terminal_row_runs_ao<T, const N_AXIS: usize, const BIT_IS_U: bool>(
    voxels: &[T],
    row_bits: u64,
    row_base_index: usize,
    bit_stride: usize,
    bit_len: usize,
    n_coord: u32,
    outer_coord: u32,
    bit_base: u32,
    row_ao_keys: &[u8],
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
        let quad_ao_key = row_ao_keys[bit_local];
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
            let next_ao_key = row_ao_keys[next_bit_local];
            if !cell_matches(voxels, next_index, &quad_value, quad_ao_key, next_ao_key) {
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

    #[cfg(feature = "internal-profiler")]
    crate::profile::record_terminal_quads(written);
}

#[inline(always)]
fn cell_matches<T>(
    voxels: &[T],
    voxel_index: usize,
    quad_value: &T::MergeValue,
    quad_ao_key: u8,
    cell_ao_key: u8,
) -> bool
where
    T: MergeVoxel,
{
    unsafe { voxels.get_unchecked(voxel_index) }
        .merge_value()
        .eq(quad_value)
        && cell_ao_key == quad_ao_key
}

#[inline(always)]
fn reset_carry_runs(carry_runs: &mut Vec<u8>, len: usize) {
    carry_runs.resize(len, 0);
    carry_runs.fill(0);
}

#[inline(always)]
fn ao_key_row(ao_keys: &[u8], outer_local: usize, bit_len: usize) -> &[u8] {
    &ao_keys[outer_local * bit_len..outer_local * bit_len + bit_len]
}

#[inline(always)]
fn build_slice_ao_keys(
    query_shape: [usize; 3],
    n_axis: usize,
    slice: SlicePlan,
    n_local: usize,
    face_sign: i32,
    opaque_cols: &[u64],
    slice_rows: &[u64],
    ao_keys: &mut Vec<u8>,
) {
    let total_cells = slice.outer_len * slice.bit_len;
    if ao_keys.len() < total_cells {
        ao_keys.resize(total_cells, 0);
    }

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
        let row_keys =
            &mut ao_keys[outer_local * slice.bit_len..outer_local * slice.bit_len + slice.bit_len];
        let source_opaque =
            (opaque_cols[source_base + outer_local * column_outer_stride] >> 1) & interior_bit_mask;
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
        let left = current_row << 1;
        let right = current_row >> 1;
        let up_left = prev_row << 1;
        let up_right = prev_row >> 1;
        let down_left = next_row << 1;
        let down_right = next_row >> 1;
        let mut bits = row_bits;
        #[cfg(feature = "internal-profiler")]
        let mut opaque_bits = 0u32;
        #[cfg(feature = "internal-profiler")]
        let mut passthrough_bits = 0u32;
        #[cfg(feature = "internal-profiler")]
        let mut first_opaque_key = 0u8;
        #[cfg(feature = "internal-profiler")]
        let mut saw_opaque_key = false;
        #[cfg(feature = "internal-profiler")]
        let mut uniform_opaque = true;

        while bits != 0 {
            let bit_local = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let bit = 1u64 << bit_local;
            row_keys[bit_local] = if source_opaque & bit != 0 {
                let key = pack_ao_signature_from_rows(
                    bit, left, right, prev_row, next_row, up_left, up_right, down_left, down_right,
                );
                #[cfg(feature = "internal-profiler")]
                {
                    opaque_bits += 1;
                    if saw_opaque_key {
                        uniform_opaque &= key == first_opaque_key;
                    } else {
                        first_opaque_key = key;
                        saw_opaque_key = true;
                    }
                }
                key
            } else {
                #[cfg(feature = "internal-profiler")]
                {
                    passthrough_bits += 1;
                }
                0
            };
        }

        #[cfg(feature = "internal-profiler")]
        crate::profile::record_key_row(
            opaque_bits,
            passthrough_bits,
            saw_opaque_key && uniform_opaque,
        );
    }
}

#[inline(always)]
fn pack_ao_signature_from_rows(
    bit: u64,
    left: u64,
    right: u64,
    up: u64,
    down: u64,
    up_left: u64,
    up_right: u64,
    down_left: u64,
    down_right: u64,
) -> u8 {
    let mut signature = 0u8;
    signature |= vertex_ao_masked(left, up, up_left, bit);
    signature |= vertex_ao_masked(right, up, up_right, bit) << 2;
    signature |= vertex_ao_masked(right, down, down_right, bit) << 4;
    signature |= vertex_ao_masked(left, down, down_left, bit) << 6;
    signature
}

#[inline(always)]
fn vertex_ao_masked(side1_mask: u64, side2_mask: u64, corner_mask: u64, bit: u64) -> u8 {
    let side1 = side1_mask & bit != 0;
    let side2 = side2_mask & bit != 0;
    let corner = corner_mask & bit != 0;
    vertex_ao(side1, side2, corner)
}

#[inline(always)]
fn vertex_ao(side1: bool, side2: bool, corner: bool) -> u8 {
    if side1 && side2 {
        0
    } else {
        3 - side1 as u8 - side2 as u8 - corner as u8
    }
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
