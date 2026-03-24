//! Alternate merge policies.
//!
//! The default `binary_greedy_quads` path stays on the existing carry-based
//! maximum-merge implementation in [`crate::merge`]. This module handles the
//! slower but more configurable policies:
//!
//! - AO-safe merging for opaque faces
//! - a conforming row-partition merge for coplanar T-junction elimination
//!
//! These modes are intentionally isolated from the vanilla path so the default
//! implementation does not pay for extra branching or scratch management.

use block_mesh::{MergeVoxel, UnorientedQuad, Voxel, VoxelVisibility};

use crate::bit_mask;
use crate::context::{MeshingContext, SlicePlan};
use crate::face::{write_quad, write_unit_quad, FaceAxes};

const NON_OPAQUE_AO_KEY: u16 = 0;

/// Reusable scratch that only alternate merge policies need.
#[derive(Default)]
pub(crate) struct FeatureScratch {
    ao_keys: Vec<u16>,
    carry_runs: Vec<u8>,
    row_runs: Vec<RowRun>,
    active_segments: Vec<ActiveSegment>,
    next_segments: Vec<ActiveSegment>,
    slice_quads: Vec<LocalQuad>,
    split_quads: Vec<LocalQuad>,
    vertical_cuts: Vec<u64>,
    horizontal_cuts: Vec<u64>,
    bit_splits: Vec<u8>,
    outer_splits: Vec<u8>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct LocalQuad {
    bit: u8,
    outer: u8,
    width: u8,
    height: u8,
}

#[derive(Clone, Copy, Debug)]
struct RowRun {
    start: u8,
    end: u8,
    voxel_index: usize,
    ao_key: u16,
}

#[derive(Clone, Copy, Debug)]
struct ActiveSegment {
    start: u8,
    end: u8,
    start_outer: u8,
    voxel_index: usize,
    ao_key: u16,
}

/// Chooses the compile-time-specialized alternate kernel for one face
/// orientation.
#[inline]
pub(crate) fn mesh_face_rows_with_features<T, const AO_SAFE: bool, const NO_T_JUNCTIONS: bool>(
    voxels: &[T],
    context: &MeshingContext,
    slice: SlicePlan,
    visible_rows: &[u64],
    unit_only: bool,
    axes: FaceAxes,
    scratch: &mut FeatureScratch,
    quads: &mut Vec<UnorientedQuad>,
) where
    T: MergeVoxel,
{
    match (axes.n_axis, slice.bit_axis == axes.u_axis) {
        (0, false) => mesh_face_rows_with_features_impl::<T, 0, false, AO_SAFE, NO_T_JUNCTIONS>(
            voxels,
            context,
            slice,
            visible_rows,
            unit_only,
            axes.n_sign,
            scratch,
            quads,
        ),
        (0, true) => mesh_face_rows_with_features_impl::<T, 0, true, AO_SAFE, NO_T_JUNCTIONS>(
            voxels,
            context,
            slice,
            visible_rows,
            unit_only,
            axes.n_sign,
            scratch,
            quads,
        ),
        (1, false) => mesh_face_rows_with_features_impl::<T, 1, false, AO_SAFE, NO_T_JUNCTIONS>(
            voxels,
            context,
            slice,
            visible_rows,
            unit_only,
            axes.n_sign,
            scratch,
            quads,
        ),
        (1, true) => mesh_face_rows_with_features_impl::<T, 1, true, AO_SAFE, NO_T_JUNCTIONS>(
            voxels,
            context,
            slice,
            visible_rows,
            unit_only,
            axes.n_sign,
            scratch,
            quads,
        ),
        (2, false) => mesh_face_rows_with_features_impl::<T, 2, false, AO_SAFE, NO_T_JUNCTIONS>(
            voxels,
            context,
            slice,
            visible_rows,
            unit_only,
            axes.n_sign,
            scratch,
            quads,
        ),
        (2, true) => mesh_face_rows_with_features_impl::<T, 2, true, AO_SAFE, NO_T_JUNCTIONS>(
            voxels,
            context,
            slice,
            visible_rows,
            unit_only,
            axes.n_sign,
            scratch,
            quads,
        ),
        _ => unreachable!(),
    }
}

#[inline]
fn mesh_face_rows_with_features_impl<
    T,
    const N_AXIS: usize,
    const BIT_IS_U: bool,
    const AO_SAFE: bool,
    const NO_T_JUNCTIONS: bool,
>(
    voxels: &[T],
    context: &MeshingContext,
    slice: SlicePlan,
    visible_rows: &[u64],
    unit_only: bool,
    face_sign: i32,
    scratch: &mut FeatureScratch,
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
    let bit_base = context.interior_min[slice.bit_axis];
    let outer_base = context.interior_min[slice.outer_axis];
    let outer_stride = context.strides[slice.outer_axis];
    let bit_stride = context.strides[slice.bit_axis];

    for n_local in 0..slice.n_len {
        let row_start = n_local * slice.outer_len;
        let slice_rows = &visible_rows[row_start..row_start + slice.outer_len];
        let n_coord = n_base + n_local as u32;
        let n_index_base = context.interior_start_index + n_local * context.strides[N_AXIS];

        if AO_SAFE {
            build_slice_ao_keys(
                voxels,
                n_index_base,
                context.strides[N_AXIS] as isize * face_sign as isize,
                outer_stride,
                bit_stride,
                slice,
                slice_rows,
                &mut scratch.ao_keys,
            );
        }

        scratch.slice_quads.clear();
        if NO_T_JUNCTIONS {
            build_slice_quads_conforming::<T, AO_SAFE>(
                voxels,
                n_index_base,
                outer_stride,
                bit_stride,
                slice,
                slice_rows,
                &scratch.ao_keys,
                &mut scratch.row_runs,
                &mut scratch.active_segments,
                &mut scratch.next_segments,
                &mut scratch.slice_quads,
            );
            subdivide_slice_quads(
                slice.bit_len,
                slice.outer_len,
                &mut scratch.slice_quads,
                &mut scratch.split_quads,
                &mut scratch.vertical_cuts,
                &mut scratch.horizontal_cuts,
                &mut scratch.bit_splits,
                &mut scratch.outer_splits,
            );
        } else {
            build_slice_quads_carry::<T, AO_SAFE>(
                voxels,
                n_index_base,
                outer_stride,
                bit_stride,
                slice,
                slice_rows,
                &scratch.ao_keys,
                &mut scratch.carry_runs,
                &mut scratch.slice_quads,
            );
        }

        emit_local_quads::<N_AXIS, BIT_IS_U>(
            &scratch.slice_quads,
            n_coord,
            outer_base,
            bit_base,
            quads,
        );
    }
}

#[inline]
fn build_slice_quads_carry<T, const AO_SAFE: bool>(
    voxels: &[T],
    n_index_base: usize,
    outer_stride: usize,
    bit_stride: usize,
    slice: SlicePlan,
    slice_rows: &[u64],
    ao_keys: &[u16],
    carry_runs: &mut Vec<u8>,
    slice_quads: &mut Vec<LocalQuad>,
) where
    T: MergeVoxel,
{
    reset_carry_runs(carry_runs, slice.bit_len);
    let mut has_incoming_carry = false;

    for outer_local in 0..slice.outer_len {
        let row_bits = slice_rows[outer_local];
        if row_bits == 0 {
            has_incoming_carry = false;
            continue;
        }

        let next_row_bits = if outer_local + 1 < slice.outer_len {
            slice_rows[outer_local + 1]
        } else {
            0
        };
        let row_base_index = n_index_base + outer_local * outer_stride;
        let overlapping_bits = row_bits & next_row_bits;

        if overlapping_bits == 0 {
            if has_incoming_carry {
                emit_local_terminal_row_runs::<T, AO_SAFE>(
                    voxels,
                    row_bits,
                    row_base_index,
                    bit_stride,
                    slice.bit_len,
                    outer_local,
                    ao_keys,
                    carry_runs,
                    slice_quads,
                );
            } else {
                emit_local_single_row_runs::<T, AO_SAFE>(
                    voxels,
                    row_bits,
                    row_base_index,
                    bit_stride,
                    slice.bit_len,
                    outer_local,
                    ao_keys,
                    slice_quads,
                );
            }
            has_incoming_carry = false;
            continue;
        }

        let continue_mask = build_continue_mask::<T, AO_SAFE>(
            voxels,
            overlapping_bits,
            row_base_index,
            bit_stride,
            outer_stride,
            slice.bit_len,
            outer_local,
            ao_keys,
            carry_runs,
        );
        let ended_bits = row_bits & !continue_mask;

        if !has_incoming_carry {
            if ended_bits != 0 {
                emit_local_single_row_runs::<T, AO_SAFE>(
                    voxels,
                    ended_bits,
                    row_base_index,
                    bit_stride,
                    slice.bit_len,
                    outer_local,
                    ao_keys,
                    slice_quads,
                );
            }
            has_incoming_carry = continue_mask != 0;
            continue;
        }

        if ended_bits == 0 {
            has_incoming_carry = continue_mask != 0;
            continue;
        }

        emit_local_mixed_row_runs::<T, AO_SAFE>(
            voxels,
            ended_bits,
            row_base_index,
            bit_stride,
            slice.bit_len,
            outer_local,
            ao_keys,
            carry_runs,
            slice_quads,
        );
        has_incoming_carry = continue_mask != 0;
    }
}

#[inline]
fn build_slice_quads_conforming<T, const AO_SAFE: bool>(
    voxels: &[T],
    n_index_base: usize,
    outer_stride: usize,
    bit_stride: usize,
    slice: SlicePlan,
    slice_rows: &[u64],
    ao_keys: &[u16],
    row_runs: &mut Vec<RowRun>,
    active_segments: &mut Vec<ActiveSegment>,
    next_segments: &mut Vec<ActiveSegment>,
    slice_quads: &mut Vec<LocalQuad>,
) where
    T: MergeVoxel,
{
    active_segments.clear();
    next_segments.clear();

    for outer_local in 0..slice.outer_len {
        let row_bits = slice_rows[outer_local];
        let row_base_index = n_index_base + outer_local * outer_stride;
        build_row_runs::<T, AO_SAFE>(
            voxels,
            row_bits,
            row_base_index,
            bit_stride,
            slice.bit_len,
            outer_local,
            ao_keys,
            row_runs,
        );

        if row_runs.is_empty() {
            flush_active_segments(active_segments, outer_local, slice_quads);
            continue;
        }

        let mut boundary_present = [false; 64];
        for run in row_runs.iter().copied() {
            boundary_present[run.start as usize] = true;
            boundary_present[run.end as usize] = true;
        }
        for segment in active_segments.iter().copied() {
            boundary_present[segment.start as usize] = true;
            boundary_present[segment.end as usize] = true;
        }

        next_segments.clear();
        let mut raw_index = 0usize;
        let mut active_index = 0usize;
        let mut interval_start = None;

        for boundary in 0..=slice.bit_len {
            if !boundary_present[boundary] {
                continue;
            }

            let Some(start) = interval_start else {
                interval_start = Some(boundary);
                continue;
            };
            let end = boundary;
            interval_start = Some(boundary);

            while raw_index < row_runs.len() && row_runs[raw_index].end as usize <= start {
                raw_index += 1;
            }
            while active_index < active_segments.len()
                && active_segments[active_index].end as usize <= start
            {
                active_index += 1;
            }

            let current_run = row_runs
                .get(raw_index)
                .copied()
                .filter(|run| run.start as usize <= start && start < run.end as usize);
            let active_segment = active_segments
                .get(active_index)
                .copied()
                .filter(|segment| segment.start as usize <= start && start < segment.end as usize);

            match (current_run, active_segment) {
                (Some(run), Some(segment)) => {
                    if segment_matches::<T, AO_SAFE>(
                        voxels,
                        active_segments,
                        active_index,
                        start,
                        end,
                        segment,
                        row_runs,
                        raw_index,
                        run,
                    ) {
                        next_segments.push(ActiveSegment {
                            start: start as u8,
                            end: end as u8,
                            start_outer: segment.start_outer,
                            voxel_index: segment.voxel_index,
                            ao_key: segment.ao_key,
                        });
                    } else {
                        emit_active_interval(segment, start, end, outer_local, slice_quads);
                        next_segments.push(ActiveSegment {
                            start: start as u8,
                            end: end as u8,
                            start_outer: outer_local as u8,
                            voxel_index: run.voxel_index,
                            ao_key: run.ao_key,
                        });
                    }
                }
                (Some(run), None) => {
                    next_segments.push(ActiveSegment {
                        start: start as u8,
                        end: end as u8,
                        start_outer: outer_local as u8,
                        voxel_index: run.voxel_index,
                        ao_key: run.ao_key,
                    });
                }
                (None, Some(segment)) => {
                    emit_active_interval(segment, start, end, outer_local, slice_quads);
                }
                (None, None) => {}
            }
        }

        std::mem::swap(active_segments, next_segments);
    }

    flush_active_segments(active_segments, slice.outer_len, slice_quads);
}

#[inline]
fn build_row_runs<T, const AO_SAFE: bool>(
    voxels: &[T],
    row_bits: u64,
    row_base_index: usize,
    bit_stride: usize,
    bit_len: usize,
    outer_local: usize,
    ao_keys: &[u16],
    row_runs: &mut Vec<RowRun>,
) where
    T: MergeVoxel,
{
    row_runs.clear();
    let row_ao_keys = ao_key_row::<AO_SAFE>(ao_keys, outer_local, bit_len);
    let mut bits_here = row_bits;

    while bits_here != 0 {
        let bit_local = bits_here.trailing_zeros() as usize;
        let voxel_index = row_base_index + bit_local * bit_stride;
        let run_ao_key = if AO_SAFE {
            row_ao_keys[bit_local]
        } else {
            NON_OPAQUE_AO_KEY
        };
        let mut run_width = 1usize;

        while bit_local + run_width < bit_len {
            let next_bit_local = bit_local + run_width;
            let next_bit = 1u64 << next_bit_local;
            if bits_here & next_bit == 0 {
                break;
            }

            let next_index = row_base_index + next_bit_local * bit_stride;
            let next_ao_key = if AO_SAFE {
                row_ao_keys[next_bit_local]
            } else {
                NON_OPAQUE_AO_KEY
            };

            if !indices_match::<T, AO_SAFE>(
                voxels,
                voxel_index,
                run_ao_key,
                next_index,
                next_ao_key,
            ) {
                break;
            }

            run_width += 1;
        }

        bits_here &= !bit_mask(bit_local, run_width);
        row_runs.push(RowRun {
            start: bit_local as u8,
            end: (bit_local + run_width) as u8,
            voxel_index,
            ao_key: run_ao_key,
        });
    }
}

#[inline]
fn flush_active_segments(
    active_segments: &mut Vec<ActiveSegment>,
    end_outer: usize,
    slice_quads: &mut Vec<LocalQuad>,
) {
    for segment in active_segments.iter().copied() {
        emit_active_interval(
            segment,
            segment.start as usize,
            segment.end as usize,
            end_outer,
            slice_quads,
        );
    }
    active_segments.clear();
}

#[inline]
fn emit_active_interval(
    segment: ActiveSegment,
    start: usize,
    end: usize,
    end_outer: usize,
    slice_quads: &mut Vec<LocalQuad>,
) {
    let height = end_outer - segment.start_outer as usize;
    if height == 0 {
        return;
    }

    slice_quads.push(LocalQuad {
        bit: start as u8,
        outer: segment.start_outer,
        width: (end - start) as u8,
        height: height as u8,
    });
}

#[inline]
fn segment_matches<T, const AO_SAFE: bool>(
    voxels: &[T],
    active_segments: &[ActiveSegment],
    active_index: usize,
    start: usize,
    end: usize,
    segment: ActiveSegment,
    row_runs: &[RowRun],
    raw_index: usize,
    run: RowRun,
) -> bool
where
    T: MergeVoxel,
{
    let active_left = active_neighbor_signature(active_segments, active_index, start, false);
    let active_right = active_neighbor_signature(active_segments, active_index, end, true);
    let current_left = row_neighbor_signature(row_runs, raw_index, start, false);
    let current_right = row_neighbor_signature(row_runs, raw_index, end, true);
    let left_neighbor = active_neighbor_segment(active_segments, active_index, start, false);
    let right_neighbor = active_neighbor_segment(active_segments, active_index, end, true);

    indices_match::<T, AO_SAFE>(
        voxels,
        segment.voxel_index,
        segment.ao_key,
        run.voxel_index,
        run.ao_key,
    ) && neighbor_matches::<T, AO_SAFE>(voxels, active_left, current_left)
        && neighbor_matches::<T, AO_SAFE>(voxels, active_right, current_right)
        && left_neighbor
            .is_none_or(|neighbor| neighbor.start_outer == segment.start_outer)
        && right_neighbor
            .is_none_or(|neighbor| neighbor.start_outer == segment.start_outer)
}

#[derive(Clone, Copy)]
struct NeighborSignature {
    voxel_index: usize,
    ao_key: u16,
}

#[inline]
fn row_neighbor_signature(
    row_runs: &[RowRun],
    run_index: usize,
    boundary: usize,
    right_side: bool,
) -> Option<NeighborSignature> {
    let run = row_runs[run_index];
    if right_side {
        if boundary < run.end as usize {
            Some(NeighborSignature {
                voxel_index: run.voxel_index,
                ao_key: run.ao_key,
            })
        } else {
            row_runs
                .get(run_index + 1)
                .filter(|next| next.start as usize == boundary)
                .map(|next| NeighborSignature {
                    voxel_index: next.voxel_index,
                    ao_key: next.ao_key,
                })
        }
    } else if boundary > run.start as usize {
        Some(NeighborSignature {
            voxel_index: run.voxel_index,
            ao_key: run.ao_key,
        })
    } else if run_index > 0 && row_runs[run_index - 1].end as usize == boundary {
        let prev = row_runs[run_index - 1];
        Some(NeighborSignature {
            voxel_index: prev.voxel_index,
            ao_key: prev.ao_key,
        })
    } else {
        None
    }
}

#[inline]
fn active_neighbor_signature(
    active_segments: &[ActiveSegment],
    segment_index: usize,
    boundary: usize,
    right_side: bool,
) -> Option<NeighborSignature> {
    let segment = active_segments[segment_index];
    if right_side {
        if boundary < segment.end as usize {
            Some(NeighborSignature {
                voxel_index: segment.voxel_index,
                ao_key: segment.ao_key,
            })
        } else {
            active_segments
                .get(segment_index + 1)
                .filter(|next| next.start as usize == boundary)
                .map(|next| NeighborSignature {
                    voxel_index: next.voxel_index,
                    ao_key: next.ao_key,
                })
        }
    } else if boundary > segment.start as usize {
        Some(NeighborSignature {
            voxel_index: segment.voxel_index,
            ao_key: segment.ao_key,
        })
    } else if segment_index > 0 && active_segments[segment_index - 1].end as usize == boundary {
        let prev = active_segments[segment_index - 1];
        Some(NeighborSignature {
            voxel_index: prev.voxel_index,
            ao_key: prev.ao_key,
        })
    } else {
        None
    }
}

#[inline]
fn active_neighbor_segment(
    active_segments: &[ActiveSegment],
    segment_index: usize,
    boundary: usize,
    right_side: bool,
) -> Option<ActiveSegment> {
    let segment = active_segments[segment_index];
    if right_side {
        if boundary < segment.end as usize {
            Some(segment)
        } else {
            active_segments
                .get(segment_index + 1)
                .copied()
                .filter(|next| next.start as usize == boundary)
        }
    } else if boundary > segment.start as usize {
        Some(segment)
    } else if segment_index > 0 && active_segments[segment_index - 1].end as usize == boundary {
        Some(active_segments[segment_index - 1])
    } else {
        None
    }
}

#[inline]
fn neighbor_matches<T, const AO_SAFE: bool>(
    voxels: &[T],
    left: Option<NeighborSignature>,
    right: Option<NeighborSignature>,
) -> bool
where
    T: MergeVoxel,
{
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => indices_match::<T, AO_SAFE>(
            voxels,
            left.voxel_index,
            left.ao_key,
            right.voxel_index,
            right.ao_key,
        ),
        _ => false,
    }
}

#[inline]
fn build_continue_mask<T, const AO_SAFE: bool>(
    voxels: &[T],
    mut overlapping_bits: u64,
    row_base_index: usize,
    bit_stride: usize,
    outer_stride: usize,
    bit_len: usize,
    outer_local: usize,
    ao_keys: &[u16],
    carry_runs: &mut [u8],
) -> u64
where
    T: MergeVoxel,
{
    let mut continue_mask = 0u64;
    let row_ao_keys = ao_key_row::<AO_SAFE>(ao_keys, outer_local, bit_len);
    let next_row_ao_keys = ao_key_row::<AO_SAFE>(ao_keys, outer_local + 1, bit_len);

    while overlapping_bits != 0 {
        let bit_local = overlapping_bits.trailing_zeros() as usize;
        let bit = 1u64 << bit_local;
        let voxel_index = row_base_index + bit_local * bit_stride;
        let quad_value = unsafe { voxels.get_unchecked(voxel_index) }.merge_value();
        let quad_ao_key = if AO_SAFE {
            row_ao_keys[bit_local]
        } else {
            NON_OPAQUE_AO_KEY
        };
        let next_ao_key = if AO_SAFE {
            next_row_ao_keys[bit_local]
        } else {
            NON_OPAQUE_AO_KEY
        };

        if cell_matches::<T, AO_SAFE>(
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

#[inline]
fn emit_local_mixed_row_runs<T, const AO_SAFE: bool>(
    voxels: &[T],
    mut row_bits: u64,
    row_base_index: usize,
    bit_stride: usize,
    bit_len: usize,
    outer_local: usize,
    ao_keys: &[u16],
    carry_runs: &mut [u8],
    slice_quads: &mut Vec<LocalQuad>,
) where
    T: MergeVoxel,
{
    let row_ao_keys = ao_key_row::<AO_SAFE>(ao_keys, outer_local, bit_len);

    while row_bits != 0 {
        let bit_local = row_bits.trailing_zeros() as usize;
        let voxel_index = row_base_index + bit_local * bit_stride;
        let quad_value = unsafe { voxels.get_unchecked(voxel_index) }.merge_value();
        let quad_ao_key = if AO_SAFE {
            row_ao_keys[bit_local]
        } else {
            NON_OPAQUE_AO_KEY
        };
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
            let next_ao_key = if AO_SAFE {
                row_ao_keys[next_bit_local]
            } else {
                NON_OPAQUE_AO_KEY
            };
            if !cell_matches::<T, AO_SAFE>(
                voxels,
                next_index,
                &quad_value,
                quad_ao_key,
                next_ao_key,
            ) {
                break;
            }

            carry_runs[next_bit_local] = 0;
            run_width += 1;
        }

        row_bits &= !bit_mask(bit_local, run_width);
        slice_quads.push(LocalQuad {
            bit: bit_local as u8,
            outer: (outer_local - run_carry as usize) as u8,
            width: run_width as u8,
            height: run_length as u8,
        });
        carry_runs[bit_local] = 0;
    }
}

#[inline]
fn emit_local_single_row_runs<T, const AO_SAFE: bool>(
    voxels: &[T],
    row_bits: u64,
    row_base_index: usize,
    bit_stride: usize,
    bit_len: usize,
    outer_local: usize,
    ao_keys: &[u16],
    slice_quads: &mut Vec<LocalQuad>,
) where
    T: MergeVoxel,
{
    let row_ao_keys = ao_key_row::<AO_SAFE>(ao_keys, outer_local, bit_len);
    let mut bits_here = row_bits;

    while bits_here != 0 {
        let bit_local = bits_here.trailing_zeros() as usize;
        let voxel_index = row_base_index + bit_local * bit_stride;
        let quad_value = unsafe { voxels.get_unchecked(voxel_index) }.merge_value();
        let quad_ao_key = if AO_SAFE {
            row_ao_keys[bit_local]
        } else {
            NON_OPAQUE_AO_KEY
        };
        let mut run_width = 1usize;

        while bit_local + run_width < bit_len {
            let next_bit_local = bit_local + run_width;
            let next_bit = 1u64 << next_bit_local;
            if bits_here & next_bit == 0 {
                break;
            }

            let next_index = row_base_index + next_bit_local * bit_stride;
            let next_ao_key = if AO_SAFE {
                row_ao_keys[next_bit_local]
            } else {
                NON_OPAQUE_AO_KEY
            };
            if !cell_matches::<T, AO_SAFE>(
                voxels,
                next_index,
                &quad_value,
                quad_ao_key,
                next_ao_key,
            ) {
                break;
            }

            run_width += 1;
        }

        bits_here &= !bit_mask(bit_local, run_width);
        slice_quads.push(LocalQuad {
            bit: bit_local as u8,
            outer: outer_local as u8,
            width: run_width as u8,
            height: 1,
        });
    }
}

#[inline]
fn emit_local_terminal_row_runs<T, const AO_SAFE: bool>(
    voxels: &[T],
    row_bits: u64,
    row_base_index: usize,
    bit_stride: usize,
    bit_len: usize,
    outer_local: usize,
    ao_keys: &[u16],
    carry_runs: &mut [u8],
    slice_quads: &mut Vec<LocalQuad>,
) where
    T: MergeVoxel,
{
    let row_ao_keys = ao_key_row::<AO_SAFE>(ao_keys, outer_local, bit_len);
    let mut bits_here = row_bits;

    while bits_here != 0 {
        let bit_local = bits_here.trailing_zeros() as usize;
        let voxel_index = row_base_index + bit_local * bit_stride;
        let quad_value = unsafe { voxels.get_unchecked(voxel_index) }.merge_value();
        let quad_ao_key = if AO_SAFE {
            row_ao_keys[bit_local]
        } else {
            NON_OPAQUE_AO_KEY
        };
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
            let next_ao_key = if AO_SAFE {
                row_ao_keys[next_bit_local]
            } else {
                NON_OPAQUE_AO_KEY
            };
            if !cell_matches::<T, AO_SAFE>(
                voxels,
                next_index,
                &quad_value,
                quad_ao_key,
                next_ao_key,
            ) {
                break;
            }

            carry_runs[next_bit_local] = 0;
            run_width += 1;
        }

        bits_here &= !bit_mask(bit_local, run_width);
        slice_quads.push(LocalQuad {
            bit: bit_local as u8,
            outer: (outer_local - run_carry as usize) as u8,
            width: run_width as u8,
            height: run_length as u8,
        });
        carry_runs[bit_local] = 0;
    }
}

#[inline]
fn cell_matches<T, const AO_SAFE: bool>(
    voxels: &[T],
    voxel_index: usize,
    quad_value: &T::MergeValue,
    quad_ao_key: u16,
    cell_ao_key: u16,
) -> bool
where
    T: MergeVoxel,
{
    unsafe { voxels.get_unchecked(voxel_index) }
        .merge_value()
        .eq(quad_value)
        && (!AO_SAFE || cell_ao_key == quad_ao_key)
}

#[inline]
fn indices_match<T, const AO_SAFE: bool>(
    voxels: &[T],
    left_index: usize,
    left_ao_key: u16,
    right_index: usize,
    right_ao_key: u16,
) -> bool
where
    T: MergeVoxel,
{
    unsafe { voxels.get_unchecked(left_index) }
        .merge_value()
        .eq(&unsafe { voxels.get_unchecked(right_index) }.merge_value())
        && (!AO_SAFE || left_ao_key == right_ao_key)
}

#[inline]
fn ao_key_row<const AO_SAFE: bool>(ao_keys: &[u16], outer_local: usize, bit_len: usize) -> &[u16] {
    if AO_SAFE {
        &ao_keys[outer_local * bit_len..outer_local * bit_len + bit_len]
    } else {
        &[]
    }
}

#[inline]
fn reset_carry_runs(carry_runs: &mut Vec<u8>, len: usize) {
    carry_runs.resize(len, 0);
    carry_runs.fill(0);
}

#[inline]
fn build_slice_ao_keys<T>(
    voxels: &[T],
    n_index_base: usize,
    normal_stride: isize,
    outer_stride: usize,
    bit_stride: usize,
    slice: SlicePlan,
    slice_rows: &[u64],
    ao_keys: &mut Vec<u16>,
) where
    T: MergeVoxel,
{
    let total_cells = slice.outer_len * slice.bit_len;
    if ao_keys.len() < total_cells {
        ao_keys.resize(total_cells, NON_OPAQUE_AO_KEY);
    }

    for (outer_local, &row_bits) in slice_rows.iter().enumerate() {
        let row_base_index = n_index_base + outer_local * outer_stride;
        let row_keys =
            &mut ao_keys[outer_local * slice.bit_len..outer_local * slice.bit_len + slice.bit_len];
        let mut bits = row_bits;

        while bits != 0 {
            let bit_local = bits.trailing_zeros() as usize;
            bits &= bits - 1;

            let voxel_index = row_base_index + bit_local * bit_stride;
            let voxel = unsafe { voxels.get_unchecked(voxel_index) };
            row_keys[bit_local] = if voxel.get_visibility() == VoxelVisibility::Opaque {
                0x100
                    | compute_ao_signature(
                        voxels,
                        voxel_index,
                        normal_stride,
                        bit_stride as isize,
                        outer_stride as isize,
                    ) as u16
            } else {
                NON_OPAQUE_AO_KEY
            };
        }
    }
}

#[inline]
fn compute_ao_signature<T>(
    voxels: &[T],
    voxel_index: usize,
    normal_stride: isize,
    bit_stride: isize,
    outer_stride: isize,
) -> u8
where
    T: Voxel,
{
    let face_neighbor_index = voxel_index as isize + normal_stride;
    let corners = [
        (-bit_stride, -outer_stride),
        (bit_stride, -outer_stride),
        (bit_stride, outer_stride),
        (-bit_stride, outer_stride),
    ];
    let mut signature = 0u8;

    for (corner_index, (bit_delta, outer_delta)) in corners.into_iter().enumerate() {
        let side1 = is_opaque(voxels, (face_neighbor_index + bit_delta) as usize);
        let side2 = is_opaque(voxels, (face_neighbor_index + outer_delta) as usize);
        let corner = is_opaque(
            voxels,
            (face_neighbor_index + bit_delta + outer_delta) as usize,
        );
        signature |= vertex_ao(side1, side2, corner) << (corner_index * 2);
    }

    signature
}

#[inline]
fn is_opaque<T>(voxels: &[T], voxel_index: usize) -> bool
where
    T: Voxel,
{
    unsafe { voxels.get_unchecked(voxel_index) }.get_visibility() == VoxelVisibility::Opaque
}

#[inline]
fn vertex_ao(side1: bool, side2: bool, corner: bool) -> u8 {
    if side1 && side2 {
        0
    } else {
        3 - side1 as u8 - side2 as u8 - corner as u8
    }
}

#[allow(clippy::too_many_arguments)]
fn subdivide_slice_quads(
    bit_len: usize,
    outer_len: usize,
    slice_quads: &mut Vec<LocalQuad>,
    split_quads: &mut Vec<LocalQuad>,
    vertical_cuts: &mut Vec<u64>,
    horizontal_cuts: &mut Vec<u64>,
    bit_splits: &mut Vec<u8>,
    outer_splits: &mut Vec<u8>,
) {
    if slice_quads.len() < 2 {
        return;
    }

    loop {
        vertical_cuts.resize(bit_len + 1, 0);
        vertical_cuts.fill(0);
        horizontal_cuts.resize(outer_len + 1, 0);
        horizontal_cuts.fill(0);

        for quad in slice_quads.iter().copied() {
            let outer_mask = boundary_mask(quad.outer as usize, quad.height as usize);
            vertical_cuts[quad.bit as usize] |= outer_mask;
            vertical_cuts[quad.bit as usize + quad.width as usize] |= outer_mask;

            let bit_mask_bits = boundary_mask(quad.bit as usize, quad.width as usize);
            horizontal_cuts[quad.outer as usize] |= bit_mask_bits;
            horizontal_cuts[quad.outer as usize + quad.height as usize] |= bit_mask_bits;
        }

        split_quads.clear();
        let mut changed = false;

        for quad in slice_quads.iter().copied() {
            bit_splits.clear();
            bit_splits.push(quad.bit);

            let outer_mask = boundary_mask(quad.outer as usize, quad.height as usize);
            for boundary in quad.bit as usize + 1..quad.bit as usize + quad.width as usize {
                if vertical_cuts[boundary] & outer_mask != 0 {
                    bit_splits.push(boundary as u8);
                }
            }
            bit_splits.push(quad.bit + quad.width);

            outer_splits.clear();
            outer_splits.push(quad.outer);

            let bit_mask_bits = boundary_mask(quad.bit as usize, quad.width as usize);
            for boundary in quad.outer as usize + 1..quad.outer as usize + quad.height as usize {
                if horizontal_cuts[boundary] & bit_mask_bits != 0 {
                    outer_splits.push(boundary as u8);
                }
            }
            outer_splits.push(quad.outer + quad.height);

            changed |= bit_splits.len() > 2 || outer_splits.len() > 2;

            for outer_window in outer_splits.windows(2) {
                for bit_window in bit_splits.windows(2) {
                    split_quads.push(LocalQuad {
                        bit: bit_window[0],
                        outer: outer_window[0],
                        width: bit_window[1] - bit_window[0],
                        height: outer_window[1] - outer_window[0],
                    });
                }
            }
        }

        if !changed {
            return;
        }

        std::mem::swap(slice_quads, split_quads);
    }
}

#[inline]
fn boundary_mask(start: usize, len_in_cells: usize) -> u64 {
    bit_mask(start, len_in_cells + 1)
}

#[inline]
fn emit_local_quads<const N_AXIS: usize, const BIT_IS_U: bool>(
    local_quads: &[LocalQuad],
    n_coord: u32,
    outer_base: u32,
    bit_base: u32,
    quads: &mut Vec<UnorientedQuad>,
) {
    let base_len = quads.len();
    quads.reserve(local_quads.len());
    let spare = quads.spare_capacity_mut();

    for (index, quad) in local_quads.iter().copied().enumerate() {
        write_quad::<N_AXIS, BIT_IS_U>(
            &mut spare[index],
            n_coord,
            outer_base + quad.outer as u32,
            bit_base + quad.bit as u32,
            quad.width as u32,
            quad.height as u32,
        );
    }

    unsafe {
        quads.set_len(base_len + local_quads.len());
    }
}

#[inline]
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
