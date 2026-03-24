//! AO-safe alternate merge policy.
//!
//! The default [`crate::binary_greedy_quads`] path stays on the existing
//! carry-based maximum-merge implementation in [`crate::merge`]. This module
//! only handles the AO-safe variant, which keeps the same carry structure but
//! refuses to merge opaque faces whose four corner AO values differ.

use block_mesh::{MergeVoxel, UnorientedQuad, Voxel, VoxelVisibility};

use crate::bit_mask;
use crate::context::{MeshingContext, SlicePlan};
use crate::face::{write_quad, write_unit_quad, FaceAxes};

const NON_OPAQUE_AO_KEY: u16 = 0;

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
    ao_keys: Vec<u16>,
    carry_runs: Vec<u8>,
    slice_quads: Vec<LocalQuad>,
}

#[derive(Clone, Copy, Debug)]
struct LocalQuad {
    bit: u8,
    outer: u8,
    width: u8,
    height: u8,
}

/// Dispatches one face slice to the requested alternate merge policy.
#[inline]
pub(crate) fn mesh_face_rows_with_features<T>(
    voxels: &[T],
    context: &MeshingContext,
    slice: SlicePlan,
    visible_rows: &[u64],
    unit_only: bool,
    axes: FaceAxes,
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
            scratch,
            quads,
        );
        return;
    }

    unreachable!("unsupported alternate merge feature set: {:?}", features);
}

/// Chooses the compile-time-specialized AO-safe kernel for one face
/// orientation.
#[inline]
fn mesh_face_rows_ao_safe<T>(
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
        (0, false) => mesh_face_rows_ao_safe_impl::<T, 0, false>(
            voxels,
            context,
            slice,
            visible_rows,
            unit_only,
            axes.n_sign,
            scratch,
            quads,
        ),
        (0, true) => mesh_face_rows_ao_safe_impl::<T, 0, true>(
            voxels,
            context,
            slice,
            visible_rows,
            unit_only,
            axes.n_sign,
            scratch,
            quads,
        ),
        (1, false) => mesh_face_rows_ao_safe_impl::<T, 1, false>(
            voxels,
            context,
            slice,
            visible_rows,
            unit_only,
            axes.n_sign,
            scratch,
            quads,
        ),
        (1, true) => mesh_face_rows_ao_safe_impl::<T, 1, true>(
            voxels,
            context,
            slice,
            visible_rows,
            unit_only,
            axes.n_sign,
            scratch,
            quads,
        ),
        (2, false) => mesh_face_rows_ao_safe_impl::<T, 2, false>(
            voxels,
            context,
            slice,
            visible_rows,
            unit_only,
            axes.n_sign,
            scratch,
            quads,
        ),
        (2, true) => mesh_face_rows_ao_safe_impl::<T, 2, true>(
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
fn mesh_face_rows_ao_safe_impl<T, const N_AXIS: usize, const BIT_IS_U: bool>(
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

        scratch.slice_quads.clear();
        build_slice_quads_carry(
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
fn build_slice_quads_carry<T>(
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
                emit_local_terminal_row_runs(
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
                emit_local_single_row_runs(
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

        let continue_mask = build_continue_mask(
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
                emit_local_single_row_runs(
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

        emit_local_mixed_row_runs(
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
fn build_continue_mask<T>(
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
    let row_ao_keys = ao_key_row(ao_keys, outer_local, bit_len);
    let next_row_ao_keys = ao_key_row(ao_keys, outer_local + 1, bit_len);

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

#[inline]
fn emit_local_mixed_row_runs<T>(
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
    let row_ao_keys = ao_key_row(ao_keys, outer_local, bit_len);

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
fn emit_local_single_row_runs<T>(
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
    let row_ao_keys = ao_key_row(ao_keys, outer_local, bit_len);
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
        slice_quads.push(LocalQuad {
            bit: bit_local as u8,
            outer: outer_local as u8,
            width: run_width as u8,
            height: 1,
        });
    }
}

#[inline]
fn emit_local_terminal_row_runs<T>(
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
    let row_ao_keys = ao_key_row(ao_keys, outer_local, bit_len);
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
fn cell_matches<T>(
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
        && cell_ao_key == quad_ao_key
}

#[inline]
fn ao_key_row(ao_keys: &[u16], outer_local: usize, bit_len: usize) -> &[u16] {
    &ao_keys[outer_local * bit_len..outer_local * bit_len + bit_len]
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
