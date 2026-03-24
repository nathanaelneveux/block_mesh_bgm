//! Alternate merge policies.
//!
//! The default `binary_greedy_quads` path stays on the existing carry-based
//! maximum-merge implementation in [`crate::merge`]. This module handles the
//! slower but more configurable policies:
//!
//! - AO-safe merging for opaque faces
//! - slice-local subdivision to eliminate coplanar T-junctions
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
    pub(crate) ao_keys: Vec<u16>,
    pub(crate) working_rows: Vec<u64>,
    pub(crate) slice_quads: Vec<LocalQuad>,
    pub(crate) split_quads: Vec<LocalQuad>,
    pub(crate) vertical_cuts: Vec<u64>,
    pub(crate) horizontal_cuts: Vec<u64>,
    pub(crate) bit_splits: Vec<u8>,
    pub(crate) outer_splits: Vec<u8>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct LocalQuad {
    bit: u8,
    outer: u8,
    width: u8,
    height: u8,
}

impl LocalQuad {
    #[inline]
    fn bit_end(self) -> usize {
        self.bit as usize + self.width as usize
    }

    #[inline]
    fn outer_end(self) -> usize {
        self.outer as usize + self.height as usize
    }
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

        scratch.working_rows.resize(slice.outer_len, 0);
        scratch.working_rows.copy_from_slice(slice_rows);

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
        build_slice_quads::<T, AO_SAFE>(
            voxels,
            n_index_base,
            outer_stride,
            bit_stride,
            slice,
            &mut scratch.working_rows,
            &scratch.ao_keys,
            &mut scratch.slice_quads,
        );

        if NO_T_JUNCTIONS {
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
fn build_slice_quads<T, const AO_SAFE: bool>(
    voxels: &[T],
    n_index_base: usize,
    outer_stride: usize,
    bit_stride: usize,
    slice: SlicePlan,
    working_rows: &mut [u64],
    ao_keys: &[u16],
    slice_quads: &mut Vec<LocalQuad>,
) where
    T: MergeVoxel,
{
    for outer_local in 0..slice.outer_len {
        while working_rows[outer_local] != 0 {
            let bit_local = working_rows[outer_local].trailing_zeros() as usize;
            let row_base_index = n_index_base + outer_local * outer_stride;
            let voxel_index = row_base_index + bit_local * bit_stride;
            let quad_value = unsafe { voxels.get_unchecked(voxel_index) }.merge_value();
            let quad_ao_key = if AO_SAFE {
                ao_keys[outer_local * slice.bit_len + bit_local]
            } else {
                NON_OPAQUE_AO_KEY
            };

            let row_ao_keys = if AO_SAFE {
                &ao_keys[outer_local * slice.bit_len..outer_local * slice.bit_len + slice.bit_len]
            } else {
                &[]
            };

            let width = row_run_width::<T, AO_SAFE>(
                voxels,
                row_base_index,
                bit_stride,
                slice.bit_len,
                working_rows[outer_local],
                bit_local,
                &quad_value,
                quad_ao_key,
                row_ao_keys,
            );
            let mask = bit_mask(bit_local, width);
            let height = row_run_height::<T, AO_SAFE>(
                voxels,
                n_index_base,
                outer_stride,
                bit_stride,
                slice,
                working_rows,
                mask,
                outer_local,
                bit_local,
                width,
                &quad_value,
                quad_ao_key,
                ao_keys,
            );

            for row in &mut working_rows[outer_local..outer_local + height] {
                *row &= !mask;
            }

            slice_quads.push(LocalQuad {
                bit: bit_local as u8,
                outer: outer_local as u8,
                width: width as u8,
                height: height as u8,
            });
        }
    }
}

#[inline]
fn row_run_width<T, const AO_SAFE: bool>(
    voxels: &[T],
    row_base_index: usize,
    bit_stride: usize,
    bit_len: usize,
    row_bits: u64,
    bit_local: usize,
    quad_value: &T::MergeValue,
    quad_ao_key: u16,
    row_ao_keys: &[u16],
) -> usize
where
    T: MergeVoxel,
{
    let mut width = 1usize;

    while bit_local + width < bit_len {
        let next_bit_local = bit_local + width;
        let next_bit = 1u64 << next_bit_local;
        if row_bits & next_bit == 0 {
            break;
        }

        let next_index = row_base_index + next_bit_local * bit_stride;
        let next_ao_key = if AO_SAFE {
            row_ao_keys[next_bit_local]
        } else {
            NON_OPAQUE_AO_KEY
        };
        if !cell_matches::<T, AO_SAFE>(voxels, next_index, quad_value, quad_ao_key, next_ao_key) {
            break;
        }

        width += 1;
    }

    width
}

#[inline]
fn row_run_height<T, const AO_SAFE: bool>(
    voxels: &[T],
    n_index_base: usize,
    outer_stride: usize,
    bit_stride: usize,
    slice: SlicePlan,
    working_rows: &[u64],
    mask: u64,
    outer_local: usize,
    bit_local: usize,
    width: usize,
    quad_value: &T::MergeValue,
    quad_ao_key: u16,
    ao_keys: &[u16],
) -> usize
where
    T: MergeVoxel,
{
    let mut height = 1usize;

    'outer: while outer_local + height < slice.outer_len {
        let next_outer = outer_local + height;
        if working_rows[next_outer] & mask != mask {
            break;
        }

        let row_base_index = n_index_base + next_outer * outer_stride;
        for offset in 0..width {
            let next_bit_local = bit_local + offset;
            let voxel_index = row_base_index + next_bit_local * bit_stride;
            let next_ao_key = if AO_SAFE {
                ao_keys[next_outer * slice.bit_len + next_bit_local]
            } else {
                NON_OPAQUE_AO_KEY
            };
            if !cell_matches::<T, AO_SAFE>(
                voxels,
                voxel_index,
                quad_value,
                quad_ao_key,
                next_ao_key,
            ) {
                break 'outer;
            }
        }

        height += 1;
    }

    height
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
    ao_keys.resize(total_cells, NON_OPAQUE_AO_KEY);
    ao_keys.fill(NON_OPAQUE_AO_KEY);

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
    loop {
        vertical_cuts.resize(bit_len + 1, 0);
        vertical_cuts.fill(0);
        horizontal_cuts.resize(outer_len + 1, 0);
        horizontal_cuts.fill(0);

        for quad in slice_quads.iter().copied() {
            let outer_mask = boundary_mask(quad.outer as usize, quad.height as usize);
            vertical_cuts[quad.bit as usize] |= outer_mask;
            vertical_cuts[quad.bit_end()] |= outer_mask;

            let bit_mask_bits = boundary_mask(quad.bit as usize, quad.width as usize);
            horizontal_cuts[quad.outer as usize] |= bit_mask_bits;
            horizontal_cuts[quad.outer_end()] |= bit_mask_bits;
        }

        split_quads.clear();
        let mut changed = false;

        for quad in slice_quads.iter().copied() {
            bit_splits.clear();
            bit_splits.push(quad.bit);

            let outer_mask = boundary_mask(quad.outer as usize, quad.height as usize);
            for boundary in quad.bit as usize + 1..quad.bit_end() {
                if vertical_cuts[boundary] & outer_mask != 0 {
                    bit_splits.push(boundary as u8);
                }
            }
            bit_splits.push(quad.bit as usize as u8 + quad.width);

            outer_splits.clear();
            outer_splits.push(quad.outer);

            let bit_mask_bits = boundary_mask(quad.bit as usize, quad.width as usize);
            for boundary in quad.outer as usize + 1..quad.outer_end() {
                if horizontal_cuts[boundary] & bit_mask_bits != 0 {
                    outer_splits.push(boundary as u8);
                }
            }
            outer_splits.push(quad.outer as usize as u8 + quad.height);

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
