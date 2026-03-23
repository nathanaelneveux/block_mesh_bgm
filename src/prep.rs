//! Stage 1 and 2 of the pipeline.
//!
//! Stage 1 walks the padded voxel query once and packs occupancy into three
//! column tables:
//!
//! - X columns: one bitset per `(y, z)`
//! - Y columns: one bitset per `(x, z)`
//! - Z columns: one bitset per `(x, y)`
//!
//! Stage 2 compares those columns against their neighbours in the face-normal
//! direction to build visible-face rows. Each row is another `u64`, now packing
//! visible cells along the chosen bit axis of one face slice.

use block_mesh::{Voxel, VoxelVisibility};

use crate::bit_mask;

/// Resizes and clears the reusable occupancy-column buffers for one query shape.
#[inline]
pub(crate) fn reset_columns(
    opaque_cols: &mut [Vec<u64>; 3],
    trans_cols: &mut [Vec<u64>; 3],
    query_shape: [usize; 3],
) {
    for bit_axis in 0..3 {
        let len = column_count(bit_axis, query_shape);
        opaque_cols[bit_axis].resize(len, 0);
        trans_cols[bit_axis].resize(len, 0);
        opaque_cols[bit_axis].fill(0);
        trans_cols[bit_axis].fill(0);
    }
}

/// Builds occupancy columns for all three candidate bit axes.
///
/// The function returns `true` if any translucent voxel was seen. That lets the
/// next stage skip translucent visibility logic entirely for opaque-only inputs.
#[inline]
pub(crate) fn build_axis_columns<T>(
    voxels: &[T],
    min: [u32; 3],
    query_shape: [usize; 3],
    strides: [usize; 3],
    opaque_cols: &mut [Vec<u64>; 3],
    trans_cols: &mut [Vec<u64>; 3],
) -> bool
where
    T: Voxel,
{
    let [opaque_x_cols, opaque_y_cols, opaque_z_cols] = opaque_cols;
    let [trans_x_cols, trans_y_cols, trans_z_cols] = trans_cols;
    let base_index = crate::context::coord_to_index(min, strides);
    let qx = query_shape[0];
    let qy = query_shape[1];
    let qz = query_shape[2];
    let x_stride = strides[0];
    let y_stride = strides[1];
    let z_stride = strides[2];
    let mut has_translucent = false;

    for z in 0..qz {
        let z_base_index = base_index + z * z_stride;
        let z_bit = 1u64 << z;
        let z_x_base = z * qx;
        let z_y_base = z * qy;

        // These temporary planes let the Y columns be built with one store per
        // `(x, z)` instead of one OR per voxel.
        let mut opaque_y_plane = [0u64; 64];
        let mut trans_y_plane = [0u64; 64];

        for y in 0..qy {
            let mut index = z_base_index + y * y_stride;
            let x_col_index = y + z_y_base;
            let y_bit = 1u64 << y;
            let z_col_row_base = y * qx;
            let opaque_z_row = &mut opaque_z_cols[z_col_row_base..z_col_row_base + qx];
            let trans_z_row = &mut trans_z_cols[z_col_row_base..z_col_row_base + qx];
            let mut opaque_x_mask = 0u64;
            let mut trans_x_mask = 0u64;
            let mut x_bit = 1u64;

            for x in 0..qx {
                match unsafe { voxels.get_unchecked(index) }.get_visibility() {
                    VoxelVisibility::Empty => {}
                    VoxelVisibility::Translucent => {
                        has_translucent = true;
                        trans_x_mask |= x_bit;
                        trans_y_plane[x] |= y_bit;
                        trans_z_row[x] |= z_bit;
                    }
                    VoxelVisibility::Opaque => {
                        opaque_x_mask |= x_bit;
                        opaque_y_plane[x] |= y_bit;
                        opaque_z_row[x] |= z_bit;
                    }
                }

                index += x_stride;
                x_bit <<= 1;
            }

            opaque_x_cols[x_col_index] = opaque_x_mask;
            trans_x_cols[x_col_index] = trans_x_mask;
        }

        opaque_y_cols[z_x_base..z_x_base + qx].copy_from_slice(&opaque_y_plane[..qx]);
        trans_y_cols[z_x_base..z_x_base + qx].copy_from_slice(&trans_y_plane[..qx]);
    }

    has_translucent
}

/// Builds visibility rows for both signs of one normal axis.
///
/// The return value reports whether each sign can be emitted as unit quads
/// directly. A slice is `unit_only` if no visible cell touches another visible
/// cell horizontally or vertically.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn build_visible_row_pair(
    query_shape: [usize; 3],
    bit_axis: usize,
    bit_len: usize,
    outer_len: usize,
    n_len: usize,
    n_axis: usize,
    outer_axis: usize,
    opaque_cols: &[u64],
    trans_cols: &[u64],
    has_translucent: bool,
    neg_rows: &mut [u64],
    pos_rows: &mut [u64],
) -> (bool, bool) {
    let interior_bit_mask = bit_mask(0, bit_len);
    let (base_offset, n_stride, outer_stride) = match (bit_axis, n_axis, outer_axis) {
        (0, 1, 2) => (query_shape[1], 1, query_shape[1]),
        (0, 2, 1) => (1, query_shape[1], 1),
        (1, 0, 2) => (query_shape[0], 1, query_shape[0]),
        (1, 2, 0) => (1, query_shape[0], 1),
        (2, 0, 1) => (query_shape[0], 1, query_shape[0]),
        (2, 1, 0) => (1, query_shape[0], 1),
        _ => unreachable!(),
    };
    let mut neg_unit_only = true;
    let mut pos_unit_only = true;

    for n_local in 0..n_len {
        let src_n = n_local + 1;
        let neg_n = src_n - 1;
        let pos_n = src_n + 1;
        let row_start = n_local * outer_len;
        let mut src = base_offset + src_n * n_stride;
        let mut neg = base_offset + neg_n * n_stride;
        let mut pos = base_offset + pos_n * n_stride;
        let mut previous_neg = 0u64;
        let mut previous_pos = 0u64;

        for outer_local in 0..outer_len {
            let src_opaque = opaque_cols[src];
            let neg_opaque = opaque_cols[neg];
            let pos_opaque = opaque_cols[pos];
            let (neg_raw_visible, pos_raw_visible) = if has_translucent {
                let src_trans = trans_cols[src];
                let neg_trans = trans_cols[neg];
                let pos_trans = trans_cols[pos];
                (
                    (src_opaque & !neg_opaque) | (src_trans & !(neg_opaque | neg_trans)),
                    (src_opaque & !pos_opaque) | (src_trans & !(pos_opaque | pos_trans)),
                )
            } else {
                (src_opaque & !neg_opaque, src_opaque & !pos_opaque)
            };

            let neg_visible = (neg_raw_visible >> 1) & interior_bit_mask;
            let pos_visible = (pos_raw_visible >> 1) & interior_bit_mask;
            neg_unit_only &= (neg_visible & (neg_visible >> 1)) == 0;
            neg_unit_only &= (neg_visible & previous_neg) == 0;
            pos_unit_only &= (pos_visible & (pos_visible >> 1)) == 0;
            pos_unit_only &= (pos_visible & previous_pos) == 0;
            neg_rows[row_start + outer_local] = neg_visible;
            pos_rows[row_start + outer_local] = pos_visible;
            previous_neg = neg_visible;
            previous_pos = pos_visible;
            src += outer_stride;
            neg += outer_stride;
            pos += outer_stride;
        }
    }

    (neg_unit_only, pos_unit_only)
}

/// Resizes the reusable visible-row buffer for the current axis family.
#[inline]
pub(crate) fn reset_visible_rows(visible_rows: &mut Vec<u64>, total_rows: usize) {
    visible_rows.resize(total_rows, 0);
}

#[inline]
fn column_count(bit_axis: usize, query_shape: [usize; 3]) -> usize {
    match bit_axis {
        0 => query_shape[1] * query_shape[2],
        1 => query_shape[0] * query_shape[2],
        2 => query_shape[0] * query_shape[1],
        _ => unreachable!(),
    }
}
