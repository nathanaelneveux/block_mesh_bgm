//! Stage 1 and 2 of the pipeline.
//!
//! Stage 1 walks the padded voxel query once and packs occupancy into two
//! column tables:
//!
//! - X columns: one bitset per `(y, z)`
//! - Y columns: one bitset per `(x, z)`
//!
//! Stage 2 compares those columns against their neighbours in the face-normal
//! direction to build visible-face rows. Each row is another `u64`, now packing
//! visible cells along the chosen bit axis of one face slice.

use block_mesh::{Voxel, VoxelVisibility};

use crate::bit_mask;

/// Fast-path classification for the slices belonging to one oriented face.
#[derive(Clone, Copy)]
pub(crate) struct FaceSliceMasks {
    /// Slices containing at least one visible face.
    pub(crate) active: u64,
    /// Active slices whose visible faces are all isolated.
    pub(crate) unit_only: u64,
}

/// Visibility classes encountered while packing the padded query.
#[derive(Clone, Copy)]
pub(crate) struct OccupancySummary {
    pub(crate) has_empty: bool,
    pub(crate) has_opaque: bool,
    pub(crate) has_translucent: bool,
}

impl OccupancySummary {
    /// Returns whether any pair of visibility classes can expose a source face.
    #[inline]
    pub(crate) fn can_have_visible_faces(self) -> bool {
        if self.has_empty {
            self.has_opaque || self.has_translucent
        } else {
            self.has_opaque && self.has_translucent
        }
    }
}

/// Resizes the reusable occupancy-column buffers for one query shape.
///
/// [`build_axis_columns`] overwrites every entry it later reads, so stale
/// contents from a previous call never need clearing.
#[inline]
pub(crate) fn reset_columns(
    opaque_cols: &mut [Vec<u64>; 2],
    trans_cols: &mut [Vec<u64>; 2],
    query_shape: [usize; 3],
) {
    for bit_axis in 0..2 {
        let len = column_count(bit_axis, query_shape);
        opaque_cols[bit_axis].resize(len, 0);
        trans_cols[bit_axis].resize(len, 0);
    }
}

/// Builds the X- and Y-packed occupancy columns used by the fixed scan plans.
///
/// This is the only stage that touches every voxel, so it is the one that
/// decides throughput on mostly-solid or mostly-empty chunks. It works one
/// Z plane at a time:
///
/// 1. Each X row is packed without branching on visibility: voxels are first
///    mapped to bytes, then eight bytes at a time are folded into bits with a
///    multiply ([`movemask8`]). Noisy terrain would otherwise mispredict a
///    branch on nearly every surface voxel.
/// 2. The plane's Y columns are the bit-matrix transpose of its X rows
///    ([`transpose64`]), which replaces a read-modify-write per voxel with a
///    fixed number of whole-word operations per plane.
///
/// The returned summary lets the next stage skip translucent visibility logic
/// for opaque-only inputs and terminate immediately when a uniform padded query
/// cannot contain a visible face.
#[inline]
pub(crate) fn build_axis_columns<T>(
    voxels: &[T],
    min: [u32; 3],
    query_shape: [usize; 3],
    strides: [usize; 3],
    opaque_cols: &mut [Vec<u64>; 2],
    trans_cols: &mut [Vec<u64>; 2],
) -> OccupancySummary
where
    T: Voxel,
{
    let [opaque_x_cols, opaque_y_cols] = opaque_cols;
    let [trans_x_cols, trans_y_cols] = trans_cols;
    let [qx, qy, qz] = query_shape;
    let [x_stride, y_stride, z_stride] = strides;
    let base_index = crate::context::coord_to_index(min, strides);
    let query_x_mask = bit_mask(0, qx);
    let mut empty_union = 0u64;
    let mut opaque_union = 0u64;
    let mut trans_union = 0u64;
    let mut scratch = [0u64; 64];

    for z in 0..qz {
        let x_plane = z * qy..(z + 1) * qy;
        let y_plane = z * qx..(z + 1) * qx;
        let opaque_rows = &mut opaque_x_cols[x_plane.clone()];
        let trans_rows = &mut trans_x_cols[x_plane.clone()];

        for y in 0..qy {
            let row_index = base_index + z * z_stride + y * y_stride;
            let (opaque, trans) = if x_stride == 1 {
                pack_row(&voxels[row_index..row_index + qx])
            } else {
                pack_strided_row(voxels, row_index, x_stride, qx)
            };
            opaque_rows[y] = opaque;
            trans_rows[y] = trans;
            opaque_union |= opaque;
            trans_union |= trans;
            empty_union |= !(opaque | trans) & query_x_mask;
        }

        transpose_plane(opaque_rows, &mut scratch, &mut opaque_y_cols[y_plane.clone()]);
        transpose_plane(trans_rows, &mut scratch, &mut trans_y_cols[y_plane]);
    }

    OccupancySummary {
        has_empty: empty_union != 0,
        has_opaque: opaque_union != 0,
        has_translucent: trans_union != 0,
    }
}

/// Encodes a voxel's visibility as one byte: bit 0 is translucent, bit 1 is
/// opaque, and empty is zero.
#[inline(always)]
fn visibility_byte<T: Voxel>(voxel: &T) -> u8 {
    match voxel.get_visibility() {
        VoxelVisibility::Empty => 0,
        VoxelVisibility::Translucent => 1,
        VoxelVisibility::Opaque => 2,
    }
}

/// Packs a contiguous row of at most 64 voxels into `(opaque, translucent)`
/// bitsets.
#[inline(always)]
fn pack_row<T: Voxel>(row: &[T]) -> (u64, u64) {
    let mut bytes = [0u8; 64];
    for (byte, voxel) in bytes.iter_mut().zip(row) {
        *byte = visibility_byte(voxel);
    }
    pack_visibility_bytes(&bytes, row.len())
}

/// [`pack_row`] for shapes whose X axis is not the contiguous one.
#[inline(always)]
fn pack_strided_row<T: Voxel>(
    voxels: &[T],
    mut index: usize,
    stride: usize,
    len: usize,
) -> (u64, u64) {
    let mut bytes = [0u8; 64];
    for byte in bytes.iter_mut().take(len) {
        // SAFETY: `MeshingContext::new` asserted that the whole query extent
        // lies inside `voxels`, and this row is part of the query.
        *byte = visibility_byte(unsafe { voxels.get_unchecked(index) });
        index += stride;
    }
    pack_visibility_bytes(&bytes, len)
}

/// Folds visibility bytes into bitsets, eight voxels per step.
#[inline(always)]
fn pack_visibility_bytes(bytes: &[u8; 64], len: usize) -> (u64, u64) {
    let mut opaque = 0u64;
    let mut trans = 0u64;
    for (chunk, word) in bytes.chunks_exact(8).take(len.div_ceil(8)).enumerate() {
        let word = u64::from_le_bytes(word.try_into().unwrap());
        opaque |= movemask8(word >> 1) << (chunk * 8);
        trans |= movemask8(word) << (chunk * 8);
    }
    (opaque, trans)
}

/// Gathers bit 0 of each of the eight bytes of `x` into the low eight bits of
/// the result: a portable `movemask`.
///
/// After masking, byte `i` holds `b_i << 8i`. The magic multiplier has byte `k`
/// equal to `1 << (7 - k)`, so the product term for `i + k = 7` lands `b_i` on
/// bit `56 + i`. Every other term lands below bit 56 or overflows past bit 63,
/// and no two terms share a bit, so there are no carries to worry about.
#[inline(always)]
fn movemask8(x: u64) -> u64 {
    (x & 0x0101_0101_0101_0101).wrapping_mul(0x0102_0408_1020_4080) >> 56
}

/// Transposes one plane of X rows (row `y`, bit `x`) into Y columns (row `x`,
/// bit `y`).
#[inline(always)]
fn transpose_plane(rows: &[u64], scratch: &mut [u64; 64], columns: &mut [u64]) {
    // Air above the terrain and solid ground below it are both common, and
    // an empty plane transposes to an empty plane.
    if rows.iter().all(|&row| row == 0) {
        columns.fill(0);
        return;
    }

    scratch[..rows.len()].copy_from_slice(rows);
    scratch[rows.len()..].fill(0);
    transpose64(scratch);
    columns.copy_from_slice(&scratch[..columns.len()]);
}

/// In-place 64x64 bit-matrix transpose, with bit `c` of word `r` as element
/// `(r, c)`.
///
/// Each round swaps the off-diagonal `J x J` blocks of every `2J x 2J` block,
/// halving `J` from 32 down to 1. The inner loops are independent lane-wise
/// operations, which LLVM vectorizes.
#[inline(always)]
fn transpose64(words: &mut [u64; 64]) {
    #[inline(always)]
    fn round<const J: usize>(words: &mut [u64; 64], low_half: u64) {
        for block in words.chunks_exact_mut(2 * J) {
            let (top, bottom) = block.split_at_mut(J);
            for (upper, lower) in top.iter_mut().zip(bottom) {
                // Swap the high J bits of `upper` with the low J bits of `lower`.
                let swap = ((*upper >> J) ^ *lower) & low_half;
                *lower ^= swap;
                *upper ^= swap << J;
            }
        }
    }

    round::<32>(words, 0x0000_0000_FFFF_FFFF);
    round::<16>(words, 0x0000_FFFF_0000_FFFF);
    round::<8>(words, 0x00FF_00FF_00FF_00FF);
    round::<4>(words, 0x0F0F_0F0F_0F0F_0F0F);
    round::<2>(words, 0x3333_3333_3333_3333);
    round::<1>(words, 0x5555_5555_5555_5555);
}

/// Builds visibility rows for both signs of one normal axis.
///
/// The return value classifies active and unit-only slices for each sign. A
/// slice is `unit_only` if no visible cell touches another visible cell
/// horizontally or vertically.
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
) -> (FaceSliceMasks, FaceSliceMasks) {
    let interior_bit_mask = bit_mask(0, bit_len);
    let (base_offset, n_stride, outer_stride) =
        column_row_layout(query_shape, bit_axis, n_axis, outer_axis);
    let mut neg_masks = FaceSliceMasks {
        active: 0,
        unit_only: 0,
    };
    let mut pos_masks = FaceSliceMasks {
        active: 0,
        unit_only: 0,
    };

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
        let mut neg_visible_union = 0u64;
        let mut pos_visible_union = 0u64;
        let mut neg_adjacency = 0u64;
        let mut pos_adjacency = 0u64;

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
            neg_visible_union |= neg_visible;
            pos_visible_union |= pos_visible;
            neg_adjacency |= neg_visible & ((neg_visible >> 1) | previous_neg);
            pos_adjacency |= pos_visible & ((pos_visible >> 1) | previous_pos);
            neg_rows[row_start + outer_local] = neg_visible;
            pos_rows[row_start + outer_local] = pos_visible;
            previous_neg = neg_visible;
            previous_pos = pos_visible;
            src += outer_stride;
            neg += outer_stride;
            pos += outer_stride;
        }

        let slice_bit = 1u64 << n_local;
        if neg_visible_union != 0 {
            neg_masks.active |= slice_bit;
            if neg_adjacency == 0 {
                neg_masks.unit_only |= slice_bit;
            }
        }
        if pos_visible_union != 0 {
            pos_masks.active |= slice_bit;
            if pos_adjacency == 0 {
                pos_masks.unit_only |= slice_bit;
            }
        }
    }

    (neg_masks, pos_masks)
}

/// Returns the index layout for scanning rows out of one occupancy-column table.
///
/// The result is `(base_offset, n_stride, outer_stride)`.
#[inline(always)]
pub(crate) fn column_row_layout(
    query_shape: [usize; 3],
    bit_axis: usize,
    n_axis: usize,
    outer_axis: usize,
) -> (usize, usize, usize) {
    match (bit_axis, n_axis, outer_axis) {
        (0, 1, 2) => (query_shape[1], 1, query_shape[1]),
        (0, 2, 1) => (1, query_shape[1], 1),
        (1, 0, 2) => (query_shape[0], 1, query_shape[0]),
        (1, 2, 0) => (1, query_shape[0], 1),
        (2, 0, 1) => (query_shape[0], 1, query_shape[0]),
        (2, 1, 0) => (1, query_shape[0], 1),
        _ => unreachable!(),
    }
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

#[cfg(test)]
mod tests {
    use super::{movemask8, transpose64};

    #[test]
    fn movemask_gathers_low_bit_of_each_byte() {
        assert_eq!(movemask8(0), 0);
        assert_eq!(movemask8(0x0101_0101_0101_0101), 0xFF);
        assert_eq!(movemask8(0x0100_0000_0000_0001), 0b1000_0001);
        assert_eq!(movemask8(0xFEFE_FEFE_FEFE_FEFE), 0);
        for i in 0..8 {
            assert_eq!(movemask8(1 << (8 * i)), 1 << i);
        }
    }

    #[test]
    fn transpose_moves_every_bit_across_the_diagonal() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut words = [0u64; 64];
        for word in &mut words {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *word = state;
        }

        let mut transposed = words;
        transpose64(&mut transposed);
        for r in 0..64 {
            for c in 0..64 {
                assert_eq!((words[r] >> c) & 1, (transposed[c] >> r) & 1);
            }
        }
    }
}
