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
use crate::merge::{emit_unit_slice, merge_rows, FaceJob, SliceGeometry};
use crate::prep::column_row_layout;

/// Merges the faces of one orientation with AO-safe merge restrictions.
#[inline(always)]
pub(crate) fn mesh_face_ao_safe<T, const N_AXIS: usize, const BIT_IS_U: bool>(
    job: &FaceJob<'_, T>,
    quads: &mut Vec<UnorientedQuad>,
) where
    T: MergeVoxel,
{
    job.for_each_active_slice(|n_local, rows, unit_only| {
        let geometry = SliceGeometry::new::<T, N_AXIS>(job, n_local);
        if unit_only {
            emit_unit_slice::<N_AXIS>(&geometry, rows, quads);
            return;
        }

        // Emit the AO classes in increasing order of generality. Each class
        // reuses the shared row merger with only its legal directions enabled.
        let classes = AoRowClasses::build::<N_AXIS, T>(job, n_local, rows);
        let len = rows.len();
        let voxels = job.voxels;
        if classes.unit_union != 0 {
            emit_unit_slice::<N_AXIS>(&geometry, &classes.unit[..len], quads);
        }
        if classes.horizontal_union != 0 {
            merge_rows::<T, N_AXIS, BIT_IS_U, true, false>(
                voxels,
                &geometry,
                &classes.horizontal[..len],
                quads,
            );
        }
        if classes.vertical_union != 0 {
            merge_rows::<T, N_AXIS, BIT_IS_U, false, true>(
                voxels,
                &geometry,
                &classes.vertical[..len],
                quads,
            );
        }
        if classes.bidir_union != 0 {
            merge_rows::<T, N_AXIS, BIT_IS_U, true, true>(
                voxels,
                &geometry,
                &classes.bidir[..len],
                quads,
            );
        }
    });
}

/// The visible cells of one face slice, split by the merge directions that
/// remain legal once AO boundaries are respected.
struct AoRowClasses {
    /// Cells whose exterior occupancy forces a `1x1` quad.
    unit: [u64; 64],
    /// Cells that may only merge along the packed row axis.
    horizontal: [u64; 64],
    /// Cells that may only merge from row to row, always at width 1.
    vertical: [u64; 64],
    /// Cells still free to use the normal bidirectional greedy merge. This
    /// includes every translucent face, which keeps vanilla merge behaviour.
    bidir: [u64; 64],
    unit_union: u64,
    horizontal_union: u64,
    vertical_union: u64,
    bidir_union: u64,
}

impl AoRowClasses {
    /// Classifies every visible cell of the slice at `n_local`.
    ///
    /// AO-safe meshing only cares about opaque occupancy in the plane just
    /// outside the visible face. From that outside plane we can derive, a row
    /// at a time, whether a visible cell must stay a unit quad, may only merge
    /// within its row, may only merge across rows at width 1, or is
    /// unconstrained. This avoids carrying a per-cell AO signature through the
    /// merge loop.
    #[inline(always)]
    fn build<const N_AXIS: usize, T>(job: &FaceJob<'_, T>, n_local: usize, rows: &[u64]) -> Self {
        let slice = job.slice;
        let mut classes = Self {
            unit: [0; 64],
            horizontal: [0; 64],
            vertical: [0; 64],
            bidir: [0; 64],
            unit_union: 0,
            horizontal_union: 0,
            vertical_union: 0,
            bidir_union: 0,
        };

        let interior_bit_mask = bit_mask(0, slice.bit_len);
        let padded_bit_mask = bit_mask(0, slice.bit_len + 2);
        let interior_padded_mask = bit_mask(1, slice.bit_len);
        let (base_offset, n_stride, outer_stride) = column_row_layout(
            job.context.query_shape,
            slice.bit_axis,
            N_AXIS,
            slice.outer_axis,
        );
        let source_n = n_local + 1;
        let outside_n = if job.n_sign < 0 {
            source_n - 1
        } else {
            source_n + 1
        };
        // `base_offset` already skips the padding row, so `base + outer_local
        // * outer_stride` is the column under interior row `outer_local`, and
        // its neighbours along the outer axis are one `outer_stride` away.
        let source_base = base_offset + source_n * n_stride;
        let outside_base = base_offset + outside_n * n_stride;
        let column =
            |base: usize, outer_local: usize| job.opaque_cols[base + outer_local * outer_stride];

        for (outer_local, &row_bits) in rows.iter().enumerate() {
            if row_bits == 0 {
                continue;
            }

            let visible_padded = (row_bits << 1) & interior_padded_mask;
            let source_opaque = column(source_base, outer_local) & interior_padded_mask;
            let opaque_visible = visible_padded & source_opaque;

            if opaque_visible == 0 {
                // Translucent and empty-visible faces do not participate in the
                // AO rule set. They keep the vanilla merge behaviour.
                classes.bidir[outer_local] = row_bits;
                classes.bidir_union |= row_bits;
                continue;
            }

            let prev_row = column(outside_base - outer_stride, outer_local) & padded_bit_mask;
            let current_row = column(outside_base, outer_local) & padded_bit_mask;
            let next_row = column(outside_base + outer_stride, outer_local) & padded_bit_mask;
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

            classes.unit[outer_local] = unit;
            classes.horizontal[outer_local] = horizontal;
            classes.vertical[outer_local] = vertical;
            classes.bidir[outer_local] = bidir;
            classes.unit_union |= unit;
            classes.horizontal_union |= horizontal;
            classes.vertical_union |= vertical;
            classes.bidir_union |= bidir;
        }

        classes
    }
}

/// Classifies the visible opaque cells of one row by the merge directions AO
/// still allows. All inputs are in padded bit coordinates.
///
/// `prev_row`, `current_row`, and `next_row` are opaque occupancy in the
/// exterior plane, one row before, at, and after the visible row. Returns
/// `(unit, horizontal, vertical)`; any opaque visible cell in none of them is
/// free to merge in both directions.
#[inline(always)]
fn classify_ao_opaque_row(
    opaque_visible: u64,
    prev_row: u64,
    current_row: u64,
    next_row: u64,
    interior_bit_mask: u64,
) -> (u64, u64, u64) {
    let neighbor_curr = ((current_row << 1) | (current_row >> 1)) & interior_bit_mask;
    let shared_opaque_depth = prev_row & current_row & next_row;
    // A continuous wall on only one side is not enough to prove a vertical
    // merge: an occupancy transition on the other side still changes the AO
    // signature. Both side columns must be stable across all three rows.
    let stable_depth = shared_opaque_depth | !(prev_row | current_row | next_row);
    let stable_on_both_sides = (stable_depth << 1) & (stable_depth >> 1);
    let opaque_on_either_side =
        ((shared_opaque_depth << 1) & interior_bit_mask) | (shared_opaque_depth >> 1);
    let vertical = opaque_visible
        & stable_on_both_sides
        & opaque_on_either_side
        & (!prev_row)
        & (!next_row)
        & interior_bit_mask;

    let edge_from_next =
        (((next_row << 1) & interior_bit_mask) ^ (next_row >> 1)) & interior_bit_mask;
    let edge_from_prev =
        (((prev_row << 1) & interior_bit_mask) ^ (prev_row >> 1)) & interior_bit_mask;
    let horiz_from_next = ((next_row << 1) & interior_bit_mask) & (next_row >> 1);
    let horiz_from_prev = ((prev_row << 1) & interior_bit_mask) & (prev_row >> 1);
    let possible_unit_from_prev = prev_row & !horiz_from_prev;
    let possible_unit_from_next = next_row & !horiz_from_next;
    let unit = opaque_visible
        & !vertical
        & (neighbor_curr
            | edge_from_next
            | edge_from_prev
            | possible_unit_from_prev
            | possible_unit_from_next)
        & interior_bit_mask;

    let horizontal = opaque_visible
        & !vertical
        & !unit
        & (horiz_from_next | horiz_from_prev)
        & interior_bit_mask;

    (unit, horizontal, vertical)
}

#[cfg(test)]
mod tests {
    use super::classify_ao_opaque_row;

    fn bits(pattern: &str) -> u64 {
        pattern
            .chars()
            .enumerate()
            .fold(0u64, |acc, (i, ch)| match ch {
                '0' => acc,
                '1' => acc | (1u64 << i),
                _ => panic!("unexpected bit {ch:?}"),
            })
    }

    fn padded(pattern: &str) -> u64 {
        let mut padded = String::with_capacity(pattern.len() + 2);
        padded.push('0');
        padded.push_str(pattern);
        padded.push('0');
        bits(&padded)
    }

    #[test]
    fn classify_matches_primary_worked_example() {
        let opaque_visible = padded("0100111111111111110");
        let prev_row = padded("1010000001111111000");
        let current_row = padded("1011000000000000001");
        let next_row = padded("1010011111110000001");
        let mask = padded("1111111111111111111");

        let (unit, horizontal, vertical) =
            classify_ao_opaque_row(opaque_visible, prev_row, current_row, next_row, mask);

        assert_eq!(unit, padded("0000110011011001110"));
        assert_eq!(horizontal, padded("0000001100100110000"));
        assert_eq!(vertical, padded("0100000000000000000"));
    }

    #[test]
    fn classify_matches_second_worked_example() {
        let opaque_visible = padded("0110110");
        let prev_row = padded("1000000");
        let current_row = padded("1001001");
        let next_row = padded("1000001");
        let mask = padded("1111111");

        let (unit, horizontal, vertical) =
            classify_ao_opaque_row(opaque_visible, prev_row, current_row, next_row, mask);

        assert_eq!(unit, padded("0010110"));
        assert_eq!(horizontal, 0);
        assert_eq!(vertical, padded("0100000"));
    }

    #[test]
    fn classify_matches_third_worked_example() {
        let opaque_visible = padded("011101");
        let prev_row = padded("111111");
        let current_row = padded("100000");
        let next_row = padded("100000");
        let mask = padded("111111");

        let (unit, horizontal, vertical) =
            classify_ao_opaque_row(opaque_visible, prev_row, current_row, next_row, mask);

        assert_eq!(unit, padded("010001"));
        assert_eq!(horizontal, padded("001100"));
        assert_eq!(vertical, 0);
    }

    #[test]
    fn classify_marks_fully_exposed_cap_as_bidir() {
        let opaque_visible = bits("011111110");
        let prev_row = 0;
        let current_row = 0;
        let next_row = 0;
        let mask = bits("011111110");

        let (unit, horizontal, vertical) =
            classify_ao_opaque_row(opaque_visible, prev_row, current_row, next_row, mask);

        assert_eq!(unit, 0);
        assert_eq!(horizontal, 0);
        assert_eq!(vertical, 0);
    }

    #[test]
    fn classify_matches_fourth_worked_example_regression() {
        let opaque_visible = padded("111111");
        let prev_row = padded("010000");
        let current_row = 0;
        let next_row = padded("000010");
        let mask = padded("111111");

        let (unit, horizontal, vertical) =
            classify_ao_opaque_row(opaque_visible, prev_row, current_row, next_row, mask);

        assert_eq!(unit, padded("111111"));
        assert_eq!(horizontal, 0);
        assert_eq!(vertical, 0);
    }

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
    fn vertical_rule_rejects_a_transition_on_the_other_side() {
        let opaque_visible = padded("010");
        let prev_row = padded("101");
        let current_row = padded("100");
        let next_row = padded("100");
        let mask = padded("111");

        let (unit, horizontal, vertical) =
            classify_ao_opaque_row(opaque_visible, prev_row, current_row, next_row, mask);

        assert_eq!(unit, padded("010"));
        assert_eq!(horizontal, 0);
        assert_eq!(vertical, 0);
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
