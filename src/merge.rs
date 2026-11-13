//! Stage 3 and 4 of the pipeline: turning visible-face rows into quads.
//!
//! Once a face slice has been reduced to visibility rows, there are two useful
//! regimes:
//!
//! - **unit-only slices**: no visible cell touches another visible cell, so the
//!   mesher can emit one quad per bit without any merge bookkeeping
//!   ([`emit_unit_slice`])
//! - **merge slices**: runs may widen along the packed row and continue across
//!   rows, so the mesher tracks how many rows each run has already been carried
//!   ([`merge_rows`])
//!
//! [`merge_rows`] is shared by every merge policy in the crate. Vanilla meshing
//! merges in both directions; the AO-safe path reuses the same loop with one or
//! both directions switched off at compile time.

use std::mem::MaybeUninit;

use block_mesh::{MergeVoxel, UnorientedQuad};

use crate::bit_mask;
use crate::context::{MeshingContext, SlicePlan};
use crate::face::oriented_quad;
use crate::prep::FaceSliceMasks;

/// Everything a merge kernel needs to know about one face orientation.
pub(crate) struct FaceJob<'a, T> {
    pub(crate) voxels: &'a [T],
    pub(crate) context: &'a MeshingContext,
    pub(crate) slice: SlicePlan,
    /// Visible-face rows for every slice of this face, `outer_len` per slice.
    pub(crate) rows: &'a [u64],
    pub(crate) slice_masks: FaceSliceMasks,
    /// `-1` or `1`: which side of the source voxel the face points to.
    pub(crate) n_sign: i32,
    /// Opaque occupancy columns packed along `slice.bit_axis`.
    pub(crate) opaque_cols: &'a [u64],
}

impl<T> FaceJob<'_, T> {
    /// The visible rows of the slice at `n_local`.
    #[inline(always)]
    pub(crate) fn slice_rows(&self, n_local: usize) -> &[u64] {
        let start = n_local * self.slice.outer_len;
        &self.rows[start..start + self.slice.outer_len]
    }

    /// Calls `visit(n_local, rows, unit_only)` for every slice with a visible
    /// face, in increasing `n_local` order.
    #[inline(always)]
    pub(crate) fn for_each_active_slice(&self, mut visit: impl FnMut(usize, &[u64], bool)) {
        let mut active = self.slice_masks.active;
        while active != 0 {
            let n_local = active.trailing_zeros() as usize;
            active &= active - 1;
            let unit_only = self.slice_masks.unit_only & (1 << n_local) != 0;
            visit(n_local, self.slice_rows(n_local), unit_only);
        }
    }
}

/// Where one face slice lives, both in the voxel array and in quad
/// coordinates.
#[derive(Clone, Copy)]
pub(crate) struct SliceGeometry {
    n_coord: u32,
    outer_base: u32,
    bit_base: u32,
    /// Voxel index of the slice's first interior cell.
    index_base: usize,
    outer_stride: usize,
    bit_stride: usize,
    bit_len: usize,
}

impl SliceGeometry {
    #[inline(always)]
    pub(crate) fn new<T, const N_AXIS: usize>(job: &FaceJob<'_, T>, n_local: usize) -> Self {
        let context = job.context;
        let slice = job.slice;
        Self {
            n_coord: context.interior_min[N_AXIS] + n_local as u32,
            outer_base: context.interior_min[slice.outer_axis],
            bit_base: context.interior_min[slice.bit_axis],
            index_base: context.interior_start_index + n_local * context.strides[N_AXIS],
            outer_stride: context.strides[slice.outer_axis],
            bit_stride: context.strides[slice.bit_axis],
            bit_len: slice.bit_len,
        }
    }

    /// Voxel index of bit 0 of row `outer_local`. Step along the row with
    /// `bit_stride` and to the next row with `outer_stride`.
    #[inline(always)]
    fn row_index(&self, outer_local: usize) -> usize {
        self.index_base + outer_local * self.outer_stride
    }
}

/// Reads a voxel's merge value without a bounds check.
///
/// # Safety
///
/// `index` must be in bounds. `MeshingContext::new` asserts that the whole
/// query extent lies inside `voxels`, and the merge kernels only index interior
/// cells of the current slice (or, for row continuation, the interior cell one
/// row further on, which is only read when that row is visible too).
#[inline(always)]
unsafe fn merge_value_at<T: MergeVoxel>(voxels: &[T], index: usize) -> T::MergeValue {
    debug_assert!(index < voxels.len());
    unsafe { voxels.get_unchecked(index) }.merge_value()
}

/// Merges the faces of one orientation with the vanilla maximum-merge policy.
#[inline(always)]
pub(crate) fn mesh_face<T, const N_AXIS: usize, const BIT_IS_U: bool>(
    job: &FaceJob<'_, T>,
    quads: &mut Vec<UnorientedQuad>,
) where
    T: MergeVoxel,
{
    job.for_each_active_slice(|n_local, rows, unit_only| {
        let geometry = SliceGeometry::new::<T, N_AXIS>(job, n_local);
        if unit_only {
            emit_unit_slice::<N_AXIS>(&geometry, rows, quads);
        } else {
            merge_rows::<T, N_AXIS, BIT_IS_U, true, true>(job.voxels, &geometry, rows, quads);
        }
    });
}

/// Greedily merges one slice's rows into quads.
///
/// `ALONG_ROW` allows runs to widen along the packed bit axis; `ACROSS_ROWS`
/// allows them to continue into the next row. A run only continues when the
/// next row's cell is visible and has the same merge value, and a run that
/// has been carried can only widen over neighbours carried exactly as far, so
/// every emitted quad is a full rectangle of matching faces.
#[inline(never)]
pub(crate) fn merge_rows<
    T,
    const N_AXIS: usize,
    const BIT_IS_U: bool,
    const ALONG_ROW: bool,
    const ACROSS_ROWS: bool,
>(
    voxels: &[T],
    geometry: &SliceGeometry,
    rows: &[u64],
    quads: &mut Vec<UnorientedQuad>,
) where
    T: MergeVoxel,
{
    // How many rows each still-open run has already been carried. Every run
    // ends by the last row (its successor row is empty), which also resets its
    // entry, so this always starts and finishes all zero.
    let mut carry_runs = [0u8; 64];
    let mut has_incoming_carry = false;

    for (outer_local, &row_bits) in rows.iter().enumerate() {
        if row_bits == 0 {
            // An empty row cannot have received carry from the previous row.
            continue;
        }

        let row_index = geometry.row_index(outer_local);
        let continuing = if ACROSS_ROWS {
            let next_row_bits = rows.get(outer_local + 1).copied().unwrap_or(0);
            continuing_bits(
                voxels,
                geometry,
                row_index,
                row_bits & next_row_bits,
                &mut carry_runs,
            )
        } else {
            0
        };

        let ended = row_bits & !continuing;
        if ended != 0 {
            // Without incoming carry every entry is zero, so the cheaper
            // emitter can skip reading and resetting carry state.
            if has_incoming_carry {
                emit_runs::<T, N_AXIS, BIT_IS_U, ALONG_ROW, true>(
                    voxels,
                    geometry,
                    outer_local,
                    row_index,
                    ended,
                    &mut carry_runs,
                    quads,
                );
            } else {
                emit_runs::<T, N_AXIS, BIT_IS_U, ALONG_ROW, false>(
                    voxels,
                    geometry,
                    outer_local,
                    row_index,
                    ended,
                    &mut carry_runs,
                    quads,
                );
            }
        }
        has_incoming_carry = continuing != 0;
    }

    debug_assert!(carry_runs.iter().all(|&carry| carry == 0));
}

/// Returns the subset of `overlapping_bits` whose face continues into the next
/// row with the same merge value, and bumps their carry counts.
#[inline(always)]
fn continuing_bits<T>(
    voxels: &[T],
    geometry: &SliceGeometry,
    row_index: usize,
    mut overlapping_bits: u64,
    carry_runs: &mut [u8; 64],
) -> u64
where
    T: MergeVoxel,
{
    let mut continuing = 0u64;

    while overlapping_bits != 0 {
        let bit_local = overlapping_bits.trailing_zeros() as usize;
        overlapping_bits &= overlapping_bits - 1;

        let index = row_index + bit_local * geometry.bit_stride;
        // SAFETY: see `merge_value_at`; both cells are visible interior cells.
        let (here, next) = unsafe {
            (
                merge_value_at(voxels, index),
                merge_value_at(voxels, index + geometry.outer_stride),
            )
        };
        if here == next {
            continuing |= 1 << bit_local;
            carry_runs[bit_local] += 1;
        }
    }

    continuing
}

/// Emits a quad for every run in `bits`, all of which end on this row.
///
/// With `ALONG_ROW`, each run widens over neighbouring set bits with the same
/// merge value (and, with `CARRIED`, the same carry count). With `CARRIED`, a
/// run's height is its carry count plus one and its start row is that many
/// rows back.
#[inline(always)]
fn emit_runs<
    T,
    const N_AXIS: usize,
    const BIT_IS_U: bool,
    const ALONG_ROW: bool,
    const CARRIED: bool,
>(
    voxels: &[T],
    geometry: &SliceGeometry,
    outer_local: usize,
    row_index: usize,
    mut bits: u64,
    carry_runs: &mut [u8; 64],
    quads: &mut Vec<UnorientedQuad>,
) where
    T: MergeVoxel,
{
    let outer_coord = geometry.outer_base + outer_local as u32;

    append_quads(quads, bits.count_ones() as usize, |out| {
        while bits != 0 {
            let start = bits.trailing_zeros() as usize;
            let carry = if CARRIED { carry_runs[start] } else { 0 };
            let mut width = 1;

            if ALONG_ROW {
                // SAFETY (both reads): see `merge_value_at`; `start` and every
                // `next` are visible interior cells of this row.
                let value_at = |bit: usize| unsafe {
                    merge_value_at(voxels, row_index + bit * geometry.bit_stride)
                };
                let value = value_at(start);
                while start + width < geometry.bit_len {
                    let next = start + width;
                    if bits & (1 << next) == 0
                        || (CARRIED && carry_runs[next] != carry)
                        || value_at(next) != value
                    {
                        break;
                    }
                    if CARRIED {
                        carry_runs[next] = 0;
                    }
                    width += 1;
                }
            }

            bits &= !bit_mask(start, width);
            if CARRIED {
                carry_runs[start] = 0;
            }
            out.push(oriented_quad::<N_AXIS, BIT_IS_U>(
                geometry.n_coord,
                outer_coord - u32::from(carry),
                geometry.bit_base + start as u32,
                width as u32,
                u32::from(carry) + 1,
            ));
        }
    });
}

/// Emits every visible cell in `rows` as a `1x1` quad.
#[inline(always)]
pub(crate) fn emit_unit_slice<const N_AXIS: usize>(
    geometry: &SliceGeometry,
    rows: &[u64],
    quads: &mut Vec<UnorientedQuad>,
) {
    let total = rows.iter().map(|row| row.count_ones() as usize).sum();

    append_quads(quads, total, |out| {
        for (outer_local, &row_bits) in rows.iter().enumerate() {
            let outer_coord = geometry.outer_base + outer_local as u32;
            let mut bits = row_bits;
            while bits != 0 {
                let bit_local = bits.trailing_zeros();
                bits &= bits - 1;
                out.push(oriented_quad::<N_AXIS, true>(
                    geometry.n_coord,
                    outer_coord,
                    geometry.bit_base + bit_local,
                    1,
                    1,
                ));
            }
        }
    });
}

/// Reserves room for `max_quads` more quads, lets `fill` write them through a
/// [`QuadSink`], then commits however many were written.
///
/// Writing into reserved spare capacity keeps growth checks out of the hot
/// emission loops.
#[inline(always)]
fn append_quads(
    quads: &mut Vec<UnorientedQuad>,
    max_quads: usize,
    fill: impl FnOnce(&mut QuadSink<'_>),
) {
    quads.reserve(max_quads);
    let len = quads.len();
    let mut sink = QuadSink {
        spare: quads.spare_capacity_mut(),
        written: 0,
    };
    fill(&mut sink);
    let written = sink.written;
    // SAFETY: `QuadSink::push` initialized exactly the first `written` slots
    // of the spare capacity. If `fill` panics we never get here, and the
    // length simply stays at `len`.
    unsafe { quads.set_len(len + written) };
}

struct QuadSink<'a> {
    spare: &'a mut [MaybeUninit<UnorientedQuad>],
    written: usize,
}

impl QuadSink<'_> {
    #[inline(always)]
    fn push(&mut self, quad: UnorientedQuad) {
        // Indexing (not `get_unchecked`) keeps an over-count a panic, not UB.
        self.spare[self.written].write(quad);
        self.written += 1;
    }
}
