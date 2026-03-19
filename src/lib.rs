//! `block-mesh`-compatible binary greedy meshing.
//!
//! The main entry point is [`binary_greedy_quads`], which mirrors the call shape
//! of [`block_mesh::greedy_quads`] while reusing `block-mesh` output types.
//!
//! Compared with classic greedy meshing, this implementation uses compact
//! per-row bit masks to find visible faces and reuse row visibility during quad
//! growth. It operates directly on the caller's voxel slice, so it does not
//! require repacking voxels into an intermediate material buffer.
//!
//! # How the algorithm works
//!
//! The implementation is organized as four stages:
//!
//! 1. Build three axis-specific column tables where each `u64` packs occupancy
//!    bits along one axis of the queried extent.
//! 2. Derive face visibility rows by comparing each source column with its
//!    neighbours in the face normal direction. Both signs of the same normal
//!    axis are built together so the column data is only walked once.
//! 3. Store those rows directly in the preferred scan order for the carry
//!    merger, so no per-face transpose step is needed.
//! 4. Carry merge lengths forward from one row to the next so compatible runs
//!    collapse into large quads with only a small number of `merge_value()`
//!    comparisons.
//!
//! The `62`-voxel interior limit comes from stage 1: each queried axis includes
//! one voxel of padding on both sides, so the padded extent must fit in a
//! single `u64`.
//!
//! # Example
//!
//! ```
//! use block_mesh::ndshape::{ConstShape, ConstShape3u32};
//! use block_mesh::{
//!     MergeVoxel, Voxel, VoxelVisibility, RIGHT_HANDED_Y_UP_CONFIG,
//! };
//! use block_mesh_bgm::{binary_greedy_quads, BinaryGreedyQuadsBuffer};
//!
//! #[derive(Clone, Copy, Debug, Eq, PartialEq)]
//! struct BoolVoxel(bool);
//!
//! const EMPTY: BoolVoxel = BoolVoxel(false);
//! const FULL: BoolVoxel = BoolVoxel(true);
//!
//! impl Voxel for BoolVoxel {
//!     fn get_visibility(&self) -> VoxelVisibility {
//!         if *self == EMPTY {
//!             VoxelVisibility::Empty
//!         } else {
//!             VoxelVisibility::Opaque
//!         }
//!     }
//! }
//!
//! impl MergeVoxel for BoolVoxel {
//!     type MergeValue = Self;
//!
//!     fn merge_value(&self) -> Self::MergeValue {
//!         *self
//!     }
//! }
//!
//! type ChunkShape = ConstShape3u32<18, 18, 18>;
//!
//! let mut voxels = [EMPTY; ChunkShape::SIZE as usize];
//! for i in 0..ChunkShape::SIZE {
//!     let [x, y, z] = ChunkShape::delinearize(i);
//!     voxels[i as usize] = if ((x * x + y * y + z * z) as f32).sqrt() < 15.0 {
//!         FULL
//!     } else {
//!         EMPTY
//!     };
//! }
//!
//! let mut buffer = BinaryGreedyQuadsBuffer::new();
//! binary_greedy_quads(
//!     &voxels,
//!     &ChunkShape {},
//!     [0; 3],
//!     [17; 3],
//!     &RIGHT_HANDED_Y_UP_CONFIG.faces,
//!     &mut buffer,
//! );
//!
//! assert!(buffer.quads.num_quads() > 0);
//! ```

use block_mesh::ilattice::glam::UVec3;
use block_mesh::ilattice::prelude::Extent;
use block_mesh::{
    MergeVoxel, OrientedBlockFace, QuadBuffer, SignedAxis, UnorientedQuad, Voxel, VoxelVisibility,
};
use ndshape::Shape;

/// Reusable output and scratch storage for [`binary_greedy_quads`].
///
/// The [`quads`](Self::quads) field uses the same public `block-mesh` data
/// model as [`block_mesh::greedy_quads`].
#[derive(Default)]
pub struct BinaryGreedyQuadsBuffer {
    pub quads: QuadBuffer,
    // For each bit axis, store one `u64` per orthogonal column.
    // `opaque_cols[axis][column]` packs opaque occupancy bits along `axis`.
    opaque_cols: [Vec<u64>; 3],
    // Mirrors `opaque_cols`, but only for translucent voxels.
    trans_cols: [Vec<u64>; 3],
    // Face-local visibility rows laid out as `[n][row] -> bits over bit_axis`,
    // where `row` and `bit_axis` are chosen per face to match the preferred
    // merge scan order.
    visible_rows: Vec<u64>,
    // Second visibility buffer so both signs of the same normal axis can be
    // built in one pass over the source columns.
    visible_rows_alt: Vec<u64>,
    // Carry lengths used by the row-merging pass.
    carry_runs: Vec<u8>,
}

impl BinaryGreedyQuadsBuffer {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Hidden diagnostics for internal benchmarking and profiling.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BinaryGreedyStageTimings {
    pub column_build: std::time::Duration,
    pub visible_rows: std::time::Duration,
    pub unit_quads: std::time::Duration,
    pub carry_merge: std::time::Duration,
}

impl BinaryGreedyStageTimings {
    #[doc(hidden)]
    pub fn total(&self) -> std::time::Duration {
        self.column_build + self.visible_rows + self.unit_quads + self.carry_merge
    }
}

/// Generates greedy quads using a binary-mask-backed implementation.
///
/// The public API mirrors [`block_mesh::greedy_quads`]. Output quads are stored
/// in [`BinaryGreedyQuadsBuffer::quads`].
///
/// This implementation preserves visible-face geometry and usually
/// stays very close to `block_mesh::greedy_quads` quad counts, but it does not
/// guarantee the exact same quad partition.
///
/// Supported extents must have a padded interior of at most `62` voxels on each
/// axis. This corresponds to a maximum queried extent of `64` voxels on each
/// axis including the one-voxel padding required by `block-mesh`.
///
/// Conceptually, the queried extent is split into:
///
/// - the padded query bounds `[min, max]`, which are only used to determine
///   face visibility, and
/// - the interior bounds `[min + 1, max - 1]`, which are the voxels that may
///   actually contribute quads to the mesh.
pub fn binary_greedy_quads<T, S>(
    voxels: &[T],
    voxels_shape: &S,
    min: [u32; 3],
    max: [u32; 3],
    faces: &[OrientedBlockFace; 6],
    output: &mut BinaryGreedyQuadsBuffer,
) where
    T: MergeVoxel,
    S: Shape<3, Coord = u32>,
{
    let mut unused_timings = BinaryGreedyStageTimings::default();
    binary_greedy_quads_impl::<T, S, false>(
        voxels,
        voxels_shape,
        min,
        max,
        faces,
        output,
        &mut unused_timings,
    )
}

/// Hidden diagnostics entry point that records coarse stage timings.
#[doc(hidden)]
pub fn binary_greedy_quads_profile<T, S>(
    voxels: &[T],
    voxels_shape: &S,
    min: [u32; 3],
    max: [u32; 3],
    faces: &[OrientedBlockFace; 6],
    output: &mut BinaryGreedyQuadsBuffer,
    timings: &mut BinaryGreedyStageTimings,
) where
    T: MergeVoxel,
    S: Shape<3, Coord = u32>,
{
    binary_greedy_quads_impl::<T, S, true>(voxels, voxels_shape, min, max, faces, output, timings)
}

fn binary_greedy_quads_impl<T, S, const PROFILE: bool>(
    voxels: &[T],
    voxels_shape: &S,
    min: [u32; 3],
    max: [u32; 3],
    faces: &[OrientedBlockFace; 6],
    output: &mut BinaryGreedyQuadsBuffer,
    timings: &mut BinaryGreedyStageTimings,
) where
    T: MergeVoxel,
    S: Shape<3, Coord = u32>,
{
    let mut stage_timings = BinaryGreedyStageTimings::default();
    assert_in_bounds(voxels, voxels_shape, min, max);

    // `block-mesh` requires a one-voxel border around the actual meshed region.
    // If any axis only contains padding, there is no interior to mesh.
    if max
        .iter()
        .zip(min.iter())
        .any(|(&max_axis, &min_axis)| max_axis <= min_axis + 1)
    {
        return;
    }

    let shape = voxels_shape.as_array();
    let strides = shape_strides(voxels_shape, shape);
    let interior_min = [min[0] + 1, min[1] + 1, min[2] + 1];
    let interior_max = [max[0] - 1, max[1] - 1, max[2] - 1];
    let interior_shape = [
        interior_max[0] - interior_min[0] + 1,
        interior_max[1] - interior_min[1] + 1,
        interior_max[2] - interior_min[2] + 1,
    ];
    let query_shape = [
        interior_shape[0] as usize + 2,
        interior_shape[1] as usize + 2,
        interior_shape[2] as usize + 2,
    ];

    for extent in interior_shape {
        assert!(
            extent <= 62,
            "binary_greedy_quads supports at most 62 interior voxels per axis; got interior_shape={interior_shape:?}"
        );
    }

    output.quads.reset();
    let interior_start_index = coord_to_index(interior_min, strides);

    let BinaryGreedyQuadsBuffer {
        quads,
        opaque_cols,
        trans_cols,
        visible_rows,
        visible_rows_alt,
        carry_runs,
    } = output;

    let face_axes = (*faces).map(|face| FaceAxes::new(&face));
    let face_indices = build_face_index_map(face_axes);

    // Build reusable occupancy columns for all three possible bit axes up
    // front, then derive face visibility from those compact masks.
    reset_columns(opaque_cols, trans_cols, query_shape);
    let has_translucent = if PROFILE {
        let start = std::time::Instant::now();
        let has_translucent =
            build_axis_columns(voxels, min, query_shape, strides, opaque_cols, trans_cols);
        stage_timings.column_build += start.elapsed();
        has_translucent
    } else {
        build_axis_columns(voxels, min, query_shape, strides, opaque_cols, trans_cols)
    };

    for n_axis in 0..3 {
        let neg_face_index = face_indices[n_axis][0];
        let pos_face_index = face_indices[n_axis][1];
        let neg_axes = face_axes[neg_face_index];
        let pos_axes = face_axes[pos_face_index];
        let (row_axis, bit_axis) = scan_axes(n_axis);
        let row_count = interior_shape[row_axis] as usize;
        let bit_count = interior_shape[bit_axis] as usize;
        let interior_n_len = interior_shape[n_axis] as usize;
        let interior_rows = interior_n_len * row_count;

        reset_visible_rows(visible_rows, interior_rows);
        reset_visible_rows(visible_rows_alt, interior_rows);
        let (neg_unit_only, pos_unit_only) = if PROFILE {
            let start = std::time::Instant::now();
            let result = build_visible_row_pair(
                query_shape,
                bit_axis,
                bit_count,
                row_count,
                interior_n_len,
                n_axis,
                row_axis,
                &opaque_cols[bit_axis],
                &trans_cols[bit_axis],
                has_translucent,
                visible_rows,
                visible_rows_alt,
            );
            stage_timings.visible_rows += start.elapsed();
            result
        } else {
            build_visible_row_pair(
                query_shape,
                bit_axis,
                bit_count,
                row_count,
                interior_n_len,
                n_axis,
                row_axis,
                &opaque_cols[bit_axis],
                &trans_cols[bit_axis],
                has_translucent,
                visible_rows,
                visible_rows_alt,
            )
        };

        mesh_face_rows::<T, PROFILE>(
            voxels,
            interior_min,
            interior_shape,
            interior_start_index,
            strides,
            visible_rows,
            neg_unit_only,
            neg_axes,
            row_axis,
            bit_axis,
            carry_runs,
            &mut quads.groups[neg_face_index],
            &mut stage_timings,
        );
        mesh_face_rows::<T, PROFILE>(
            voxels,
            interior_min,
            interior_shape,
            interior_start_index,
            strides,
            visible_rows_alt,
            pos_unit_only,
            pos_axes,
            row_axis,
            bit_axis,
            carry_runs,
            &mut quads.groups[pos_face_index],
            &mut stage_timings,
        );
    }

    if PROFILE {
        *timings = stage_timings;
    }
}

fn mesh_face_rows<T, const PROFILE: bool>(
    voxels: &[T],
    interior_min: [u32; 3],
    interior_shape: [u32; 3],
    interior_start_index: usize,
    strides: [usize; 3],
    visible_rows: &[u64],
    unit_only: bool,
    axes: FaceAxes,
    row_axis: usize,
    bit_axis: usize,
    carry_runs: &mut Vec<u8>,
    quads: &mut Vec<UnorientedQuad>,
    timings: &mut BinaryGreedyStageTimings,
) where
    T: MergeVoxel,
{
    let row_count = interior_shape[row_axis] as usize;
    let bit_count = interior_shape[bit_axis] as usize;
    let interior_n_len = interior_shape[axes.n_axis] as usize;
    let row_stride = strides[row_axis];
    let bit_stride = strides[bit_axis];

    if unit_only {
        let start = if PROFILE {
            Some(std::time::Instant::now())
        } else {
            None
        };
        for n_local in 0..interior_n_len {
            emit_unit_slice(
                interior_min,
                axes,
                row_axis,
                bit_axis,
                interior_min[axes.n_axis] + n_local as u32,
                &visible_rows[n_local * row_count..n_local * row_count + row_count],
                quads,
            );
        }
        if let Some(start) = start {
            timings.unit_quads += start.elapsed();
        }
        return;
    }

    let start = if PROFILE {
        Some(std::time::Instant::now())
    } else {
        None
    };
    mesh_face_carry(
        voxels,
        interior_min,
        interior_start_index,
        strides,
        axes,
        row_count,
        bit_count,
        interior_n_len,
        row_stride,
        bit_stride,
        row_axis,
        bit_axis,
        visible_rows,
        carry_runs,
        quads,
    );
    if let Some(start) = start {
        timings.carry_merge += start.elapsed();
    }
}

fn mesh_face_carry<T>(
    voxels: &[T],
    interior_min: [u32; 3],
    interior_start_index: usize,
    strides: [usize; 3],
    axes: FaceAxes,
    row_count: usize,
    bit_count: usize,
    interior_n_len: usize,
    row_stride: usize,
    bit_stride: usize,
    outer_axis: usize,
    bit_axis: usize,
    rows: &[u64],
    carry_runs: &mut Vec<u8>,
    quads: &mut Vec<UnorientedQuad>,
) where
    T: MergeVoxel,
{
    reset_carry_runs(carry_runs, bit_count);

    for n_local in 0..interior_n_len {
        carry_runs.fill(0);
        let n_index_base = interior_start_index + n_local * strides[axes.n_axis];
        let n_coord = interior_min[axes.n_axis] + n_local as u32;
        let slice_start = n_local * row_count;
        let slice_rows = &rows[slice_start..slice_start + row_count];

        for outer_local in 0..row_count {
            let mut bits_here = slice_rows[outer_local];
            if bits_here == 0 {
                continue;
            }

            let bits_next = if outer_local + 1 < row_count {
                slice_rows[outer_local + 1]
            } else {
                0
            };
            let row_base_index = n_index_base + outer_local * row_stride;

            while bits_here != 0 {
                let bit_local = bits_here.trailing_zeros() as usize;
                let bit = 1u64 << bit_local;
                let voxel_index = row_base_index + bit_local * bit_stride;
                let quad_value = unsafe { voxels.get_unchecked(voxel_index) }.merge_value();

                if unsafe {
                    row_bit_continues(voxels, bits_next, bit, voxel_index, row_stride, &quad_value)
                } {
                    carry_runs[bit_local] += 1;
                    bits_here &= !bit;
                    continue;
                }

                let run_length = carry_runs[bit_local] as usize + 1;
                let mut run_width = 1usize;

                while bit_local + run_width < bit_count {
                    let next_bit_local = bit_local + run_width;
                    let next_bit = 1u64 << next_bit_local;
                    if bits_here & next_bit == 0
                        || carry_runs[next_bit_local] != carry_runs[bit_local]
                    {
                        break;
                    }

                    let next_index = row_base_index + next_bit_local * bit_stride;
                    if unsafe { voxels.get_unchecked(next_index) }
                        .merge_value()
                        .ne(&quad_value)
                    {
                        break;
                    }

                    if unsafe {
                        row_bit_continues(
                            voxels,
                            bits_next,
                            next_bit,
                            next_index,
                            row_stride,
                            &quad_value,
                        )
                    } {
                        break;
                    }

                    carry_runs[next_bit_local] = 0;
                    run_width += 1;
                }

                bits_here &= !bit_mask(bit_local, run_width);

                let start_outer = outer_local - carry_runs[bit_local] as usize;
                let mut minimum = [0; 3];
                minimum[axes.n_axis] = n_coord;
                minimum[outer_axis] = interior_min[outer_axis] + start_outer as u32;
                minimum[bit_axis] = interior_min[bit_axis] + bit_local as u32;

                let (width, height) = if bit_axis == axes.u_axis {
                    (run_width as u32, run_length as u32)
                } else {
                    (run_length as u32, run_width as u32)
                };

                quads.push(UnorientedQuad {
                    minimum,
                    width,
                    height,
                });

                carry_runs[bit_local] = 0;
            }
        }
    }
}

#[inline]
fn reset_carry_runs(carry_runs: &mut Vec<u8>, len: usize) {
    carry_runs.resize(len, 0);
    carry_runs.fill(0);
}

#[inline]
unsafe fn row_bit_continues<T>(
    voxels: &[T],
    next_row_bits: u64,
    bit: u64,
    voxel_index: usize,
    row_stride: usize,
    quad_value: &T::MergeValue,
) -> bool
where
    T: MergeVoxel,
{
    next_row_bits & bit != 0
        && voxels
            .get_unchecked(voxel_index + row_stride)
            .merge_value()
            .eq(quad_value)
}

fn reset_columns(
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

fn build_axis_columns<T>(
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
    let base_index = coord_to_index(min, strides);
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

fn build_visible_row_pair(
    query_shape: [usize; 3],
    bit_axis: usize,
    bit_len: usize,
    row_count: usize,
    interior_n_len: usize,
    n_axis: usize,
    row_axis: usize,
    opaque_cols: &[u64],
    trans_cols: &[u64],
    has_translucent: bool,
    neg_rows: &mut [u64],
    pos_rows: &mut [u64],
) -> (bool, bool) {
    let interior_bit_mask = bit_mask(0, bit_len);
    let (base_offset, n_stride, row_stride) = match (bit_axis, n_axis, row_axis) {
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
    for n_local in 0..interior_n_len {
        let src_n = n_local + 1;
        let neg_n = src_n - 1;
        let pos_n = src_n + 1;
        let row_start = n_local * row_count;
        let mut src = base_offset + src_n * n_stride;
        let mut neg = base_offset + neg_n * n_stride;
        let mut pos = base_offset + pos_n * n_stride;
        let mut previous_neg = 0u64;
        let mut previous_pos = 0u64;

        for row_local in 0..row_count {
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
            neg_rows[row_start + row_local] = neg_visible;
            pos_rows[row_start + row_local] = pos_visible;
            previous_neg = neg_visible;
            previous_pos = pos_visible;
            src += row_stride;
            neg += row_stride;
            pos += row_stride;
        }
    }

    (neg_unit_only, pos_unit_only)
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

#[derive(Clone, Copy)]
struct FaceAxes {
    n_axis: usize,
    u_axis: usize,
    n_sign: i32,
}

impl FaceAxes {
    fn new(face: &OrientedBlockFace) -> Self {
        let unit_quad = UnorientedQuad {
            minimum: [0; 3],
            width: 1,
            height: 1,
        };
        let corners = face.quad_corners(&unit_quad);
        let normal = SignedAxis::from_vector(face.signed_normal()).expect("axis-aligned face");
        let u_axis = SignedAxis::from_vector(corners[1].as_ivec3() - corners[0].as_ivec3())
            .expect("axis-aligned face edge")
            .unsigned_axis();
        let _v_axis = SignedAxis::from_vector(corners[2].as_ivec3() - corners[0].as_ivec3())
            .expect("axis-aligned face edge")
            .unsigned_axis();

        Self {
            n_axis: normal.unsigned_axis().index(),
            u_axis: u_axis.index(),
            n_sign: normal.signum(),
        }
    }
}

fn build_face_index_map(face_axes: [FaceAxes; 6]) -> [[usize; 2]; 3] {
    let mut indices = [[usize::MAX; 2]; 3];

    for (i, axes) in face_axes.into_iter().enumerate() {
        let sign_index = if axes.n_sign < 0 { 0 } else { 1 };
        indices[axes.n_axis][sign_index] = i;
    }

    for axis_indices in indices {
        debug_assert!(axis_indices[0] != usize::MAX);
        debug_assert!(axis_indices[1] != usize::MAX);
    }

    indices
}

fn scan_axes(n_axis: usize) -> (usize, usize) {
    match n_axis {
        0 => (2, 1),
        1 => (2, 0),
        2 => (1, 0),
        _ => unreachable!(),
    }
}

fn emit_unit_slice(
    interior_min: [u32; 3],
    axes: FaceAxes,
    row_axis: usize,
    bit_axis: usize,
    n_coord: u32,
    visible_rows: &[u64],
    quads: &mut Vec<UnorientedQuad>,
) {
    let additional_quads = visible_rows
        .iter()
        .map(|row_bits| row_bits.count_ones() as usize)
        .sum::<usize>();
    let base_len = quads.len();
    quads.reserve(additional_quads);
    let spare = quads.spare_capacity_mut();
    let mut written = 0usize;

    for (row_local, &row_bits) in visible_rows.iter().enumerate() {
        let mut bits = row_bits;
        let row_coord = interior_min[row_axis] + row_local as u32;

        while bits != 0 {
            let bit_local = bits.trailing_zeros() as usize;
            bits &= bits - 1;

            let mut minimum = [0; 3];
            minimum[axes.n_axis] = n_coord;
            minimum[row_axis] = row_coord;
            minimum[bit_axis] = interior_min[bit_axis] + bit_local as u32;

            spare[written].write(UnorientedQuad {
                minimum,
                width: 1,
                height: 1,
            });
            written += 1;
        }
    }

    unsafe {
        quads.set_len(base_len + written);
    }
}

fn reset_visible_rows(visible_rows: &mut Vec<u64>, interior_rows: usize) {
    visible_rows.resize(interior_rows, 0);
}

#[inline]
fn bit_mask(start: usize, width: usize) -> u64 {
    ((1u64 << width) - 1) << start
}

#[inline]
fn coord_to_index(coord: [u32; 3], strides: [usize; 3]) -> usize {
    coord[0] as usize * strides[0] + coord[1] as usize * strides[1] + coord[2] as usize * strides[2]
}

// `ndshape` exposes `linearize`, but not the raw per-axis strides that make
// the hot loops cheaper to write. Derive them once from unit offsets.
fn shape_strides<S>(shape: &S, shape_array: [u32; 3]) -> [usize; 3]
where
    S: Shape<3, Coord = u32>,
{
    [
        if shape_array[0] > 1 {
            shape.linearize([1, 0, 0]) as usize
        } else {
            0
        },
        if shape_array[1] > 1 {
            shape.linearize([0, 1, 0]) as usize
        } else {
            0
        },
        if shape_array[2] > 1 {
            shape.linearize([0, 0, 1]) as usize
        } else {
            0
        },
    ]
}

fn assert_in_bounds<T, S>(voxels: &[T], voxels_shape: &S, min: [u32; 3], max: [u32; 3])
where
    S: Shape<3, Coord = u32>,
{
    assert!(
        voxels_shape.size() as usize <= voxels.len(),
        "voxel buffer size {:?} is less than the shape size {:?}; would cause access out of bounds",
        voxels.len(),
        voxels_shape.size()
    );

    let shape = voxels_shape.as_array();
    let local_extent = Extent::from_min_and_shape(UVec3::ZERO, UVec3::from(shape));
    local_extent
        .check_positive_shape()
        .unwrap_or_else(|| panic!("Invalid shape={shape:?}"));

    let query_extent = Extent::from_min_and_max(UVec3::from(min), UVec3::from(max));
    query_extent.check_positive_shape().unwrap_or_else(|| {
        panic!("Invalid extent min={min:?} max={max:?}; has non-positive shape")
    });

    assert!(
        query_extent.is_subset_of(&local_extent),
        "min={min:?} max={max:?} would access out of bounds"
    );
}
