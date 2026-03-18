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
#[cfg(feature = "profiling")]
use std::time::{Duration, Instant};

/// Reusable output and scratch storage for [`binary_greedy_quads`].
///
/// The [`quads`](Self::quads) field uses the same public `block-mesh` data
/// model as [`block_mesh::greedy_quads`].
#[derive(Default)]
pub struct BinaryGreedyQuadsBuffer {
    pub quads: QuadBuffer,
    opaque_cols: [Vec<u64>; 3],
    trans_cols: [Vec<u64>; 3],
    visible_rows: Vec<u64>,
    scan_rows: Vec<u64>,
}

impl BinaryGreedyQuadsBuffer {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg(feature = "profiling")]
#[derive(Clone, Copy, Debug, Default)]
#[doc(hidden)]
pub struct BinaryGreedyPhaseProfile {
    pub source_masks: Duration,
    pub visibility_masks: Duration,
    pub scan_masks: Duration,
    pub unit_emit: Duration,
    pub merge: Duration,
    pub total: Duration,
}

/// Generates greedy quads using a binary-mask-backed implementation.
///
/// The public API mirrors [`block_mesh::greedy_quads`]. Output quads are stored
/// in [`BinaryGreedyQuadsBuffer::quads`].
///
/// Supported extents must have a padded interior of at most `62` voxels on each
/// axis. This corresponds to a maximum queried extent of `64` voxels on each
/// axis including the one-voxel padding required by `block-mesh`.
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
    #[cfg(feature = "profiling")]
    {
        let _ = binary_greedy_quads_impl(voxels, voxels_shape, min, max, faces, output, None);
        return;
    }

    #[cfg(not(feature = "profiling"))]
    {
        binary_greedy_quads_impl(voxels, voxels_shape, min, max, faces, output);
    }
}

#[cfg(feature = "profiling")]
#[doc(hidden)]
pub fn binary_greedy_quads_profile<T, S>(
    voxels: &[T],
    voxels_shape: &S,
    min: [u32; 3],
    max: [u32; 3],
    faces: &[OrientedBlockFace; 6],
    output: &mut BinaryGreedyQuadsBuffer,
) -> BinaryGreedyPhaseProfile
where
    T: MergeVoxel,
    S: Shape<3, Coord = u32>,
{
    binary_greedy_quads_impl(
        voxels,
        voxels_shape,
        min,
        max,
        faces,
        output,
        Some(BinaryGreedyPhaseProfile::default()),
    )
}

#[cfg(feature = "profiling")]
fn binary_greedy_quads_impl<T, S>(
    voxels: &[T],
    voxels_shape: &S,
    min: [u32; 3],
    max: [u32; 3],
    faces: &[OrientedBlockFace; 6],
    output: &mut BinaryGreedyQuadsBuffer,
    mut profile: Option<BinaryGreedyPhaseProfile>,
) -> BinaryGreedyPhaseProfile
where
    T: MergeVoxel,
    S: Shape<3, Coord = u32>,
{
    let total_start = Instant::now();
    binary_greedy_quads_impl_inner(voxels, voxels_shape, min, max, faces, output, &mut profile);
    let mut result = profile.unwrap_or_default();
    result.total = total_start.elapsed();
    result
}

#[cfg(not(feature = "profiling"))]
fn binary_greedy_quads_impl<T, S>(
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
    binary_greedy_quads_impl_inner(voxels, voxels_shape, min, max, faces, output);
}

#[cfg(feature = "profiling")]
fn binary_greedy_quads_impl_inner<T, S>(
    voxels: &[T],
    voxels_shape: &S,
    min: [u32; 3],
    max: [u32; 3],
    faces: &[OrientedBlockFace; 6],
    output: &mut BinaryGreedyQuadsBuffer,
    profile: &mut Option<BinaryGreedyPhaseProfile>,
) where
    T: MergeVoxel,
    S: Shape<3, Coord = u32>,
{
    assert_in_bounds(voxels, voxels_shape, min, max);

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
        scan_rows,
    } = output;

    let face_axes = (*faces).map(|face| FaceAxes::new(&face));

    let source_start = Instant::now();
    reset_columns(opaque_cols, trans_cols, query_shape);
    build_axis_columns(voxels, min, query_shape, strides, opaque_cols, trans_cols);
    if let Some(profile) = profile.as_mut() {
        profile.source_masks += source_start.elapsed();
    }

    for i in 0..6 {
        let axes = face_axes[i];
        mesh_face(
            voxels,
            interior_min,
            interior_shape,
            query_shape,
            interior_start_index,
            strides,
            axes,
            &opaque_cols[axes.u_axis],
            &trans_cols[axes.u_axis],
            visible_rows,
            scan_rows,
            &mut quads.groups[i],
            profile,
        );
    }
}

#[cfg(not(feature = "profiling"))]
fn binary_greedy_quads_impl_inner<T, S>(
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
    assert_in_bounds(voxels, voxels_shape, min, max);

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
        scan_rows,
    } = output;

    let face_axes = (*faces).map(|face| FaceAxes::new(&face));

    reset_columns(opaque_cols, trans_cols, query_shape);
    build_axis_columns(voxels, min, query_shape, strides, opaque_cols, trans_cols);

    for i in 0..6 {
        let axes = face_axes[i];
        mesh_face(
            voxels,
            interior_min,
            interior_shape,
            query_shape,
            interior_start_index,
            strides,
            axes,
            &opaque_cols[axes.u_axis],
            &trans_cols[axes.u_axis],
            visible_rows,
            scan_rows,
            &mut quads.groups[i],
        );
    }
}

fn mesh_face<T>(
    voxels: &[T],
    interior_min: [u32; 3],
    interior_shape: [u32; 3],
    query_shape: [usize; 3],
    interior_start_index: usize,
    strides: [usize; 3],
    axes: FaceAxes,
    opaque_cols: &[u64],
    trans_cols: &[u64],
    visible_rows: &mut Vec<u64>,
    scan_rows: &mut Vec<u64>,
    quads: &mut Vec<UnorientedQuad>,
    #[cfg(feature = "profiling")] profile: &mut Option<BinaryGreedyPhaseProfile>,
) where
    T: MergeVoxel,
{
    let v_len = interior_shape[axes.v_axis] as usize;
    let u_len = interior_shape[axes.u_axis] as usize;
    let interior_n_len = interior_shape[axes.n_axis] as usize;
    let interior_rows = interior_n_len * v_len;
    let u_stride = strides[axes.u_axis];
    let v_stride = strides[axes.v_axis];

    #[cfg(feature = "profiling")]
    let visibility_start = Instant::now();
    reset_visible_rows(visible_rows, interior_rows);
    build_visible_rows(
        query_shape,
        axes.u_axis,
        u_len,
        v_len,
        interior_n_len,
        axes.n_sign,
        axes.n_axis,
        axes.v_axis,
        opaque_cols,
        trans_cols,
        visible_rows,
    );
    #[cfg(feature = "profiling")]
    if let Some(profile) = profile.as_mut() {
        profile.visibility_masks += visibility_start.elapsed();
    }

    let (outer_axis, inner_axis) = scan_axes(axes.n_axis);
    let outer_len = interior_shape[outer_axis] as usize;
    let unit_only_slices = unit_only_slice_mask(visible_rows, v_len, interior_n_len);
    let all_slices_mask = (1u64 << interior_n_len) - 1;

    if unit_only_slices == all_slices_mask {
        #[cfg(feature = "profiling")]
        let unit_emit_start = Instant::now();
        for n_local in 0..interior_n_len {
            emit_unit_slice(
                interior_min,
                axes,
                outer_axis,
                inner_axis,
                u_len,
                interior_min[axes.n_axis] + n_local as u32,
                &visible_rows[n_local * v_len..n_local * v_len + v_len],
                quads,
            );
        }
        #[cfg(feature = "profiling")]
        if let Some(profile) = profile.as_mut() {
            profile.unit_emit += unit_emit_start.elapsed();
        }
        return;
    }

    #[cfg(feature = "profiling")]
    let scan_start = Instant::now();
    reset_scan_rows(scan_rows, interior_n_len * outer_len);
    build_scan_rows(
        visible_rows,
        scan_rows,
        v_len,
        outer_len,
        interior_n_len,
        axes,
        outer_axis,
        inner_axis,
    );
    #[cfg(feature = "profiling")]
    if let Some(profile) = profile.as_mut() {
        profile.scan_masks += scan_start.elapsed();
    }

    #[cfg(feature = "profiling")]
    let merge_start = Instant::now();
    for n_local in 0..interior_n_len {
        let n_index_base = interior_start_index + n_local * strides[axes.n_axis];
        let n_coord = interior_min[axes.n_axis] + n_local as u32;
        let slice_start = n_local * v_len;
        let slice_rows = &mut visible_rows[slice_start..slice_start + v_len];

        for outer_local in 0..outer_len {
            let mut pending = scan_rows[n_local * outer_len + outer_local];
            if pending == 0 {
                continue;
            }

            if outer_axis == axes.v_axis {
                let row_base_index = n_index_base + outer_local * strides[axes.v_axis];
                let v_coord = interior_min[axes.v_axis] + outer_local as u32;

                while pending != 0 {
                    let u_local = pending.trailing_zeros() as usize;
                    pending &= pending - 1;

                    let mut minimum = [0; 3];
                    minimum[axes.n_axis] = n_coord;
                    minimum[axes.u_axis] = interior_min[axes.u_axis] + u_local as u32;
                    minimum[axes.v_axis] = v_coord;

                    mesh_cell(
                        voxels,
                        minimum,
                        row_base_index + u_local * strides[axes.u_axis],
                        u_len,
                        v_len,
                        u_stride,
                        v_stride,
                        u_local,
                        outer_local,
                        slice_rows,
                        quads,
                    );
                }
            } else {
                let column_base_index = n_index_base + outer_local * strides[axes.u_axis];
                let u_coord = interior_min[axes.u_axis] + outer_local as u32;

                while pending != 0 {
                    let v_local = pending.trailing_zeros() as usize;
                    pending &= pending - 1;

                    let mut minimum = [0; 3];
                    minimum[axes.n_axis] = n_coord;
                    minimum[axes.u_axis] = u_coord;
                    minimum[axes.v_axis] = interior_min[axes.v_axis] + v_local as u32;

                    mesh_cell(
                        voxels,
                        minimum,
                        column_base_index + v_local * strides[axes.v_axis],
                        u_len,
                        v_len,
                        u_stride,
                        v_stride,
                        outer_local,
                        v_local,
                        slice_rows,
                        quads,
                    );
                }
            }
        }
    }
    #[cfg(feature = "profiling")]
    if let Some(profile) = profile.as_mut() {
        profile.merge += merge_start.elapsed();
    }
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
) where
    T: Voxel,
{
    let base_index = coord_to_index(min, strides);
    let qx = query_shape[0];
    let qy = query_shape[1];
    let qz = query_shape[2];
    let x_stride = strides[0];
    let y_stride = strides[1];
    let z_stride = strides[2];

    for z in 0..qz {
        let z_base_index = base_index + z * z_stride;
        let z_bit = 1u64 << z;
        let z_x_base = z * qx;
        let z_y_base = z * qy;

        for y in 0..qy {
            let mut index = z_base_index + y * y_stride;
            let x_col_index = y + z_y_base;
            let y_bit = 1u64 << y;
            let z_col_row_base = y * qx;

            for x in 0..qx {
                match unsafe { voxels.get_unchecked(index) }.get_visibility() {
                    VoxelVisibility::Empty => {}
                    VoxelVisibility::Translucent => {
                        trans_cols[0][x_col_index] |= 1u64 << x;
                        trans_cols[1][x + z_x_base] |= y_bit;
                        trans_cols[2][x + z_col_row_base] |= z_bit;
                    }
                    VoxelVisibility::Opaque => {
                        opaque_cols[0][x_col_index] |= 1u64 << x;
                        opaque_cols[1][x + z_x_base] |= y_bit;
                        opaque_cols[2][x + z_col_row_base] |= z_bit;
                    }
                }

                index += x_stride;
            }
        }
    }
}

fn build_visible_rows(
    query_shape: [usize; 3],
    u_axis: usize,
    u_len: usize,
    v_len: usize,
    interior_n_len: usize,
    n_sign: i32,
    n_axis: usize,
    v_axis: usize,
    opaque_cols: &[u64],
    trans_cols: &[u64],
    visible_rows: &mut [u64],
) {
    let interior_u_mask = bit_mask(0, u_len);

    for n_local in 0..interior_n_len {
        let src_n = n_local + 1;
        let neighbour_n = if n_sign > 0 { src_n + 1 } else { src_n - 1 };

        for v_local in 0..v_len {
            let mut coord = [0usize; 3];
            coord[n_axis] = src_n;
            coord[v_axis] = v_local + 1;
            let src = column_index_from_coord(u_axis, coord, query_shape);

            coord[n_axis] = neighbour_n;
            let neighbour = column_index_from_coord(u_axis, coord, query_shape);

            let src_opaque = opaque_cols[src];
            let src_trans = trans_cols[src];
            let neighbour_opaque = opaque_cols[neighbour];
            let neighbour_trans = trans_cols[neighbour];

            let raw_visible = (src_opaque & !neighbour_opaque)
                | (src_trans & !(neighbour_opaque | neighbour_trans));

            visible_rows[n_local * v_len + v_local] = (raw_visible >> 1) & interior_u_mask;
        }
    }
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

#[inline]
fn column_index_from_coord(bit_axis: usize, coord: [usize; 3], query_shape: [usize; 3]) -> usize {
    match bit_axis {
        0 => coord[1] + coord[2] * query_shape[1],
        1 => coord[0] + coord[2] * query_shape[0],
        2 => coord[0] + coord[1] * query_shape[0],
        _ => unreachable!(),
    }
}

fn mesh_cell<T>(
    voxels: &[T],
    minimum: [u32; 3],
    voxel_index: usize,
    u_len: usize,
    v_len: usize,
    u_stride: usize,
    v_stride: usize,
    u_local: usize,
    v_local: usize,
    visible_rows: &mut [u64],
    quads: &mut Vec<UnorientedQuad>,
) where
    T: MergeVoxel,
{
    let row_index = v_local;
    if visible_rows[row_index] & (1u64 << u_local) == 0 {
        return;
    }

    let quad_value = unsafe { voxels.get_unchecked(voxel_index) }.merge_value();

    let max_width = u_len - u_local;
    let quad_width = unsafe {
        row_width_start(
            voxels,
            visible_rows[row_index],
            u_local,
            max_width,
            voxel_index,
            u_stride,
            &quad_value,
        )
    };

    let mut quad_height = 1usize;
    let max_height = v_len - v_local;
    let mut next_row_index = row_index + 1;
    let mut next_voxel_index = voxel_index + v_stride;

    if quad_width == 1 {
        while quad_height < max_height {
            let row_bits = visible_rows[next_row_index];
            if row_bits & (1u64 << u_local) == 0 {
                break;
            }
            if unsafe { voxels.get_unchecked(next_voxel_index) }
                .merge_value()
                .ne(&quad_value)
            {
                break;
            }

            quad_height += 1;
            next_row_index += 1;
            next_voxel_index += v_stride;
        }
    } else {
        while quad_height < max_height {
            let width = unsafe {
                row_width(
                    voxels,
                    visible_rows[next_row_index],
                    u_local,
                    quad_width,
                    next_voxel_index,
                    u_stride,
                    &quad_value,
                )
            };

            if width < quad_width {
                break;
            }

            quad_height += 1;
            next_row_index += 1;
            next_voxel_index += v_stride;
        }
    }

    let visit_mask = bit_mask(u_local, quad_width);
    for row in 0..quad_height {
        visible_rows[row_index + row] &= !visit_mask;
    }

    quads.push(UnorientedQuad {
        minimum,
        width: quad_width as u32,
        height: quad_height as u32,
    });
}

unsafe fn row_width_start<T>(
    voxels: &[T],
    visible_row: u64,
    start_u: usize,
    max_width: usize,
    start_index: usize,
    u_stride: usize,
    quad_value: &T::MergeValue,
) -> usize
where
    T: MergeVoxel,
{
    let available = visible_row >> start_u;
    let run_width = (available.trailing_ones() as usize).min(max_width);
    if run_width <= 1 {
        return run_width;
    }

    let mut index = start_index + u_stride;
    let mut width = 1usize;

    while width < run_width {
        let voxel = voxels.get_unchecked(index);
        if !voxel.merge_value().eq(quad_value) {
            break;
        }

        width += 1;
        index += u_stride;
    }

    width
}

unsafe fn row_width<T>(
    voxels: &[T],
    visible_row: u64,
    start_u: usize,
    max_width: usize,
    start_index: usize,
    u_stride: usize,
    quad_value: &T::MergeValue,
) -> usize
where
    T: MergeVoxel,
{
    let available = visible_row >> start_u;
    let run_width = (available.trailing_ones() as usize).min(max_width);
    let mut index = start_index;
    let mut width = 0usize;

    while width < run_width {
        let voxel = voxels.get_unchecked(index);
        if !voxel.merge_value().eq(quad_value) {
            break;
        }

        width += 1;
        index += u_stride;
    }

    width
}

#[derive(Clone, Copy)]
struct FaceAxes {
    n_axis: usize,
    u_axis: usize,
    v_axis: usize,
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
        let v_axis = SignedAxis::from_vector(corners[2].as_ivec3() - corners[0].as_ivec3())
            .expect("axis-aligned face edge")
            .unsigned_axis();

        Self {
            n_axis: normal.unsigned_axis().index(),
            u_axis: u_axis.index(),
            v_axis: v_axis.index(),
            n_sign: normal.signum(),
        }
    }
}

fn scan_axes(n_axis: usize) -> (usize, usize) {
    match n_axis {
        0 => (2, 1),
        1 => (2, 0),
        2 => (1, 0),
        _ => unreachable!(),
    }
}

fn reset_scan_rows(scan_rows: &mut Vec<u64>, len: usize) {
    scan_rows.resize(len, 0);
    scan_rows.fill(0);
}

fn build_scan_rows(
    visible_rows: &[u64],
    scan_rows: &mut [u64],
    v_len: usize,
    outer_len: usize,
    interior_n_len: usize,
    axes: FaceAxes,
    outer_axis: usize,
    inner_axis: usize,
) {
    if outer_axis == axes.v_axis && inner_axis == axes.u_axis {
        scan_rows.copy_from_slice(visible_rows);
        return;
    }

    debug_assert_eq!(outer_axis, axes.u_axis);
    debug_assert_eq!(inner_axis, axes.v_axis);
    for n_local in 0..interior_n_len {
        let src = &visible_rows[n_local * v_len..n_local * v_len + v_len];
        let dst = &mut scan_rows[n_local * outer_len..n_local * outer_len + outer_len];
        for (v_local, &row_bits) in src.iter().enumerate() {
            let mut bits = row_bits;
            while bits != 0 {
                let u_local = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                dst[u_local] |= 1u64 << v_local;
            }
        }
    }
}

#[inline]
fn unit_only_slice_mask(visible_rows: &[u64], v_len: usize, interior_n_len: usize) -> u64 {
    let mut unit_only_mask = 0u64;

    for n_local in 0..interior_n_len {
        let slice_rows = &visible_rows[n_local * v_len..n_local * v_len + v_len];
        let mut previous_row = 0u64;
        let mut can_merge = false;

        for &row in slice_rows {
            if (row & (row >> 1)) != 0 || (row & previous_row) != 0 {
                can_merge = true;
                break;
            }
            previous_row = row;
        }

        if !can_merge {
            unit_only_mask |= 1u64 << n_local;
        }
    }

    unit_only_mask
}

fn emit_unit_slice(
    interior_min: [u32; 3],
    axes: FaceAxes,
    outer_axis: usize,
    inner_axis: usize,
    u_len: usize,
    n_coord: u32,
    visible_rows: &[u64],
    quads: &mut Vec<UnorientedQuad>,
) {
    if outer_axis == axes.v_axis && inner_axis == axes.u_axis {
        for (v_local, &row_bits) in visible_rows.iter().enumerate() {
            let mut bits = row_bits;
            let v_coord = interior_min[axes.v_axis] + v_local as u32;

            while bits != 0 {
                let u_local = bits.trailing_zeros() as usize;
                bits &= bits - 1;

                let mut minimum = [0; 3];
                minimum[axes.n_axis] = n_coord;
                minimum[axes.u_axis] = interior_min[axes.u_axis] + u_local as u32;
                minimum[axes.v_axis] = v_coord;

                quads.push(UnorientedQuad {
                    minimum,
                    width: 1,
                    height: 1,
                });
            }
        }
        return;
    }

    debug_assert_eq!(outer_axis, axes.u_axis);
    debug_assert_eq!(inner_axis, axes.v_axis);
    for u_local in 0..u_len {
        let u_bit = 1u64 << u_local;
        let u_coord = interior_min[axes.u_axis] + u_local as u32;
        for (v_local, &row_bits) in visible_rows.iter().enumerate() {
            if row_bits & u_bit == 0 {
                continue;
            }

            let mut minimum = [0; 3];
            minimum[axes.n_axis] = n_coord;
            minimum[axes.u_axis] = u_coord;
            minimum[axes.v_axis] = interior_min[axes.v_axis] + v_local as u32;

            quads.push(UnorientedQuad {
                minimum,
                width: 1,
                height: 1,
            });
        }
    }
}

fn reset_visible_rows(visible_rows: &mut Vec<u64>, interior_rows: usize) {
    visible_rows.resize(interior_rows, 0);
    visible_rows.fill(0);
}

fn bit_mask(start: usize, width: usize) -> u64 {
    ((1u64 << width) - 1) << start
}

fn coord_to_index(coord: [u32; 3], strides: [usize; 3]) -> usize {
    coord[0] as usize * strides[0] + coord[1] as usize * strides[1] + coord[2] as usize * strides[2]
}

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
