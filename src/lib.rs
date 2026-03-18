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

/// Reusable output and scratch storage for [`binary_greedy_quads`].
///
/// The [`quads`](Self::quads) field uses the same public `block-mesh` data
/// model as [`block_mesh::greedy_quads`].
#[derive(Default)]
pub struct BinaryGreedyQuadsBuffer {
    pub quads: QuadBuffer,
    opaque_rows: Vec<u64>,
    trans_rows: Vec<u64>,
    visible_rows: Vec<u64>,
    visited_rows: Vec<u64>,
}

impl BinaryGreedyQuadsBuffer {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }
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

    for extent in interior_shape {
        assert!(
            extent <= 62,
            "binary_greedy_quads supports at most 62 interior voxels per axis; got interior_shape={interior_shape:?}"
        );
    }

    output.quads.reset();

    let BinaryGreedyQuadsBuffer {
        quads,
        opaque_rows,
        trans_rows,
        visible_rows,
        visited_rows,
    } = output;

    for (group, face) in quads.groups.iter_mut().zip(faces.iter()) {
        mesh_face(
            voxels,
            min,
            max,
            interior_min,
            interior_shape,
            strides,
            face,
            opaque_rows,
            trans_rows,
            visible_rows,
            visited_rows,
            group,
        );
    }
}

fn mesh_face<T>(
    voxels: &[T],
    min: [u32; 3],
    max: [u32; 3],
    interior_min: [u32; 3],
    interior_shape: [u32; 3],
    strides: [usize; 3],
    face: &OrientedBlockFace,
    opaque_rows: &mut Vec<u64>,
    trans_rows: &mut Vec<u64>,
    visible_rows: &mut Vec<u64>,
    visited_rows: &mut Vec<u64>,
    quads: &mut Vec<UnorientedQuad>,
) where
    T: MergeVoxel,
{
    let axes = FaceAxes::new(face);
    let u_len = interior_shape[axes.u_axis] as usize;
    let v_len = interior_shape[axes.v_axis] as usize;
    let interior_n_len = interior_shape[axes.n_axis] as usize;
    let query_n_len = (max[axes.n_axis] - min[axes.n_axis] + 1) as usize;
    let source_rows = query_n_len * v_len;
    let interior_rows = interior_n_len * v_len;

    reset_rows(
        opaque_rows,
        trans_rows,
        visible_rows,
        visited_rows,
        source_rows,
        interior_rows,
    );

    build_source_rows(
        voxels,
        min,
        interior_min,
        v_len,
        u_len,
        query_n_len,
        strides,
        axes,
        opaque_rows,
        trans_rows,
    );
    build_visible_rows(
        v_len,
        interior_n_len,
        axes.n_sign,
        opaque_rows,
        trans_rows,
        visible_rows,
    );

    let (outer_axis, inner_axis) = scan_axes(axes.n_axis);
    let outer_len = interior_shape[outer_axis] as usize;
    reset_scan_rows(opaque_rows, interior_n_len * outer_len);
    build_scan_rows(
        visible_rows,
        opaque_rows,
        interior_shape,
        axes,
        outer_axis,
        inner_axis,
    );

    for n_local in 0..interior_n_len {
        for outer_local in 0..outer_len {
            let mut pending = opaque_rows[n_local * outer_len + outer_local];
            while pending != 0 {
                let inner_local = pending.trailing_zeros() as usize;
                pending &= pending - 1;

                let mut coord = [0; 3];
                coord[axes.n_axis] = interior_min[axes.n_axis] + n_local as u32;
                coord[outer_axis] = interior_min[outer_axis] + outer_local as u32;
                coord[inner_axis] = interior_min[inner_axis] + inner_local as u32;

                mesh_cell(
                    voxels,
                    coord,
                    interior_min,
                    interior_shape,
                    strides,
                    axes,
                    n_local,
                    visible_rows,
                    visited_rows,
                    quads,
                );
            }
        }
    }
}

fn build_source_rows<T>(
    voxels: &[T],
    min: [u32; 3],
    interior_min: [u32; 3],
    v_len: usize,
    u_len: usize,
    query_n_len: usize,
    strides: [usize; 3],
    axes: FaceAxes,
    opaque_rows: &mut [u64],
    trans_rows: &mut [u64],
) where
    T: Voxel,
{
    for n_local in 0..query_n_len {
        for v_local in 0..v_len {
            let mut coord = [0; 3];
            coord[axes.n_axis] = min[axes.n_axis] + n_local as u32;
            coord[axes.u_axis] = interior_min[axes.u_axis];
            coord[axes.v_axis] = interior_min[axes.v_axis] + v_local as u32;

            let mut index = coord_to_index(coord, strides);
            let row_index = n_local * v_len + v_local;
            let mut opaque = 0u64;
            let mut trans = 0u64;

            for u_local in 0..u_len {
                match unsafe { voxels.get_unchecked(index) }.get_visibility() {
                    VoxelVisibility::Empty => {}
                    VoxelVisibility::Translucent => trans |= 1u64 << u_local,
                    VoxelVisibility::Opaque => opaque |= 1u64 << u_local,
                }

                index += strides[axes.u_axis];
            }

            opaque_rows[row_index] = opaque;
            trans_rows[row_index] = trans;
        }
    }
}

fn build_visible_rows(
    v_len: usize,
    interior_n_len: usize,
    n_sign: i32,
    opaque_rows: &[u64],
    trans_rows: &[u64],
    visible_rows: &mut [u64],
) {
    for n_local in 0..interior_n_len {
        let src_n = n_local + 1;
        let neighbour_n = if n_sign > 0 { src_n + 1 } else { src_n - 1 };

        for v_local in 0..v_len {
            let src = src_n * v_len + v_local;
            let neighbour = neighbour_n * v_len + v_local;

            let src_opaque = opaque_rows[src];
            let src_trans = trans_rows[src];
            let neighbour_opaque = opaque_rows[neighbour];
            let neighbour_trans = trans_rows[neighbour];

            visible_rows[n_local * v_len + v_local] = (src_opaque & !neighbour_opaque)
                | (src_trans & !(neighbour_opaque | neighbour_trans));
        }
    }
}

fn mesh_cell<T>(
    voxels: &[T],
    coord: [u32; 3],
    interior_min: [u32; 3],
    interior_shape: [u32; 3],
    strides: [usize; 3],
    axes: FaceAxes,
    n_local: usize,
    visible_rows: &mut [u64],
    visited_rows: &mut [u64],
    quads: &mut Vec<UnorientedQuad>,
) where
    T: MergeVoxel,
{
    let u_local = (coord[axes.u_axis] - interior_min[axes.u_axis]) as usize;
    let v_local = (coord[axes.v_axis] - interior_min[axes.v_axis]) as usize;
    let v_len = interior_shape[axes.v_axis] as usize;
    let row_index = n_local * v_len + v_local;
    let bit = 1u64 << u_local;

    if visible_rows[row_index] & bit == 0 || visited_rows[row_index] & bit != 0 {
        return;
    }

    let voxel_index = coord_to_index(coord, strides);
    let quad_value = unsafe { voxels.get_unchecked(voxel_index) }.merge_value();

    let max_width = interior_shape[axes.u_axis] as usize - u_local;
    let quad_width = unsafe {
        row_width(
            voxels,
            visible_rows[row_index],
            visited_rows[row_index],
            u_local,
            max_width,
            voxel_index,
            strides[axes.u_axis],
            &quad_value,
        )
    };

    let mut quad_height = 1usize;
    let max_height = interior_shape[axes.v_axis] as usize - v_local;
    let mut next_row_index = row_index + 1;
    let mut next_voxel_index = voxel_index + strides[axes.v_axis];

    while quad_height < max_height {
        let width = unsafe {
            row_width(
                voxels,
                visible_rows[next_row_index],
                visited_rows[next_row_index],
                u_local,
                quad_width,
                next_voxel_index,
                strides[axes.u_axis],
                &quad_value,
            )
        };

        if width < quad_width {
            break;
        }

        quad_height += 1;
        next_row_index += 1;
        next_voxel_index += strides[axes.v_axis];
    }

    let visit_mask = bit_mask(u_local, quad_width);
    for row in 0..quad_height {
        visited_rows[row_index + row] |= visit_mask;
    }

    quads.push(UnorientedQuad {
        minimum: coord,
        width: quad_width as u32,
        height: quad_height as u32,
    });
}

unsafe fn row_width<T>(
    voxels: &[T],
    visible_row: u64,
    visited_row: u64,
    start_u: usize,
    max_width: usize,
    start_index: usize,
    u_stride: usize,
    quad_value: &T::MergeValue,
) -> usize
where
    T: MergeVoxel,
{
    let available = (visible_row & !visited_row) >> start_u;
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
    interior_shape: [u32; 3],
    axes: FaceAxes,
    outer_axis: usize,
    inner_axis: usize,
) {
    let v_len = interior_shape[axes.v_axis] as usize;
    let outer_len = interior_shape[outer_axis] as usize;
    let interior_n_len = interior_shape[axes.n_axis] as usize;

    for n_local in 0..interior_n_len {
        for v_local in 0..v_len {
            let mut bits = visible_rows[n_local * v_len + v_local];
            while bits != 0 {
                let u_local = bits.trailing_zeros() as usize;
                bits &= bits - 1;

                let mut coord = [0usize; 3];
                coord[axes.n_axis] = n_local;
                coord[axes.u_axis] = u_local;
                coord[axes.v_axis] = v_local;

                let outer_local = coord[outer_axis];
                let inner_local = coord[inner_axis];
                scan_rows[n_local * outer_len + outer_local] |= 1u64 << inner_local;
            }
        }
    }
}

fn reset_rows(
    opaque_rows: &mut Vec<u64>,
    trans_rows: &mut Vec<u64>,
    visible_rows: &mut Vec<u64>,
    visited_rows: &mut Vec<u64>,
    source_rows: usize,
    interior_rows: usize,
) {
    opaque_rows.resize(source_rows, 0);
    trans_rows.resize(source_rows, 0);
    visible_rows.resize(interior_rows, 0);
    visited_rows.resize(interior_rows, 0);

    opaque_rows.fill(0);
    trans_rows.fill(0);
    visible_rows.fill(0);
    visited_rows.fill(0);
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
