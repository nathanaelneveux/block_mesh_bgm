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
    scan_rows: Vec<u64>,
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
    let interior_start_index = coord_to_index(interior_min, strides);

    let BinaryGreedyQuadsBuffer {
        quads,
        opaque_rows,
        trans_rows,
        visible_rows,
        scan_rows,
    } = output;

    let face_axes = (*faces).map(|face| FaceAxes::new(&face));
    let mut processed = [false; 6];

    for i in 0..6 {
        if processed[i] {
            continue;
        }

        prepare_face_basis(
            voxels,
            min,
            max,
            interior_min,
            interior_shape,
            strides,
            face_axes[i],
            opaque_rows,
            trans_rows,
        );

        for j in i..6 {
            if processed[j] || !face_axes[j].same_basis(face_axes[i]) {
                continue;
            }

            mesh_face(
                voxels,
                interior_min,
                interior_shape,
                interior_start_index,
                strides,
                face_axes[j],
                opaque_rows,
                trans_rows,
                visible_rows,
                scan_rows,
                &mut quads.groups[j],
            );
            processed[j] = true;
        }
    }
}

fn prepare_face_basis<T>(
    voxels: &[T],
    min: [u32; 3],
    max: [u32; 3],
    interior_min: [u32; 3],
    interior_shape: [u32; 3],
    strides: [usize; 3],
    axes: FaceAxes,
    opaque_rows: &mut Vec<u64>,
    trans_rows: &mut Vec<u64>,
) where
    T: MergeVoxel,
{
    let u_len = interior_shape[axes.u_axis] as usize;
    let v_len = interior_shape[axes.v_axis] as usize;
    let query_n_len = (max[axes.n_axis] - min[axes.n_axis] + 1) as usize;
    let source_rows = query_n_len * v_len;

    reset_source_rows(opaque_rows, trans_rows, source_rows);

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
}

fn mesh_face<T>(
    voxels: &[T],
    interior_min: [u32; 3],
    interior_shape: [u32; 3],
    interior_start_index: usize,
    strides: [usize; 3],
    axes: FaceAxes,
    opaque_rows: &[u64],
    trans_rows: &[u64],
    visible_rows: &mut Vec<u64>,
    scan_rows: &mut Vec<u64>,
    quads: &mut Vec<UnorientedQuad>,
) where
    T: MergeVoxel,
{
    let v_len = interior_shape[axes.v_axis] as usize;
    let interior_n_len = interior_shape[axes.n_axis] as usize;
    let interior_rows = interior_n_len * v_len;

    reset_visible_rows(visible_rows, interior_rows);
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
    reset_scan_rows(scan_rows, interior_n_len * outer_len);
    build_scan_rows(
        visible_rows,
        scan_rows,
        interior_shape,
        axes,
        outer_axis,
        inner_axis,
    );

    for n_local in 0..interior_n_len {
        let n_index_base = interior_start_index + n_local * strides[axes.n_axis];
        let n_coord = interior_min[axes.n_axis] + n_local as u32;

        for outer_local in 0..outer_len {
            let scan_row_index = n_local * outer_len + outer_local;
            let mut pending = scan_rows[scan_row_index];
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
                        interior_shape,
                        strides,
                        axes,
                        n_local,
                        u_local,
                        outer_local,
                        visible_rows,
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
                        interior_shape,
                        strides,
                        axes,
                        n_local,
                        outer_local,
                        v_local,
                        visible_rows,
                        quads,
                    );
                }
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
    let n_stride = strides[axes.n_axis];
    let u_stride = strides[axes.u_axis];
    let v_stride = strides[axes.v_axis];
    let source_start_index = min[axes.n_axis] as usize * n_stride
        + interior_min[axes.u_axis] as usize * u_stride
        + interior_min[axes.v_axis] as usize * v_stride;

    for n_local in 0..query_n_len {
        let n_base_index = source_start_index + n_local * n_stride;
        for v_local in 0..v_len {
            let mut index = n_base_index + v_local * v_stride;
            let row_index = n_local * v_len + v_local;
            let mut opaque = 0u64;
            let mut trans = 0u64;

            for u_local in 0..u_len {
                match unsafe { voxels.get_unchecked(index) }.get_visibility() {
                    VoxelVisibility::Empty => {}
                    VoxelVisibility::Translucent => trans |= 1u64 << u_local,
                    VoxelVisibility::Opaque => opaque |= 1u64 << u_local,
                }

                index += u_stride;
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
    minimum: [u32; 3],
    voxel_index: usize,
    interior_shape: [u32; 3],
    strides: [usize; 3],
    axes: FaceAxes,
    n_local: usize,
    u_local: usize,
    v_local: usize,
    visible_rows: &mut [u64],
    quads: &mut Vec<UnorientedQuad>,
) where
    T: MergeVoxel,
{
    let v_len = interior_shape[axes.v_axis] as usize;
    let row_index = n_local * v_len + v_local;
    if visible_rows[row_index] & (1u64 << u_local) == 0 {
        return;
    }

    let quad_value = unsafe { voxels.get_unchecked(voxel_index) }.merge_value();

    let max_width = interior_shape[axes.u_axis] as usize - u_local;
    let quad_width = unsafe {
        row_width_start(
            voxels,
            visible_rows[row_index],
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
            next_voxel_index += strides[axes.v_axis];
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

    #[inline]
    fn same_basis(self, other: Self) -> bool {
        self.n_axis == other.n_axis && self.u_axis == other.u_axis && self.v_axis == other.v_axis
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

    if outer_axis == axes.v_axis && inner_axis == axes.u_axis {
        for n_local in 0..interior_n_len {
            let src = &visible_rows[n_local * v_len..n_local * v_len + v_len];
            let dst = &mut scan_rows[n_local * outer_len..n_local * outer_len + outer_len];
            dst.copy_from_slice(src);
        }
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

fn reset_source_rows(opaque_rows: &mut Vec<u64>, trans_rows: &mut Vec<u64>, source_rows: usize) {
    opaque_rows.resize(source_rows, 0);
    trans_rows.resize(source_rows, 0);

    opaque_rows.fill(0);
    trans_rows.fill(0);
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
