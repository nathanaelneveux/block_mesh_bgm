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
//! 2. Derive per-face visibility rows by comparing each source column with its
//!    neighbour column in the face normal direction.
//! 3. Arrange each face slice into scan rows, using a transposed bitset when
//!    the face's `(u, v)` axes do not match the preferred scan order.
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
    // Face-local visibility rows laid out as `[n][v] -> bits over u`.
    visible_rows: Vec<u64>,
    // Transposed view of `visible_rows` used when a face prefers scanning by
    // columns instead of rows.
    scan_rows: Vec<u64>,
    // Carry lengths used by the row-merging pass.
    carry_runs: Vec<u8>,
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
    binary_greedy_quads_impl(voxels, voxels_shape, min, max, faces, output)
}

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
        scan_rows,
        carry_runs,
    } = output;

    let face_axes = (*faces).map(|face| FaceAxes::new(&face));

    // Build reusable occupancy columns for all three possible bit axes up
    // front, then derive face visibility from those compact masks.
    reset_columns(opaque_cols, trans_cols, query_shape);
    let has_translucent =
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
            has_translucent,
            visible_rows,
            scan_rows,
            carry_runs,
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
    has_translucent: bool,
    visible_rows: &mut Vec<u64>,
    scan_rows: &mut Vec<u64>,
    carry_runs: &mut Vec<u8>,
    quads: &mut Vec<UnorientedQuad>,
) where
    T: MergeVoxel,
{
    let v_len = interior_shape[axes.v_axis] as usize;
    let u_len = interior_shape[axes.u_axis] as usize;
    let interior_n_len = interior_shape[axes.n_axis] as usize;
    let interior_rows = interior_n_len * v_len;
    let u_stride = strides[axes.u_axis];
    let v_stride = strides[axes.v_axis];

    reset_visible_rows(visible_rows, interior_rows);
    let unit_only = build_visible_rows(
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
        has_translucent,
        visible_rows,
    );

    if unit_only {
        for n_local in 0..interior_n_len {
            emit_unit_slice(
                interior_min,
                axes,
                interior_min[axes.n_axis] + n_local as u32,
                &visible_rows[n_local * v_len..n_local * v_len + v_len],
                quads,
            );
        }
        return;
    }

    // Some face orientations already match the row-major bit layout, while the
    // others are faster to scan by first transposing into `[n][u] -> bits over v`.
    let (outer_axis, inner_axis) = scan_axes(axes.n_axis);
    let scan_aligned = outer_axis == axes.v_axis && inner_axis == axes.u_axis;
    let outer_len = interior_shape[outer_axis] as usize;

    if !scan_aligned {
        reset_scan_rows(scan_rows, interior_n_len * outer_len);
        build_transposed_scan_rows(visible_rows, scan_rows, v_len, outer_len, interior_n_len);
    }

    if scan_aligned {
        mesh_face_carry(
            voxels,
            interior_min,
            interior_start_index,
            strides,
            axes,
            v_len,
            u_len,
            interior_n_len,
            v_stride,
            u_stride,
            axes.v_axis,
            axes.u_axis,
            visible_rows,
            carry_runs,
            quads,
        );
    } else {
        mesh_face_carry(
            voxels,
            interior_min,
            interior_start_index,
            strides,
            axes,
            u_len,
            v_len,
            interior_n_len,
            u_stride,
            v_stride,
            axes.u_axis,
            axes.v_axis,
            scan_rows,
            carry_runs,
            quads,
        );
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

        for y in 0..qy {
            let mut index = z_base_index + y * y_stride;
            let x_col_index = y + z_y_base;
            let y_bit = 1u64 << y;
            let z_col_row_base = y * qx;

            for x in 0..qx {
                match unsafe { voxels.get_unchecked(index) }.get_visibility() {
                    VoxelVisibility::Empty => {}
                    VoxelVisibility::Translucent => {
                        has_translucent = true;
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

    has_translucent
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
    has_translucent: bool,
    visible_rows: &mut [u64],
) -> bool {
    let interior_u_mask = bit_mask(0, u_len);
    let (base_offset, n_stride, v_stride) = match (u_axis, n_axis, v_axis) {
        (0, 1, 2) => (query_shape[1], 1, query_shape[1]),
        (0, 2, 1) => (1, query_shape[1], 1),
        (1, 0, 2) => (query_shape[0], 1, query_shape[0]),
        (1, 2, 0) => (1, query_shape[0], 1),
        (2, 0, 1) => (query_shape[0], 1, query_shape[0]),
        (2, 1, 0) => (1, query_shape[0], 1),
        _ => unreachable!(),
    };

    let mut unit_only = true;
    for n_local in 0..interior_n_len {
        let src_n = n_local + 1;
        let neighbour_n = if n_sign > 0 { src_n + 1 } else { src_n - 1 };
        let row_start = n_local * v_len;
        let mut src = base_offset + src_n * n_stride;
        let mut neighbour = base_offset + neighbour_n * n_stride;
        let mut previous_row = 0u64;

        for row in &mut visible_rows[row_start..row_start + v_len] {
            let src_opaque = opaque_cols[src];
            let neighbour_opaque = opaque_cols[neighbour];
            let raw_visible = if has_translucent {
                let src_trans = trans_cols[src];
                let neighbour_trans = trans_cols[neighbour];
                (src_opaque & !neighbour_opaque)
                    | (src_trans & !(neighbour_opaque | neighbour_trans))
            } else {
                src_opaque & !neighbour_opaque
            };

            let visible = (raw_visible >> 1) & interior_u_mask;
            unit_only &= (visible & (visible >> 1)) == 0;
            unit_only &= (visible & previous_row) == 0;
            *row = visible;
            previous_row = visible;
            src += v_stride;
            neighbour += v_stride;
        }
    }

    unit_only
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

fn build_transposed_scan_rows(
    visible_rows: &[u64],
    scan_rows: &mut [u64],
    v_len: usize,
    outer_len: usize,
    interior_n_len: usize,
) {
    // `visible_rows` is stored as `[n][v] -> bits over u`. Some face
    // orientations are cheaper to scan as `[n][u] -> bits over v`, so we build
    // that transposed view once per face instead of re-walking rows for every
    // probe.
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

fn emit_unit_slice(
    interior_min: [u32; 3],
    axes: FaceAxes,
    n_coord: u32,
    visible_rows: &[u64],
    quads: &mut Vec<UnorientedQuad>,
) {
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
