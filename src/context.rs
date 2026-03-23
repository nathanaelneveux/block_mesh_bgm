//! Query validation and precomputed layout facts.
//!
//! The mesher uses the same derived facts over and over: interior bounds,
//! padded query size, per-axis strides, and the face-index lookup for the
//! caller's `OrientedBlockFace` array. Computing them once keeps the main entry
//! point short enough to read top-to-bottom.

use block_mesh::ilattice::glam::UVec3;
use block_mesh::ilattice::prelude::Extent;
use block_mesh::OrientedBlockFace;
use ndshape::Shape;

use crate::face::{build_face_index_map, scan_axes, FaceAxes};

/// Precomputed facts about one call to `binary_greedy_quads`.
pub(crate) struct MeshingContext {
    pub(crate) strides: [usize; 3],
    pub(crate) interior_min: [u32; 3],
    pub(crate) interior_shape: [u32; 3],
    pub(crate) query_shape: [usize; 3],
    pub(crate) interior_start_index: usize,
    pub(crate) face_axes: [FaceAxes; 6],
    pub(crate) face_indices: [[usize; 2]; 3],
}

/// The scan layout for one family of faces that shares the same normal axis.
#[derive(Clone, Copy)]
pub(crate) struct SlicePlan {
    /// The axis that advances from one row to the next in a face slice.
    pub(crate) outer_axis: usize,
    /// The axis packed into the row's `u64` bitset.
    pub(crate) bit_axis: usize,
    /// Number of rows per face slice.
    pub(crate) outer_len: usize,
    /// Number of bits stored in each row.
    pub(crate) bit_len: usize,
    /// Number of slices along the normal axis.
    pub(crate) n_len: usize,
}

impl SlicePlan {
    #[inline]
    pub(crate) fn total_rows(self) -> usize {
        self.n_len * self.outer_len
    }
}

impl MeshingContext {
    /// Validates the caller's query and derives the indexing facts used by the
    /// later stages.
    pub(crate) fn new<T, S>(
        voxels: &[T],
        voxels_shape: &S,
        min: [u32; 3],
        max: [u32; 3],
        faces: &[OrientedBlockFace; 6],
    ) -> Option<Self>
    where
        S: Shape<3, Coord = u32>,
    {
        assert_in_bounds(voxels, voxels_shape, min, max);

        // `block-mesh` requires one voxel of padding on every side of the
        // meshed region. If any axis contains only padding, there is nothing to emit.
        if max
            .iter()
            .zip(min.iter())
            .any(|(&max_axis, &min_axis)| max_axis <= min_axis + 1)
        {
            return None;
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

        let interior_start_index = coord_to_index(interior_min, strides);
        let face_axes = (*faces).map(|face| FaceAxes::new(&face));
        let face_indices = build_face_index_map(face_axes);

        Some(Self {
            strides,
            interior_min,
            interior_shape,
            query_shape,
            interior_start_index,
            face_axes,
            face_indices,
        })
    }

    /// Describes how to scan faces whose normal points along `n_axis`.
    #[inline]
    pub(crate) fn slice_plan(&self, n_axis: usize) -> SlicePlan {
        let (outer_axis, bit_axis) = scan_axes(n_axis);

        SlicePlan {
            outer_axis,
            bit_axis,
            outer_len: self.interior_shape[outer_axis] as usize,
            bit_len: self.interior_shape[bit_axis] as usize,
            n_len: self.interior_shape[n_axis] as usize,
        }
    }
}

#[inline]
pub(crate) fn coord_to_index(coord: [u32; 3], strides: [usize; 3]) -> usize {
    coord[0] as usize * strides[0] + coord[1] as usize * strides[1] + coord[2] as usize * strides[2]
}

// `ndshape` exposes `linearize`, but not the raw per-axis strides that make
// the hot loops cheaper to write. Derive them once from unit offsets.
pub(crate) fn shape_strides<S>(shape: &S, shape_array: [u32; 3]) -> [usize; 3]
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
