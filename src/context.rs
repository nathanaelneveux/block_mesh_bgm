//! Query validation, tiling, and precomputed layout facts.
//!
//! The mesher uses the same derived facts over and over: interior bounds,
//! padded query size, per-axis strides, and the face-index lookup for the
//! caller's `OrientedBlockFace` array. [`Query`] validates the caller's request
//! once; [`MeshingContext`] holds the facts for one tile of it.
//!
//! # Terminology
//!
//! The source names the three axes of a face scan:
//!
//! - `n_axis`: the face normal axis
//! - `outer_axis`: the axis that advances from one row to the next within a
//!   face slice
//! - `bit_axis`: the axis packed into a row's `u64` bitset
//!
//! A query `[min, max]` is the **query extent**, including one voxel of
//! padding; `[min + 1, max - 1]` is the **interior extent**, whose faces are
//! emitted.
//!
//! # Tiling
//!
//! Each axis of a tile is packed into one `u64`, so a tile's interior is at
//! most [`MAX_TILE_INTERIOR`] voxels wide. Larger queries are split into
//! evenly sized tiles. Each tile reads its neighbours' voxels as its padding,
//! so visibility and AO stay exact across seams; the only cost is that quads
//! never merge across a seam.

use block_mesh::ilattice::glam::UVec3;
use block_mesh::ilattice::prelude::Extent;
use block_mesh::OrientedBlockFace;
use ndshape::Shape;

use crate::face::{build_face_index_map, scan_axes, FaceAxes};

/// The widest interior a single tile can mesh: 62 voxels plus one voxel of
/// padding on each side fill a `u64` exactly.
pub(crate) const MAX_TILE_INTERIOR: u32 = 62;

/// A validated caller query, split into tiles of at most
/// [`MAX_TILE_INTERIOR`] interior voxels per axis.
pub(crate) struct Query {
    pub(crate) strides: [usize; 3],
    pub(crate) face_axes: [FaceAxes; 6],
    /// `face_indices[n_axis][0 for negative, 1 for positive]` is the caller's
    /// face-group index for that face.
    pub(crate) face_indices: [[usize; 2]; 3],
    interior_min: [u32; 3],
    interior_max: [u32; 3],
}

impl Query {
    /// Validates the caller's query. Returns `None` when the interior is
    /// empty, so there is nothing to mesh.
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
        if (0..3).any(|axis| max[axis] <= min[axis] + 1) {
            return None;
        }

        let face_axes = (*faces).map(|face| FaceAxes::new(&face));
        Some(Self {
            strides: shape_strides(voxels_shape),
            face_axes,
            face_indices: build_face_index_map(face_axes),
            interior_min: min.map(|coord| coord + 1),
            interior_max: max.map(|coord| coord - 1),
        })
    }

    /// Calls `visit(interior_min, interior_max)` for every tile, in Z, Y, X
    /// order. Bounds are inclusive.
    pub(crate) fn for_each_tile(&self, mut visit: impl FnMut([u32; 3], [u32; 3])) {
        let tiles =
            [0, 1, 2].map(|axis| AxisTiles::new(self.interior_min[axis], self.interior_max[axis]));
        for z in 0..tiles[2].count {
            for y in 0..tiles[1].count {
                for x in 0..tiles[0].count {
                    let [x, y, z] = [tiles[0].get(x), tiles[1].get(y), tiles[2].get(z)];
                    visit([x.0, y.0, z.0], [x.1, y.1, z.1]);
                }
            }
        }
    }
}

/// An inclusive range `[lo, hi]` split into `count` near-equal tiles.
#[derive(Clone, Copy)]
struct AxisTiles {
    lo: u32,
    count: u32,
    base_len: u32,
    /// The first `longer` tiles are one voxel longer than `base_len`.
    longer: u32,
}

impl AxisTiles {
    fn new(lo: u32, hi: u32) -> Self {
        let len = hi - lo + 1;
        let count = len.div_ceil(MAX_TILE_INTERIOR);
        Self {
            lo,
            count,
            base_len: len / count,
            longer: len % count,
        }
    }

    /// Inclusive bounds of tile `index`.
    fn get(self, index: u32) -> (u32, u32) {
        let start = self.lo + index * self.base_len + index.min(self.longer);
        let len = self.base_len + u32::from(index < self.longer);
        (start, start + len - 1)
    }
}

/// Precomputed facts about one tile of a query.
pub(crate) struct MeshingContext {
    pub(crate) strides: [usize; 3],
    pub(crate) interior_min: [u32; 3],
    pub(crate) interior_shape: [u32; 3],
    /// Interior shape plus one voxel of padding on each side.
    pub(crate) query_shape: [usize; 3],
    pub(crate) interior_start_index: usize,
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
    /// Derives the indexing facts for the tile with inclusive interior bounds
    /// `[interior_min, interior_max]`.
    pub(crate) fn for_tile(query: &Query, interior_min: [u32; 3], interior_max: [u32; 3]) -> Self {
        let interior_shape = [0, 1, 2].map(|axis| interior_max[axis] - interior_min[axis] + 1);
        debug_assert!(interior_shape.iter().all(|&len| len <= MAX_TILE_INTERIOR));

        Self {
            strides: query.strides,
            interior_min,
            interior_shape,
            query_shape: interior_shape.map(|len| len as usize + 2),
            interior_start_index: coord_to_index(interior_min, query.strides),
        }
    }

    /// The padded query's minimum corner.
    #[inline]
    pub(crate) fn query_min(&self) -> [u32; 3] {
        self.interior_min.map(|coord| coord - 1)
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
pub(crate) fn shape_strides<S>(shape: &S) -> [usize; 3]
where
    S: Shape<3, Coord = u32>,
{
    let dims = shape.as_array();
    let units = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    // A length-1 axis is never stepped along, and `linearize` of a unit
    // offset on it would point past the shape.
    std::array::from_fn(|axis| {
        if dims[axis] > 1 {
            shape.linearize(units[axis]) as usize
        } else {
            0
        }
    })
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

#[cfg(test)]
mod tests {
    use super::{AxisTiles, MAX_TILE_INTERIOR};

    #[test]
    fn axis_tiles_cover_the_range_exactly_with_balanced_sizes() {
        for lo in [0, 1, 5] {
            for len in 1..=400 {
                let hi = lo + len - 1;
                let tiles = AxisTiles::new(lo, hi);
                let mut next = lo;
                let mut sizes = Vec::new();
                for index in 0..tiles.count {
                    let (start, end) = tiles.get(index);
                    assert_eq!(start, next);
                    sizes.push(end - start + 1);
                    next = end + 1;
                }
                assert_eq!(next, hi + 1);
                assert!(sizes.iter().all(|&size| size <= MAX_TILE_INTERIOR));
                assert!(sizes.iter().max().unwrap() - sizes.iter().min().unwrap() <= 1);
            }
        }
    }
}
