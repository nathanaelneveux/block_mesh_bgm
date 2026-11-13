//! Per-quad vertex ambient occlusion for meshes from
//! [`binary_greedy_quads_ao_safe`](crate::binary_greedy_quads_ao_safe).
//!
//! AO-safe meshing guarantees that every unit face inside a merged quad has
//! the same four vertex AO values. Rendering therefore only needs to sample
//! the eight voxels around one face of each final quad, not every voxel.

use block_mesh::{OrientedBlockFace, UnorientedQuad};
use ndshape::Shape;

/// Samples the standard four-corner voxel AO for quads of one face
/// orientation.
///
/// Construct one sampler per face group, outside the quad loop; it
/// precomputes the linear offsets of the eight voxels around a face in the
/// plane just outside it. Each [`sample`](Self::sample) then costs eight
/// loads and one table lookup.
///
/// ```
/// # use block_mesh::ndshape::{ConstShape, ConstShape3u32};
/// # use block_mesh::{MergeVoxel, Voxel, VoxelVisibility, RIGHT_HANDED_Y_UP_CONFIG};
/// # #[derive(Clone, Copy, PartialEq, Eq)]
/// # struct Block(bool);
/// # impl Voxel for Block {
/// #     fn get_visibility(&self) -> VoxelVisibility {
/// #         if self.0 { VoxelVisibility::Opaque } else { VoxelVisibility::Empty }
/// #     }
/// # }
/// # impl MergeVoxel for Block {
/// #     type MergeValue = bool;
/// #     fn merge_value(&self) -> bool { self.0 }
/// # }
/// use block_mesh_bgm::{binary_greedy_quads_ao_safe, BinaryGreedyQuadsBuffer, FaceAoSampler};
///
/// type ChunkShape = ConstShape3u32<34, 34, 34>;
/// let voxels = [Block(false); ChunkShape::SIZE as usize];
/// let faces = RIGHT_HANDED_Y_UP_CONFIG.faces;
///
/// let mut buffer = BinaryGreedyQuadsBuffer::new();
/// binary_greedy_quads_ao_safe(&voxels, &ChunkShape {}, [0; 3], [33; 3], &faces, &mut buffer);
///
/// for (group, face) in buffer.quads.groups.iter().zip(faces) {
///     let sampler = FaceAoSampler::for_shape(face, &ChunkShape {});
///     for quad in group {
///         // 0 = fully occluded ..= 3 = open, in `quad_corners` vertex order.
///         let vertex_ao: [u8; 4] = sampler.sample(&voxels, quad.minimum, |block| block.0);
///     }
/// }
/// ```
#[derive(Clone, Copy, Debug)]
pub struct FaceAoSampler {
    strides: [usize; 3],
    neighbor_offsets: [isize; 8],
    backward_extent: usize,
    forward_extent: usize,
}

impl FaceAoSampler {
    /// Builds the sampler for one oriented block face of voxels laid out by
    /// `shape`.
    pub fn for_shape<S>(face: OrientedBlockFace, shape: &S) -> Self
    where
        S: Shape<3, Coord = u32>,
    {
        Self::new(face, crate::context::shape_strides(shape))
    }

    /// Builds the sampler for one oriented block face of a dense linear voxel
    /// array with the given per-axis `strides`. For a `34³` padded chunk in
    /// X/Y/Z order that is `[1, 34, 34 * 34]`.
    pub fn new(face: OrientedBlockFace, strides: [usize; 3]) -> Self {
        let unit_quad = UnorientedQuad {
            minimum: [0; 3],
            width: 1,
            height: 1,
        };
        let corners = face.quad_corners(&unit_quad);
        let normal = face.signed_normal();
        let u = corners[1].as_ivec3() - corners[0].as_ivec3();
        let v = corners[2].as_ivec3() - corners[0].as_ivec3();

        // Ring order around the exterior sample plane:
        //
        //     7 -- 6 -- 5
        //     |         |
        //     0         4
        //     |         |
        //     1 -- 2 -- 3
        //
        // This lines up with block_mesh's [minU/minV, maxU/minV,
        // minU/maxV, maxU/maxV] vertex order.
        let neighbor_offsets = [
            normal - u,
            normal - u - v,
            normal - v,
            normal + u - v,
            normal + u,
            normal + u + v,
            normal + v,
            normal - u + v,
        ]
        .map(|direction| vector_offset(direction.to_array(), strides));
        let min_offset = *neighbor_offsets.iter().min().unwrap();
        let max_offset = *neighbor_offsets.iter().max().unwrap();
        let backward_extent = if min_offset < 0 {
            min_offset.unsigned_abs()
        } else {
            0
        };
        let forward_extent = if max_offset > 0 {
            max_offset as usize
        } else {
            0
        };

        Self {
            strides,
            neighbor_offsets,
            backward_extent,
            forward_extent,
        }
    }

    /// Samples the four vertex AO values for a quad whose source voxel is
    /// `minimum`, in the vertex order of [`OrientedBlockFace::quad_corners`].
    ///
    /// Each value is `0` (fully occluded) to `3` (unoccluded), following the
    /// standard voxel AO rule: two occluding sides make a corner fully dark
    /// regardless of the diagonal. `is_opaque` decides which voxels occlude.
    ///
    /// For a merged quad this is exact only if the quad came from
    /// [`binary_greedy_quads_ao_safe`](crate::binary_greedy_quads_ao_safe),
    /// which keeps AO uniform across each opaque quad.
    ///
    /// # Panics
    ///
    /// Panics if any of the eight neighbours lies outside `voxels`. Meshing
    /// requires a one-voxel border, so quads from a padded query always have
    /// their whole neighbourhood in bounds.
    #[inline]
    pub fn sample<T>(
        &self,
        voxels: &[T],
        minimum: [u32; 3],
        is_opaque: impl Fn(&T) -> bool,
    ) -> [u8; 4] {
        let base_index = minimum[0] as usize * self.strides[0]
            + minimum[1] as usize * self.strides[1]
            + minimum[2] as usize * self.strides[2];
        assert!(
            base_index >= self.backward_extent
                && self.forward_extent < voxels.len()
                && base_index < voxels.len() - self.forward_extent,
            "AO neighborhood is outside the padded voxel array"
        );

        let mut opaque_mask = 0u8;
        for (bit, offset) in self.neighbor_offsets.into_iter().enumerate() {
            let index = base_index.wrapping_add_signed(offset);
            // SAFETY: the backward/forward range check covers every
            // precomputed neighbor offset.
            opaque_mask |= u8::from(is_opaque(unsafe { voxels.get_unchecked(index) })) << bit;
        }

        AO_BY_RING[opaque_mask as usize]
    }
}

#[inline]
fn vector_offset([x, y, z]: [i32; 3], strides: [usize; 3]) -> isize {
    [x, y, z]
        .into_iter()
        .zip(strides)
        .try_fold(0isize, |offset, (component, stride)| {
            let stride = isize::try_from(stride).ok()?;
            offset.checked_add((component as isize).checked_mul(stride)?)
        })
        .expect("AO strides exceed the supported range")
}

#[inline]
#[cfg(test)]
fn ring_aos(opaque: [bool; 8]) -> [u8; 4] {
    let mut mask = 0u8;
    let mut bit = 0;
    while bit < 8 {
        mask |= (opaque[bit] as u8) << bit;
        bit += 1;
    }
    AO_BY_RING[mask as usize]
}

const AO_BY_RING: [[u8; 4]; 256] = build_ao_lookup();

const fn build_ao_lookup() -> [[u8; 4]; 256] {
    let mut lookup = [[0; 4]; 256];
    let mut mask = 0;
    while mask < 256 {
        lookup[mask] = [
            masked_vertex_ao(mask, 0, 1, 2),
            masked_vertex_ao(mask, 2, 3, 4),
            masked_vertex_ao(mask, 6, 7, 0),
            masked_vertex_ao(mask, 4, 5, 6),
        ];
        mask += 1;
    }
    lookup
}

const fn masked_vertex_ao(mask: usize, side1: usize, corner: usize, side2: usize) -> u8 {
    vertex_ao(
        mask & (1 << side1) != 0,
        mask & (1 << corner) != 0,
        mask & (1 << side2) != 0,
    )
}

const fn vertex_ao(side1: bool, corner: bool, side2: bool) -> u8 {
    if side1 && side2 {
        0
    } else {
        3 - side1 as u8 - corner as u8 - side2 as u8
    }
}

#[cfg(test)]
mod tests {
    use block_mesh::{SignedAxis, RIGHT_HANDED_Y_UP_CONFIG};

    use super::{ring_aos, FaceAoSampler};

    const EDGE: usize = 7;
    const STRIDES: [usize; 3] = [1, EDGE, EDGE * EDGE];

    #[test]
    fn vertex_values_match_the_standard_voxel_ao_rule() {
        assert_eq!(ring_aos([false; 8]), [3; 4]);

        for ring_bits in 0u16..=255 {
            let opaque = core::array::from_fn(|bit| ring_bits & (1 << bit) != 0);
            let values = ring_aos(opaque);
            for value in values {
                assert!(value <= 3);
            }
        }
    }

    #[test]
    fn sampler_matches_coordinate_reference_for_every_face_orientation() {
        let voxels = core::array::from_fn::<_, { EDGE * EDGE * EDGE }, _>(|index| {
            let x = index % EDGE;
            let y = index / EDGE % EDGE;
            let z = index / (EDGE * EDGE);
            (x * 17 + y * 7 + z * 13) % 5 <= 1
        });
        let minimum = [3, 3, 3];
        let mut faces = RIGHT_HANDED_Y_UP_CONFIG.faces.to_vec();
        faces.extend([
            block_mesh::OrientedBlockFace::canonical(SignedAxis::PosX),
            block_mesh::OrientedBlockFace::canonical(SignedAxis::PosY),
            block_mesh::OrientedBlockFace::canonical(SignedAxis::PosZ),
        ]);

        for face in faces {
            let sampler = FaceAoSampler::new(face, STRIDES);
            let actual = sampler.sample(&voxels, minimum, |&opaque| opaque);
            let expected = coordinate_reference(face, minimum, &voxels);
            assert_eq!(actual, expected);
        }
    }

    fn coordinate_reference(
        face: block_mesh::OrientedBlockFace,
        minimum: [u32; 3],
        voxels: &[bool; EDGE * EDGE * EDGE],
    ) -> [u8; 4] {
        let quad = block_mesh::UnorientedQuad {
            minimum,
            width: 1,
            height: 1,
        };
        let corners = face.quad_corners(&quad);
        let normal = face.signed_normal();
        let u = corners[1].as_ivec3() - corners[0].as_ivec3();
        let v = corners[2].as_ivec3() - corners[0].as_ivec3();
        let center =
            block_mesh::ilattice::glam::IVec3::from(minimum.map(|value| value as i32)) + normal;
        let coords = [
            center - u,
            center - u - v,
            center - v,
            center + u - v,
            center + u,
            center + u + v,
            center + v,
            center - u + v,
        ];
        let opaque = coords.map(|coord| {
            let [x, y, z] = coord.to_array().map(|value| value as usize);
            voxels[x + y * EDGE + z * EDGE * EDGE]
        });
        ring_aos(opaque)
    }
}
