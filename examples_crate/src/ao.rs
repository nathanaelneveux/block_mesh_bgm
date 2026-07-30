//! Fast ambient-occlusion sampling for meshes built from padded voxel arrays.
//!
//! AO-safe meshing guarantees that every vertex covered by a merged quad has a
//! compatible AO value. That means rendering only needs to sample the eight
//! voxels around one face of each final quad.

use block_mesh::{OrientedBlockFace, UnorientedQuad};

/// Precomputed linear offsets for sampling the AO neighborhood of one face.
///
/// Construct one sampler per face group, outside the quad loop. `strides` must
/// describe a dense linear voxel array in X/Y/Z order; for a `34³` padded chunk
/// that is `[1, 34, 34 * 34]`.
#[derive(Clone, Copy, Debug)]
pub struct FaceAoSampler {
    strides: [usize; 3],
    neighbor_offsets: [isize; 8],
    backward_extent: usize,
    forward_extent: usize,
}

impl FaceAoSampler {
    /// Builds the sampler for one oriented block face.
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

    /// Samples the four AO values for a quad whose source voxel is `minimum`.
    ///
    /// The padded voxel array must contain the complete one-voxel AO
    /// neighborhood. The function checks that contract once, then reads all
    /// eight neighbors without repeated bounds checks.
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
