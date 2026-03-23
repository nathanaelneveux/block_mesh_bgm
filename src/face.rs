//! Face-orientation helpers.
//!
//! `block-mesh` lets callers choose the orientation of each face group. The
//! binary mesher still needs to recover a stable internal convention:
//!
//! - which axis is normal to the face
//! - which in-face axis should be treated as the packed bit axis
//! - whether the caller's face belongs to the negative or positive sign of that
//!   normal axis

use std::mem::MaybeUninit;

use block_mesh::{OrientedBlockFace, SignedAxis, UnorientedQuad};

/// The parts of an `OrientedBlockFace` that matter to the binary mesher.
#[derive(Clone, Copy)]
pub(crate) struct FaceAxes {
    pub(crate) n_axis: usize,
    pub(crate) u_axis: usize,
    pub(crate) n_sign: i32,
}

impl FaceAxes {
    /// Extracts the face normal axis, the `u` axis, and the face sign.
    pub(crate) fn new(face: &OrientedBlockFace) -> Self {
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

        Self {
            n_axis: normal.unsigned_axis().index(),
            u_axis: u_axis.index(),
            n_sign: normal.signum(),
        }
    }
}

/// Builds a lookup table from normal axis + sign to the caller's face-group index.
pub(crate) fn build_face_index_map(face_axes: [FaceAxes; 6]) -> [[usize; 2]; 3] {
    let mut indices = [[usize::MAX; 2]; 3];

    for (face_index, axes) in face_axes.into_iter().enumerate() {
        let sign_index = if axes.n_sign < 0 { 0 } else { 1 };
        indices[axes.n_axis][sign_index] = face_index;
    }

    for axis_indices in indices {
        debug_assert!(axis_indices[0] != usize::MAX);
        debug_assert!(axis_indices[1] != usize::MAX);
    }

    indices
}

/// Chooses the two in-face scan axes for faces whose normal points along `n_axis`.
///
/// The return value is `(outer_axis, bit_axis)`.
#[inline]
pub(crate) fn scan_axes(n_axis: usize) -> (usize, usize) {
    match n_axis {
        0 => (2, 1),
        1 => (2, 0),
        2 => (1, 0),
        _ => unreachable!(),
    }
}

/// Writes one merged quad using the mesher's internal axis naming.
#[inline(always)]
pub(crate) fn write_quad<const N_AXIS: usize, const BIT_IS_U: bool>(
    quad: &mut MaybeUninit<UnorientedQuad>,
    n_coord: u32,
    outer_coord: u32,
    bit_coord: u32,
    run_width: u32,
    run_length: u32,
) {
    let minimum = match N_AXIS {
        0 => [n_coord, bit_coord, outer_coord],
        1 => [bit_coord, n_coord, outer_coord],
        2 => [bit_coord, outer_coord, n_coord],
        _ => unreachable!(),
    };
    let (width, height) = if BIT_IS_U {
        (run_width, run_length)
    } else {
        (run_length, run_width)
    };

    quad.write(UnorientedQuad {
        minimum,
        width,
        height,
    });
}

/// Writes a `1x1` quad in the same coordinate convention as [`write_quad`].
#[inline(always)]
pub(crate) fn write_unit_quad<const N_AXIS: usize>(
    quad: &mut MaybeUninit<UnorientedQuad>,
    n_coord: u32,
    outer_coord: u32,
    bit_coord: u32,
) {
    let minimum = match N_AXIS {
        0 => [n_coord, bit_coord, outer_coord],
        1 => [bit_coord, n_coord, outer_coord],
        2 => [bit_coord, outer_coord, n_coord],
        _ => unreachable!(),
    };
    quad.write(UnorientedQuad {
        minimum,
        width: 1,
        height: 1,
    });
}
