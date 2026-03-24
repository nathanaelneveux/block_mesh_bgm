use std::collections::BTreeMap;

use block_mesh::ndshape::{RuntimeShape, Shape};
use block_mesh::{
    greedy_quads, GreedyQuadsBuffer, MergeVoxel, OrientedBlockFace, QuadBuffer, SignedAxis, Voxel,
    VoxelVisibility, RIGHT_HANDED_Y_UP_CONFIG,
};
use block_mesh_bgm::{
    binary_greedy_quads, binary_greedy_quads_with_config, BinaryGreedyQuadsBuffer,
    BinaryGreedyQuadsConfig,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TestVoxel {
    visibility: VoxelVisibility,
    material: u8,
    facing: u8,
}

impl TestVoxel {
    const fn empty(facing: u8) -> Self {
        Self {
            visibility: VoxelVisibility::Empty,
            material: 0,
            facing,
        }
    }

    const fn opaque(material: u8, facing: u8) -> Self {
        Self {
            visibility: VoxelVisibility::Opaque,
            material,
            facing,
        }
    }

    const fn translucent(material: u8, facing: u8) -> Self {
        Self {
            visibility: VoxelVisibility::Translucent,
            material,
            facing,
        }
    }
}

impl Default for TestVoxel {
    fn default() -> Self {
        Self::empty(0)
    }
}

impl Voxel for TestVoxel {
    fn get_visibility(&self) -> VoxelVisibility {
        self.visibility
    }
}

impl MergeVoxel for TestVoxel {
    type MergeValue = (VoxelVisibility, u8);

    fn merge_value(&self) -> Self::MergeValue {
        (self.visibility, self.material)
    }
}

#[derive(Clone)]
struct CarryCase {
    name: &'static str,
    shape: RuntimeShape<u32, 3>,
    voxels: Vec<TestVoxel>,
    min: [u32; 3],
    max: [u32; 3],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct UnitFace {
    face_index: u8,
    cell: [u32; 3],
    merge_key: u16,
}

#[derive(Clone, Copy)]
struct FaceAxes {
    n_axis: usize,
    u_axis: usize,
    v_axis: usize,
    n_sign: i32,
}

#[test]
fn matches_block_mesh_geometry_for_opaque_sphere() {
    let shape = RuntimeShape::<u32, 3>::new([18, 18, 18]);
    let voxels = make_voxels([18, 18, 18], |x, y, z| {
        let cx = 8.5f32;
        let cy = 8.5f32;
        let cz = 8.5f32;
        let dx = x as f32 - cx;
        let dy = y as f32 - cy;
        let dz = z as f32 - cz;
        if dx * dx + dy * dy + dz * dz <= 45.0 {
            TestVoxel::opaque(((x + y + z) % 4 + 1) as u8, (x ^ y ^ z) as u8 & 3)
        } else {
            TestVoxel::empty(((x + y + z) % 3) as u8)
        }
    });

    assert_same_geometry(
        &voxels,
        &shape,
        [0; 3],
        [17; 3],
        &RIGHT_HANDED_Y_UP_CONFIG.faces,
    );
}

#[test]
fn matches_block_mesh_geometry_for_translucent_mix() {
    let shape = RuntimeShape::<u32, 3>::new([18, 18, 18]);
    let voxels = make_voxels([18, 18, 18], |x, y, z| {
        if x == 0 || y == 0 || z == 0 || x == 17 || y == 17 || z == 17 {
            return TestVoxel::empty(0);
        }

        let pattern = (x * 17 + y * 7 + z * 13) % 9;
        match pattern {
            0 | 1 | 2 => TestVoxel::opaque((pattern + 1) as u8, ((x + z) % 4) as u8),
            3 | 4 => TestVoxel::translucent((pattern + 1) as u8, ((y + z) % 5) as u8),
            _ => TestVoxel::empty(((x + y + z) % 6) as u8),
        }
    });

    assert_same_geometry(
        &voxels,
        &shape,
        [0; 3],
        [17; 3],
        &RIGHT_HANDED_Y_UP_CONFIG.faces,
    );
}

#[test]
fn matches_block_mesh_geometry_for_non_zero_subextent() {
    let shape = RuntimeShape::<u32, 3>::new([14, 13, 12]);
    let voxels = make_voxels([14, 13, 12], |x, y, z| {
        let pattern = (x * 3 + y * 5 + z * 11) % 10;
        match pattern {
            0..=3 => TestVoxel::opaque((x % 5 + 1) as u8, (z % 3) as u8),
            4..=5 => TestVoxel::translucent((y % 4 + 1) as u8, (x % 7) as u8),
            _ => TestVoxel::empty((y % 6) as u8),
        }
    });

    assert_same_geometry(
        &voxels,
        &shape,
        [2, 1, 1],
        [11, 10, 9],
        &RIGHT_HANDED_Y_UP_CONFIG.faces,
    );
}

#[test]
fn matches_block_mesh_geometry_for_alternate_face_config() {
    let shape = RuntimeShape::<u32, 3>::new([12, 12, 12]);
    let faces = canonical_faces();
    let voxels = make_voxels([12, 12, 12], |x, y, z| {
        if x == 0 || y == 0 || z == 0 || x == 11 || y == 11 || z == 11 {
            TestVoxel::empty(0)
        } else if (x + y + z) % 2 == 0 {
            TestVoxel::opaque(((x ^ y ^ z) % 6 + 1) as u8, ((x + 2 * y + z) % 5) as u8)
        } else {
            TestVoxel::translucent(((x + z) % 4 + 1) as u8, ((x + y) % 7) as u8)
        }
    });

    assert_same_geometry(&voxels, &shape, [0; 3], [11; 3], &faces);
}

#[test]
fn default_config_matches_vanilla_exactly() {
    let shape = RuntimeShape::<u32, 3>::new([18, 18, 18]);
    let voxels = make_voxels([18, 18, 18], |x, y, z| {
        if x == 0 || y == 0 || z == 0 || x == 17 || y == 17 || z == 17 {
            TestVoxel::empty(0)
        } else {
            match (x * 13 + y * 7 + z * 3) % 7 {
                0..=2 => TestVoxel::opaque(((x + z) % 5 + 1) as u8, 0),
                3 => TestVoxel::translucent(((x + y) % 4 + 1) as u8, 0),
                _ => TestVoxel::empty(0),
            }
        }
    });

    let vanilla = mesh_with_binary_bgm(
        &voxels,
        &shape,
        [0; 3],
        [17; 3],
        &RIGHT_HANDED_Y_UP_CONFIG.faces,
    );
    let configured = mesh_with_binary_bgm_config(
        &voxels,
        &shape,
        [0; 3],
        [17; 3],
        &RIGHT_HANDED_Y_UP_CONFIG.faces,
        BinaryGreedyQuadsConfig::default(),
    );

    assert_eq!(vanilla.groups, configured.groups);
}

#[test]
fn ao_safe_mode_preserves_geometry_and_splits_ao_boundaries() {
    let shape = RuntimeShape::<u32, 3>::new([8, 6, 8]);
    let voxels = make_voxels([8, 6, 8], |x, y, z| {
        if x == 0 || y == 0 || z == 0 || x == 7 || y == 5 || z == 7 {
            return TestVoxel::empty(0);
        }

        if y == 2 && (2..=4).contains(&x) && (2..=4).contains(&z) {
            TestVoxel::opaque(1, 0)
        } else if y == 3 && (2..=4).contains(&x) && z == 1 {
            TestVoxel::opaque(1, 0)
        } else {
            TestVoxel::empty(0)
        }
    });
    let config = BinaryGreedyQuadsConfig {
        ambient_occlusion_safe: true,
        eliminate_t_junctions: false,
    };

    let vanilla = mesh_with_binary_bgm(
        &voxels,
        &shape,
        [0; 3],
        [7, 5, 7],
        &RIGHT_HANDED_Y_UP_CONFIG.faces,
    );
    let ao_safe = mesh_with_binary_bgm_config(
        &voxels,
        &shape,
        [0; 3],
        [7, 5, 7],
        &RIGHT_HANDED_Y_UP_CONFIG.faces,
        config,
    );
    let top_face_index = RIGHT_HANDED_Y_UP_CONFIG
        .faces
        .iter()
        .position(|face| {
            let axes = face_axes(face);
            axes.n_axis == 1 && axes.n_sign > 0
        })
        .expect("top face should exist");
    let vanilla_lower_top = vanilla.groups[top_face_index]
        .iter()
        .filter(|quad| quad.minimum[1] == 2)
        .count();
    let ao_safe_lower_top = ao_safe.groups[top_face_index]
        .iter()
        .filter(|quad| quad.minimum[1] == 2)
        .count();

    assert_same_geometry_buffers(
        &voxels,
        &shape,
        &RIGHT_HANDED_Y_UP_CONFIG.faces,
        &vanilla,
        &ao_safe,
    );
    assert!(ao_safe.num_quads() > vanilla.num_quads());
    assert_eq!(vanilla_lower_top, 1);
    assert_eq!(ao_safe_lower_top, 4);
    assert_uniform_ao_per_quad(&voxels, &shape, &RIGHT_HANDED_Y_UP_CONFIG.faces, &ao_safe);
}

#[test]
fn ao_safe_mode_keeps_fully_exposed_caps_merged() {
    let shape = RuntimeShape::<u32, 3>::new([7, 5, 7]);
    let voxels = make_voxels([7, 5, 7], |x, y, z| {
        if x == 0 || y == 0 || z == 0 || x == 6 || y == 4 || z == 6 {
            return TestVoxel::empty(0);
        }

        if y == 2 && (2..=4).contains(&x) && (2..=4).contains(&z) {
            TestVoxel::opaque(1, 0)
        } else {
            TestVoxel::empty(0)
        }
    });

    let ao_safe = mesh_with_binary_bgm_config(
        &voxels,
        &shape,
        [0; 3],
        [6, 4, 6],
        &RIGHT_HANDED_Y_UP_CONFIG.faces,
        BinaryGreedyQuadsConfig {
            ambient_occlusion_safe: true,
            eliminate_t_junctions: false,
        },
    );
    let top_face_index = RIGHT_HANDED_Y_UP_CONFIG
        .faces
        .iter()
        .position(|face| {
            let axes = face_axes(face);
            axes.n_axis == 1 && axes.n_sign > 0
        })
        .expect("top face should exist");

    assert_eq!(
        ao_safe.groups[top_face_index].len(),
        1,
        "AO-safe mode should not split a face whose entire exposed side is empty"
    );
}

#[test]
fn t_junction_free_mode_preserves_geometry_and_removes_coplanar_t_junctions() {
    let shape = RuntimeShape::<u32, 3>::new([8, 5, 8]);
    let voxels = make_voxels([8, 5, 8], |x, y, z| {
        if x == 0 || y == 0 || z == 0 || x == 7 || y == 4 || z == 7 {
            return TestVoxel::empty(0);
        }

        let filled =
            y == 2 && matches!((x, z), (2, 2) | (3, 2) | (4, 2) | (5, 2) | (2, 3) | (3, 3));

        if filled {
            TestVoxel::opaque(1, 0)
        } else {
            TestVoxel::empty(0)
        }
    });
    let config = BinaryGreedyQuadsConfig {
        ambient_occlusion_safe: false,
        eliminate_t_junctions: true,
    };

    let vanilla = mesh_with_binary_bgm(
        &voxels,
        &shape,
        [0; 3],
        [7, 4, 7],
        &RIGHT_HANDED_Y_UP_CONFIG.faces,
    );
    let no_t_junctions = mesh_with_binary_bgm_config(
        &voxels,
        &shape,
        [0; 3],
        [7, 4, 7],
        &RIGHT_HANDED_Y_UP_CONFIG.faces,
        config,
    );

    assert_same_geometry_buffers(
        &voxels,
        &shape,
        &RIGHT_HANDED_Y_UP_CONFIG.faces,
        &vanilla,
        &no_t_junctions,
    );
    assert!(has_coplanar_t_junctions(
        &RIGHT_HANDED_Y_UP_CONFIG.faces,
        &vanilla
    ));
    assert!(!has_coplanar_t_junctions(
        &RIGHT_HANDED_Y_UP_CONFIG.faces,
        &no_t_junctions
    ));
}

#[test]
fn combined_mode_preserves_geometry_and_enforces_both_invariants() {
    let shape = RuntimeShape::<u32, 3>::new([8, 6, 8]);
    let voxels = make_voxels([8, 6, 8], |x, y, z| {
        if x == 0 || y == 0 || z == 0 || x == 7 || y == 5 || z == 7 {
            return TestVoxel::empty(0);
        }

        let filled = y == 2
            && matches!(
                (x, z),
                (1, 2) | (2, 2) | (3, 2) | (4, 2) | (5, 2) | (2, 3) | (3, 3)
            );

        if filled {
            TestVoxel::opaque(1, 0)
        } else {
            TestVoxel::empty(0)
        }
    });
    let config = BinaryGreedyQuadsConfig {
        ambient_occlusion_safe: true,
        eliminate_t_junctions: true,
    };

    let expected = mesh_with_block_mesh(
        &voxels,
        &shape,
        [0; 3],
        [7, 5, 7],
        &RIGHT_HANDED_Y_UP_CONFIG.faces,
    );
    let combined = mesh_with_binary_bgm_config(
        &voxels,
        &shape,
        [0; 3],
        [7, 5, 7],
        &RIGHT_HANDED_Y_UP_CONFIG.faces,
        config,
    );

    assert_same_geometry_buffers(
        &voxels,
        &shape,
        &RIGHT_HANDED_Y_UP_CONFIG.faces,
        &expected,
        &combined,
    );
    assert_uniform_ao_per_quad(&voxels, &shape, &RIGHT_HANDED_Y_UP_CONFIG.faces, &combined);
    assert!(!has_coplanar_t_junctions(
        &RIGHT_HANDED_Y_UP_CONFIG.faces,
        &combined
    ));
}

#[test]
fn randomized_property_cases_match_block_mesh_geometry() {
    let mut rng = StdRng::seed_from_u64(0x5eed_baad_f00d);

    for _ in 0..96 {
        let dims = [
            rng.random_range(3..=10),
            rng.random_range(3..=10),
            rng.random_range(3..=10),
        ];
        let shape = RuntimeShape::<u32, 3>::new(dims);
        let voxels = make_random_voxels(&mut rng, dims);

        let min = [
            rng.random_range(0..=(dims[0] - 3)),
            rng.random_range(0..=(dims[1] - 3)),
            rng.random_range(0..=(dims[2] - 3)),
        ];
        let max = [
            rng.random_range((min[0] + 2)..dims[0]),
            rng.random_range((min[1] + 2)..dims[1]),
            rng.random_range((min[2] + 2)..dims[2]),
        ];

        let faces = if rng.random_bool(0.5) {
            RIGHT_HANDED_Y_UP_CONFIG.faces
        } else {
            canonical_faces()
        };

        assert_same_geometry(&voxels, &shape, min, max, &faces);
    }
}

#[test]
#[ignore = "informational benchmark-case report"]
fn reports_quad_count_similarity_for_benchmark_cases() {
    for case in benchmark_like_cases() {
        let greedy = mesh_with_block_mesh(
            &case.voxels,
            &case.shape,
            case.min,
            case.max,
            &RIGHT_HANDED_Y_UP_CONFIG.faces,
        );
        let binary = mesh_with_binary_bgm(
            &case.voxels,
            &case.shape,
            case.min,
            case.max,
            &RIGHT_HANDED_Y_UP_CONFIG.faces,
        );

        assert_same_geometry_buffers(
            &case.voxels,
            &case.shape,
            &RIGHT_HANDED_Y_UP_CONFIG.faces,
            &greedy,
            &binary,
        );

        eprintln!(
            "{}: greedy={} binary={} ratio={:.3}",
            case.name,
            greedy.num_quads(),
            binary.num_quads(),
            binary.num_quads() as f64 / greedy.num_quads() as f64,
        );
    }
}

#[test]
#[should_panic(expected = "supports at most 62 interior voxels per axis")]
fn panics_when_interior_axis_exceeds_limit() {
    let shape = RuntimeShape::<u32, 3>::new([65, 4, 4]);
    let voxels = vec![TestVoxel::default(); shape.size() as usize];
    let mut buffer = BinaryGreedyQuadsBuffer::new();

    binary_greedy_quads(
        &voxels,
        &shape,
        [0; 3],
        [64, 3, 3],
        &RIGHT_HANDED_Y_UP_CONFIG.faces,
        &mut buffer,
    );
}

fn assert_same_geometry(
    voxels: &[TestVoxel],
    shape: &RuntimeShape<u32, 3>,
    min: [u32; 3],
    max: [u32; 3],
    faces: &[OrientedBlockFace; 6],
) {
    let expected = mesh_with_block_mesh(voxels, shape, min, max, faces);
    let actual = mesh_with_binary_bgm(voxels, shape, min, max, faces);

    assert_same_geometry_buffers(voxels, shape, faces, &expected, &actual);
}

fn mesh_with_block_mesh(
    voxels: &[TestVoxel],
    shape: &RuntimeShape<u32, 3>,
    min: [u32; 3],
    max: [u32; 3],
    faces: &[OrientedBlockFace; 6],
) -> QuadBuffer {
    let mut buffer = GreedyQuadsBuffer::new(voxels.len());
    greedy_quads(voxels, shape, min, max, faces, &mut buffer);
    buffer.quads
}

fn mesh_with_binary_bgm(
    voxels: &[TestVoxel],
    shape: &RuntimeShape<u32, 3>,
    min: [u32; 3],
    max: [u32; 3],
    faces: &[OrientedBlockFace; 6],
) -> QuadBuffer {
    let mut buffer = BinaryGreedyQuadsBuffer::new();
    binary_greedy_quads(voxels, shape, min, max, faces, &mut buffer);
    buffer.quads
}

fn mesh_with_binary_bgm_config(
    voxels: &[TestVoxel],
    shape: &RuntimeShape<u32, 3>,
    min: [u32; 3],
    max: [u32; 3],
    faces: &[OrientedBlockFace; 6],
    config: BinaryGreedyQuadsConfig,
) -> QuadBuffer {
    let mut buffer = BinaryGreedyQuadsBuffer::new();
    binary_greedy_quads_with_config(voxels, shape, min, max, faces, &config, &mut buffer);
    buffer.quads
}

fn assert_same_geometry_buffers(
    voxels: &[TestVoxel],
    shape: &RuntimeShape<u32, 3>,
    faces: &[OrientedBlockFace; 6],
    expected: &QuadBuffer,
    actual: &QuadBuffer,
) {
    let expected_faces = expand_quad_buffer(voxels, shape, faces, expected);
    let actual_faces = expand_quad_buffer(voxels, shape, faces, actual);

    assert_eq!(
        expected_faces,
        actual_faces,
        "binary mesh changed the visible unit-face set\nexpected quads: {}\nactual quads: {}",
        expected.num_quads(),
        actual.num_quads(),
    );
}

fn expand_quad_buffer(
    voxels: &[TestVoxel],
    shape: &RuntimeShape<u32, 3>,
    faces: &[OrientedBlockFace; 6],
    buffer: &QuadBuffer,
) -> Vec<UnitFace> {
    let mut unit_faces = Vec::new();

    for (face_index, face) in faces.iter().enumerate() {
        let axes = face_axes(face);

        for quad in &buffer.groups[face_index] {
            let merge_key = face_merge_key(voxels, shape, quad.minimum);

            for dv in 0..quad.height {
                for du in 0..quad.width {
                    let mut cell = quad.minimum;
                    cell[axes.u_axis] += du;
                    cell[axes.v_axis] += dv;
                    unit_faces.push(UnitFace {
                        face_index: face_index as u8,
                        cell,
                        merge_key,
                    });
                }
            }
        }
    }

    unit_faces.sort_unstable();
    unit_faces
}

fn face_axes(face: &OrientedBlockFace) -> FaceAxes {
    let unit_quad = block_mesh::UnorientedQuad {
        minimum: [0; 3],
        width: 1,
        height: 1,
    };
    let corners = face.quad_corners(&unit_quad);
    let u_axis = SignedAxis::from_vector(corners[1].as_ivec3() - corners[0].as_ivec3())
        .expect("axis-aligned face edge")
        .unsigned_axis();
    let v_axis = SignedAxis::from_vector(corners[2].as_ivec3() - corners[0].as_ivec3())
        .expect("axis-aligned face edge")
        .unsigned_axis();
    let normal = SignedAxis::from_vector(face.signed_normal()).expect("axis-aligned face");

    FaceAxes {
        n_axis: normal.unsigned_axis().index(),
        u_axis: u_axis.index(),
        v_axis: v_axis.index(),
        n_sign: normal.signum(),
    }
}

fn assert_uniform_ao_per_quad(
    voxels: &[TestVoxel],
    shape: &RuntimeShape<u32, 3>,
    faces: &[OrientedBlockFace; 6],
    buffer: &QuadBuffer,
) {
    for (face_index, face) in faces.iter().enumerate() {
        let axes = face_axes(face);

        for quad in &buffer.groups[face_index] {
            let mut expected = None;

            for dv in 0..quad.height {
                for du in 0..quad.width {
                    let mut cell = quad.minimum;
                    cell[axes.u_axis] += du;
                    cell[axes.v_axis] += dv;
                    let ao = face_ao_signature(voxels, shape, axes, cell);

                    match expected {
                        None => expected = Some(ao),
                        Some(current) => assert_eq!(
                            current, ao,
                            "quad spans differing AO signatures: face_index={face_index} quad={quad:?}"
                        ),
                    }
                }
            }
        }
    }
}

fn face_ao_signature(
    voxels: &[TestVoxel],
    shape: &RuntimeShape<u32, 3>,
    axes: FaceAxes,
    coord: [u32; 3],
) -> Option<u8> {
    let voxel = voxels[shape.linearize(coord) as usize];
    if voxel.visibility != VoxelVisibility::Opaque {
        return None;
    }

    let corners = [
        ([axes.u_axis, axes.v_axis], [-1i32, -1i32]),
        ([axes.u_axis, axes.v_axis], [1, -1]),
        ([axes.u_axis, axes.v_axis], [1, 1]),
        ([axes.u_axis, axes.v_axis], [-1, 1]),
    ];
    let mut signature = 0u8;

    for (corner_index, (axes_pair, deltas)) in corners.into_iter().enumerate() {
        let side1 = offset_is_opaque(shape, voxels, coord, axes, axes_pair[0], deltas[0]);
        let side2 = offset_is_opaque(shape, voxels, coord, axes, axes_pair[1], deltas[1]);
        let corner = offset2_is_opaque(
            shape,
            voxels,
            coord,
            axes,
            axes_pair[0],
            deltas[0],
            axes_pair[1],
            deltas[1],
        );
        signature |= vertex_ao(side1, side2, corner) << (corner_index * 2);
    }

    Some(signature)
}

fn vertex_ao(side1: bool, side2: bool, corner: bool) -> u8 {
    if side1 && side2 {
        0
    } else {
        3 - side1 as u8 - side2 as u8 - corner as u8
    }
}

fn offset_is_opaque(
    shape: &RuntimeShape<u32, 3>,
    voxels: &[TestVoxel],
    coord: [u32; 3],
    axes: FaceAxes,
    axis: usize,
    delta: i32,
) -> bool {
    let mut offset_coord = coord;
    offset_coord[axes.n_axis] = (offset_coord[axes.n_axis] as i32 + axes.n_sign) as u32;
    offset_coord[axis] = (offset_coord[axis] as i32 + delta) as u32;
    voxels[shape.linearize(offset_coord) as usize].visibility == VoxelVisibility::Opaque
}

fn offset2_is_opaque(
    shape: &RuntimeShape<u32, 3>,
    voxels: &[TestVoxel],
    coord: [u32; 3],
    axes: FaceAxes,
    axis_a: usize,
    delta_a: i32,
    axis_b: usize,
    delta_b: i32,
) -> bool {
    let mut offset_coord = coord;
    offset_coord[axes.n_axis] = (offset_coord[axes.n_axis] as i32 + axes.n_sign) as u32;
    offset_coord[axis_a] = (offset_coord[axis_a] as i32 + delta_a) as u32;
    offset_coord[axis_b] = (offset_coord[axis_b] as i32 + delta_b) as u32;
    voxels[shape.linearize(offset_coord) as usize].visibility == VoxelVisibility::Opaque
}

fn has_coplanar_t_junctions(faces: &[OrientedBlockFace; 6], buffer: &QuadBuffer) -> bool {
    for (face_index, face) in faces.iter().enumerate() {
        let axes = face_axes(face);
        let mut planes = BTreeMap::<u32, Vec<PlaneQuad>>::new();

        for quad in &buffer.groups[face_index] {
            planes
                .entry(quad.minimum[axes.n_axis])
                .or_default()
                .push(PlaneQuad {
                    u: quad.minimum[axes.u_axis],
                    v: quad.minimum[axes.v_axis],
                    width: quad.width,
                    height: quad.height,
                });
        }

        for mut quads in planes.into_values() {
            let mut subdivided = subdivide_plane_quads(&quads);
            quads.sort_unstable();
            subdivided.sort_unstable();
            if quads != subdivided {
                return true;
            }
        }
    }

    false
}

fn subdivide_plane_quads(quads: &[PlaneQuad]) -> Vec<PlaneQuad> {
    let mut max_u = 0usize;
    let mut max_v = 0usize;
    for quad in quads {
        max_u = max_u.max((quad.u + quad.width) as usize);
        max_v = max_v.max((quad.v + quad.height) as usize);
    }

    let mut vertical_cuts = vec![0u64; max_u + 1];
    let mut horizontal_cuts = vec![0u64; max_v + 1];

    for quad in quads {
        let v_mask = boundary_mask(quad.v, quad.height);
        vertical_cuts[quad.u as usize] |= v_mask;
        vertical_cuts[(quad.u + quad.width) as usize] |= v_mask;

        let u_mask = boundary_mask(quad.u, quad.width);
        horizontal_cuts[quad.v as usize] |= u_mask;
        horizontal_cuts[(quad.v + quad.height) as usize] |= u_mask;
    }

    let mut split = Vec::new();
    let mut u_splits = Vec::new();
    let mut v_splits = Vec::new();

    for quad in quads {
        u_splits.clear();
        u_splits.push(quad.u);
        let v_mask = boundary_mask(quad.v, quad.height);
        for boundary in quad.u + 1..quad.u + quad.width {
            if vertical_cuts[boundary as usize] & v_mask != 0 {
                u_splits.push(boundary);
            }
        }
        u_splits.push(quad.u + quad.width);

        v_splits.clear();
        v_splits.push(quad.v);
        let u_mask = boundary_mask(quad.u, quad.width);
        for boundary in quad.v + 1..quad.v + quad.height {
            if horizontal_cuts[boundary as usize] & u_mask != 0 {
                v_splits.push(boundary);
            }
        }
        v_splits.push(quad.v + quad.height);

        for v_window in v_splits.windows(2) {
            for u_window in u_splits.windows(2) {
                split.push(PlaneQuad {
                    u: u_window[0],
                    v: v_window[0],
                    width: u_window[1] - u_window[0],
                    height: v_window[1] - v_window[0],
                });
            }
        }
    }

    split
}

fn boundary_mask(start: u32, len_in_cells: u32) -> u64 {
    ((1u64 << (len_in_cells + 1)) - 1) << start
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct PlaneQuad {
    u: u32,
    v: u32,
    width: u32,
    height: u32,
}

fn face_merge_key(voxels: &[TestVoxel], shape: &RuntimeShape<u32, 3>, coord: [u32; 3]) -> u16 {
    let voxel = voxels[shape.linearize(coord) as usize];
    merge_key(voxel.merge_value())
}

fn merge_key((visibility, material): (VoxelVisibility, u8)) -> u16 {
    ((visibility_key(visibility) as u16) << 8) | material as u16
}

fn visibility_key(visibility: VoxelVisibility) -> u8 {
    match visibility {
        VoxelVisibility::Empty => 0,
        VoxelVisibility::Translucent => 1,
        VoxelVisibility::Opaque => 2,
    }
}

fn make_voxels(dims: [u32; 3], mut fill: impl FnMut(u32, u32, u32) -> TestVoxel) -> Vec<TestVoxel> {
    let shape = RuntimeShape::<u32, 3>::new(dims);
    let mut voxels = Vec::with_capacity(shape.size() as usize);
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                voxels.push(fill(x, y, z));
            }
        }
    }
    voxels
}

fn make_random_voxels(rng: &mut StdRng, dims: [u32; 3]) -> Vec<TestVoxel> {
    make_voxels(dims, |x, y, z| {
        if x == 0 || y == 0 || z == 0 || x == dims[0] - 1 || y == dims[1] - 1 || z == dims[2] - 1 {
            return TestVoxel::empty(rng.random_range(0..=7));
        }

        match rng.random_range(0..=6) {
            0..=2 => TestVoxel::opaque(rng.random_range(1..=5), rng.random_range(0..=7)),
            3..=4 => TestVoxel::translucent(rng.random_range(1..=5), rng.random_range(0..=7)),
            _ => TestVoxel::empty(rng.random_range(0..=7)),
        }
    })
}

fn canonical_faces() -> [OrientedBlockFace; 6] {
    [
        OrientedBlockFace::canonical(SignedAxis::NegX),
        OrientedBlockFace::canonical(SignedAxis::NegY),
        OrientedBlockFace::canonical(SignedAxis::NegZ),
        OrientedBlockFace::canonical(SignedAxis::PosX),
        OrientedBlockFace::canonical(SignedAxis::PosY),
        OrientedBlockFace::canonical(SignedAxis::PosZ),
    ]
}

fn benchmark_like_cases() -> Vec<CarryCase> {
    vec![
        build_carry_case(
            "dense-sphere",
            [34, 34, 34],
            [0; 3],
            [33; 3],
            |x, y, z, dims| {
                let cx = (dims[0] - 1) as f32 * 0.5;
                let cy = (dims[1] - 1) as f32 * 0.5;
                let cz = (dims[2] - 1) as f32 * 0.5;
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dz = z as f32 - cz;
                let radius = dims[0].min(dims[1]).min(dims[2]) as f32 * 0.46;

                if dx * dx + dy * dy + dz * dz <= radius * radius {
                    TestVoxel::opaque((1 + ((x + y + z) % 4)) as u8, 0)
                } else {
                    TestVoxel::default()
                }
            },
        ),
        build_carry_case(
            "translucent-sphere",
            [34, 34, 34],
            [0; 3],
            [33; 3],
            |x, y, z, dims| {
                let cx = (dims[0] - 1) as f32 * 0.5;
                let cy = (dims[1] - 1) as f32 * 0.5;
                let cz = (dims[2] - 1) as f32 * 0.5;
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dz = z as f32 - cz;
                let radius = dims[0].min(dims[1]).min(dims[2]) as f32 * 0.46;

                if dx * dx + dy * dy + dz * dz <= radius * radius {
                    TestVoxel::translucent((1 + ((x + y + z) % 4)) as u8, 0)
                } else {
                    TestVoxel::default()
                }
            },
        ),
        build_carry_case(
            "translucent-shell-sphere",
            [34, 34, 34],
            [0; 3],
            [33; 3],
            |x, y, z, dims| translucent_shell_sphere_voxel(x, y, z, dims),
        ),
        build_carry_case(
            "layered-caves",
            [34, 34, 34],
            [0; 3],
            [33; 3],
            |x, y, z, dims| {
                let base = y <= dims[1] * 7 / 10;
                let cave = hash3(x, y, z, 23) % 100 > 24;
                let ridge = hash3(x, y, z, 91) % 100 > 14;

                if base && cave && ridge {
                    TestVoxel::opaque((1 + (hash3(x, y, z, 7) % 6)) as u8, 0)
                } else if y < dims[1] / 3 && x < dims[0] / 2 {
                    TestVoxel::opaque(2, 0)
                } else {
                    TestVoxel::default()
                }
            },
        ),
        build_carry_case(
            "checkerboard",
            [34, 34, 34],
            [0; 3],
            [33; 3],
            |x, y, z, _dims| {
                if (x + y + z) % 2 == 0 {
                    TestVoxel::opaque((1 + ((x ^ y ^ z) % 4)) as u8, 0)
                } else {
                    TestVoxel::default()
                }
            },
        ),
        build_carry_case(
            "partial-extent",
            [34, 34, 34],
            [4, 4, 4],
            [29, 29, 29],
            |x, y, z, _dims| match hash3(x, y, z, 131) % 8 {
                0..=4 => TestVoxel::opaque((1 + (hash3(x, y, z, 11) % 5)) as u8, 0),
                _ => TestVoxel::default(),
            },
        ),
        build_carry_case(
            "translucent-mix",
            [34, 34, 34],
            [0; 3],
            [33; 3],
            |x, y, z, _dims| match hash3(x, y, z, 131) % 8 {
                0..=3 => TestVoxel::opaque((1 + (hash3(x, y, z, 11) % 5)) as u8, 0),
                4 => TestVoxel::translucent((1 + (hash3(x, y, z, 17) % 3)) as u8, 0),
                _ => TestVoxel::default(),
            },
        ),
    ]
}

fn build_carry_case(
    name: &'static str,
    dims: [u32; 3],
    min: [u32; 3],
    max: [u32; 3],
    mut fill: impl FnMut(u32, u32, u32, [u32; 3]) -> TestVoxel,
) -> CarryCase {
    let shape = RuntimeShape::<u32, 3>::new(dims);
    let mut voxels = Vec::with_capacity(shape.size() as usize);

    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                if x == 0
                    || y == 0
                    || z == 0
                    || x == dims[0] - 1
                    || y == dims[1] - 1
                    || z == dims[2] - 1
                {
                    voxels.push(TestVoxel::default());
                } else {
                    voxels.push(fill(x, y, z, dims));
                }
            }
        }
    }

    CarryCase {
        name,
        shape,
        voxels,
        min,
        max,
    }
}

fn hash3(x: u32, y: u32, z: u32, seed: u32) -> u32 {
    let mut v = x.wrapping_mul(0x9E37_79B1);
    v ^= y.wrapping_mul(0x85EB_CA77);
    v ^= z.wrapping_mul(0xC2B2_AE3D);
    v ^= seed.wrapping_mul(0x27D4_EB2D);
    v ^ (v >> 16)
}

fn translucent_shell_sphere_voxel(x: u32, y: u32, z: u32, dims: [u32; 3]) -> TestVoxel {
    const OUTER_RADIUS_VOXELS: f32 = 14.4;
    const SHELL_THICKNESS_VOXELS: f32 = 3.0;

    let cx = (dims[0] - 1) as f32 * 0.5;
    let cy = (dims[1] - 1) as f32 * 0.5;
    let cz = (dims[2] - 1) as f32 * 0.5;
    let dx = x as f32 - cx;
    let dy = y as f32 - cy;
    let dz = z as f32 - cz;
    let radius_voxels = (dx * dx + dy * dy + dz * dz).sqrt();

    if radius_voxels >= OUTER_RADIUS_VOXELS {
        return TestVoxel::default();
    }

    if radius_voxels < OUTER_RADIUS_VOXELS - SHELL_THICKNESS_VOXELS {
        return TestVoxel::opaque(2, 0);
    }

    let stripe = ((x / 3) + 2 * (y / 3) + (z / 3)) % 4;
    TestVoxel::translucent((stripe + 3) as u8, 0)
}
