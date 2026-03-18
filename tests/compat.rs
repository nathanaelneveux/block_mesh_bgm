use block_mesh::ndshape::{RuntimeShape, Shape};
use block_mesh::{
    greedy_quads, GreedyQuadsBuffer, MergeVoxel, OrientedBlockFace, QuadBuffer, SignedAxis, Voxel,
    VoxelVisibility, RIGHT_HANDED_Y_UP_CONFIG,
};
use block_mesh_bgm::{binary_greedy_quads, BinaryGreedyQuadsBuffer};
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

#[test]
fn matches_block_mesh_for_opaque_sphere() {
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

    assert_same_output(
        &voxels,
        &shape,
        [0; 3],
        [17; 3],
        &RIGHT_HANDED_Y_UP_CONFIG.faces,
    );
}

#[test]
fn matches_block_mesh_for_translucent_mix() {
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

    assert_same_output(
        &voxels,
        &shape,
        [0; 3],
        [17; 3],
        &RIGHT_HANDED_Y_UP_CONFIG.faces,
    );
}

#[test]
fn matches_block_mesh_for_non_zero_subextent() {
    let shape = RuntimeShape::<u32, 3>::new([14, 13, 12]);
    let voxels = make_voxels([14, 13, 12], |x, y, z| {
        let pattern = (x * 3 + y * 5 + z * 11) % 10;
        match pattern {
            0..=3 => TestVoxel::opaque((x % 5 + 1) as u8, (z % 3) as u8),
            4..=5 => TestVoxel::translucent((y % 4 + 1) as u8, (x % 7) as u8),
            _ => TestVoxel::empty((y % 6) as u8),
        }
    });

    assert_same_output(
        &voxels,
        &shape,
        [2, 1, 1],
        [11, 10, 9],
        &RIGHT_HANDED_Y_UP_CONFIG.faces,
    );
}

#[test]
fn matches_block_mesh_for_alternate_face_config() {
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

    assert_same_output(&voxels, &shape, [0; 3], [11; 3], &faces);
}

#[test]
fn randomized_property_cases_match_block_mesh() {
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

        assert_same_output(&voxels, &shape, min, max, &faces);
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

fn assert_same_output(
    voxels: &[TestVoxel],
    shape: &RuntimeShape<u32, 3>,
    min: [u32; 3],
    max: [u32; 3],
    faces: &[OrientedBlockFace; 6],
) {
    let expected = mesh_with_block_mesh(voxels, shape, min, max, faces);
    let actual = mesh_with_binary_bgm(voxels, shape, min, max, faces);

    assert_quad_buffer_eq(&expected, &actual);
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

fn assert_quad_buffer_eq(expected: &QuadBuffer, actual: &QuadBuffer) {
    for i in 0..6 {
        assert_eq!(
            expected.groups[i], actual.groups[i],
            "quad group {i} differed\nexpected: {:?}\nactual: {:?}",
            expected.groups[i], actual.groups[i]
        );
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
