use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, Mesh, VertexAttributeValues},
    render::render_resource::PrimitiveTopology,
};
use block_mesh::ilattice::glam::Vec3A;
use block_mesh::ndshape::{ConstShape, ConstShape3u32};
use block_mesh::{
    MergeVoxel, OrientedBlockFace, QuadBuffer, UnitQuadBuffer, UnorientedQuad, Voxel,
    VoxelVisibility,
};

pub type SampleShape = ConstShape3u32<34, 34, 34>;

pub const SAMPLE_MIN: [u32; 3] = [0; 3];
pub const SAMPLE_MAX: [u32; 3] = [33; 3];
pub const EMPTY: DemoVoxel = DemoVoxel {
    visibility: VoxelVisibility::Empty,
    material: 0,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DemoVoxel {
    pub visibility: VoxelVisibility,
    pub material: u8,
}

#[derive(Clone, Copy, Debug)]
pub struct MeshStats {
    pub quads: usize,
    pub vertices: usize,
    pub triangles: usize,
}

impl DemoVoxel {
    pub const fn solid(material: u8) -> Self {
        Self {
            visibility: VoxelVisibility::Opaque,
            material,
        }
    }

    pub const fn translucent(material: u8) -> Self {
        Self {
            visibility: VoxelVisibility::Translucent,
            material,
        }
    }
}

impl Default for DemoVoxel {
    fn default() -> Self {
        EMPTY
    }
}

impl Voxel for DemoVoxel {
    fn get_visibility(&self) -> VoxelVisibility {
        self.visibility
    }
}

impl MergeVoxel for DemoVoxel {
    type MergeValue = (VoxelVisibility, u8);

    fn merge_value(&self) -> Self::MergeValue {
        (self.visibility, self.material)
    }
}

pub fn build_demo_samples(
    mut field: impl FnMut([u32; 3], Vec3A) -> DemoVoxel,
) -> [DemoVoxel; SampleShape::SIZE as usize] {
    let mut samples = [EMPTY; SampleShape::SIZE as usize];

    for i in 0..SampleShape::SIZE {
        let coord = SampleShape::delinearize(i);
        samples[i as usize] = field(coord, into_domain(32, coord));
    }

    samples
}

pub fn striped_sphere(coord: [u32; 3], p: Vec3A) -> DemoVoxel {
    if p.length() >= 0.9 {
        return EMPTY;
    }

    let stripe = ((coord[0] / 4) + 2 * (coord[1] / 4) + 3 * (coord[2] / 4)) % 6;
    DemoVoxel::solid((stripe + 1) as u8)
}

pub fn translucent_shell_sphere(coord: [u32; 3], p: Vec3A) -> DemoVoxel {
    const OUTER_RADIUS_VOXELS: f32 = 14.4;
    const SHELL_THICKNESS_VOXELS: f32 = 3.0;

    let radius_voxels = p.length() * 16.0;
    if radius_voxels >= OUTER_RADIUS_VOXELS {
        return EMPTY;
    }

    if radius_voxels < OUTER_RADIUS_VOXELS - SHELL_THICKNESS_VOXELS {
        return DemoVoxel::solid(2);
    }

    let stripe = ((coord[0] / 3) + 2 * (coord[1] / 3) + (coord[2] / 3)) % 4;
    DemoVoxel::translucent((stripe + 3) as u8)
}

pub fn into_domain(array_dim: u32, [x, y, z]: [u32; 3]) -> Vec3A {
    (2.0 / array_dim as f32) * Vec3A::new(x as f32, y as f32, z as f32) - 1.0
}

pub fn mesh_from_unit_quads(
    buffer: UnitQuadBuffer,
    faces: &[OrientedBlockFace; 6],
    samples: &[DemoVoxel; SampleShape::SIZE as usize],
) -> (Mesh, MeshStats) {
    mesh_from_unit_quads_impl(buffer, faces, samples, false)
}

pub fn mesh_from_unit_quads_with_ao(
    buffer: UnitQuadBuffer,
    faces: &[OrientedBlockFace; 6],
    samples: &[DemoVoxel; SampleShape::SIZE as usize],
) -> (Mesh, MeshStats) {
    mesh_from_unit_quads_impl(buffer, faces, samples, true)
}

fn mesh_from_unit_quads_impl(
    buffer: UnitQuadBuffer,
    faces: &[OrientedBlockFace; 6],
    samples: &[DemoVoxel; SampleShape::SIZE as usize],
    apply_ao: bool,
) -> (Mesh, MeshStats) {
    let num_quads = buffer.num_quads();
    let mut mesh = MeshBuffers::default();

    for (group, face) in buffer.groups.into_iter().zip(faces.iter().copied()) {
        for quad in group {
            let quad: UnorientedQuad = quad.into();
            append_quad(
                face,
                quad,
                quad_voxel(samples, quad.minimum),
                samples,
                apply_ao,
                &mut mesh,
            );
        }
    }

    build_demo_render_mesh(num_quads, mesh)
}

pub fn mesh_from_quads(
    buffer: QuadBuffer,
    faces: &[OrientedBlockFace; 6],
    samples: &[DemoVoxel; SampleShape::SIZE as usize],
) -> (Mesh, MeshStats) {
    mesh_from_quads_impl(buffer, faces, samples, false)
}

pub fn mesh_from_quads_with_ao(
    buffer: QuadBuffer,
    faces: &[OrientedBlockFace; 6],
    samples: &[DemoVoxel; SampleShape::SIZE as usize],
) -> (Mesh, MeshStats) {
    mesh_from_quads_impl(buffer, faces, samples, true)
}

fn mesh_from_quads_impl(
    buffer: QuadBuffer,
    faces: &[OrientedBlockFace; 6],
    samples: &[DemoVoxel; SampleShape::SIZE as usize],
    apply_ao: bool,
) -> (Mesh, MeshStats) {
    let num_quads = buffer.num_quads();
    let mut mesh = MeshBuffers::default();

    for (group, face) in buffer.groups.into_iter().zip(faces.iter().copied()) {
        for quad in group {
            append_quad(
                face,
                quad,
                quad_voxel(samples, quad.minimum),
                samples,
                apply_ao,
                &mut mesh,
            );
        }
    }

    build_demo_render_mesh(num_quads, mesh)
}

#[derive(Default)]
struct MeshBuffers {
    indices: Vec<u32>,
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    colors: Vec<[f32; 4]>,
}

impl MeshBuffers {
    fn push_quad(&mut self, face: OrientedBlockFace, quad: UnorientedQuad, color: [f32; 4]) {
        self.indices
            .extend_from_slice(&face.quad_mesh_indices(self.positions.len() as u32));
        self.positions
            .extend_from_slice(&face.quad_mesh_positions(&quad, 1.0));
        self.normals.extend_from_slice(&face.quad_mesh_normals());
        self.colors.extend_from_slice(&[color; 4]);
    }

    fn push_quad_colors(
        &mut self,
        face: OrientedBlockFace,
        quad: UnorientedQuad,
        colors: [[f32; 4]; 4],
    ) {
        self.indices
            .extend_from_slice(&face.quad_mesh_indices(self.positions.len() as u32));
        self.positions
            .extend_from_slice(&face.quad_mesh_positions(&quad, 1.0));
        self.normals.extend_from_slice(&face.quad_mesh_normals());
        self.colors.extend_from_slice(&colors);
    }

    fn build(self) -> Mesh {
        build_render_mesh(self.indices, self.positions, self.normals, self.colors)
    }
}

fn append_quad(
    face: OrientedBlockFace,
    quad: UnorientedQuad,
    voxel: DemoVoxel,
    samples: &[DemoVoxel; SampleShape::SIZE as usize],
    apply_ao: bool,
    mesh: &mut MeshBuffers,
) {
    if voxel.visibility != VoxelVisibility::Empty {
        if apply_ao && voxel.visibility == VoxelVisibility::Opaque {
            mesh.push_quad_colors(
                face,
                quad,
                quad_vertex_colors_with_ao(face, quad, voxel, samples),
            );
        } else {
            mesh.push_quad(face, quad, voxel_color(voxel));
        }
    }
}

fn build_demo_render_mesh(num_quads: usize, mesh: MeshBuffers) -> (Mesh, MeshStats) {
    (
        mesh.build(),
        MeshStats {
            quads: num_quads,
            vertices: num_quads * 4,
            triangles: num_quads * 2,
        },
    )
}

fn build_render_mesh(
    indices: Vec<u32>,
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    colors: Vec<[f32; 4]>,
) -> Mesh {
    let mut render_mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    render_mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(positions),
    );
    render_mesh.insert_attribute(
        Mesh::ATTRIBUTE_NORMAL,
        VertexAttributeValues::Float32x3(normals),
    );
    render_mesh.insert_attribute(
        Mesh::ATTRIBUTE_UV_0,
        VertexAttributeValues::Float32x2(vec![[0.0; 2]; colors.len()]),
    );
    render_mesh.insert_attribute(
        Mesh::ATTRIBUTE_COLOR,
        VertexAttributeValues::Float32x4(colors),
    );
    render_mesh.insert_indices(Indices::U32(indices));
    render_mesh
}

fn quad_voxel(samples: &[DemoVoxel; SampleShape::SIZE as usize], minimum: [u32; 3]) -> DemoVoxel {
    samples[SampleShape::linearize(minimum) as usize]
}

fn voxel_color(voxel: DemoVoxel) -> [f32; 4] {
    let mut color = material_color(voxel.material);
    if voxel.visibility == VoxelVisibility::Translucent {
        color[3] = 0.34;
    }
    color
}

fn material_color(material: u8) -> [f32; 4] {
    match material {
        1 => [0.92, 0.39, 0.27, 1.0],
        2 => [0.97, 0.79, 0.31, 1.0],
        3 => [0.33, 0.71, 0.45, 1.0],
        4 => [0.23, 0.57, 0.84, 1.0],
        5 => [0.48, 0.36, 0.84, 1.0],
        6 => [0.86, 0.41, 0.69, 1.0],
        _ => [0.85, 0.85, 0.85, 1.0],
    }
}

fn quad_vertex_colors_with_ao(
    face: OrientedBlockFace,
    quad: UnorientedQuad,
    voxel: DemoVoxel,
    samples: &[DemoVoxel; SampleShape::SIZE as usize],
) -> [[f32; 4]; 4] {
    let ao = face_aos(face, quad.minimum, samples);
    let base = voxel_color(voxel);
    ao.map(|ao_value| shade_color_with_ao(base, ao_value))
}

fn shade_color_with_ao(mut color: [f32; 4], ao_value: u32) -> [f32; 4] {
    let shade = match ao_value {
        0 => 0.10,
        1 => 0.30,
        2 => 0.50,
        _ => 1.00,
    };
    color[0] *= shade;
    color[1] *= shade;
    color[2] *= shade;
    color
}

fn face_aos(
    face: OrientedBlockFace,
    minimum: [u32; 3],
    samples: &[DemoVoxel; SampleShape::SIZE as usize],
) -> [u32; 4] {
    let normal = face.signed_normal();
    let [x, y, z] = minimum;

    match [normal.x, normal.y, normal.z] {
        [-1, 0, 0] => side_aos(
            samples,
            [
                [x - 1, y, z - 1],
                [x - 1, y - 1, z - 1],
                [x - 1, y - 1, z],
                [x - 1, y - 1, z + 1],
                [x - 1, y, z + 1],
                [x - 1, y + 1, z + 1],
                [x - 1, y + 1, z],
                [x - 1, y + 1, z - 1],
            ],
        ),
        [1, 0, 0] => side_aos(
            samples,
            [
                [x + 1, y, z - 1],
                [x + 1, y - 1, z - 1],
                [x + 1, y - 1, z],
                [x + 1, y - 1, z + 1],
                [x + 1, y, z + 1],
                [x + 1, y + 1, z + 1],
                [x + 1, y + 1, z],
                [x + 1, y + 1, z - 1],
            ],
        ),
        [0, -1, 0] => side_aos(
            samples,
            [
                [x, y - 1, z - 1],
                [x - 1, y - 1, z - 1],
                [x - 1, y - 1, z],
                [x - 1, y - 1, z + 1],
                [x, y - 1, z + 1],
                [x + 1, y - 1, z + 1],
                [x + 1, y - 1, z],
                [x + 1, y - 1, z - 1],
            ],
        ),
        [0, 1, 0] => side_aos(
            samples,
            [
                [x, y + 1, z - 1],
                [x - 1, y + 1, z - 1],
                [x - 1, y + 1, z],
                [x - 1, y + 1, z + 1],
                [x, y + 1, z + 1],
                [x + 1, y + 1, z + 1],
                [x + 1, y + 1, z],
                [x + 1, y + 1, z - 1],
            ],
        ),
        [0, 0, -1] => side_aos(
            samples,
            [
                [x - 1, y, z - 1],
                [x - 1, y - 1, z - 1],
                [x, y - 1, z - 1],
                [x + 1, y - 1, z - 1],
                [x + 1, y, z - 1],
                [x + 1, y + 1, z - 1],
                [x, y + 1, z - 1],
                [x - 1, y + 1, z - 1],
            ],
        ),
        [0, 0, 1] => side_aos(
            samples,
            [
                [x - 1, y, z + 1],
                [x - 1, y - 1, z + 1],
                [x, y - 1, z + 1],
                [x + 1, y - 1, z + 1],
                [x + 1, y, z + 1],
                [x + 1, y + 1, z + 1],
                [x, y + 1, z + 1],
                [x - 1, y + 1, z + 1],
            ],
        ),
        _ => unreachable!(),
    }
}

fn side_aos(
    samples: &[DemoVoxel; SampleShape::SIZE as usize],
    neighbors: [[u32; 3]; 8],
) -> [u32; 4] {
    let opaque = neighbors.map(|coord| voxel_is_opaque(samples, coord));
    [
        ao_value(opaque[0], opaque[1], opaque[2]),
        ao_value(opaque[2], opaque[3], opaque[4]),
        ao_value(opaque[6], opaque[7], opaque[0]),
        ao_value(opaque[4], opaque[5], opaque[6]),
    ]
}

fn voxel_is_opaque(samples: &[DemoVoxel; SampleShape::SIZE as usize], coord: [u32; 3]) -> bool {
    samples[SampleShape::linearize(coord) as usize].visibility == VoxelVisibility::Opaque
}

fn ao_value(side1: bool, corner: bool, side2: bool) -> u32 {
    match (side1, corner, side2) {
        (true, _, true) => 0,
        (true, true, false) | (false, true, true) => 1,
        (false, false, false) => 3,
        _ => 2,
    }
}
