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

pub fn into_domain(array_dim: u32, [x, y, z]: [u32; 3]) -> Vec3A {
    (2.0 / array_dim as f32) * Vec3A::new(x as f32, y as f32, z as f32) - 1.0
}

pub fn mesh_from_unit_quads(
    buffer: UnitQuadBuffer,
    faces: &[OrientedBlockFace; 6],
    samples: &[DemoVoxel; SampleShape::SIZE as usize],
) -> (Mesh, MeshStats) {
    let num_quads = buffer.num_quads();
    let num_vertices = num_quads * 4;
    let mut indices = Vec::with_capacity(num_quads * 6);
    let mut positions = Vec::with_capacity(num_vertices);
    let mut normals = Vec::with_capacity(num_vertices);
    let mut colors = Vec::with_capacity(num_vertices);

    for (group, face) in buffer.groups.into_iter().zip(faces.iter().copied()) {
        for quad in group {
            let quad: UnorientedQuad = quad.into();
            let color = quad_color(samples, quad.minimum);

            indices.extend_from_slice(&face.quad_mesh_indices(positions.len() as u32));
            positions.extend_from_slice(&face.quad_mesh_positions(&quad, 1.0));
            normals.extend_from_slice(&face.quad_mesh_normals());
            colors.extend_from_slice(&[color; 4]);
        }
    }

    (
        build_render_mesh(indices, positions, normals, colors),
        MeshStats {
            quads: num_quads,
            vertices: num_vertices,
            triangles: num_quads * 2,
        },
    )
}

pub fn mesh_from_quads(
    buffer: QuadBuffer,
    faces: &[OrientedBlockFace; 6],
    samples: &[DemoVoxel; SampleShape::SIZE as usize],
) -> (Mesh, MeshStats) {
    let num_quads = buffer.num_quads();
    let num_vertices = num_quads * 4;
    let mut indices = Vec::with_capacity(num_quads * 6);
    let mut positions = Vec::with_capacity(num_vertices);
    let mut normals = Vec::with_capacity(num_vertices);
    let mut colors = Vec::with_capacity(num_vertices);

    for (group, face) in buffer.groups.into_iter().zip(faces.iter().copied()) {
        for quad in group {
            let color = quad_color(samples, quad.minimum);

            indices.extend_from_slice(&face.quad_mesh_indices(positions.len() as u32));
            positions.extend_from_slice(&face.quad_mesh_positions(&quad, 1.0));
            normals.extend_from_slice(&face.quad_mesh_normals());
            colors.extend_from_slice(&[color; 4]);
        }
    }

    (
        build_render_mesh(indices, positions, normals, colors),
        MeshStats {
            quads: num_quads,
            vertices: num_vertices,
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

fn quad_color(samples: &[DemoVoxel; SampleShape::SIZE as usize], minimum: [u32; 3]) -> [f32; 4] {
    material_color(samples[SampleShape::linearize(minimum) as usize].material)
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
