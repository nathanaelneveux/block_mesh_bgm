use std::{
    cell::RefCell,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use bevy::{
    asset::RenderAssetUsages,
    color::palettes::css::WHITE,
    light::CascadeShadowConfigBuilder,
    mesh::{Indices, VertexAttributeValues},
    pbr::wireframe::{WireframeConfig, WireframePlugin},
    platform::collections::HashMap,
    prelude::*,
    render::{
        render_resource::{PrimitiveTopology, WgpuFeatures},
        settings::{RenderCreation, WgpuSettings},
        RenderPlugin,
    },
};
use bevy_voxel_world::{
    custom_meshing::{PaddedChunkShape, VoxelArray, CHUNK_SIZE_U},
    prelude::*,
    rendering::ATTRIBUTE_TEX_INDEX,
};
use block_mesh::{
    greedy_quads, visible_block_faces, GreedyQuadsBuffer, OrientedBlockFace, QuadBuffer,
    UnitQuadBuffer, UnorientedQuad, Voxel, VoxelVisibility, RIGHT_HANDED_Y_UP_CONFIG,
};
use block_mesh_bgm::{binary_greedy_quads, binary_greedy_quads_ao_safe, BinaryGreedyQuadsBuffer};
use ndshape::ConstShape;
use noise::{HybridMulti, NoiseFn, Perlin};

const PADDED_CHUNK_EDGE: usize = CHUNK_SIZE_U as usize + 2;
const PADDED_CHUNK_LEN: usize = PADDED_CHUNK_EDGE * PADDED_CHUNK_EDGE * PADDED_CHUNK_EDGE;

thread_local! {
    static SIMPLE_BUFFER: RefCell<UnitQuadBuffer> = RefCell::new(UnitQuadBuffer::new());
    static GREEDY_BUFFER: RefCell<GreedyQuadsBuffer> =
        RefCell::new(GreedyQuadsBuffer::new(PADDED_CHUNK_LEN));
    static BINARY_BUFFER: RefCell<BinaryGreedyQuadsBuffer> =
        RefCell::new(BinaryGreedyQuadsBuffer::new());
}

#[derive(Clone, Copy)]
struct ColumnSample {
    height: f64,
    surface_y: i32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum MeshingAlgorithm {
    VisibleFaces,
    GreedyQuads,
    #[default]
    BinaryGreedyQuads,
}

impl MeshingAlgorithm {
    fn as_u8(self) -> u8 {
        match self {
            Self::VisibleFaces => 0,
            Self::GreedyQuads => 1,
            Self::BinaryGreedyQuads => 2,
        }
    }

    fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::VisibleFaces,
            1 => Self::GreedyQuads,
            _ => Self::BinaryGreedyQuads,
        }
    }

    fn from_input(keyboard: &ButtonInput<KeyCode>) -> Option<Self> {
        if keyboard.just_pressed(KeyCode::Digit1) {
            Some(Self::VisibleFaces)
        } else if keyboard.just_pressed(KeyCode::Digit2) {
            Some(Self::GreedyQuads)
        } else if keyboard.just_pressed(KeyCode::Digit3) {
            Some(Self::BinaryGreedyQuads)
        } else {
            None
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::VisibleFaces => "visible faces",
            Self::GreedyQuads => "greedy quads",
            Self::BinaryGreedyQuads => "binary greedy quads",
        }
    }

    fn supports_ao(self) -> bool {
        matches!(self, Self::VisibleFaces | Self::BinaryGreedyQuads)
    }
}

#[derive(Clone)]
struct MeshingSharedState {
    algorithm: Arc<AtomicU8>,
    ao_safe: Arc<AtomicBool>,
    timing_generation: Arc<AtomicU64>,
    total_nanos: Arc<AtomicU64>,
    sample_count: Arc<AtomicU64>,
}

impl Default for MeshingSharedState {
    fn default() -> Self {
        Self {
            algorithm: Arc::new(AtomicU8::new(MeshingAlgorithm::default().as_u8())),
            ao_safe: Arc::new(AtomicBool::new(false)),
            timing_generation: Arc::new(AtomicU64::new(0)),
            total_nanos: Arc::new(AtomicU64::new(0)),
            sample_count: Arc::new(AtomicU64::new(0)),
        }
    }
}

impl MeshingSharedState {
    fn algorithm(&self) -> MeshingAlgorithm {
        MeshingAlgorithm::from_u8(self.algorithm.load(Ordering::Acquire))
    }

    fn ao_safe(&self) -> bool {
        self.ao_safe.load(Ordering::Acquire)
    }

    fn timing_generation(&self) -> u64 {
        self.timing_generation.load(Ordering::Acquire)
    }

    fn reset_timing(&self) {
        self.timing_generation.fetch_add(1, Ordering::AcqRel);
        self.total_nanos.store(0, Ordering::Release);
        self.sample_count.store(0, Ordering::Release);
    }

    fn select_algorithm(&self, algorithm: MeshingAlgorithm) -> bool {
        let previous = self.algorithm.swap(algorithm.as_u8(), Ordering::AcqRel);
        if previous == algorithm.as_u8() {
            return false;
        }

        self.reset_timing();
        true
    }

    fn set_ao_safe(&self, ao_safe: bool) -> bool {
        let previous = self.ao_safe.swap(ao_safe, Ordering::AcqRel);
        previous != ao_safe
    }

    fn record_sample(&self, generation: u64, elapsed: Duration) {
        if self.timing_generation() != generation {
            return;
        }

        let elapsed_nanos = elapsed.as_nanos().min(u64::MAX as u128) as u64;
        self.total_nanos.fetch_add(elapsed_nanos, Ordering::Relaxed);
        self.sample_count.fetch_add(1, Ordering::Relaxed);
    }

    fn snapshot(&self) -> MeshingTimingSnapshot {
        let algorithm = self.algorithm();
        let ao_safe = self.ao_safe();
        let sample_count = self.sample_count.load(Ordering::Acquire);
        let total_nanos = self.total_nanos.load(Ordering::Acquire);
        let average = if sample_count == 0 {
            None
        } else {
            Some(Duration::from_nanos(total_nanos / sample_count))
        };

        MeshingTimingSnapshot {
            algorithm,
            ao_safe,
            sample_count,
            average,
        }
    }
}

struct MeshingTimingSnapshot {
    algorithm: MeshingAlgorithm,
    ao_safe: bool,
    sample_count: u64,
    average: Option<Duration>,
}

#[derive(Resource, Clone, Default)]
struct MainWorld {
    meshing: MeshingSharedState,
}

#[derive(Component)]
struct OverlayText;

impl VoxelWorldConfig for MainWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        25
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(move |_chunk_pos, _lod, _previous| get_voxel_fn())
    }

    fn texture_index_mapper(&self) -> Arc<dyn Fn(Self::MaterialIndex) -> [u32; 3] + Send + Sync> {
        Arc::new(|mat| match mat {
            0 => [0, 0, 0],
            1 => [1, 1, 1],
            2 => [2, 2, 2],
            3 => [3, 3, 3],
            _ => [0, 0, 0],
        })
    }

    fn chunk_meshing_delegate(
        &self,
    ) -> ChunkMeshingDelegate<Self::MaterialIndex, Self::ChunkUserBundle> {
        let meshing = self.meshing.clone();
        Some(Box::new(
            move |_pos: IVec3, _lod, _data_shape, _mesh_shape, _previous| {
                let meshing = meshing.clone();
                Box::new(
                    move |voxels: VoxelArray<Self::MaterialIndex>,
                          _data_shape_in: UVec3,
                          _mesh_shape_in: UVec3,
                          texture_index_mapper: TextureIndexMapperFn<Self::MaterialIndex>| {
                        let algorithm = meshing.algorithm();
                        let ao_safe = meshing.ao_safe();
                        let generation = meshing.timing_generation();
                        let start = Instant::now();

                        let render_mesh = mesh_chunk(algorithm, ao_safe, &voxels, &texture_index_mapper);

                        meshing.record_sample(generation, start.elapsed());
                        (render_mesh, None)
                    },
                )
            },
        ))
    }
}

struct RenderMeshBuffers {
    indices: Vec<u32>,
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    tex_coords: Vec<[f32; 2]>,
    material_types: Vec<[u32; 3]>,
    colors: Vec<[f32; 4]>,
}

impl RenderMeshBuffers {
    fn with_quad_capacity(num_quads: usize) -> Self {
        let num_indices = num_quads * 6;
        let num_vertices = num_quads * 4;
        Self {
            indices: Vec::with_capacity(num_indices),
            positions: Vec::with_capacity(num_vertices),
            normals: Vec::with_capacity(num_vertices),
            tex_coords: Vec::with_capacity(num_vertices),
            material_types: Vec::with_capacity(num_vertices),
            colors: Vec::with_capacity(num_vertices),
        }
    }

    fn append_quad(
        &mut self,
        face: OrientedBlockFace,
        quad: &UnorientedQuad,
        voxels: &[WorldVoxel<u8>],
        texture_index_mapper: &TextureIndexMapperFn<u8>,
        ao: Option<[u32; 4]>,
    ) {
        self.indices
            .extend_from_slice(&face.quad_mesh_indices(self.positions.len() as u32));
        self.positions
            .extend_from_slice(&face.quad_mesh_positions(quad, 1.0));
        self.normals.extend_from_slice(&face.quad_mesh_normals());
        self.tex_coords.extend_from_slice(&face.tex_coords(
            RIGHT_HANDED_Y_UP_CONFIG.u_flip_face,
            true,
            quad,
        ));

        let voxel_index = PaddedChunkShape::linearize(quad.minimum) as usize;
        let material_type = match voxels[voxel_index] {
            WorldVoxel::Solid(mat) => (*texture_index_mapper)(mat),
            _ => [0, 0, 0],
        };
        self.material_types
            .extend(std::iter::repeat_n(material_type, 4));
        match ao {
            Some(values) => self.colors.extend(values.map(ao_vertex_color)),
            None => self.colors.extend(std::iter::repeat_n([1.0; 4], 4)),
        }
    }

    fn build(self) -> Mesh {
        let mut render_mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::RENDER_WORLD,
        );
        render_mesh.insert_attribute(
            Mesh::ATTRIBUTE_POSITION,
            VertexAttributeValues::Float32x3(self.positions),
        );
        render_mesh.insert_attribute(
            Mesh::ATTRIBUTE_NORMAL,
            VertexAttributeValues::Float32x3(self.normals),
        );
        render_mesh.insert_attribute(
            Mesh::ATTRIBUTE_UV_0,
            VertexAttributeValues::Float32x2(self.tex_coords),
        );
        render_mesh.insert_attribute(
            ATTRIBUTE_TEX_INDEX,
            VertexAttributeValues::Uint32x3(self.material_types),
        );
        render_mesh.insert_attribute(
            Mesh::ATTRIBUTE_COLOR,
            VertexAttributeValues::Float32x4(self.colors),
        );
        render_mesh.insert_indices(Indices::U32(self.indices));
        render_mesh
    }
}

fn mesh_chunk(
    algorithm: MeshingAlgorithm,
    ao_safe: bool,
    voxels: &[WorldVoxel<u8>],
    texture_index_mapper: &TextureIndexMapperFn<u8>,
) -> Mesh {
    let faces = RIGHT_HANDED_Y_UP_CONFIG.faces;

    match algorithm {
        MeshingAlgorithm::VisibleFaces => SIMPLE_BUFFER.with(|buffer| {
            let mut buffer = buffer.borrow_mut();
            buffer.reset();
            visible_block_faces(
                voxels,
                &PaddedChunkShape {},
                [0; 3],
                [CHUNK_SIZE_U + 1; 3],
                &faces,
                &mut buffer,
            );
            build_mesh_from_unit_quad_buffer(&buffer, &faces, voxels, texture_index_mapper, ao_safe)
        }),
        MeshingAlgorithm::GreedyQuads => GREEDY_BUFFER.with(|buffer| {
            let mut buffer = buffer.borrow_mut();
            greedy_quads(
                voxels,
                &PaddedChunkShape {},
                [0; 3],
                [CHUNK_SIZE_U + 1; 3],
                &faces,
                &mut buffer,
            );
            build_mesh_from_quad_buffer(&buffer.quads, &faces, voxels, texture_index_mapper, false)
        }),
        MeshingAlgorithm::BinaryGreedyQuads => BINARY_BUFFER.with(|buffer| {
            let mut buffer = buffer.borrow_mut();
            if ao_safe {
                binary_greedy_quads_ao_safe(
                    voxels,
                    &PaddedChunkShape {},
                    [0; 3],
                    [CHUNK_SIZE_U + 1; 3],
                    &faces,
                    &mut buffer,
                );
            } else {
                binary_greedy_quads(
                    voxels,
                    &PaddedChunkShape {},
                    [0; 3],
                    [CHUNK_SIZE_U + 1; 3],
                    &faces,
                    &mut buffer,
                );
            }
            build_mesh_from_quad_buffer(
                &buffer.quads,
                &faces,
                voxels,
                texture_index_mapper,
                ao_safe,
            )
        }),
    }
}

fn build_mesh_from_unit_quad_buffer(
    buffer: &UnitQuadBuffer,
    faces: &[OrientedBlockFace; 6],
    voxels: &[WorldVoxel<u8>],
    texture_index_mapper: &TextureIndexMapperFn<u8>,
    ao_safe: bool,
) -> Mesh {
    let mut mesh = RenderMeshBuffers::with_quad_capacity(buffer.num_quads());

    for (group, face) in buffer.groups.iter().zip(faces.iter().copied()) {
        for quad in group {
            let quad: UnorientedQuad = (*quad).into();
            let ao = ao_safe.then(|| face_aos(face, quad.minimum, voxels));
            mesh.append_quad(face, &quad, voxels, texture_index_mapper, ao);
        }
    }

    mesh.build()
}

fn build_mesh_from_quad_buffer(
    buffer: &QuadBuffer,
    faces: &[OrientedBlockFace; 6],
    voxels: &[WorldVoxel<u8>],
    texture_index_mapper: &TextureIndexMapperFn<u8>,
    ao_safe: bool,
) -> Mesh {
    let mut mesh = RenderMeshBuffers::with_quad_capacity(buffer.num_quads());

    for (group, face) in buffer.groups.iter().zip(faces.iter().copied()) {
        for quad in group {
            let ao = ao_safe.then(|| face_aos(face, quad.minimum, voxels));
            mesh.append_quad(face, quad, voxels, texture_index_mapper, ao);
        }
    }

    mesh.build()
}

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(RenderPlugin {
                render_creation: RenderCreation::Automatic(WgpuSettings {
                    features: WgpuFeatures::POLYGON_MODE_LINE,
                    ..default()
                }),
                ..default()
            }),
            WireframePlugin::default(),
        ))
        .add_plugins(VoxelWorldPlugin::with_config(MainWorld::default()))
        .insert_resource(WireframeConfig {
            global: true,
            default_color: WHITE.into(),
        })
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                (switch_meshing_settings, update_overlay).chain(),
                move_camera,
            ),
        )
        .run();
}

fn setup(mut commands: Commands) {
    info!("bevy_voxel_world custom meshing example");
    info!("Press 1 for visible faces, 2 for greedy quads, 3 for binary greedy quads.");
    info!(
        "Press A to toggle ambient occlusion for visible faces and AO-safe binary greedy meshing."
    );
    info!("Switching the mesher resets average timing and respawns the loaded chunks.");
    info!(
        "Initial meshing algorithm: {}",
        MeshingAlgorithm::default().label()
    );

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-50.0, 50.0, -50.0).looking_at(Vec3::ZERO, Vec3::Y),
        VoxelWorldCamera::<MainWorld>::default(),
    ));

    let cascade_shadow_config = CascadeShadowConfigBuilder { ..default() }.build();
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(0.98, 0.95, 0.82),
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, 0.0).looking_at(Vec3::new(-0.15, -0.1, 0.15), Vec3::Y),
        cascade_shadow_config,
    ));

    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.98, 0.95, 0.82),
        brightness: 100.0,
        affects_lightmapped_meshes: true,
    });

    spawn_overlay(&mut commands);
}

fn spawn_overlay(commands: &mut Commands) {
    commands.spawn((
        OverlayText,
        Text::new(""),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(12.0),
            left: Val::Px(12.0),
            ..default()
        },
    ));
}

fn update_overlay(mut overlay: Query<&mut Text, With<OverlayText>>, world: Res<MainWorld>) {
    let snapshot = world.meshing.snapshot();
    let average_text = match snapshot.average {
        Some(average) => format!(
            "{:.3} ms over {} chunk meshes",
            average.as_secs_f64() * 1_000.0,
            snapshot.sample_count,
        ),
        None => "waiting for chunk meshes".to_owned(),
    };

    if let Ok(mut text) = overlay.single_mut() {
        *text = Text::new(format!(
            "Custom meshing comparison\n\
             1: visible faces\n\
             2: greedy quads\n\
             3: binary greedy quads\n\
             A: visible-face AO / binary greedy AO-safe [{}]\n\
             Current: {}\n\
             Average meshing time: {}\n\
             Switching clears the average and respawns loaded chunks.",
            toggle_label(snapshot.ao_safe),
            snapshot.algorithm.label(),
            average_text,
        ));
    }
}

fn switch_meshing_settings(
    keyboard: Res<ButtonInput<KeyCode>>,
    world: Res<MainWorld>,
    mut commands: Commands,
    chunks: Query<Entity, (With<Chunk<MainWorld>>, Without<NeedsDespawn>)>,
) {
    let current_algorithm = world.meshing.algorithm();
    let next_algorithm = MeshingAlgorithm::from_input(&keyboard);
    let ao_toggled = keyboard.just_pressed(KeyCode::KeyA);

    if next_algorithm.is_none() && !ao_toggled {
        return;
    }

    let mut changed = false;
    let mut remesh_needed = false;

    if let Some(next_algorithm) = next_algorithm {
        if world.meshing.select_algorithm(next_algorithm) {
            changed = true;
            remesh_needed = true;
        }
    }

    if ao_toggled {
        let next_ao_safe = !world.meshing.ao_safe();
        if world.meshing.set_ao_safe(next_ao_safe) {
            changed = true;
            if current_algorithm.supports_ao() {
                world.meshing.reset_timing();
                remesh_needed = true;
            }
        }
    }

    if !changed {
        return;
    }

    if !remesh_needed {
        info!(
            "Ambient occlusion: {} (stored for the next visible-face or binary greedy remesh)",
            toggle_label(world.meshing.ao_safe()),
        );
        return;
    }

    let mut chunk_count = 0usize;
    for entity in &chunks {
        commands.entity(entity).try_insert(NeedsDespawn);
        chunk_count += 1;
    }

    info!(
        "Meshing algorithm: {} | ambient occlusion: {} (respawning {chunk_count} loaded chunks)",
        world.meshing.algorithm().label(),
        toggle_label(world.meshing.ao_safe()),
    );
}

fn toggle_label(enabled: bool) -> &'static str {
    if enabled {
        "on"
    } else {
        "off"
    }
}

fn ao_vertex_color(ao_value: u32) -> [f32; 4] {
    match ao_value {
        0 => [0.10, 0.10, 0.10, 1.0],
        1 => [0.30, 0.30, 0.30, 1.0],
        2 => [0.50, 0.50, 0.50, 1.0],
        _ => [1.0, 1.0, 1.0, 1.0],
    }
}

fn ao_value(side1: bool, corner: bool, side2: bool) -> u32 {
    match (side1, corner, side2) {
        (true, _, true) => 0,
        (true, true, false) | (false, true, true) => 1,
        (false, false, false) => 3,
        _ => 2,
    }
}

fn side_aos(neighbours: [WorldVoxel<u8>; 8]) -> [u32; 4] {
    let opaque = neighbours.map(|voxel| voxel.get_visibility() == VoxelVisibility::Opaque);

    [
        ao_value(opaque[0], opaque[1], opaque[2]),
        ao_value(opaque[2], opaque[3], opaque[4]),
        ao_value(opaque[6], opaque[7], opaque[0]),
        ao_value(opaque[4], opaque[5], opaque[6]),
    ]
}

fn face_aos(face: OrientedBlockFace, minimum: [u32; 3], voxels: &[WorldVoxel<u8>]) -> [u32; 4] {
    let normal = face.signed_normal();
    let [x, y, z] = minimum;

    match [normal.x, normal.y, normal.z] {
        [-1, 0, 0] => side_aos([
            voxels[PaddedChunkShape::linearize([x - 1, y, z - 1]) as usize],
            voxels[PaddedChunkShape::linearize([x - 1, y - 1, z - 1]) as usize],
            voxels[PaddedChunkShape::linearize([x - 1, y - 1, z]) as usize],
            voxels[PaddedChunkShape::linearize([x - 1, y - 1, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x - 1, y, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x - 1, y + 1, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x - 1, y + 1, z]) as usize],
            voxels[PaddedChunkShape::linearize([x - 1, y + 1, z - 1]) as usize],
        ]),
        [1, 0, 0] => side_aos([
            voxels[PaddedChunkShape::linearize([x + 1, y, z - 1]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y - 1, z - 1]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y - 1, z]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y - 1, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y + 1, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y + 1, z]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y + 1, z - 1]) as usize],
        ]),
        [0, -1, 0] => side_aos([
            voxels[PaddedChunkShape::linearize([x, y - 1, z - 1]) as usize],
            voxels[PaddedChunkShape::linearize([x - 1, y - 1, z - 1]) as usize],
            voxels[PaddedChunkShape::linearize([x - 1, y - 1, z]) as usize],
            voxels[PaddedChunkShape::linearize([x - 1, y - 1, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x, y - 1, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y - 1, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y - 1, z]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y - 1, z - 1]) as usize],
        ]),
        [0, 1, 0] => side_aos([
            voxels[PaddedChunkShape::linearize([x, y + 1, z - 1]) as usize],
            voxels[PaddedChunkShape::linearize([x - 1, y + 1, z - 1]) as usize],
            voxels[PaddedChunkShape::linearize([x - 1, y + 1, z]) as usize],
            voxels[PaddedChunkShape::linearize([x - 1, y + 1, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x, y + 1, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y + 1, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y + 1, z]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y + 1, z - 1]) as usize],
        ]),
        [0, 0, -1] => side_aos([
            voxels[PaddedChunkShape::linearize([x - 1, y, z - 1]) as usize],
            voxels[PaddedChunkShape::linearize([x - 1, y - 1, z - 1]) as usize],
            voxels[PaddedChunkShape::linearize([x, y - 1, z - 1]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y - 1, z - 1]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y, z - 1]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y + 1, z - 1]) as usize],
            voxels[PaddedChunkShape::linearize([x, y + 1, z - 1]) as usize],
            voxels[PaddedChunkShape::linearize([x - 1, y + 1, z - 1]) as usize],
        ]),
        [0, 0, 1] => side_aos([
            voxels[PaddedChunkShape::linearize([x - 1, y, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x - 1, y - 1, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x, y - 1, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y - 1, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x + 1, y + 1, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x, y + 1, z + 1]) as usize],
            voxels[PaddedChunkShape::linearize([x - 1, y + 1, z + 1]) as usize],
        ]),
        _ => unreachable!(),
    }
}

fn get_voxel_fn() -> Box<dyn FnMut(IVec3, Option<WorldVoxel>) -> WorldVoxel + Send + Sync> {
    let mut noise = HybridMulti::<Perlin>::new(1234);
    noise.octaves = 5;
    noise.frequency = 1.1;
    noise.lacunarity = 2.8;
    noise.persistence = 0.4;
    let mut cache = HashMap::<(i32, i32), ColumnSample>::new();

    Box::new(move |pos: IVec3, _previous| {
        if pos.y < 1 {
            return WorldVoxel::Solid(3);
        }

        let [x, y, z] = pos.as_dvec3().to_array();
        let column = match cache.get(&(pos.x, pos.z)) {
            Some(sample) => *sample,
            None => {
                let height = noise.get([x / 4000.0, z / 4000.0]) * 20.0;
                let sample = ColumnSample {
                    height,
                    surface_y: height.floor() as i32,
                };
                cache.insert((pos.x, pos.z), sample);
                sample
            }
        };

        if y < column.height {
            // Keep material regions broad so the mesh can actually merge.
            // The previous per-voxel checker pattern destroyed greedy runs.
            let material = if column.surface_y <= 2 {
                if pos.y >= column.surface_y - 1 {
                    2
                } else {
                    3
                }
            } else if pos.y >= column.surface_y - 1 {
                0
            } else if pos.y >= column.surface_y - 4 {
                1
            } else {
                2
            };

            WorldVoxel::Solid(material)
        } else {
            WorldVoxel::Air
        }
    })
}

fn move_camera(
    time: Res<Time>,
    mut camera: Query<&mut Transform, With<VoxelWorldCamera<MainWorld>>>,
) {
    if let Ok(mut transform) = camera.single_mut() {
        transform.translation.x += time.delta_secs() * 5.0;
        transform.translation.z += time.delta_secs() * 10.0;
    }
}
