use std::{cell::RefCell, sync::Arc};

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
use block_mesh::RIGHT_HANDED_Y_UP_CONFIG;
use block_mesh_bgm::{binary_greedy_quads, BinaryGreedyQuadsBuffer};
use ndshape::ConstShape;
use noise::{HybridMulti, NoiseFn, Perlin};

thread_local! {
    // Meshing runs on Bevy's async worker pool, so keep one scratch buffer per
    // worker thread and reuse its allocations across chunk jobs.
    static MESH_BUFFER: RefCell<BinaryGreedyQuadsBuffer> =
        RefCell::new(BinaryGreedyQuadsBuffer::new());
}

#[derive(Clone, Copy)]
struct ColumnSample {
    height: f64,
    surface_y: i32,
}

#[derive(Resource, Clone, Default)]
struct MainWorld;

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
        Some(Box::new(
            |_pos: IVec3, _lod, _data_shape, _mesh_shape, _previous| {
                Box::new(
                    |voxels: VoxelArray<Self::MaterialIndex>,
                     _data_shape_in: UVec3,
                     _mesh_shape_in: UVec3,
                     texture_index_mapper: TextureIndexMapperFn<Self::MaterialIndex>| {
                        let faces = RIGHT_HANDED_Y_UP_CONFIG.faces;
                        let (num_vertices, indices, positions, normals, tex_coords, material_types) =
                            MESH_BUFFER.with(|buffer| {
                                let mut buffer = buffer.borrow_mut();

                                binary_greedy_quads(
                                    &voxels,
                                    &PaddedChunkShape {},
                                    [0; 3],
                                    [CHUNK_SIZE_U + 1; 3],
                                    &faces,
                                    &mut buffer,
                                );

                                let num_indices = buffer.quads.num_quads() * 6;
                                let num_vertices = buffer.quads.num_quads() * 4;
                                let mut indices = Vec::with_capacity(num_indices);
                                let mut positions = Vec::with_capacity(num_vertices);
                                let mut normals = Vec::with_capacity(num_vertices);
                                let mut tex_coords = Vec::with_capacity(num_vertices);
                                let mut material_types = Vec::with_capacity(num_vertices);

                                for (group, face) in
                                    buffer.quads.groups.iter().zip(faces.iter().copied())
                                {
                                    for quad in group {
                                        indices.extend_from_slice(
                                            &face.quad_mesh_indices(positions.len() as u32),
                                        );
                                        positions.extend_from_slice(
                                            &face.quad_mesh_positions(quad, 1.0),
                                        );
                                        normals.extend_from_slice(&face.quad_mesh_normals());
                                        tex_coords.extend_from_slice(&face.tex_coords(
                                            RIGHT_HANDED_Y_UP_CONFIG.u_flip_face,
                                            true,
                                            quad,
                                        ));

                                        let voxel_index =
                                            PaddedChunkShape::linearize(quad.minimum) as usize;
                                        let material_type = match voxels[voxel_index] {
                                            WorldVoxel::Solid(mat) => texture_index_mapper(mat),
                                            _ => [0, 0, 0],
                                        };
                                        material_types
                                            .extend(std::iter::repeat_n(material_type, 4));
                                    }
                                }

                                (
                                    num_vertices,
                                    indices,
                                    positions,
                                    normals,
                                    tex_coords,
                                    material_types,
                                )
                            });

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
                            VertexAttributeValues::Float32x2(tex_coords),
                        );
                        render_mesh.insert_attribute(
                            ATTRIBUTE_TEX_INDEX,
                            VertexAttributeValues::Uint32x3(material_types),
                        );
                        render_mesh.insert_attribute(
                            Mesh::ATTRIBUTE_COLOR,
                            vec![[1.0; 4]; num_vertices],
                        );
                        render_mesh.insert_indices(Indices::U32(indices));

                        (render_mesh, None)
                    },
                )
            },
        ))
    }
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
        .add_plugins(VoxelWorldPlugin::with_config(MainWorld))
        .insert_resource(WireframeConfig {
            global: true,
            default_color: WHITE.into(),
        })
        .add_systems(Startup, setup)
        .add_systems(Update, move_camera)
        .run();
}

fn setup(mut commands: Commands) {
    info!("bevy_voxel_world custom meshing example using binary greedy meshing");

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
