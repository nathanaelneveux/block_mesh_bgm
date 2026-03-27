use bevy::{
    color::palettes::css::{BLACK, WHITE},
    light::CascadeShadowConfigBuilder,
    pbr::wireframe::{WireframeConfig, WireframePlugin},
    prelude::*,
    render::{
        render_resource::WgpuFeatures,
        settings::{RenderCreation, WgpuSettings},
        RenderPlugin,
    },
};
use block_mesh::ilattice::glam::Vec3A;
use block_mesh::ndshape::ConstShape;
use block_mesh::{visible_block_faces, UnitQuadBuffer, RIGHT_HANDED_Y_UP_CONFIG};
use block_mesh_bgm::{binary_greedy_quads, binary_greedy_quads_ao_safe, BinaryGreedyQuadsBuffer};
use block_mesh_bgm_examples::{
    build_demo_samples, mesh_from_quads, mesh_from_quads_with_ao, mesh_from_unit_quads,
    mesh_from_unit_quads_with_ao, DemoVoxel, MeshStats, SampleShape, EMPTY, SAMPLE_MAX, SAMPLE_MIN,
};

#[derive(Component)]
struct OverlayText;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SampleMode {
    Sphere,
    Torus,
}

impl SampleMode {
    fn next(self) -> Self {
        match self {
            Self::Sphere => Self::Torus,
            Self::Torus => Self::Sphere,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Sphere => "opaque sphere",
            Self::Torus => "opaque torus",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MeshMode {
    BinaryGreedy,
    UnitQuads,
}

impl MeshMode {
    fn next(self) -> Self {
        match self {
            Self::BinaryGreedy => Self::UnitQuads,
            Self::UnitQuads => Self::BinaryGreedy,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::BinaryGreedy => "binary greedy",
            Self::UnitQuads => "unit quads",
        }
    }
}

#[derive(Resource)]
struct AmbientOcclusionScene {
    ao_safe: bool,
    sample_mode: SampleMode,
    mesh_mode: MeshMode,
    mesh: Handle<Mesh>,
    samples: [DemoVoxel; SampleShape::SIZE as usize],
    stats: MeshStats,
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
        .insert_resource(WireframeConfig {
            global: true,
            default_color: BLACK.into(),
        })
        .add_systems(Startup, setup)
        .add_systems(Update, toggle_ambient_occlusion_viewer)
        .run();
}

fn setup(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let ao_safe = false;
    let sample_mode = SampleMode::Sphere;
    let mesh_mode = MeshMode::BinaryGreedy;
    let samples = build_samples(sample_mode);
    let (mesh, stats) = build_mesh(&samples, ao_safe, mesh_mode);
    let mesh = meshes.add(mesh);
    let material = materials.add(StandardMaterial {
        base_color: WHITE.into(),
        perceptual_roughness: 0.94,
        metallic: 0.0,
        ..default()
    });

    spawn_camera(&mut commands);
    spawn_lights(&mut commands);
    spawn_overlay(&mut commands, sample_mode, mesh_mode, ao_safe, stats);
    spawn_mesh(&mut commands, mesh.clone(), material);

    log_stats(sample_mode, mesh_mode, ao_safe, stats);
    info!("Press U to toggle binary greedy vs unit quads.");
    info!("Press A to toggle AO-safe merging.");
    info!("Press S to toggle sphere vs torus.");

    commands.insert_resource(AmbientOcclusionScene {
        ao_safe,
        sample_mode,
        mesh_mode,
        mesh,
        samples,
        stats,
    });
}

fn toggle_ambient_occlusion_viewer(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut scene: ResMut<AmbientOcclusionScene>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut overlay_text: Query<&mut Text, With<OverlayText>>,
) {
    let mut changed = false;

    if keyboard.just_pressed(KeyCode::KeyU) {
        scene.mesh_mode = scene.mesh_mode.next();
        changed = true;
    }

    if keyboard.just_pressed(KeyCode::KeyS) {
        scene.sample_mode = scene.sample_mode.next();
        scene.samples = build_samples(scene.sample_mode);
        changed = true;
    }

    if keyboard.just_pressed(KeyCode::KeyA) {
        scene.ao_safe = !scene.ao_safe;
        changed = true;
    }

    if !changed {
        return;
    }

    let ao_safe = scene.ao_safe;
    let sample_mode = scene.sample_mode;
    let mesh_mode = scene.mesh_mode;
    let (mesh, stats) = build_mesh(&scene.samples, ao_safe, mesh_mode);
    *meshes
        .get_mut(scene.mesh.id())
        .expect("example mesh handle should remain valid") = mesh;
    scene.stats = stats;

    overlay_text.single_mut().expect("overlay should exist").0 =
        overlay_string(sample_mode, mesh_mode, ao_safe, stats);

    log_stats(sample_mode, mesh_mode, ao_safe, stats);
}

fn build_mesh(
    samples: &[DemoVoxel; SampleShape::SIZE as usize],
    ao_safe: bool,
    mesh_mode: MeshMode,
) -> (Mesh, MeshStats) {
    let faces = RIGHT_HANDED_Y_UP_CONFIG.faces;

    match mesh_mode {
        MeshMode::BinaryGreedy => {
            let mut buffer = BinaryGreedyQuadsBuffer::new();

            if ao_safe {
                binary_greedy_quads_ao_safe(
                    samples,
                    &SampleShape {},
                    SAMPLE_MIN,
                    SAMPLE_MAX,
                    &faces,
                    &mut buffer,
                );
            } else {
                binary_greedy_quads(
                    samples,
                    &SampleShape {},
                    SAMPLE_MIN,
                    SAMPLE_MAX,
                    &faces,
                    &mut buffer,
                );
            }

            if ao_safe {
                mesh_from_quads_with_ao(buffer.quads, &faces, samples)
            } else {
                mesh_from_quads(buffer.quads, &faces, samples)
            }
        }
        MeshMode::UnitQuads => {
            let mut buffer = UnitQuadBuffer::new();
            visible_block_faces(
                samples,
                &SampleShape {},
                SAMPLE_MIN,
                SAMPLE_MAX,
                &faces,
                &mut buffer,
            );

            if ao_safe {
                mesh_from_unit_quads_with_ao(buffer, &faces, samples)
            } else {
                mesh_from_unit_quads(buffer, &faces, samples)
            }
        }
    }
}

fn build_samples(sample_mode: SampleMode) -> [DemoVoxel; SampleShape::SIZE as usize] {
    match sample_mode {
        SampleMode::Sphere => build_demo_samples(solid_sphere),
        SampleMode::Torus => build_demo_samples(solid_torus),
    }
}

fn solid_sphere(_: [u32; 3], p: Vec3A) -> DemoVoxel {
    if p.length() >= 0.9 {
        EMPTY
    } else {
        DemoVoxel::solid(2)
    }
}

fn solid_torus(_: [u32; 3], p: Vec3A) -> DemoVoxel {
    let radial = (p.x * p.x + p.z * p.z).sqrt();
    let major_radius = 0.56;
    let minor_radius = 0.24;
    let ring = radial - major_radius;

    if ring * ring + p.y * p.y >= minor_radius * minor_radius {
        EMPTY
    } else {
        DemoVoxel::solid(4)
    }
}

fn spawn_overlay(
    commands: &mut Commands,
    sample_mode: SampleMode,
    mesh_mode: MeshMode,
    ao_safe: bool,
    stats: MeshStats,
) {
    commands.spawn((
        Text::new(overlay_string(sample_mode, mesh_mode, ao_safe, stats)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(12.0),
            left: Val::Px(12.0),
            ..default()
        },
        OverlayText,
    ));
}

fn overlay_string(
    sample_mode: SampleMode,
    mesh_mode: MeshMode,
    ao_safe: bool,
    stats: MeshStats,
) -> String {
    format!(
        "Binary greedy ambient occlusion viewer\n\
         U: toggle mesh mode [{}]\n\
         S: toggle sample shape [{}]\n\
         A: toggle AO-safe merges + ambient occlusion [{}]\n\
         Wireframe: always on\n\
         Mesh: {}\n\
         Sample: {}\n\
         Quads: {}\n\
         Vertices: {}\n\
        Triangles: {}",
        mesh_mode.label(),
        sample_mode.label(),
        toggle_label(ao_safe),
        mesh_mode.label(),
        sample_mode.label(),
        stats.quads,
        stats.vertices,
        stats.triangles,
    )
}

fn toggle_label(enabled: bool) -> &'static str {
    if enabled {
        "on"
    } else {
        "off"
    }
}

fn spawn_mesh(commands: &mut Commands, mesh: Handle<Mesh>, material: Handle<StandardMaterial>) {
    commands.spawn((Mesh3d(mesh), MeshMaterial3d(material), Transform::default()));
}

fn spawn_camera(commands: &mut Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(38.0, 28.0, 42.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

fn spawn_lights(commands: &mut Commands) {
    let cascade_shadow_config = CascadeShadowConfigBuilder { ..default() }.build();
    commands.spawn((
        DirectionalLight {
            color: WHITE.into(),
            shadows_enabled: true,
            illuminance: 16_000.0,
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, 0.0).looking_at(Vec3::new(-0.3, -0.5, -0.2), Vec3::Y),
        cascade_shadow_config,
    ));

    commands.insert_resource(GlobalAmbientLight {
        color: WHITE.into(),
        brightness: 220.0,
        affects_lightmapped_meshes: true,
    });
}

fn log_stats(sample_mode: SampleMode, mesh_mode: MeshMode, ao_safe: bool, stats: MeshStats) {
    info!(
        "Mesh: {} | Sample: {} | AO-safe: {} | quads: {} | vertices: {} | triangles: {}",
        mesh_mode.label(),
        sample_mode.label(),
        toggle_label(ao_safe),
        stats.quads,
        stats.vertices,
        stats.triangles,
    );
}
