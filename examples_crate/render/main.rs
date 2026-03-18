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
use block_mesh::{
    greedy_quads, visible_block_faces, GreedyQuadsBuffer, UnitQuadBuffer, RIGHT_HANDED_Y_UP_CONFIG,
};
use block_mesh_bgm::{
    binary_greedy_quads, binary_greedy_quads_carry_merge, BinaryGreedyQuadsBuffer,
};
use block_mesh_bgm_examples::{
    build_demo_samples, mesh_from_quads, mesh_from_unit_quads, striped_sphere, MeshStats,
    SampleShape, SAMPLE_MAX, SAMPLE_MIN,
};

#[derive(Component)]
struct ExampleMesh;

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
            global: false,
            default_color: BLACK.into(),
        })
        .add_systems(Startup, setup)
        .add_systems(Update, toggle_wireframe)
        .run();
}

fn setup(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let faces = RIGHT_HANDED_Y_UP_CONFIG.faces;
    let samples = build_demo_samples(striped_sphere);

    let mut simple_buffer = UnitQuadBuffer::new();
    visible_block_faces(
        &samples,
        &SampleShape {},
        SAMPLE_MIN,
        SAMPLE_MAX,
        &faces,
        &mut simple_buffer,
    );
    let (simple_mesh, simple_stats) = mesh_from_unit_quads(simple_buffer, &faces, &samples);

    let mut greedy_buffer = GreedyQuadsBuffer::new(samples.len());
    greedy_quads(
        &samples,
        &SampleShape {},
        SAMPLE_MIN,
        SAMPLE_MAX,
        &faces,
        &mut greedy_buffer,
    );
    let (greedy_mesh, greedy_stats) = mesh_from_quads(greedy_buffer.quads, &faces, &samples);

    let mut binary_buffer = BinaryGreedyQuadsBuffer::new();
    binary_greedy_quads(
        &samples,
        &SampleShape {},
        SAMPLE_MIN,
        SAMPLE_MAX,
        &faces,
        &mut binary_buffer,
    );
    let (binary_mesh, binary_stats) = mesh_from_quads(binary_buffer.quads, &faces, &samples);

    let mut carry_buffer = BinaryGreedyQuadsBuffer::new();
    binary_greedy_quads_carry_merge(
        &samples,
        &SampleShape {},
        SAMPLE_MIN,
        SAMPLE_MAX,
        &faces,
        &mut carry_buffer,
    );
    let (carry_mesh, carry_stats) = mesh_from_quads(carry_buffer.quads, &faces, &samples);

    log_stats("simple", Vec3::new(-22.0, 0.0, -22.0), simple_stats);
    log_stats("greedy", Vec3::new(22.0, 0.0, -22.0), greedy_stats);
    log_stats("binary exact", Vec3::new(-22.0, 0.0, 22.0), binary_stats);
    log_stats("binary carry", Vec3::new(22.0, 0.0, 22.0), carry_stats);
    info!("Press Space to toggle wireframe.");

    spawn_camera(&mut commands);
    spawn_lights(&mut commands);
    spawn_overlay(&mut commands);

    spawn_mesh(
        &mut commands,
        &mut meshes,
        &mut materials,
        simple_mesh,
        Vec3::new(-22.0, 0.0, -22.0),
    );
    spawn_mesh(
        &mut commands,
        &mut meshes,
        &mut materials,
        greedy_mesh,
        Vec3::new(22.0, 0.0, -22.0),
    );
    spawn_mesh(
        &mut commands,
        &mut meshes,
        &mut materials,
        binary_mesh,
        Vec3::new(-22.0, 0.0, 22.0),
    );
    spawn_mesh(
        &mut commands,
        &mut meshes,
        &mut materials,
        carry_mesh,
        Vec3::new(22.0, 0.0, 22.0),
    );
}

fn spawn_camera(commands: &mut Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(70.0, 48.0, 78.0).looking_at(Vec3::ZERO, Vec3::Y),
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

fn spawn_overlay(commands: &mut Commands) {
    commands.spawn((
        Text::new(
            "Render comparison\n\
             Space: toggle quad wireframe\n\
             Top row: visible faces | greedy quads\n\
             Bottom row: binary exact | binary carry",
        ),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(12.0),
            left: Val::Px(12.0),
            ..default()
        },
    ));
}

fn spawn_mesh(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    mesh: Mesh,
    translation: Vec3,
) {
    let mesh = meshes.add(mesh);
    let material = materials.add(StandardMaterial {
        base_color: WHITE.into(),
        perceptual_roughness: 0.92,
        metallic: 0.0,
        ..default()
    });

    commands.spawn((
        Mesh3d(mesh),
        MeshMaterial3d(material),
        Transform::from_translation(translation),
        ExampleMesh,
    ));
}

fn toggle_wireframe(keyboard: Res<ButtonInput<KeyCode>>, mut wireframe: ResMut<WireframeConfig>) {
    if keyboard.just_pressed(KeyCode::Space) {
        wireframe.global = !wireframe.global;
        info!(
            "Wireframe {}",
            if wireframe.global {
                "enabled"
            } else {
                "disabled"
            }
        );
    }
}

fn log_stats(label: &str, position: Vec3, stats: MeshStats) {
    info!(
        "{label:<12} at ({:>5.1}, {:>5.1}, {:>5.1}) -> quads: {}, vertices: {}, triangles: {}",
        position.x, position.y, position.z, stats.quads, stats.vertices, stats.triangles,
    );
}
