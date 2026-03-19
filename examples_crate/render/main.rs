use bevy::{
    color::palettes::css::{BLACK, WHITE},
    core_pipeline::oit::OrderIndependentTransparencySettings,
    light::CascadeShadowConfigBuilder,
    pbr::wireframe::{WireframeConfig, WireframePlugin},
    prelude::*,
    render::{
        render_resource::{TextureUsages, WgpuFeatures},
        settings::{RenderCreation, WgpuSettings},
        RenderPlugin,
    },
};
use block_mesh::{
    greedy_quads, visible_block_faces, GreedyQuadsBuffer, UnitQuadBuffer, RIGHT_HANDED_Y_UP_CONFIG,
};
use block_mesh_bgm::{binary_greedy_quads, BinaryGreedyQuadsBuffer};
use block_mesh_bgm_examples::{
    build_demo_samples, mesh_from_quads, mesh_from_unit_quads, striped_sphere,
    translucent_shell_sphere, MeshStats, SampleShape, SAMPLE_MAX, SAMPLE_MIN,
};

#[derive(Component)]
struct ExampleMesh;

#[derive(Resource)]
struct ExampleScene {
    mode: DemoMode,
    transparency: TransparencyMode,
    camera: Entity,
    material: Handle<StandardMaterial>,
    columns: [ExampleColumn; 3],
}

struct ExampleColumn {
    mesh: Handle<Mesh>,
}

struct BuiltColumn {
    label: &'static str,
    translation: Vec3,
    mesh: Mesh,
    stats: MeshStats,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DemoMode {
    StripedOpaqueSphere,
    TranslucentShellSphere,
}

impl DemoMode {
    fn next(self) -> Self {
        match self {
            Self::StripedOpaqueSphere => Self::TranslucentShellSphere,
            Self::TranslucentShellSphere => Self::StripedOpaqueSphere,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::StripedOpaqueSphere => "striped opaque sphere",
            Self::TranslucentShellSphere => "solid core + translucent shell sphere",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TransparencyMode {
    AlphaToCoverage,
    OrderIndependentTransparency,
}

impl TransparencyMode {
    fn next(self) -> Self {
        match self {
            Self::AlphaToCoverage => Self::OrderIndependentTransparency,
            Self::OrderIndependentTransparency => Self::AlphaToCoverage,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::AlphaToCoverage => "AlphaToCoverage",
            Self::OrderIndependentTransparency => "OIT",
        }
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
        .insert_resource(WireframeConfig {
            global: false,
            default_color: BLACK.into(),
        })
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (toggle_wireframe, toggle_demo_mode, toggle_transparency_mode),
        )
        .run();
}

fn setup(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let initial_mode = DemoMode::StripedOpaqueSphere;
    let initial_transparency = TransparencyMode::AlphaToCoverage;
    let surface_material = materials.add(StandardMaterial {
        base_color: WHITE.into(),
        perceptual_roughness: 0.92,
        metallic: 0.0,
        ..default()
    });
    let [simple, greedy, binary] = build_demo_columns(initial_mode);

    info!("Press Space to toggle wireframe.");
    info!("Press T to toggle between the opaque and translucent sphere demos.");
    info!("Press O to toggle AlphaToCoverage and OIT.");
    info!("Demo mode: {}", initial_mode.label());
    info!("Transparency mode: {}", initial_transparency.label());

    let camera = spawn_camera(&mut commands);
    apply_transparency_mode(
        &mut commands,
        materials
            .get_mut(&surface_material)
            .expect("shared surface material should remain valid"),
        camera,
        initial_transparency,
    );
    spawn_lights(&mut commands);
    spawn_overlay(&mut commands);

    let columns = [
        spawn_column(&mut commands, &mut meshes, &surface_material, simple),
        spawn_column(&mut commands, &mut meshes, &surface_material, greedy),
        spawn_column(&mut commands, &mut meshes, &surface_material, binary),
    ];

    commands.insert_resource(ExampleScene {
        mode: initial_mode,
        transparency: initial_transparency,
        camera,
        material: surface_material,
        columns,
    });
}

fn build_demo_columns(mode: DemoMode) -> [BuiltColumn; 3] {
    let faces = RIGHT_HANDED_Y_UP_CONFIG.faces;
    let samples = match mode {
        DemoMode::StripedOpaqueSphere => build_demo_samples(striped_sphere),
        DemoMode::TranslucentShellSphere => build_demo_samples(translucent_shell_sphere),
    };

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

    [
        BuiltColumn {
            label: "simple",
            translation: Vec3::new(-32.0, 0.0, 0.0),
            mesh: simple_mesh,
            stats: simple_stats,
        },
        BuiltColumn {
            label: "greedy",
            translation: Vec3::new(0.0, 0.0, 0.0),
            mesh: greedy_mesh,
            stats: greedy_stats,
        },
        BuiltColumn {
            label: "binary",
            translation: Vec3::new(32.0, 0.0, 0.0),
            mesh: binary_mesh,
            stats: binary_stats,
        },
    ]
}

fn spawn_column(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    material: &Handle<StandardMaterial>,
    built: BuiltColumn,
) -> ExampleColumn {
    let BuiltColumn {
        label,
        translation,
        mesh,
        stats,
    } = built;

    let mesh = meshes.add(mesh);
    spawn_mesh(commands, mesh.clone(), material.clone(), translation);

    log_stats(label, translation, stats);

    ExampleColumn { mesh }
}

fn spawn_camera(commands: &mut Commands) -> Entity {
    let mut camera_3d = Camera3d::default();
    let mut depth_texture_usages = TextureUsages::from(camera_3d.depth_texture_usages);
    depth_texture_usages |= TextureUsages::TEXTURE_BINDING;
    camera_3d.depth_texture_usages = depth_texture_usages.into();

    commands
        .spawn((
            camera_3d,
            Transform::from_xyz(70.0, 48.0, 78.0).looking_at(Vec3::ZERO, Vec3::Y),
        ))
        .id()
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
             T: toggle opaque vs translucent sample\n\
             O: toggle AlphaToCoverage vs OIT\n\
             Left to right: visible faces | greedy quads | binary greedy",
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
    mesh: Handle<Mesh>,
    material: Handle<StandardMaterial>,
    translation: Vec3,
) -> Entity {
    commands
        .spawn((
            Mesh3d(mesh),
            MeshMaterial3d(material),
            Transform::from_translation(translation),
            ExampleMesh,
        ))
        .id()
}

fn toggle_demo_mode(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut scene: ResMut<ExampleScene>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    if keyboard.just_pressed(KeyCode::KeyT) {
        let next_mode = scene.mode.next();
        let columns = build_demo_columns(next_mode);

        for (column, built) in scene.columns.iter().zip(columns) {
            apply_column_meshes(&mut meshes, column, built);
        }

        scene.mode = next_mode;
        info!("Demo mode: {}", scene.mode.label());
    }
}

fn toggle_transparency_mode(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut commands: Commands,
    mut scene: ResMut<ExampleScene>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if keyboard.just_pressed(KeyCode::KeyO) {
        scene.transparency = scene.transparency.next();
        apply_transparency_mode(
            &mut commands,
            materials
                .get_mut(&scene.material)
                .expect("shared surface material should remain valid"),
            scene.camera,
            scene.transparency,
        );
        info!("Transparency mode: {}", scene.transparency.label());
    }
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

fn apply_column_meshes(meshes: &mut Assets<Mesh>, column: &ExampleColumn, built: BuiltColumn) {
    let BuiltColumn {
        label,
        translation,
        mesh,
        stats,
    } = built;

    *meshes
        .get_mut(column.mesh.id())
        .expect("mesh handle should remain valid") = mesh;

    log_stats(label, translation, stats);
}

fn apply_transparency_mode(
    commands: &mut Commands,
    material: &mut StandardMaterial,
    camera: Entity,
    mode: TransparencyMode,
) {
    match mode {
        TransparencyMode::AlphaToCoverage => {
            material.alpha_mode = AlphaMode::AlphaToCoverage;
            commands
                .entity(camera)
                .remove::<OrderIndependentTransparencySettings>()
                .insert(Msaa::Sample4);
        }
        TransparencyMode::OrderIndependentTransparency => {
            material.alpha_mode = AlphaMode::Blend;
            commands
                .entity(camera)
                .insert(OrderIndependentTransparencySettings::default())
                .insert(Msaa::Off);
        }
    }
}

fn log_stats(label: &str, position: Vec3, stats: MeshStats) {
    info!(
        "{label:<12} at ({:>5.1}, {:>5.1}, {:>5.1}) -> quads: {}, vertices: {}, triangles: {}",
        position.x, position.y, position.z, stats.quads, stats.vertices, stats.triangles,
    );
}
