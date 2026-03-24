use std::hint::black_box;

use block_mesh::ndshape::{RuntimeShape, Shape};
use block_mesh::{
    greedy_quads, visible_block_faces, GreedyQuadsBuffer, MergeVoxel, UnitQuadBuffer, Voxel,
    VoxelVisibility, RIGHT_HANDED_Y_UP_CONFIG,
};
use block_mesh_bgm::{
    binary_greedy_quads, binary_greedy_quads_with_config, BinaryGreedyQuadsBuffer,
    BinaryGreedyQuadsConfig,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BenchVoxel {
    visibility: VoxelVisibility,
    material: u8,
}

impl Default for BenchVoxel {
    fn default() -> Self {
        Self {
            visibility: VoxelVisibility::Empty,
            material: 0,
        }
    }
}

impl Voxel for BenchVoxel {
    fn get_visibility(&self) -> VoxelVisibility {
        self.visibility
    }
}

impl MergeVoxel for BenchVoxel {
    type MergeValue = (VoxelVisibility, u8);

    fn merge_value(&self) -> Self::MergeValue {
        (self.visibility, self.material)
    }
}

#[derive(Clone)]
struct Case {
    name: &'static str,
    shape: RuntimeShape<u32, 3>,
    voxels: Vec<BenchVoxel>,
    min: [u32; 3],
    max: [u32; 3],
}

#[derive(Clone)]
struct MultiCase {
    name: &'static str,
    chunks: Vec<Case>,
}

fn bench_meshers(c: &mut Criterion) {
    let cases = benchmark_cases();
    let multi_cases = benchmark_multi_chunk_cases();
    let strategy_cases = strategy_benchmark_cases();
    let faces = RIGHT_HANDED_Y_UP_CONFIG.faces;
    let mut group = c.benchmark_group("meshers");
    let configs = [
        (
            "binary_greedy_quads_ao_safe",
            BinaryGreedyQuadsConfig {
                ambient_occlusion_safe: true,
                eliminate_t_junctions: false,
            },
        ),
        (
            "binary_greedy_quads_no_t_junctions",
            BinaryGreedyQuadsConfig {
                ambient_occlusion_safe: false,
                eliminate_t_junctions: true,
            },
        ),
        (
            "binary_greedy_quads_ao_safe_no_t_junctions",
            BinaryGreedyQuadsConfig {
                ambient_occlusion_safe: true,
                eliminate_t_junctions: true,
            },
        ),
    ];

    for case in &cases {
        group.bench_with_input(
            BenchmarkId::new("visible_block_faces", case.name),
            case,
            |b, case| {
                let mut buffer = UnitQuadBuffer::new();
                b.iter(|| {
                    buffer.reset();
                    visible_block_faces(
                        black_box(&case.voxels),
                        &case.shape,
                        case.min,
                        case.max,
                        &faces,
                        &mut buffer,
                    );
                    black_box(buffer.num_quads());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("greedy_quads", case.name),
            case,
            |b, case| {
                let mut buffer = GreedyQuadsBuffer::new(case.voxels.len());
                b.iter(|| {
                    greedy_quads(
                        black_box(&case.voxels),
                        &case.shape,
                        case.min,
                        case.max,
                        &faces,
                        &mut buffer,
                    );
                    black_box(buffer.quads.num_quads());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("binary_greedy_quads", case.name),
            case,
            |b, case| {
                let mut buffer = BinaryGreedyQuadsBuffer::new();
                b.iter(|| {
                    binary_greedy_quads(
                        black_box(&case.voxels),
                        &case.shape,
                        case.min,
                        case.max,
                        &faces,
                        &mut buffer,
                    );
                    black_box(buffer.quads.num_quads());
                });
            },
        );

        for (name, config) in configs {
            group.bench_with_input(BenchmarkId::new(name, case.name), case, |b, case| {
                let mut buffer = BinaryGreedyQuadsBuffer::new();
                b.iter(|| {
                    binary_greedy_quads_with_config(
                        black_box(&case.voxels),
                        &case.shape,
                        case.min,
                        case.max,
                        &faces,
                        &config,
                        &mut buffer,
                    );
                    black_box(buffer.quads.num_quads());
                });
            });
        }
    }

    for case in &multi_cases {
        group.bench_with_input(
            BenchmarkId::new("visible_block_faces", case.name),
            case,
            |b, case| {
                let mut buffer = UnitQuadBuffer::new();
                b.iter(|| {
                    let mut total_quads = 0usize;
                    for chunk in &case.chunks {
                        buffer.reset();
                        visible_block_faces(
                            black_box(&chunk.voxels),
                            &chunk.shape,
                            chunk.min,
                            chunk.max,
                            &faces,
                            &mut buffer,
                        );
                        total_quads += buffer.num_quads();
                    }
                    black_box(total_quads);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("greedy_quads", case.name),
            case,
            |b, case| {
                let mut buffer = GreedyQuadsBuffer::new(case.chunks[0].voxels.len());
                b.iter(|| {
                    let mut total_quads = 0usize;
                    for chunk in &case.chunks {
                        greedy_quads(
                            black_box(&chunk.voxels),
                            &chunk.shape,
                            chunk.min,
                            chunk.max,
                            &faces,
                            &mut buffer,
                        );
                        total_quads += buffer.quads.num_quads();
                    }
                    black_box(total_quads);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("binary_greedy_quads", case.name),
            case,
            |b, case| {
                let mut buffer = BinaryGreedyQuadsBuffer::new();
                b.iter(|| {
                    let mut total_quads = 0usize;
                    for chunk in &case.chunks {
                        binary_greedy_quads(
                            black_box(&chunk.voxels),
                            &chunk.shape,
                            chunk.min,
                            chunk.max,
                            &faces,
                            &mut buffer,
                        );
                        total_quads += buffer.quads.num_quads();
                    }
                    black_box(total_quads);
                });
            },
        );

        for (name, config) in configs {
            group.bench_with_input(BenchmarkId::new(name, case.name), case, |b, case| {
                let mut buffer = BinaryGreedyQuadsBuffer::new();
                b.iter(|| {
                    let mut total_quads = 0usize;
                    for chunk in &case.chunks {
                        binary_greedy_quads_with_config(
                            black_box(&chunk.voxels),
                            &chunk.shape,
                            chunk.min,
                            chunk.max,
                            &faces,
                            &config,
                            &mut buffer,
                        );
                        total_quads += buffer.quads.num_quads();
                    }
                    black_box(total_quads);
                });
            });
        }
    }

    for case in &strategy_cases {
        group.bench_with_input(
            BenchmarkId::new("visible_block_faces", case.name),
            case,
            |b, case| {
                let mut buffer = UnitQuadBuffer::new();
                b.iter(|| {
                    buffer.reset();
                    visible_block_faces(
                        black_box(&case.voxels),
                        &case.shape,
                        case.min,
                        case.max,
                        &faces,
                        &mut buffer,
                    );
                    black_box(buffer.num_quads());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("greedy_quads", case.name),
            case,
            |b, case| {
                let mut buffer = GreedyQuadsBuffer::new(case.voxels.len());
                b.iter(|| {
                    greedy_quads(
                        black_box(&case.voxels),
                        &case.shape,
                        case.min,
                        case.max,
                        &faces,
                        &mut buffer,
                    );
                    black_box(buffer.quads.num_quads());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("binary_greedy_quads", case.name),
            case,
            |b, case| {
                let mut buffer = BinaryGreedyQuadsBuffer::new();
                b.iter(|| {
                    binary_greedy_quads(
                        black_box(&case.voxels),
                        &case.shape,
                        case.min,
                        case.max,
                        &faces,
                        &mut buffer,
                    );
                    black_box(buffer.quads.num_quads());
                });
            },
        );

        for (name, config) in configs {
            group.bench_with_input(BenchmarkId::new(name, case.name), case, |b, case| {
                let mut buffer = BinaryGreedyQuadsBuffer::new();
                b.iter(|| {
                    binary_greedy_quads_with_config(
                        black_box(&case.voxels),
                        &case.shape,
                        case.min,
                        case.max,
                        &faces,
                        &config,
                        &mut buffer,
                    );
                    black_box(buffer.quads.num_quads());
                });
            });
        }
    }

    group.finish();
}

fn benchmark_cases() -> Vec<Case> {
    vec![
        build_case(
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
                    BenchVoxel {
                        visibility: VoxelVisibility::Opaque,
                        material: (1 + ((x + y + z) % 4)) as u8,
                    }
                } else {
                    BenchVoxel::default()
                }
            },
        ),
        build_case(
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
                    BenchVoxel {
                        visibility: VoxelVisibility::Translucent,
                        material: (1 + ((x + y + z) % 4)) as u8,
                    }
                } else {
                    BenchVoxel::default()
                }
            },
        ),
        build_case(
            "translucent-shell-sphere",
            [34, 34, 34],
            [0; 3],
            [33; 3],
            |x, y, z, dims| translucent_shell_sphere_voxel(x, y, z, dims),
        ),
        build_case(
            "layered-caves",
            [34, 34, 34],
            [0; 3],
            [33; 3],
            |x, y, z, dims| {
                let base = y <= dims[1] * 7 / 10;
                let cave = hash3(x, y, z, 23) % 100 > 24;
                let ridge = hash3(x, y, z, 91) % 100 > 14;

                if base && cave && ridge {
                    BenchVoxel {
                        visibility: VoxelVisibility::Opaque,
                        material: (1 + (hash3(x, y, z, 7) % 6)) as u8,
                    }
                } else if y < dims[1] / 3 && x < dims[0] / 2 {
                    BenchVoxel {
                        visibility: VoxelVisibility::Opaque,
                        material: 2,
                    }
                } else {
                    BenchVoxel::default()
                }
            },
        ),
        build_case(
            "checkerboard",
            [34, 34, 34],
            [0; 3],
            [33; 3],
            |x, y, z, _dims| {
                if (x + y + z) % 2 == 0 {
                    BenchVoxel {
                        visibility: VoxelVisibility::Opaque,
                        material: (1 + ((x ^ y ^ z) % 4)) as u8,
                    }
                } else {
                    BenchVoxel::default()
                }
            },
        ),
        build_case(
            "partial-extent",
            [34, 34, 34],
            [4, 4, 4],
            [29, 29, 29],
            |x, y, z, _dims| match hash3(x, y, z, 131) % 8 {
                0..=4 => BenchVoxel {
                    visibility: VoxelVisibility::Opaque,
                    material: (1 + (hash3(x, y, z, 11) % 5)) as u8,
                },
                _ => BenchVoxel::default(),
            },
        ),
        build_case(
            "translucent-mix",
            [34, 34, 34],
            [0; 3],
            [33; 3],
            |x, y, z, _dims| match hash3(x, y, z, 131) % 8 {
                0..=3 => BenchVoxel {
                    visibility: VoxelVisibility::Opaque,
                    material: (1 + (hash3(x, y, z, 11) % 5)) as u8,
                },
                4 => BenchVoxel {
                    visibility: VoxelVisibility::Translucent,
                    material: (1 + (hash3(x, y, z, 17) % 3)) as u8,
                },
                _ => BenchVoxel::default(),
            },
        ),
    ]
}

fn benchmark_multi_chunk_cases() -> Vec<MultiCase> {
    vec![build_multi_chunk_case(
        "layered-caves-2x2x2",
        [2, 2, 2],
        |x, y, z| {
            let base = y <= 22;
            let cave = hash3(x as u32, y as u32, z as u32, 23) % 100 > 24;
            let ridge = hash3(x as u32, y as u32, z as u32, 91) % 100 > 14;

            if base && cave && ridge {
                BenchVoxel {
                    visibility: VoxelVisibility::Opaque,
                    material: (1 + (hash3(x as u32, y as u32, z as u32, 7) % 6)) as u8,
                }
            } else if y < 11 && (x & 63) < 32 {
                BenchVoxel {
                    visibility: VoxelVisibility::Opaque,
                    material: 2,
                }
            } else {
                BenchVoxel::default()
            }
        },
    )]
}

fn strategy_benchmark_cases() -> Vec<Case> {
    let base_cases = benchmark_cases();

    vec![
        {
            let mut case = base_cases
                .iter()
                .find(|case| case.name == "dense-sphere")
                .expect("dense-sphere benchmark case")
                .clone();
            case.name = "dense-sphere-merge-modes";
            case
        },
        {
            let mut case = base_cases
                .iter()
                .find(|case| case.name == "translucent-shell-sphere")
                .expect("translucent-shell-sphere benchmark case")
                .clone();
            case.name = "translucent-shell-sphere-merge-modes";
            case
        },
        build_case(
            "ao-t-junction-stress",
            [34, 34, 34],
            [0; 3],
            [33; 3],
            |x, y, z, _dims| {
                let terrace = y == 16
                    && ((8..=11).contains(&z) && (8..=25).contains(&x)
                        || (12..=15).contains(&z) && (8..=19).contains(&x)
                        || (16..=19).contains(&z) && (8..=13).contains(&x));
                let ribs = y == 16 && matches!((x, z), (8, 12..=19) | (14, 12..=19) | (20, 8..=15));

                if terrace || ribs {
                    BenchVoxel {
                        visibility: VoxelVisibility::Opaque,
                        material: 1,
                    }
                } else {
                    BenchVoxel::default()
                }
            },
        ),
    ]
}

fn build_case(
    name: &'static str,
    dims: [u32; 3],
    min: [u32; 3],
    max: [u32; 3],
    mut fill: impl FnMut(u32, u32, u32, [u32; 3]) -> BenchVoxel,
) -> Case {
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
                    voxels.push(BenchVoxel::default());
                } else {
                    voxels.push(fill(x, y, z, dims));
                }
            }
        }
    }

    Case {
        name,
        shape,
        voxels,
        min,
        max,
    }
}

fn build_multi_chunk_case(
    name: &'static str,
    grid: [u32; 3],
    mut fill_world: impl FnMut(i32, i32, i32) -> BenchVoxel,
) -> MultiCase {
    let dims = [34, 34, 34];
    let shape = RuntimeShape::<u32, 3>::new(dims);
    let mut chunks = Vec::with_capacity((grid[0] * grid[1] * grid[2]) as usize);

    for chunk_z in 0..grid[2] {
        for chunk_y in 0..grid[1] {
            for chunk_x in 0..grid[0] {
                let world_origin = [
                    chunk_x as i32 * 32,
                    chunk_y as i32 * 32,
                    chunk_z as i32 * 32,
                ];
                let mut voxels = Vec::with_capacity(shape.size() as usize);

                for z in 0..dims[2] {
                    for y in 0..dims[1] {
                        for x in 0..dims[0] {
                            let world_x = world_origin[0] + x as i32 - 1;
                            let world_y = world_origin[1] + y as i32 - 1;
                            let world_z = world_origin[2] + z as i32 - 1;
                            voxels.push(fill_world(world_x, world_y, world_z));
                        }
                    }
                }

                chunks.push(Case {
                    name,
                    shape: shape.clone(),
                    voxels,
                    min: [0; 3],
                    max: [33; 3],
                });
            }
        }
    }

    MultiCase { name, chunks }
}

fn hash3(x: u32, y: u32, z: u32, seed: u32) -> u32 {
    let mut v = x.wrapping_mul(0x9E37_79B1);
    v ^= y.wrapping_mul(0x85EB_CA77);
    v ^= z.wrapping_mul(0xC2B2_AE3D);
    v ^= seed.wrapping_mul(0x27D4_EB2D);
    v ^ (v >> 16)
}

fn translucent_shell_sphere_voxel(x: u32, y: u32, z: u32, dims: [u32; 3]) -> BenchVoxel {
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
        return BenchVoxel::default();
    }

    if radius_voxels < OUTER_RADIUS_VOXELS - SHELL_THICKNESS_VOXELS {
        return BenchVoxel {
            visibility: VoxelVisibility::Opaque,
            material: 2,
        };
    }

    let stripe = ((x / 3) + 2 * (y / 3) + (z / 3)) % 4;
    BenchVoxel {
        visibility: VoxelVisibility::Translucent,
        material: (stripe + 3) as u8,
    }
}

criterion_group!(benches, bench_meshers);
criterion_main!(benches);
