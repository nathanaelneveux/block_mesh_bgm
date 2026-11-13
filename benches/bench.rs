use std::hint::black_box;

use block_mesh::ndshape::{RuntimeShape, Shape};
use block_mesh::{
    greedy_quads, visible_block_faces, GreedyQuadsBuffer, MergeVoxel, UnitQuadBuffer, Voxel,
    VoxelVisibility, RIGHT_HANDED_Y_UP_CONFIG,
};
use block_mesh_bgm::{
    binary_greedy_quads, binary_greedy_quads_ao_safe, binary_visible_block_faces,
    BinaryGreedyQuadsBuffer, BinaryUnitQuadBuffer,
};
use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion};

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
    let faces = RIGHT_HANDED_Y_UP_CONFIG.faces;
    let cases: Vec<MultiCase> = benchmark_cases()
        .into_iter()
        .map(MultiCase::single)
        .chain(benchmark_multi_chunk_cases())
        .chain(ao_benchmark_cases().into_iter().map(MultiCase::single))
        .chain(large_model_cases().into_iter().map(MultiCase::single))
        .collect();
    let mut group = c.benchmark_group("meshers");

    for case in &cases {
        bench_mesher(
            &mut group,
            "visible_block_faces",
            case,
            UnitQuadBuffer::new,
            |chunk, buffer| {
                buffer.reset();
                visible_block_faces(
                    &chunk.voxels,
                    &chunk.shape,
                    chunk.min,
                    chunk.max,
                    &faces,
                    buffer,
                );
                buffer.num_quads()
            },
        );
        bench_mesher(
            &mut group,
            "binary_visible_block_faces",
            case,
            BinaryUnitQuadBuffer::new,
            |chunk, buffer| {
                binary_visible_block_faces(
                    &chunk.voxels,
                    &chunk.shape,
                    chunk.min,
                    chunk.max,
                    &faces,
                    buffer,
                );
                buffer.quads.num_quads()
            },
        );
        let greedy_len = case.chunks[0].voxels.len();
        bench_mesher(
            &mut group,
            "greedy_quads",
            case,
            || GreedyQuadsBuffer::new(greedy_len),
            |chunk, buffer| {
                greedy_quads(
                    &chunk.voxels,
                    &chunk.shape,
                    chunk.min,
                    chunk.max,
                    &faces,
                    buffer,
                );
                buffer.quads.num_quads()
            },
        );
        bench_mesher(
            &mut group,
            "binary_greedy_quads",
            case,
            BinaryGreedyQuadsBuffer::new,
            |chunk, buffer| {
                binary_greedy_quads(
                    &chunk.voxels,
                    &chunk.shape,
                    chunk.min,
                    chunk.max,
                    &faces,
                    buffer,
                );
                buffer.quads.num_quads()
            },
        );
        bench_mesher(
            &mut group,
            "binary_greedy_quads_ao_safe",
            case,
            BinaryGreedyQuadsBuffer::new,
            |chunk, buffer| {
                binary_greedy_quads_ao_safe(
                    &chunk.voxels,
                    &chunk.shape,
                    chunk.min,
                    chunk.max,
                    &faces,
                    buffer,
                );
                buffer.quads.num_quads()
            },
        );
    }

    group.finish();
}

/// Benchmarks one mesher over every chunk of `case`, reusing one buffer.
fn bench_mesher<B>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    mesher: &str,
    case: &MultiCase,
    new_buffer: impl Fn() -> B,
    mesh: impl Fn(&Case, &mut B) -> usize,
) {
    group.bench_with_input(BenchmarkId::new(mesher, case.name), case, |b, case| {
        let mut buffer = new_buffer();
        b.iter(|| {
            let total_quads: usize = case
                .chunks
                .iter()
                .map(|chunk| mesh(black_box(chunk), &mut buffer))
                .sum();
            black_box(total_quads)
        });
    });
}

impl MultiCase {
    fn single(case: Case) -> Self {
        Self {
            name: case.name,
            chunks: vec![case],
        }
    }
}

/// Whole models wider than one 62-voxel tile, like a `.vox` file meshed in
/// one call.
fn large_model_cases() -> Vec<Case> {
    vec![build_case(
        "large-model-130x66x98",
        [130, 66, 98],
        [0; 3],
        [129, 65, 97],
        |x, y, z, dims| {
            let height = dims[1] / 3 + hash3(x / 7, 0, z / 7, 5) % (dims[1] / 3);
            if y < height && hash3(x, y, z, 29) % 100 > 8 {
                BenchVoxel {
                    visibility: VoxelVisibility::Opaque,
                    material: (1 + hash3(x / 4, y / 4, z / 4, 3) % 3) as u8,
                }
            } else {
                BenchVoxel::default()
            }
        },
    )]
}

fn benchmark_cases() -> Vec<Case> {
    vec![
        build_case(
            "empty",
            [34, 34, 34],
            [0; 3],
            [33; 3],
            |_x, _y, _z, _dims| BenchVoxel::default(),
        ),
        {
            let mut case = build_case(
                "uniform-opaque",
                [34, 34, 34],
                [0; 3],
                [33; 3],
                |_x, _y, _z, _dims| BenchVoxel::default(),
            );
            case.voxels.fill(BenchVoxel {
                visibility: VoxelVisibility::Opaque,
                material: 1,
            });
            case
        },
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
            translucent_shell_sphere_voxel,
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

fn ao_benchmark_cases() -> Vec<Case> {
    let base_cases = benchmark_cases();

    vec![
        {
            let mut case = base_cases
                .iter()
                .find(|case| case.name == "dense-sphere")
                .expect("dense-sphere benchmark case")
                .clone();
            case.name = "dense-sphere-ao";
            case
        },
        {
            let mut case = base_cases
                .iter()
                .find(|case| case.name == "translucent-shell-sphere")
                .expect("translucent-shell-sphere benchmark case")
                .clone();
            case.name = "translucent-shell-sphere-ao";
            case
        },
        build_case(
            "ao-boundary-stress",
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
        build_case(
            "ao-unit-patterns",
            [34, 34, 34],
            [0; 3],
            [33; 3],
            |x, y, z, _dims| {
                let lower_plate = y == 16
                    && ((6..=10).contains(&x) && (6..=10).contains(&z)
                        || (14..=18).contains(&x) && (6..=10).contains(&z)
                        || (22..=26).contains(&x) && (6..=10).contains(&z));

                let upper_lips = y == 17
                    && (
                        // Interior-corner pattern: a two-sided upper lip.
                        ((6..=8).contains(&x) && z == 5) || (x == 5 && (6..=8).contains(&z))
                        // Diagonal-from-corner pattern: a single upper corner.
                        || (x == 13 && z == 5)
                        // Side-run corner pattern: a one-voxel-deep upper strip.
                        || ((22..=24).contains(&x) && z == 5)
                    );

                if lower_plate || upper_lips {
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
