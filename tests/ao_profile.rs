#![cfg(feature = "internal-profiler")]

use block_mesh::ndshape::{RuntimeShape, Shape};
use block_mesh::{MergeVoxel, Voxel, VoxelVisibility, RIGHT_HANDED_Y_UP_CONFIG};
use block_mesh_bgm::{
    binary_greedy_quads_with_config, with_ao_profile, AoProfile, BinaryGreedyQuadsBuffer,
    BinaryGreedyQuadsConfig,
};

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

#[test]
#[ignore]
fn prints_ao_profile_for_selected_cases() {
    let config = BinaryGreedyQuadsConfig {
        ambient_occlusion_safe: true,
    };

    for case in benchmark_cases() {
        let profile = profile_case(&case, &config);
        print_profile(case.name, &profile);
    }

    let multi = profile_multi_case("layered-caves-2x2x2", build_layered_caves_2x2x2(), &config);
    print_profile("layered-caves-2x2x2", &multi);
}

fn profile_case(case: &Case, config: &BinaryGreedyQuadsConfig) -> AoProfile {
    let mut buffer = BinaryGreedyQuadsBuffer::new();
    let mut profile = AoProfile::default();

    with_ao_profile(&mut profile, || {
        binary_greedy_quads_with_config(
            &case.voxels,
            &case.shape,
            case.min,
            case.max,
            &RIGHT_HANDED_Y_UP_CONFIG.faces,
            config,
            &mut buffer,
        );
    });

    profile
}

fn profile_multi_case(
    name: &'static str,
    cases: Vec<Case>,
    config: &BinaryGreedyQuadsConfig,
) -> AoProfile {
    let mut total = AoProfile::default();
    let chunk_count = cases.len();
    for case in cases {
        let profile = profile_case(&case, config);
        accumulate_profile(&mut total, &profile);
    }
    println!("multi-case chunks: {name} = {chunk_count}");
    total
}

fn accumulate_profile(total: &mut AoProfile, next: &AoProfile) {
    total.key_build += next.key_build;
    total.carry_total += next.carry_total;
    total.continue_mask += next.continue_mask;
    total.emit_single += next.emit_single;
    total.emit_terminal += next.emit_terminal;
    total.emit_mixed += next.emit_mixed;
    total.unit_emit += next.unit_emit;
    total.slices += next.slices;
    total.unit_slices += next.unit_slices;
    total.carry_slices += next.carry_slices;
    total.carry_rows += next.carry_rows;
    total.single_rows += next.single_rows;
    total.terminal_rows += next.terminal_rows;
    total.mixed_rows += next.mixed_rows;
    total.visible_bits += next.visible_bits;
    total.overlapping_bits += next.overlapping_bits;
    total.ao_compatible_overlap_bits += next.ao_compatible_overlap_bits;
    total.continued_bits += next.continued_bits;
    total.opaque_key_bits += next.opaque_key_bits;
    total.passthrough_key_bits += next.passthrough_key_bits;
    total.uniform_opaque_rows += next.uniform_opaque_rows;
    total.passthrough_rows += next.passthrough_rows;
    total.single_quads += next.single_quads;
    total.terminal_quads += next.terminal_quads;
    total.mixed_quads += next.mixed_quads;
    total.unit_quads += next.unit_quads;
    total.ao_rejected_rows += next.ao_rejected_rows;
}

fn print_profile(name: &str, profile: &AoProfile) {
    println!(
        "\n{name}\n  key_build={:?}\n  carry_total={:?}\n  continue_mask={:?}\n  emit_single={:?}\n  emit_terminal={:?}\n  emit_mixed={:?}\n  unit_emit={:?}\n  slices={} unit_slices={} carry_slices={}\n  carry_rows={} single_rows={} terminal_rows={} mixed_rows={}\n  visible_bits={} overlapping_bits={} ao_compatible_overlap_bits={} continued_bits={}\n  opaque_key_bits={} passthrough_key_bits={} uniform_opaque_rows={} passthrough_rows={} ao_rejected_rows={}\n  single_quads={} terminal_quads={} mixed_quads={} unit_quads={}",
        profile.key_build,
        profile.carry_total,
        profile.continue_mask,
        profile.emit_single,
        profile.emit_terminal,
        profile.emit_mixed,
        profile.unit_emit,
        profile.slices,
        profile.unit_slices,
        profile.carry_slices,
        profile.carry_rows,
        profile.single_rows,
        profile.terminal_rows,
        profile.mixed_rows,
        profile.visible_bits,
        profile.overlapping_bits,
        profile.ao_compatible_overlap_bits,
        profile.continued_bits,
        profile.opaque_key_bits,
        profile.passthrough_key_bits,
        profile.uniform_opaque_rows,
        profile.passthrough_rows,
        profile.ao_rejected_rows,
        profile.single_quads,
        profile.terminal_quads,
        profile.mixed_quads,
        profile.unit_quads,
    );
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
                    && (((6..=8).contains(&x) && z == 5)
                        || (x == 5 && (6..=8).contains(&z))
                        || (x == 13 && z == 5)
                        || ((22..=24).contains(&x) && z == 5));

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

fn build_layered_caves_2x2x2() -> Vec<Case> {
    let dims = [34, 34, 34];
    let shape = RuntimeShape::<u32, 3>::new(dims);
    let mut chunks = Vec::with_capacity(8);

    for chunk_z in 0..2 {
        for chunk_y in 0..2 {
            for chunk_x in 0..2 {
                let world_origin = [chunk_x * 32, chunk_y * 32, chunk_z * 32];
                let mut voxels = Vec::with_capacity(shape.size() as usize);

                for z in 0..dims[2] {
                    for y in 0..dims[1] {
                        for x in 0..dims[0] {
                            let world_x = world_origin[0] + x as i32 - 1;
                            let world_y = world_origin[1] + y as i32 - 1;
                            let world_z = world_origin[2] + z as i32 - 1;
                            voxels.push(layered_caves_world_voxel(world_x, world_y, world_z));
                        }
                    }
                }

                chunks.push(Case {
                    name: "layered-caves-2x2x2",
                    shape: shape.clone(),
                    voxels,
                    min: [0; 3],
                    max: [33; 3],
                });
            }
        }
    }

    chunks
}

fn layered_caves_world_voxel(x: i32, y: i32, z: i32) -> BenchVoxel {
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

fn hash3(x: u32, y: u32, z: u32, seed: u32) -> u32 {
    let mut v = x.wrapping_mul(0x9E37_79B1);
    v ^= y.wrapping_mul(0x85EB_CA77);
    v ^= z.wrapping_mul(0xC2B2_AE3D);
    v ^= seed.wrapping_mul(0x27D4_EB2D);
    v ^ (v >> 16)
}
