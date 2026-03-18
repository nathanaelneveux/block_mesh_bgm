#![cfg(feature = "profiling")]

use block_mesh::ndshape::{RuntimeShape, Shape};
use block_mesh::{MergeVoxel, Voxel, VoxelVisibility, RIGHT_HANDED_Y_UP_CONFIG};
use block_mesh_bgm::{
    binary_greedy_quads_profile, BinaryGreedyPhaseProfile, BinaryGreedyQuadsBuffer,
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

fn main() {
    let iterations = env_usize("PROFILE_ITERS", 300);
    let warmup = env_usize("PROFILE_WARMUP", 50);
    let faces = RIGHT_HANDED_Y_UP_CONFIG.faces;
    let cases = important_cases();

    println!("Phase profile");
    println!("iterations: {iterations}, warmup: {warmup}");
    println!();

    for case in cases {
        let mut buffer = BinaryGreedyQuadsBuffer::new();

        for _ in 0..warmup {
            let _ = binary_greedy_quads_profile(
                &case.voxels,
                &case.shape,
                case.min,
                case.max,
                &faces,
                &mut buffer,
            );
        }

        let mut sum = BinaryGreedyPhaseProfile::default();
        for _ in 0..iterations {
            let profile = binary_greedy_quads_profile(
                &case.voxels,
                &case.shape,
                case.min,
                case.max,
                &faces,
                &mut buffer,
            );
            sum.source_masks += profile.source_masks;
            sum.visibility_masks += profile.visibility_masks;
            sum.scan_masks += profile.scan_masks;
            sum.unit_emit += profile.unit_emit;
            sum.merge += profile.merge;
            sum.total += profile.total;
        }

        let total = avg_ns(sum.total, iterations);
        let source_masks = avg_ns(sum.source_masks, iterations);
        let visibility_masks = avg_ns(sum.visibility_masks, iterations);
        let scan_masks = avg_ns(sum.scan_masks, iterations);
        let unit_emit = avg_ns(sum.unit_emit, iterations);
        let merge = avg_ns(sum.merge, iterations);

        println!("Case: {}", case.name);
        println!("  total: {:.2} us", total / 1_000.0);
        println!(
            "  source_masks: {:.2} us ({:.1}%)",
            source_masks / 1_000.0,
            pct(source_masks, total)
        );
        println!(
            "  visibility_masks: {:.2} us ({:.1}%)",
            visibility_masks / 1_000.0,
            pct(visibility_masks, total)
        );
        println!(
            "  scan_masks: {:.2} us ({:.1}%)",
            scan_masks / 1_000.0,
            pct(scan_masks, total)
        );
        println!(
            "  unit_emit: {:.2} us ({:.1}%)",
            unit_emit / 1_000.0,
            pct(unit_emit, total)
        );
        println!(
            "  merge: {:.2} us ({:.1}%)",
            merge / 1_000.0,
            pct(merge, total)
        );
        println!(
            "  accounted: {:.2} us ({:.1}%)",
            (source_masks + visibility_masks + scan_masks + unit_emit + merge) / 1_000.0,
            pct(
                source_masks + visibility_masks + scan_masks + unit_emit + merge,
                total
            )
        );
        println!();
    }
}

fn important_cases() -> Vec<Case> {
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

fn hash3(x: u32, y: u32, z: u32, seed: u32) -> u32 {
    let mut v = x.wrapping_mul(0x9E37_79B1);
    v ^= y.wrapping_mul(0x85EB_CA77);
    v ^= z.wrapping_mul(0xC2B2_AE3D);
    v ^= seed.wrapping_mul(0x27D4_EB2D);
    v ^ (v >> 16)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn avg_ns(duration: std::time::Duration, iterations: usize) -> f64 {
    duration.as_nanos() as f64 / iterations as f64
}

fn pct(part_ns: f64, total_ns: f64) -> f64 {
    if total_ns == 0.0 {
        0.0
    } else {
        100.0 * part_ns / total_ns
    }
}
