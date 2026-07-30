use std::hint::black_box;

use block_mesh::{QuadBuffer, RIGHT_HANDED_Y_UP_CONFIG};
use block_mesh_bgm::{binary_greedy_quads_ao_safe, BinaryGreedyQuadsBuffer};
use block_mesh_bgm_examples::{
    ao::FaceAoSampler, build_demo_samples, mesh_from_quads_with_ao, striped_sphere, SampleShape,
    SAMPLE_MAX, SAMPLE_MIN, SAMPLE_STRIDES,
};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};

fn bench_render_ao(c: &mut Criterion) {
    let samples = build_demo_samples(striped_sphere);
    let faces = RIGHT_HANDED_Y_UP_CONFIG.faces;
    let mut buffer = BinaryGreedyQuadsBuffer::new();
    binary_greedy_quads_ao_safe(
        &samples,
        &SampleShape {},
        SAMPLE_MIN,
        SAMPLE_MAX,
        &faces,
        &mut buffer,
    );
    let samplers = faces.map(|face| FaceAoSampler::new(face, SAMPLE_STRIDES));

    c.bench_function("render_ao/striped-sphere", |b| {
        b.iter(|| {
            let mut checksum = 0u32;
            for (group, sampler) in buffer.quads.groups.iter().zip(samplers.iter()) {
                for quad in group {
                    let ao = sampler.sample(black_box(&samples), quad.minimum, |voxel| {
                        voxel.visibility == block_mesh::VoxelVisibility::Opaque
                    });
                    checksum = checksum.wrapping_add(ao.into_iter().map(u32::from).sum::<u32>());
                }
            }
            black_box(checksum)
        });
    });

    let groups = buffer.quads.groups.clone();
    c.bench_function("render_mesh/striped-sphere-with-ao", |b| {
        b.iter_batched(
            || QuadBuffer {
                groups: groups.clone(),
            },
            |buffer| black_box(mesh_from_quads_with_ao(buffer, &faces, &samples)),
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, bench_render_ao);
criterion_main!(benches);
