use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use std::hint::black_box;
use similarity::spectral_entropy::*;
use rand::prelude::*;

fn create_test_spectrum(size: usize) -> Spectrum<f64> {
    let mut rng = rand::rng();
    let mzs: Vec<f64> = (0..size).map(|i| 100.0 + i as f64).collect();
    let intensities: Vec<f64> = (0..size)
        .map(|_| rng.random_range(0.0..1.0))
        .collect();
    Spectrum::from_arrays(&mzs, &intensities).unwrap()
}

fn bench_entropy_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Entropy Calculation");
    group.sample_size(100);
    group.measurement_time(std::time::Duration::from_secs(10));

    for size in [100, 1000, 10000].iter() {
        let spectrum = create_test_spectrum(*size);
        
        group.bench_with_input(BenchmarkId::new("Original", size), size, |b, _| {
            b.iter(|| calculate_entropy(black_box(&spectrum)))
        });
        

    }
    group.finish();
}

fn bench_tsallis_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("Tsallis Entropy");
    group.sample_size(100);
    group.measurement_time(std::time::Duration::from_secs(10));

    for size in [100, 1000, 10000].iter() {
        let spectrum = create_test_spectrum(*size);
        let q = 2.0;
        
        group.bench_with_input(BenchmarkId::new("Original", size), size, |b, _| {
            b.iter(|| calculate_tsallis_entropy(black_box(&spectrum), black_box(q)))
        });
        
        group.bench_with_input(BenchmarkId::new("Optimized", size), size, |b, _| {
            b.iter(|| calculate_tsallis_entropy_optimized(black_box(&spectrum), black_box(q)))
        });
    }
    group.finish();
}

fn bench_entropy_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("Entropy Similarity");
    group.sample_size(100);
    group.measurement_time(std::time::Duration::from_secs(10));

    for size in [100, 1000, 10000].iter() {
        let spectrum1 = create_test_spectrum(*size);
        let spectrum2 = create_test_spectrum(*size);
        
        group.bench_with_input(BenchmarkId::new("Original", size), size, |b, _| {
            b.iter(|| calculate_entropy_similarity(black_box(&spectrum1), black_box(&spectrum2)))
        });
        

    }
    group.finish();
}

criterion_group!(
    benches,
    bench_entropy_calculation,
    bench_tsallis_entropy,
    bench_entropy_similarity
);
criterion_main!(benches); 