use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::Rng;
use similarity::*;
use std::hint::black_box;

fn generate_random_vectors_f64(size: usize) -> (Vec<f64>, Vec<f64>) {
    let mut rng = rand::rng();
    let a: Vec<f64> = (0..size).map(|_| rng.random_range(-1.0..1.0)).collect();
    let b: Vec<f64> = (0..size).map(|_| rng.random_range(-1.0..1.0)).collect();
    (a, b)
}

// ---------------------------------------------------------------------------
// Dot product
// ---------------------------------------------------------------------------

fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product_comparison");
    for size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536, 262144] {
        let (a, b) = generate_random_vectors_f64(size);

        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |bench, _| {
            bench.iter(|| {
                black_box(&a)
                    .iter()
                    .zip(black_box(&b).iter())
                    .map(|(&x, &y)| x * y)
                    .sum::<f64>()
            });
        });
        group.bench_with_input(BenchmarkId::new("simd", size), &size, |bench, _| {
            bench.iter(|| dot_product_simd(black_box(&a), black_box(&b)));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Cosine similarity
// ---------------------------------------------------------------------------

fn bench_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity_comparison");
    for size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536, 262144] {
        let (a, b) = generate_random_vectors_f64(size);

        group.bench_with_input(BenchmarkId::new("generic", size), &size, |bench, _| {
            bench.iter(|| cosine_similarity(black_box(&a), black_box(&b)));
        });
        group.bench_with_input(BenchmarkId::new("simd", size), &size, |bench, _| {
            bench.iter(|| cosine_similarity_simd(black_box(&a), black_box(&b)));
        });
        #[cfg(feature = "parallel")]
        group.bench_with_input(BenchmarkId::new("parallel", size), &size, |bench, _| {
            bench.iter(|| cosine_similarity_parallel(black_box(&a), black_box(&b)));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Euclidean distance
// ---------------------------------------------------------------------------

fn bench_euclidean_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("euclidean_distance_comparison");
    for size in [8, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536, 262144] {
        let (a, b) = generate_random_vectors_f64(size);

        group.bench_with_input(BenchmarkId::new("generic", size), &size, |bench, _| {
            bench.iter(|| euclidean_distance(black_box(&a), black_box(&b)));
        });
        group.bench_with_input(BenchmarkId::new("simd", size), &size, |bench, _| {
            bench.iter(|| euclidean_distance_simd(black_box(&a), black_box(&b)));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Squared Euclidean distance
// ---------------------------------------------------------------------------

fn bench_squared_euclidean_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("squared_euclidean_distance_comparison");
    for size in [8, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536, 262144] {
        let (a, b) = generate_random_vectors_f64(size);

        group.bench_with_input(BenchmarkId::new("generic", size), &size, |bench, _| {
            bench.iter(|| squared_euclidean_distance(black_box(&a), black_box(&b)));
        });
        group.bench_with_input(BenchmarkId::new("simd", size), &size, |bench, _| {
            bench.iter(|| squared_euclidean_distance_simd(black_box(&a), black_box(&b)));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Manhattan distance
// ---------------------------------------------------------------------------

fn bench_manhattan_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("manhattan_distance_comparison");
    for size in [8, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536, 262144] {
        let (a, b) = generate_random_vectors_f64(size);

        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |bench, _| {
            bench.iter(|| {
                black_box(&a)
                    .iter()
                    .zip(black_box(&b).iter())
                    .map(|(&x, &y)| (x - y).abs())
                    .sum::<f64>()
            });
        });
        group.bench_with_input(BenchmarkId::new("simd", size), &size, |bench, _| {
            bench.iter(|| manhattan_distance_simd(black_box(&a), black_box(&b)));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Pearson correlation distance
// ---------------------------------------------------------------------------

fn bench_pearson_correlation_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("pearson_correlation_distance_comparison");
    for size in [8, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536, 262144] {
        let (a, b) = generate_random_vectors_f64(size);

        group.bench_with_input(BenchmarkId::new("generic", size), &size, |bench, _| {
            bench.iter(|| pearson_correlation_distance(black_box(&a), black_box(&b)));
        });
        group.bench_with_input(BenchmarkId::new("simd", size), &size, |bench, _| {
            bench.iter(|| pearson_correlation_distance_simd(black_box(&a), black_box(&b)));
        });
        #[cfg(feature = "parallel")]
        group.bench_with_input(BenchmarkId::new("parallel", size), &size, |bench, _| {
            bench.iter(|| pearson_correlation_distance_parallel(black_box(&a), black_box(&b)));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Cross-correlation  (O(n²) — use smaller sizes)
// ---------------------------------------------------------------------------

fn bench_cross_correlate(c: &mut Criterion) {
    let mut group = c.benchmark_group("cross_correlate_comparison");
    for size in [32, 64, 128, 256, 512, 1024] {
        let (a, b) = generate_random_vectors_f64(size);

        group.bench_with_input(BenchmarkId::new("generic", size), &size, |bench, _| {
            bench.iter(|| cross_correlate(black_box(&a), black_box(&b)));
        });
        group.bench_with_input(BenchmarkId::new("simd", size), &size, |bench, _| {
            bench.iter(|| cross_correlate_simd(black_box(&a), black_box(&b)));
        });
        #[cfg(feature = "parallel")]
        group.bench_with_input(BenchmarkId::new("parallel", size), &size, |bench, _| {
            bench.iter(|| cross_correlate_parallel(black_box(&a), black_box(&b)));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_dot_product,
    bench_cosine_similarity,
    bench_euclidean_distance,
    bench_squared_euclidean_distance,
    bench_manhattan_distance,
    bench_pearson_correlation_distance,
    bench_cross_correlate,
);
criterion_main!(benches);
