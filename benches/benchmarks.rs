use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use std::hint::black_box;
use similarity::similarity_metrics::*;
use rand::prelude::*;
use rand;
use std::time::Duration;

fn create_test_data(size: usize) -> Vec<f64> {
    let mut rng = rand::rng();
    let mut data = Vec::with_capacity(size);
    for _ in 0..size {
        // Generate data between -1.0 and 1.0 to be more realistic
        data.push(rng.random_range(-1.0..1.0));
    }
    data
}

fn create_shifted_data(original: &[f64], shift: usize) -> Vec<f64> {
    let mut shifted = vec![0.0; original.len()];
    for (i, &val) in original.iter().enumerate() {
        if i + shift < original.len() {
            shifted[i + shift] = val;
        }
    }
    shifted
}

pub fn bench_cross_correlate(c: &mut Criterion) {
    let mut group = c.benchmark_group("Cross Correlation");
    group.sample_size(100); // Increased sample size
    group.measurement_time(Duration::from_secs(10)); // Longer measurement time
    
    let sizes = [100, 1000, 10000];
    
    for size in sizes {
        let data = create_test_data(size);
        let shifted_data = create_shifted_data(&data, size / 4);
        
        group.bench_with_input(BenchmarkId::new("time_domain_optimized", size), &size, |b, _| {
            b.iter(|| {
                let result = cross_correlate(black_box(&data), black_box(&shifted_data));
                black_box(result)
            });
        });

        group.bench_with_input(BenchmarkId::new("time_domain_parallel", size), &size, |b, _| {
            b.iter(|| {
                let result = cross_correlate_parallel(black_box(&data), black_box(&shifted_data));
                black_box(result)
            });
        });

        group.bench_with_input(BenchmarkId::new("fft_optimized", size), &size, |b, _| {
            b.iter(|| {
                let result = cross_correlate_fft(black_box(&data), black_box(&shifted_data));
                black_box(result)
            });
        });
    }

    group.finish();
}

pub fn bench_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("Cosine Similarity");
    group.sample_size(100);
    group.measurement_time(Duration::from_secs(10));
    
    let sizes = [100, 1000, 10000];
    
    for size in sizes {
        let data1 = create_test_data(size);
        let data2 = create_test_data(size);
        
        group.bench_with_input(BenchmarkId::new("original", size), &size, |b, _| {
            b.iter(|| {
                let result = cosine_similarity(black_box(&data1), black_box(&data2));
                black_box(result)
            });
        });



        group.bench_with_input(BenchmarkId::new("parallel", size), &size, |b, _| {
            b.iter(|| {
                let result = cosine_similarity_parallel(black_box(&data1), black_box(&data2));
                black_box(result)
            });
        });
    }

    group.finish();
}

pub fn bench_cosine_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("Cosine Distance");
    group.sample_size(100);
    group.measurement_time(Duration::from_secs(10));
    
    let sizes = [100, 1000, 10000];
    
    for size in sizes {
        let data1 = create_test_data(size);
        let data2 = create_test_data(size);
        
        group.bench_with_input(BenchmarkId::new("original", size), &size, |b, _| {
            b.iter(|| {
                let result = cosine_distance(black_box(&data1), black_box(&data2));
                black_box(result)
            });
        });



        group.bench_with_input(BenchmarkId::new("parallel", size), &size, |b, _| {
            b.iter(|| {
                let result = cosine_distance_parallel(black_box(&data1), black_box(&data2));
                black_box(result)
            });
        });
    }

    group.finish();
}

pub fn bench_pearson_correlation(c: &mut Criterion) {
    let mut group = c.benchmark_group("pearson_correlation");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(100);

    let sizes = [100, 1000, 10000];
    for size in sizes {
        let data1 = create_test_data(size);
        let data2 = create_test_data(size);

        group.bench_with_input(BenchmarkId::new("standard", size), &size, |b, _| {
            b.iter(|| black_box(pearson_correlation_distance(&data1, &data2)))
        });



        group.bench_with_input(BenchmarkId::new("parallel", size), &size, |b, _| {
            b.iter(|| black_box(pearson_correlation_distance_parallel(&data1, &data2)))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_cross_correlate,
    bench_cosine_similarity,
    bench_cosine_distance,
    bench_pearson_correlation
);
criterion_main!(benches);
