use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use similarity::*;
use similarity::similarity_traits::*;
use std::collections::HashSet;
use std::time::Duration;
use std::hint::black_box;
use rand::prelude::*;

// ============================================================================
// TEST DATA GENERATORS
// ============================================================================

fn create_test_data(size: usize) -> Vec<f64> {
    let mut rng = rand::rng();
    (0..size).map(|_| rng.random_range(-1.0..1.0)).collect()
}

fn create_test_sets(size: usize) -> (HashSet<u32>, HashSet<u32>) {
    let mut rng = rand::rng();
    let mut set1 = HashSet::new();
    let mut set2 = HashSet::new();
    
    // Create overlapping sets for realistic Jaccard testing
    for _ in 0..size {
        let val = rng.random_range(0..size as u32 * 2);
        if rng.random_bool(0.6) {
            set1.insert(val);
        }
        if rng.random_bool(0.6) {
            set2.insert(val);
        }
    }
    
    (set1, set2)
}

fn create_prediction_data(size: usize) -> (Vec<f64>, Vec<f64>) {
    let mut rng = rand::rng();
    let actual: Vec<f64> = (0..size).map(|_| rng.random_range(0.0..10.0)).collect();
    let predicted: Vec<f64> = actual.iter()
        .map(|&x| x + rng.random_range(-0.5..0.5)) // Add some noise
        .collect();
    (actual, predicted)
}

// ============================================================================
// COSINE SIMILARITY BENCHMARKS
// ============================================================================

fn bench_cosine_similarity_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Cosine Similarity - Function vs Trait");
    group.sample_size(200);
    group.measurement_time(Duration::from_secs(15));
    
    let sizes = [100, 1000, 10000, 100000];
    
    for size in sizes {
        let data1 = create_test_data(size);
        let data2 = create_test_data(size);
        
        // Functional API
        group.bench_with_input(BenchmarkId::new("Function/Standard", size), &size, |b, _| {
            b.iter(|| black_box(cosine_similarity(black_box(&data1), black_box(&data2))))
        });
        
        #[cfg(feature = "parallel")]
        group.bench_with_input(BenchmarkId::new("Function/Parallel", size), &size, |b, _| {
            b.iter(|| black_box(cosine_similarity_parallel(black_box(&data1), black_box(&data2))))
        });
        
        // Trait API - should have identical performance
        group.bench_with_input(BenchmarkId::new("Trait/Standard", size), &size, |b, _| {
            b.iter(|| black_box(CosineSimilarity::similarity(black_box((&data1, &data2)))))
        });
        
        #[cfg(feature = "parallel")]
        group.bench_with_input(BenchmarkId::new("Trait/Parallel", size), &size, |b, _| {
            b.iter(|| black_box(CosineSimilarityParallel::similarity(black_box((&data1, &data2)))))
        });
    }
    
    group.finish();
}

// ============================================================================
// COSINE DISTANCE BENCHMARKS
// ============================================================================

fn bench_cosine_distance_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Cosine Distance - Function vs Trait");
    group.sample_size(200);
    group.measurement_time(Duration::from_secs(10));
    
    let sizes = [100, 1000, 10000];
    
    for size in sizes {
        let data1 = create_test_data(size);
        let data2 = create_test_data(size);
        
        // Functional API
        group.bench_with_input(BenchmarkId::new("Function/Standard", size), &size, |b, _| {
            b.iter(|| black_box(cosine_distance(black_box(&data1), black_box(&data2))))
        });
        
        #[cfg(feature = "parallel")]
        group.bench_with_input(BenchmarkId::new("Function/Parallel", size), &size, |b, _| {
            b.iter(|| black_box(cosine_distance_parallel(black_box(&data1), black_box(&data2))))
        });
        
        // Trait API
        group.bench_with_input(BenchmarkId::new("Trait/Standard", size), &size, |b, _| {
            b.iter(|| black_box(CosineDistance::similarity(black_box((&data1, &data2)))))
        });
        
        #[cfg(feature = "parallel")]
        group.bench_with_input(BenchmarkId::new("Trait/Parallel", size), &size, |b, _| {
            b.iter(|| black_box(CosineDistanceParallel::similarity(black_box((&data1, &data2)))))
        });
    }
    
    group.finish();
}

// ============================================================================
// EUCLIDEAN DISTANCE BENCHMARKS
// ============================================================================

fn bench_euclidean_distance_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Euclidean Distance - Function vs Trait");
    group.sample_size(200);
    group.measurement_time(Duration::from_secs(10));
    
    let sizes = [100, 1000, 10000];
    
    for size in sizes {
        let data1 = create_test_data(size);
        let data2 = create_test_data(size);
        
        // Functional API
        group.bench_with_input(BenchmarkId::new("Function/Euclidean", size), &size, |b, _| {
            b.iter(|| black_box(euclidean_distance(black_box(&data1), black_box(&data2))))
        });
        
        group.bench_with_input(BenchmarkId::new("Function/Squared", size), &size, |b, _| {
            b.iter(|| black_box(squared_euclidean_distance(black_box(&data1), black_box(&data2))))
        });
        
        // Trait API
        group.bench_with_input(BenchmarkId::new("Trait/Euclidean", size), &size, |b, _| {
            b.iter(|| black_box(EuclideanDistance::similarity(black_box((&data1, &data2)))))
        });
        
        group.bench_with_input(BenchmarkId::new("Trait/Squared", size), &size, |b, _| {
            b.iter(|| black_box(SquaredEuclideanDistance::similarity(black_box((&data1, &data2)))))
        });
    }
    
    group.finish();
}

// ============================================================================
// PEARSON CORRELATION BENCHMARKS
// ============================================================================

fn bench_pearson_correlation_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Pearson Correlation - Function vs Trait");
    group.sample_size(200);
    group.measurement_time(Duration::from_secs(10));
    
    let sizes = [100, 1000, 10000];
    
    for size in sizes {
        let data1 = create_test_data(size);
        let data2 = create_test_data(size);
        
        // Functional API
        group.bench_with_input(BenchmarkId::new("Function/Standard", size), &size, |b, _| {
            b.iter(|| black_box(pearson_correlation_distance(black_box(&data1), black_box(&data2))))
        });
        
        #[cfg(feature = "parallel")]
        group.bench_with_input(BenchmarkId::new("Function/Parallel", size), &size, |b, _| {
            b.iter(|| black_box(pearson_correlation_distance_parallel(black_box(&data1), black_box(&data2))))
        });
        
        // Trait API
        group.bench_with_input(BenchmarkId::new("Trait/Standard", size), &size, |b, _| {
            b.iter(|| black_box(PearsonCorrelationDistance::similarity(black_box((&data1, &data2)))))
        });
        
        #[cfg(feature = "parallel")]
        group.bench_with_input(BenchmarkId::new("Trait/Parallel", size), &size, |b, _| {
            b.iter(|| black_box(PearsonCorrelationDistanceParallel::similarity(black_box((&data1, &data2)))))
        });
    }
    
    group.finish();
}

// ============================================================================
// JACCARD INDEX BENCHMARKS
// ============================================================================

fn bench_jaccard_index_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Jaccard Index - Function vs Trait");
    group.sample_size(200);
    group.measurement_time(Duration::from_secs(10));
    
    let sizes = [100, 1000, 10000];
    
    for size in sizes {
        let (set1, set2) = create_test_sets(size);
        
        // Functional API
        group.bench_with_input(BenchmarkId::new("Function", size), &size, |b, _| {
            b.iter(|| black_box(jaccard_index(black_box(&set1), black_box(&set2))))
        });
        
        // Trait API
        group.bench_with_input(BenchmarkId::new("Trait", size), &size, |b, _| {
            b.iter(|| black_box(JaccardIndex::similarity(black_box((&set1, &set2)))))
        });
    }
    
    group.finish();
}

// ============================================================================
// PREDICTION ACCURACY BENCHMARKS
// ============================================================================

fn bench_prediction_accuracy_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Prediction Accuracy - Function vs Trait");
    group.sample_size(200);
    group.measurement_time(Duration::from_secs(10));
    
    let sizes = [100, 1000, 10000];
    let tolerance = 0.5;
    
    for size in sizes {
        let (actual, predicted) = create_prediction_data(size);
        
        // Hit Rate
        group.bench_with_input(BenchmarkId::new("HitRate/Function", size), &size, |b, _| {
            b.iter(|| black_box(hit_rate(black_box(&actual), black_box(&predicted), black_box(tolerance))))
        });
        
        group.bench_with_input(BenchmarkId::new("HitRate/Trait", size), &size, |b, _| {
            b.iter(|| black_box(HitRate::similarity(black_box((&actual, &predicted, tolerance)))))
        });
        
        // Overshoot Rate
        group.bench_with_input(BenchmarkId::new("OvershootRate/Function", size), &size, |b, _| {
            b.iter(|| black_box(overshoot_rate(black_box(&actual), black_box(&predicted), black_box(tolerance))))
        });
        
        group.bench_with_input(BenchmarkId::new("OvershootRate/Trait", size), &size, |b, _| {
            b.iter(|| black_box(OvershootRate::similarity(black_box((&actual, &predicted, tolerance)))))
        });
    }
    
    group.finish();
}

// ============================================================================
// CROSS-CORRELATION BENCHMARKS
// ============================================================================

fn bench_cross_correlation_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Cross-Correlation - Function vs Trait");
    group.sample_size(100);
    group.measurement_time(Duration::from_secs(15));
    
    let sizes = [100, 1000, 5000]; // Smaller sizes for cross-correlation due to O(n²) complexity
    
    for size in sizes {
        let data1 = create_test_data(size);
        let data2 = create_test_data(size);
        
        // Functional API
        group.bench_with_input(BenchmarkId::new("Function/Optimized", size), &size, |b, _| {
            b.iter(|| black_box(cross_correlate(black_box(&data1), black_box(&data2))))
        });
        
        #[cfg(feature = "parallel")]
        group.bench_with_input(BenchmarkId::new("Function/Parallel", size), &size, |b, _| {
            b.iter(|| black_box(cross_correlate_parallel(black_box(&data1), black_box(&data2))))
        });
        
        #[cfg(feature = "fft")]
        group.bench_with_input(BenchmarkId::new("Function/FFT", size), &size, |b, _| {
            b.iter(|| black_box(cross_correlate_fft(black_box(&data1), black_box(&data2))))
        });
        
        // Trait API
        group.bench_with_input(BenchmarkId::new("Trait/Optimized", size), &size, |b, _| {
            b.iter(|| black_box(CrossCorrelationOptimized::similarity(black_box((&data1, &data2)))))
        });
        
        #[cfg(feature = "parallel")]
        group.bench_with_input(BenchmarkId::new("Trait/Parallel", size), &size, |b, _| {
            b.iter(|| black_box(CrossCorrelationParallel::similarity(black_box((&data1, &data2)))))
        });
        
        #[cfg(feature = "fft")]
        group.bench_with_input(BenchmarkId::new("Trait/FFT", size), &size, |b, _| {
            b.iter(|| black_box(CrossCorrelationFFT::similarity(black_box((&data1, &data2)))))
        });
    }
    
    group.finish();
}

// ============================================================================
// TIME SHIFT DETECTION BENCHMARKS
// ============================================================================

fn bench_time_shift_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Time Shift Detection - Function vs Trait");
    group.sample_size(100);
    group.measurement_time(Duration::from_secs(10));
    
    let sizes = [100, 1000, 5000];
    
    for size in sizes {
        let data1 = create_test_data(size);
        let data2 = create_test_data(size);
        
        // Functional API
        group.bench_with_input(BenchmarkId::new("Function/Standard", size), &size, |b, _| {
            b.iter(|| black_box(find_time_shift(black_box(&data1), black_box(&data2))))
        });
        
        #[cfg(feature = "fft")]
        group.bench_with_input(BenchmarkId::new("Function/FFT", size), &size, |b, _| {
            b.iter(|| black_box(find_time_shift_fft(black_box(&data1), black_box(&data2))))
        });
        
        // Trait API
        group.bench_with_input(BenchmarkId::new("Trait/Standard", size), &size, |b, _| {
            b.iter(|| black_box(TimeShiftFinder::similarity(black_box((&data1, &data2)))))
        });
        
        #[cfg(feature = "fft")]
        group.bench_with_input(BenchmarkId::new("Trait/FFT", size), &size, |b, _| {
            b.iter(|| black_box(TimeShiftFinderFFT::similarity(black_box((&data1, &data2)))))
        });
    }
    
    group.finish();
}

criterion_group!(
    similarity_benches,
    bench_cosine_similarity_comparison,
    bench_cosine_distance_comparison,
    bench_euclidean_distance_comparison,
    bench_pearson_correlation_comparison,
    bench_jaccard_index_comparison,
    bench_prediction_accuracy_comparison,
    bench_cross_correlation_comparison,
    bench_time_shift_comparison
);

criterion_main!(similarity_benches); 