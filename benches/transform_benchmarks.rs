use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use similarity::*;
use similarity::transform_traits::*;
use std::time::Duration;
use std::hint::black_box;
use rand::prelude::*;

// ============================================================================
// TEST DATA GENERATORS
// ============================================================================

fn create_test_mz_intensity_data(size: usize) -> (Vec<f64>, Vec<f64>) {
    let mut rng = rand::rng();
    
    // Realistic m/z values (mass-to-charge ratio)
    let mzs: Vec<f64> = (0..size)
        .map(|i| 50.0 + i as f64 * 2.5) // Spread from 50 to 50 + size*2.5
        .collect();
    
    // Realistic intensity values with varying magnitudes
    let intensities: Vec<f64> = (0..size)
        .map(|_| rng.random_range(0.001..1.0)) // Avoid zero intensities
        .collect();
    
    (mzs, intensities)
}

fn create_test_weight_factors() -> (f64, f64) {
    let mut rng = rand::rng();
    let wf_mz = rng.random_range(0.1..2.0);
    let wf_int = rng.random_range(0.1..5.0);
    (wf_mz, wf_int)
}

// ============================================================================
// WEIGHT FACTOR TRANSFORMATION BENCHMARKS
// ============================================================================

fn bench_weight_factor_transformation_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Weight Factor Transformation - Function vs Trait");
    group.sample_size(200);
    group.measurement_time(Duration::from_secs(15));
    
    let sizes = [100, 1000, 10000, 50000, 100000];
    
    for size in sizes {
        let (mzs, intensities) = create_test_mz_intensity_data(size);
        let (wf_mz, wf_int) = create_test_weight_factors();
        
        // Functional API
        group.bench_with_input(BenchmarkId::new("Function/Standard", size), &size, |b, _| {
            b.iter(|| {
                black_box(weight_factor_transformation(
                    black_box(&mzs), 
                    black_box(&intensities), 
                    black_box(wf_mz), 
                    black_box(wf_int)
                ))
            })
        });
        
        // Trait API - should have identical performance
        group.bench_with_input(BenchmarkId::new("Trait/Standard", size), &size, |b, _| {
            b.iter(|| {
                black_box(WeightFactorTransformation::transform(
                    black_box((&mzs, &intensities, wf_mz, wf_int))
                ))
            })
        });
        
        // Optimized versions (note: they currently use the same implementation)
        #[cfg(feature = "parallel")]
        group.bench_with_input(BenchmarkId::new("Trait/Optimized", size), &size, |b, _| {
            b.iter(|| {
                black_box(WeightFactorTransformationOptimized::transform(
                    black_box((&mzs, &intensities, wf_mz, wf_int))
                ))
            })
        });
    }
    
    group.finish();
}

// ============================================================================
// WEIGHT FACTOR PARAMETER ANALYSIS
// ============================================================================

fn bench_weight_factor_parameter_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("Weight Factor Parameter Analysis");
    group.sample_size(200);
    group.measurement_time(Duration::from_secs(10));
    
    let (mzs, intensities) = create_test_mz_intensity_data(10000);
    
    // Test different weight factor combinations
    let wf_combinations = [
        (0.1, 0.1), (0.1, 1.0), (0.1, 5.0),
        (0.5, 0.1), (0.5, 1.0), (0.5, 5.0),
        (1.0, 0.1), (1.0, 1.0), (1.0, 5.0),
        (2.0, 0.1), (2.0, 1.0), (2.0, 5.0),
        (5.0, 0.1), (5.0, 1.0), (5.0, 5.0),
    ];
    
    for (wf_mz, wf_int) in wf_combinations {
        let param_name = format!("mz={:.1}_int={:.1}", wf_mz, wf_int);
        
        // Functional API
        group.bench_with_input(BenchmarkId::new("Function", &param_name), &param_name, |b, _| {
            b.iter(|| {
                black_box(weight_factor_transformation(
                    black_box(&mzs), 
                    black_box(&intensities), 
                    black_box(wf_mz), 
                    black_box(wf_int)
                ))
            })
        });
        
        // Trait API
        group.bench_with_input(BenchmarkId::new("Trait", &param_name), &param_name, |b, _| {
            b.iter(|| {
                black_box(WeightFactorTransformation::transform(
                    black_box((&mzs, &intensities, wf_mz, wf_int))
                ))
            })
        });
    }
    
    group.finish();
}

// ============================================================================
// TRANSFORM SCALING ANALYSIS
// ============================================================================

fn bench_transform_scaling_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("Transform Scaling Analysis");
    group.sample_size(150);
    group.measurement_time(Duration::from_secs(20));
    
    // Test how transformations scale with data size
    let sizes = [50, 100, 500, 1000, 5000, 10000, 50000, 100000];
    let (wf_mz, wf_int) = (0.5, 2.0); // Fixed weight factors
    
    for size in sizes {
        let (mzs, intensities) = create_test_mz_intensity_data(size);
        
        // Weight factor transformation scaling
        group.bench_with_input(BenchmarkId::new("WeightTransform/Function", size), &size, |b, _| {
            b.iter(|| {
                black_box(weight_factor_transformation(
                    black_box(&mzs), 
                    black_box(&intensities), 
                    black_box(wf_mz), 
                    black_box(wf_int)
                ))
            })
        });
        
        group.bench_with_input(BenchmarkId::new("WeightTransform/Trait", size), &size, |b, _| {
            b.iter(|| {
                black_box(WeightFactorTransformation::transform(
                    black_box((&mzs, &intensities, wf_mz, wf_int))
                ))
            })
        });
    }
    
    group.finish();
}

// ============================================================================
// MEMORY ALLOCATION BENCHMARKS
// ============================================================================

fn bench_transform_memory_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Transform Memory Allocation");
    group.sample_size(100);
    group.measurement_time(Duration::from_secs(15));
    
    let sizes = [1000, 10000, 100000];
    let (wf_mz, wf_int) = (0.5, 2.0);
    
    for size in sizes {
        let (mzs, intensities) = create_test_mz_intensity_data(size);
        
        // Measure allocation patterns - functional API
        group.bench_with_input(BenchmarkId::new("Allocation/Function", size), &size, |b, _| {
            b.iter(|| {
                let result = weight_factor_transformation(
                    black_box(&mzs), 
                    black_box(&intensities), 
                    black_box(wf_mz), 
                    black_box(wf_int)
                );
                black_box(result.len()); // Ensure result is used
                drop(result); // Explicit drop for benchmarking
            })
        });
        
        // Measure allocation patterns - trait API
        group.bench_with_input(BenchmarkId::new("Allocation/Trait", size), &size, |b, _| {
            b.iter(|| {
                let result = WeightFactorTransformation::transform(
                    black_box((&mzs, &intensities, wf_mz, wf_int))
                );
                black_box(result.len()); // Ensure result is used
                drop(result); // Explicit drop for benchmarking
            })
        });
    }
    
    group.finish();
}

// ============================================================================
// EDGE CASE PERFORMANCE BENCHMARKS
// ============================================================================

fn bench_transform_edge_cases(c: &mut Criterion) {
    let mut group = c.benchmark_group("Transform Edge Cases");
    group.sample_size(200);
    group.measurement_time(Duration::from_secs(10));
    
    // Test performance with different data characteristics
    let edge_cases = [
        ("VerySmall", create_test_mz_intensity_data(10)),
        ("Small", create_test_mz_intensity_data(100)),
        ("ZeroIntensities", create_zero_intensity_data(1000)),
        ("UniformIntensities", create_uniform_intensity_data(1000)),
        ("ExtremeWeights", create_test_mz_intensity_data(1000)),
    ];
    
    for (case_name, (mzs, intensities)) in edge_cases {
        let (wf_mz, wf_int) = if case_name == "ExtremeWeights" {
            (10.0, 0.01) // Extreme weight factors
        } else {
            (0.5, 2.0) // Normal weight factors
        };
        
        // Functional API
        group.bench_function(&format!("Function/{}", case_name), |b| {
            b.iter(|| {
                black_box(weight_factor_transformation(
                    black_box(&mzs), 
                    black_box(&intensities), 
                    black_box(wf_mz), 
                    black_box(wf_int)
                ))
            })
        });
        
        // Trait API
        group.bench_function(&format!("Trait/{}", case_name), |b| {
            b.iter(|| {
                black_box(WeightFactorTransformation::transform(
                    black_box((&mzs, &intensities, wf_mz, wf_int))
                ))
            })
        });
    }
    
    group.finish();
}

// Helper functions for creating specific data patterns
fn create_zero_intensity_data(size: usize) -> (Vec<f64>, Vec<f64>) {
    let mzs: Vec<f64> = (0..size).map(|i| 100.0 + i as f64).collect();
    let intensities: Vec<f64> = vec![0.0; size];
    (mzs, intensities)
}

fn create_uniform_intensity_data(size: usize) -> (Vec<f64>, Vec<f64>) {
    let mzs: Vec<f64> = (0..size).map(|i| 100.0 + i as f64).collect();
    let intensities: Vec<f64> = vec![1.0; size];
    (mzs, intensities)
}

// ============================================================================
// MATHEMATICAL PRECISION BENCHMARKS
// ============================================================================

fn bench_transform_precision_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("Transform Precision Analysis");
    group.sample_size(200);
    group.measurement_time(Duration::from_secs(10));
    
    let (mzs, intensities) = create_test_mz_intensity_data(10000);
    
    // Test with different floating-point precision scenarios
    let precision_cases = [
        ("HighPrecision", (0.123456789, 1.987654321)),
        ("LowPrecision", (0.1, 2.0)),
        ("VerySmall", (0.000001, 0.000002)),
        ("VeryLarge", (1000.0, 2000.0)),
        ("NearZero", (0.0001, 0.0002)),
        ("NearOne", (0.9999, 1.0001)),
    ];
    
    for (case_name, (wf_mz, wf_int)) in precision_cases {
        // Functional API
        group.bench_function(&format!("Function/{}", case_name), |b| {
            b.iter(|| {
                black_box(weight_factor_transformation(
                    black_box(&mzs), 
                    black_box(&intensities), 
                    black_box(wf_mz), 
                    black_box(wf_int)
                ))
            })
        });
        
        // Trait API
        group.bench_function(&format!("Trait/{}", case_name), |b| {
            b.iter(|| {
                black_box(WeightFactorTransformation::transform(
                    black_box((&mzs, &intensities, wf_mz, wf_int))
                ))
            })
        });
    }
    
    group.finish();
}

criterion_group!(
    transform_benches,
    bench_weight_factor_transformation_comparison,
    bench_weight_factor_parameter_analysis,
    bench_transform_scaling_analysis,
    bench_transform_memory_allocation,
    bench_transform_edge_cases,
    bench_transform_precision_analysis
);

criterion_main!(transform_benches); 