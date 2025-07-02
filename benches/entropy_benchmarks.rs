use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use similarity::*;
use similarity::entropy_traits::*;
use similarity::similarity_traits::EntropySimilarity;
use std::time::Duration;
use rand::prelude::*;
use std::hint::black_box;

// ============================================================================
// TEST DATA GENERATORS
// ============================================================================

fn create_test_spectrum(size: usize) -> Spectrum<f64> {
    let mut rng = rand::rng();
    let mzs: Vec<f64> = (0..size).map(|i| 100.0 + i as f64).collect();
    let intensities: Vec<f64> = (0..size)
        .map(|_| rng.random_range(0.0..1.0))
        .collect();
    Spectrum::from_arrays(&mzs, &intensities).unwrap()
}

fn create_paired_spectra(size: usize) -> (Spectrum<f64>, Spectrum<f64>) {
    let mut rng = rand::rng();
    
    // Create first spectrum
    let mzs1: Vec<f64> = (0..size).map(|i| 100.0 + i as f64).collect();
    let intensities1: Vec<f64> = (0..size)
        .map(|_| rng.random_range(0.0..1.0))
        .collect();
    let spectrum1 = Spectrum::from_arrays(&mzs1, &intensities1).unwrap();
    
    // Create second spectrum with some overlap for realistic entropy similarity testing
    let mzs2: Vec<f64> = (0..size).map(|i| 110.0 + i as f64).collect();
    let intensities2: Vec<f64> = (0..size)
        .map(|_| rng.random_range(0.0..1.0))
        .collect();
    let spectrum2 = Spectrum::from_arrays(&mzs2, &intensities2).unwrap();
    
    (spectrum1, spectrum2)
}

// ============================================================================
// SHANNON ENTROPY BENCHMARKS
// ============================================================================

fn bench_shannon_entropy_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Shannon Entropy - Function vs Trait");
    group.sample_size(200);
    group.measurement_time(Duration::from_secs(15));
    
    let sizes = [100, 1000, 10000, 50000];
    
    for size in sizes {
        let spectrum = create_test_spectrum(size);
        
        // Functional API
        group.bench_with_input(BenchmarkId::new("Function/Standard", size), &size, |b, _| {
            b.iter(|| black_box(calculate_entropy(black_box(&spectrum))))
        });
        
        // Trait API - should have identical performance
        group.bench_with_input(BenchmarkId::new("Trait/Standard", size), &size, |b, _| {
            b.iter(|| black_box(ShannonEntropy::entropy(black_box(&spectrum))))
        });
    }
    
    group.finish();
}

// ============================================================================
// TSALLIS ENTROPY BENCHMARKS
// ============================================================================

fn bench_tsallis_entropy_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Tsallis Entropy - Function vs Trait");
    group.sample_size(200);
    group.measurement_time(Duration::from_secs(15));
    
    let sizes = [100, 1000, 10000, 50000];
    let q_values = [0.5, 1.0, 2.0, 3.0];
    
    for size in sizes {
        let spectrum = create_test_spectrum(size);
        
        for q in q_values {
            // Functional API
            group.bench_with_input(
                BenchmarkId::new(format!("Function/Standard/q={}", q), size), 
                &size, 
                |b, _| {
                    b.iter(|| black_box(calculate_tsallis_entropy(black_box(&spectrum), black_box(q))))
                }
            );
            
            #[cfg(feature = "parallel")]
            group.bench_with_input(
                BenchmarkId::new(format!("Function/Optimized/q={}", q), size), 
                &size, 
                |b, _| {
                    b.iter(|| black_box(calculate_tsallis_entropy_optimized(black_box(&spectrum), black_box(q))))
                }
            );
            
            // Trait API
            group.bench_with_input(
                BenchmarkId::new(format!("Trait/Standard/q={}", q), size), 
                &size, 
                |b, _| {
                    b.iter(|| black_box(TsallisEntropy::entropy(black_box((&spectrum, q)))))
                }
            );
            
            #[cfg(feature = "parallel")]
            group.bench_with_input(
                BenchmarkId::new(format!("Trait/Optimized/q={}", q), size), 
                &size, 
                |b, _| {
                    b.iter(|| black_box(TsallisEntropyOptimized::entropy(black_box((&spectrum, q)))))
                }
            );
        }
    }
    
    group.finish();
}

// ============================================================================
// ENTROPY SIMILARITY BENCHMARKS
// ============================================================================

fn bench_entropy_similarity_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Entropy Similarity - Function vs Trait");
    group.sample_size(150);
    group.measurement_time(Duration::from_secs(15));
    
    let sizes = [100, 1000, 10000];
    
    for size in sizes {
        let (spectrum1, spectrum2) = create_paired_spectra(size);
        
        // Functional API
        group.bench_with_input(BenchmarkId::new("Function/Standard", size), &size, |b, _| {
            b.iter(|| black_box(calculate_entropy_similarity(black_box(&spectrum1), black_box(&spectrum2))))
        });
        
        // Trait API
        group.bench_with_input(BenchmarkId::new("Trait/Standard", size), &size, |b, _| {
            b.iter(|| black_box(EntropySimilarity::similarity(black_box((&spectrum1, &spectrum2)))))
        });
    }
    
    group.finish();
}

// ============================================================================
// ENTROPY SCALING ANALYSIS
// ============================================================================

fn bench_entropy_scaling_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("Entropy Scaling Analysis");
    group.sample_size(100);
    group.measurement_time(Duration::from_secs(20));
    
    // Test how entropy calculations scale with spectrum size
    let sizes = [50, 100, 500, 1000, 5000, 10000, 50000];
    
    for size in sizes {
        let spectrum = create_test_spectrum(size);
        
        // Shannon entropy scaling
        group.bench_with_input(BenchmarkId::new("Shannon/Standard", size), &size, |b, _| {
            b.iter(|| black_box(calculate_entropy(black_box(&spectrum))))
        });
        

        
        // Tsallis entropy scaling (with q=2.0)
        group.bench_with_input(BenchmarkId::new("Tsallis/Standard", size), &size, |b, _| {
            b.iter(|| black_box(calculate_tsallis_entropy(black_box(&spectrum), black_box(2.0))))
        });
        
        #[cfg(feature = "parallel")]
        group.bench_with_input(BenchmarkId::new("Tsallis/Optimized", size), &size, |b, _| {
            b.iter(|| black_box(calculate_tsallis_entropy_optimized(black_box(&spectrum), black_box(2.0))))
        });
    }
    
    group.finish();
}

// ============================================================================
// DIFFERENT Q PARAMETER ANALYSIS FOR TSALLIS ENTROPY
// ============================================================================

fn bench_tsallis_q_parameter_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("Tsallis Entropy - Q Parameter Analysis");
    group.sample_size(200);
    group.measurement_time(Duration::from_secs(10));
    
    let spectrum = create_test_spectrum(10000);
    let q_values: Vec<f64> = vec![0.1, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0];
    
    for q in q_values {
        // Functional API
        group.bench_with_input(BenchmarkId::new("Function", q), &q, |b, _| {
            b.iter(|| black_box(calculate_tsallis_entropy(black_box(&spectrum), black_box(q))))
        });
        
        // Trait API
        group.bench_with_input(BenchmarkId::new("Trait", q), &q, |b, _| {
            b.iter(|| black_box(TsallisEntropy::entropy(black_box((&spectrum, q)))))
        });
    }
    
    group.finish();
}

// ============================================================================
// ENTROPY CONVERGENCE TESTING
// ============================================================================

fn bench_entropy_convergence_testing(c: &mut Criterion) {
    let mut group = c.benchmark_group("Entropy Convergence Testing");
    group.sample_size(150);
    group.measurement_time(Duration::from_secs(10));
    
    // Test entropy calculations with different spectrum types
    let uniform_spectrum = create_uniform_spectrum(1000);
    let exponential_spectrum = create_exponential_spectrum(1000);
    let sparse_spectrum = create_sparse_spectrum(1000);
    
    // Shannon entropy on different distributions
    group.bench_function("Shannon/Uniform", |b| {
        b.iter(|| black_box(calculate_entropy(black_box(&uniform_spectrum))))
    });
    
    group.bench_function("Shannon/Exponential", |b| {
        b.iter(|| black_box(calculate_entropy(black_box(&exponential_spectrum))))
    });
    
    group.bench_function("Shannon/Sparse", |b| {
        b.iter(|| black_box(calculate_entropy(black_box(&sparse_spectrum))))
    });
    
    // Tsallis entropy on different distributions
    group.bench_function("Tsallis/Uniform", |b| {
        b.iter(|| black_box(calculate_tsallis_entropy(black_box(&uniform_spectrum), black_box(2.0))))
    });
    
    group.bench_function("Tsallis/Exponential", |b| {
        b.iter(|| black_box(calculate_tsallis_entropy(black_box(&exponential_spectrum), black_box(2.0))))
    });
    
    group.bench_function("Tsallis/Sparse", |b| {
        b.iter(|| black_box(calculate_tsallis_entropy(black_box(&sparse_spectrum), black_box(2.0))))
    });
    
    group.finish();
}

// Helper functions for creating specific spectrum types
fn create_uniform_spectrum(size: usize) -> Spectrum<f64> {
    let mzs: Vec<f64> = (0..size).map(|i| 100.0 + i as f64).collect();
    let intensities: Vec<f64> = vec![1.0 / size as f64; size]; // Uniform distribution
    Spectrum::from_arrays(&mzs, &intensities).unwrap()
}

fn create_exponential_spectrum(size: usize) -> Spectrum<f64> {
    let mzs: Vec<f64> = (0..size).map(|i| 100.0 + i as f64).collect();
    let intensities: Vec<f64> = (0..size)
        .map(|i| (-0.01 * i as f64).exp()) // Exponential decay
        .collect();
    Spectrum::from_arrays(&mzs, &intensities).unwrap()
}

fn create_sparse_spectrum(size: usize) -> Spectrum<f64> {
    let mut rng = rand::rng();
    let mzs: Vec<f64> = (0..size).map(|i| 100.0 + i as f64).collect();
    let intensities: Vec<f64> = (0..size)
        .map(|_| {
            if rng.random_bool(0.1) { // 10% chance of non-zero intensity
                rng.random_range(0.5..1.0)
            } else {
                0.0
            }
        })
        .collect();
    Spectrum::from_arrays(&mzs, &intensities).unwrap()
}

criterion_group!(
    entropy_benches,
    bench_shannon_entropy_comparison,
    bench_tsallis_entropy_comparison,
    bench_entropy_similarity_comparison,
    bench_entropy_scaling_analysis,
    bench_tsallis_q_parameter_analysis,
    bench_entropy_convergence_testing
);

criterion_main!(entropy_benches); 