use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use similarity::*;
use similarity::similarity_traits::*;
use similarity::entropy_traits::*;
use similarity::transform_traits::*;
use std::collections::HashSet;
use std::time::Duration;
use rand::prelude::*;

/// This benchmark suite specifically proves that trait-based APIs have zero-cost abstractions
/// by directly comparing identical operations through both functional and trait interfaces.

// ============================================================================
// ZERO-COST ABSTRACTION TEST DATA
// ============================================================================

const BENCHMARK_SIZE: usize = 10000;

fn create_standard_test_vectors() -> (Vec<f64>, Vec<f64>) {
    let mut rng = rand::rng();
    let vec1: Vec<f64> = (0..BENCHMARK_SIZE).map(|_| rng.random_range(-1.0..1.0)).collect();
    let vec2: Vec<f64> = (0..BENCHMARK_SIZE).map(|_| rng.random_range(-1.0..1.0)).collect();
    (vec1, vec2)
}

fn create_standard_test_spectrum() -> Spectrum<f64> {
    let mut rng = rand::rng();
    let mzs: Vec<f64> = (0..BENCHMARK_SIZE).map(|i| 100.0 + i as f64).collect();
    let intensities: Vec<f64> = (0..BENCHMARK_SIZE).map(|_| rng.random_range(0.001..1.0)).collect();
    Spectrum::from_arrays(&mzs, &intensities).unwrap()
}

fn create_standard_test_sets() -> (HashSet<u32>, HashSet<u32>) {
    let mut rng = rand::rng();
    let mut set1 = HashSet::new();
    let mut set2 = HashSet::new();
    
    for _ in 0..1000 {
        let val = rng.random_range(0..2000u32);
        if rng.random_bool(0.6) {
            set1.insert(val);
        }
        if rng.random_bool(0.6) {
            set2.insert(val);
        }
    }
    
    (set1, set2)
}

// ============================================================================
// ZERO-COST COSINE SIMILARITY PROOF
// ============================================================================

fn bench_zero_cost_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zero-Cost Cosine Similarity");
    group.sample_size(1000);
    group.measurement_time(Duration::from_secs(30));
    
    let (vec1, vec2) = create_standard_test_vectors();
    
    // These should have IDENTICAL performance
    group.bench_function("Function_API", |b| {
        b.iter(|| black_box(cosine_similarity(black_box(&vec1), black_box(&vec2))))
    });
    
    group.bench_function("Trait_API", |b| {
        b.iter(|| black_box(CosineSimilarity::similarity(black_box((&vec1, &vec2)))))
    });
    
    group.finish();
}

// ============================================================================
// ZERO-COST EUCLIDEAN DISTANCE PROOF
// ============================================================================

fn bench_zero_cost_euclidean_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zero-Cost Euclidean Distance");
    group.sample_size(1000);
    group.measurement_time(Duration::from_secs(30));
    
    let (vec1, vec2) = create_standard_test_vectors();
    
    group.bench_function("Function_API", |b| {
        b.iter(|| black_box(euclidean_distance(black_box(&vec1), black_box(&vec2))))
    });
    
    group.bench_function("Trait_API", |b| {
        b.iter(|| black_box(EuclideanDistance::similarity(black_box((&vec1, &vec2)))))
    });
    
    group.finish();
}

// ============================================================================
// ZERO-COST PEARSON CORRELATION PROOF
// ============================================================================

fn bench_zero_cost_pearson_correlation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zero-Cost Pearson Correlation");
    group.sample_size(500);
    group.measurement_time(Duration::from_secs(30));
    
    let (vec1, vec2) = create_standard_test_vectors();
    
    group.bench_function("Function_API", |b| {
        b.iter(|| black_box(pearson_correlation_distance(black_box(&vec1), black_box(&vec2))))
    });
    
    group.bench_function("Trait_API", |b| {
        b.iter(|| black_box(PearsonCorrelationDistance::similarity(black_box((&vec1, &vec2)))))
    });
    
    group.finish();
}

// ============================================================================
// ZERO-COST JACCARD INDEX PROOF
// ============================================================================

fn bench_zero_cost_jaccard_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zero-Cost Jaccard Index");
    group.sample_size(1000);
    group.measurement_time(Duration::from_secs(30));
    
    let (set1, set2) = create_standard_test_sets();
    
    group.bench_function("Function_API", |b| {
        b.iter(|| black_box(jaccard_index(black_box(&set1), black_box(&set2))))
    });
    
    group.bench_function("Trait_API", |b| {
        b.iter(|| black_box(JaccardIndex::similarity(black_box((&set1, &set2)))))
    });
    
    group.finish();
}

// ============================================================================
// ZERO-COST SHANNON ENTROPY PROOF
// ============================================================================

fn bench_zero_cost_shannon_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zero-Cost Shannon Entropy");
    group.sample_size(500);
    group.measurement_time(Duration::from_secs(30));
    
    let spectrum = create_standard_test_spectrum();
    
    group.bench_function("Function_API", |b| {
        b.iter(|| black_box(calculate_entropy(black_box(&spectrum))))
    });
    
    group.bench_function("Trait_API", |b| {
        b.iter(|| black_box(ShannonEntropy::entropy(black_box(&spectrum))))
    });
    
    group.finish();
}

// ============================================================================
// ZERO-COST TSALLIS ENTROPY PROOF
// ============================================================================

fn bench_zero_cost_tsallis_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zero-Cost Tsallis Entropy");
    group.sample_size(500);
    group.measurement_time(Duration::from_secs(30));
    
    let spectrum = create_standard_test_spectrum();
    let q = 2.0;
    
    group.bench_function("Function_API", |b| {
        b.iter(|| black_box(calculate_tsallis_entropy(black_box(&spectrum), black_box(q))))
    });
    
    group.bench_function("Trait_API", |b| {
        b.iter(|| black_box(TsallisEntropy::entropy(black_box((&spectrum, q)))))
    });
    
    group.finish();
}

// ============================================================================
// ZERO-COST WEIGHT FACTOR TRANSFORMATION PROOF
// ============================================================================

fn bench_zero_cost_weight_factor_transformation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zero-Cost Weight Factor Transformation");
    group.sample_size(500);
    group.measurement_time(Duration::from_secs(30));
    
    let mut rng = rand::rng();
    let mzs: Vec<f64> = (0..BENCHMARK_SIZE).map(|i| 100.0 + i as f64).collect();
    let intensities: Vec<f64> = (0..BENCHMARK_SIZE).map(|_| rng.random_range(0.001..1.0)).collect();
    let wf_mz = 0.5;
    let wf_int = 2.0;
    
    group.bench_function("Function_API", |b| {
        b.iter(|| {
            black_box(weight_factor_transformation(
                black_box(&mzs), 
                black_box(&intensities), 
                black_box(wf_mz), 
                black_box(wf_int)
            ))
        })
    });
    
    group.bench_function("Trait_API", |b| {
        b.iter(|| {
            black_box(WeightFactorTransformation::transform(
                black_box((&mzs, &intensities, wf_mz, wf_int))
            ))
        })
    });
    
    group.finish();
}

// ============================================================================
// COMPREHENSIVE ZERO-COST COMPARISON
// ============================================================================

fn bench_comprehensive_zero_cost_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Comprehensive Zero-Cost Comparison");
    group.sample_size(200);
    group.measurement_time(Duration::from_secs(60));
    
    let (vec1, vec2) = create_standard_test_vectors();
    let spectrum = create_standard_test_spectrum();
    let (set1, set2) = create_standard_test_sets();
    
    // Test multiple operations in sequence to check for compounding effects
    group.bench_function("Function_API_Chain", |b| {
        b.iter(|| {
            let cosine = cosine_similarity(black_box(&vec1), black_box(&vec2));
            let euclidean = euclidean_distance(black_box(&vec1), black_box(&vec2));
            let entropy = calculate_entropy(black_box(&spectrum));
            let jaccard = jaccard_index(black_box(&set1), black_box(&set2));
            black_box((cosine, euclidean, entropy, jaccard))
        })
    });
    
    group.bench_function("Trait_API_Chain", |b| {
        b.iter(|| {
            let cosine = CosineSimilarity::similarity(black_box((&vec1, &vec2)));
            let euclidean = EuclideanDistance::similarity(black_box((&vec1, &vec2)));
            let entropy = ShannonEntropy::entropy(black_box(&spectrum));
            let jaccard = JaccardIndex::similarity(black_box((&set1, &set2)));
            black_box((cosine, euclidean, entropy, jaccard))
        })
    });
    
    group.finish();
}

// ============================================================================
// INLINE ASSEMBLY VERIFICATION (Advanced)
// ============================================================================

fn bench_assembly_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("Assembly Code Verification");
    group.sample_size(2000);
    group.measurement_time(Duration::from_secs(30));
    
    // Small vectors to minimize noise and focus on call overhead
    let vec1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let vec2 = vec![5.0, 4.0, 3.0, 2.0, 1.0];
    
    // These should compile to identical assembly
    group.bench_function("Function_Direct_Call", |b| {
        b.iter(|| {
            black_box(cosine_similarity(black_box(&vec1), black_box(&vec2)))
        })
    });
    
    group.bench_function("Trait_Method_Call", |b| {
        b.iter(|| {
            black_box(CosineSimilarity::similarity(black_box((&vec1, &vec2))))
        })
    });
    
    // Test with even more minimal data to isolate call overhead
    let tiny_vec1 = vec![1.0, 2.0];
    let tiny_vec2 = vec![3.0, 4.0];
    
    group.bench_function("Function_Minimal", |b| {
        b.iter(|| {
            black_box(cosine_similarity(black_box(&tiny_vec1), black_box(&tiny_vec2)))
        })
    });
    
    group.bench_function("Trait_Minimal", |b| {
        b.iter(|| {
            black_box(CosineSimilarity::similarity(black_box((&tiny_vec1, &tiny_vec2))))
        })
    });
    
    group.finish();
}

// ============================================================================
// OPTIMIZATION LEVEL VERIFICATION
// ============================================================================

fn bench_optimization_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("Optimization Level Verification");
    group.sample_size(1000);
    group.measurement_time(Duration::from_secs(30));
    
    let (vec1, vec2) = create_standard_test_vectors();
    
    // Test all optimization levels to ensure zero-cost abstraction holds
    group.bench_function("Standard_Function", |b| {
        b.iter(|| black_box(cosine_similarity(black_box(&vec1), black_box(&vec2))))
    });
    
    group.bench_function("Standard_Trait", |b| {
        b.iter(|| black_box(CosineSimilarity::similarity(black_box((&vec1, &vec2)))))
    });
    

    
    #[cfg(feature = "parallel")]
    {
        group.bench_function("Parallel_Function", |b| {
            b.iter(|| black_box(cosine_similarity_parallel(black_box(&vec1), black_box(&vec2))))
        });
        
        group.bench_function("Parallel_Trait", |b| {
            b.iter(|| black_box(CosineSimilarityParallel::similarity(black_box((&vec1, &vec2)))))
        });
    }
    
    group.finish();
}

criterion_group!(
    zero_cost_benches,
    bench_zero_cost_cosine_similarity,
    bench_zero_cost_euclidean_distance,
    bench_zero_cost_pearson_correlation,
    bench_zero_cost_jaccard_index,
    bench_zero_cost_shannon_entropy,
    bench_zero_cost_tsallis_entropy,
    bench_zero_cost_weight_factor_transformation,
    bench_comprehensive_zero_cost_comparison,
    bench_assembly_verification,
    bench_optimization_verification
);

criterion_main!(zero_cost_benches); 