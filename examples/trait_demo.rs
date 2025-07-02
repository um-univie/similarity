use similarity::*;
use similarity::similarity_traits::*;
use similarity::entropy_traits::*;
use similarity::transform_traits::*;
use std::collections::HashSet;

fn main() {
    println!("=== Similarity Crate - Trait-Based Demo ===\n");

    // ============================================================================
    // SIMILARITY AND DISTANCE METRICS (comparing two entities)
    // ============================================================================
    
    println!("SIMILARITY AND DISTANCE METRICS");
    println!("--------------------------------");

    // Cosine similarity
    let vec_a = [1.0, 2.0, 3.0, 4.0];
    let vec_b = [2.0, 4.0, 6.0, 8.0];
    let cosine_sim = CosineSimilarity::similarity((&vec_a, &vec_b));
    println!("Cosine similarity: {:?}", cosine_sim);

    // Cosine distance
    let cosine_dist = CosineDistance::similarity((&vec_a, &vec_b));
    println!("Cosine distance: {:?}", cosine_dist);

    // Euclidean distance
    let euclidean_dist = EuclideanDistance::similarity((&vec_a, &vec_b));
    println!("Euclidean distance: {:?}", euclidean_dist);

    // Pearson correlation distance
    let pearson_dist = PearsonCorrelationDistance::similarity((&vec_a, &vec_b));
    println!("Pearson correlation distance: {:?}", pearson_dist);

    // Jaccard index for sets
    let mut set1 = HashSet::new();
    set1.insert(1);
    set1.insert(2);
    set1.insert(3);

    let mut set2 = HashSet::new();
    set2.insert(2);
    set2.insert(3);
    set2.insert(4);

    let jaccard = JaccardIndex::similarity((&set1, &set2));
    println!("Jaccard index: {}", jaccard);

    // Hit rate and overshoot rate for prediction accuracy
    let actual = [1.0, 2.0, 3.0];
    let predicted = [1.1, 1.9, 3.2];
    let tolerance = 0.3;
    let hit_rate = HitRate::similarity((&actual, &predicted, tolerance));
    let overshoot_rate = OvershootRate::similarity((&actual, &predicted, tolerance));
    println!("Hit rate: {:?}", hit_rate);
    println!("Overshoot rate: {:?}", overshoot_rate);

    // Cross-correlation for time series similarity
    let signal1 = [1.0, 2.0, 3.0, 2.0, 1.0];
    let signal2 = [0.0, 1.0, 2.0, 3.0, 2.0, 1.0];
    let xcorr = CrossCorrelationOptimized::similarity((&signal1, &signal2));
    println!("Cross-correlation (first 5 values): {:?}", &xcorr[..5]);

    // Time shift detection
    let shift = TimeShiftFinder::similarity((&signal1, &signal2));
    println!("Time shift: {:?}", shift);

    println!();

    // ============================================================================
    // ENTROPY MEASURES (single entity analysis)
    // ============================================================================
    
    println!("ENTROPY MEASURES");
    println!("----------------");

    // Create test spectra
    let peaks1 = vec![
        Peak { mz: 100.0, intensity: 0.4 },
        Peak { mz: 200.0, intensity: 0.3 },
        Peak { mz: 300.0, intensity: 0.3 },
    ];
    let spectrum1 = Spectrum::from_peaks(peaks1);

    let peaks2 = vec![
        Peak { mz: 150.0, intensity: 0.5 },
        Peak { mz: 250.0, intensity: 0.5 },
    ];
    let spectrum2 = Spectrum::from_peaks(peaks2);

    // Shannon entropy
    let shannon_entropy1 = ShannonEntropy::entropy(&spectrum1);
    let shannon_entropy2 = ShannonEntropy::entropy(&spectrum2);
    println!("Shannon entropy spectrum 1: {:.4}", shannon_entropy1);
    println!("Shannon entropy spectrum 2: {:.4}", shannon_entropy2);

    // Tsallis entropy with different q parameters
    let q_values = [0.5, 1.0, 2.0, 3.0];
    for q in q_values {
        let tsallis_entropy = TsallisEntropy::entropy((&spectrum1, q));
        println!("Tsallis entropy (q={}): {:.4}", q, tsallis_entropy);
    }

    // Entropy similarity between two spectra
    let entropy_similarity = EntropySimilarity::similarity((&spectrum1, &spectrum2));
    println!("Entropy similarity between spectra: {:.4}", entropy_similarity);

    println!();

    // ============================================================================
    // DATA TRANSFORMATIONS (preprocessing operations)
    // ============================================================================
    
    println!("DATA TRANSFORMATIONS");
    println!("--------------------");

    // Weight factor transformation for spectral preprocessing
    let mzs = [100.0, 200.0, 300.0, 400.0];
    let intensities = [0.4, 0.3, 0.2, 0.1];
    let wf_mz = 0.5;
    let wf_int = 2.0;

    let transformed = WeightFactorTransformation::transform((&mzs, &intensities, wf_mz, wf_int));
    println!("Original intensities: {:?}", intensities);
    println!("Transformed weights: {:.3?}", transformed);

    println!();

    // ============================================================================
    // PERFORMANCE COMPARISON: Optimized vs Parallel implementations
    // ============================================================================
    
    #[cfg(feature = "parallel")]
    {
        println!("PERFORMANCE COMPARISON");
        println!("----------------------");

        let large_vec_a: Vec<f64> = (0..10000).map(|i| i as f64).collect();
        let large_vec_b: Vec<f64> = (0..10000).map(|i| (i as f64) * 1.1).collect();

        // Regular vs parallel implementations
        let cosine_regular = CosineSimilarity::similarity((&large_vec_a, &large_vec_b));
        let cosine_parallel = CosineSimilarityParallel::similarity((&large_vec_a, &large_vec_b));

        println!("Cosine similarity (regular): {:?}", cosine_regular);
        println!("Cosine similarity (parallel): {:?}", cosine_parallel);

        // Both should give same or very similar results
        if let (Some(regular), Some(parallel)) = (cosine_regular, cosine_parallel) {
            let diff = (regular - parallel).abs();
            println!("Difference regular vs parallel: {:.10}", diff);
        }

        println!();
    }

    // ============================================================================
    // FFT-OPTIMIZED OPERATIONS (when FFT feature is enabled)
    // ============================================================================
    
    #[cfg(feature = "fft")]
    {
        println!("FFT-OPTIMIZED OPERATIONS");
        println!("-------------------------");

        let signal_x = [1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0];
        let signal_y = [0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];

        // FFT-based cross-correlation
        let fft_xcorr = CrossCorrelationFFT::similarity((&signal_x, &signal_y));
        println!("FFT cross-correlation (first 5): {:.3?}", &fft_xcorr[..5]);

        // FFT-based time shift detection
        let fft_shift = TimeShiftFinderFFT::similarity((&signal_x, &signal_y));
        println!("FFT time shift: {:?}", fft_shift);

        println!();
    }

    // ============================================================================
    // SUMMARY: Trait-based API Benefits
    // ============================================================================
    
    println!("TRAIT-BASED API BENEFITS");
    println!("------------------------");
    println!("1. Type Safety: Each trait has specific input/output types");
    println!("2. Semantic Clarity: Separate traits for similarity, entropy, and transformations");  
    println!("3. Zero-Cost Abstraction: Trait calls compile to direct function calls");
    println!("4. Composability: Easy to combine different algorithms");
    println!("5. Extensibility: Add new implementations without changing existing code");
    println!("6. Feature Gates: Optional parallelization and FFT optimizations");
    println!();
    
    println!("Trait implementations demonstrated.");
} 