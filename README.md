# Similarity

[![Crates.io](https://img.shields.io/crates/v/similarity.svg)](https://crates.io/crates/similarity)
[![Documentation](https://docs.rs/similarity/badge.svg)](https://docs.rs/similarity)
[![CI](https://github.com/um-univie/similarity/actions/workflows/ci.yml/badge.svg)](https://github.com/um-univie/similarity/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Rust library for calculating similarity metrics between vectors, collections, and spectral data. Features both functional and trait-based APIs with optional parallel processing and FFT optimizations.

## Features

- **Portable SIMD for `f32` and `f64`**: 40+ kernels powered by [`pulp`](https://crates.io/crates/pulp) with runtime CPU dispatch (AVX-512, AVX2+FMA, NEON), 4–8 independent accumulators, fused multiply-add, single-pass multi-sum reductions and vectorised tails
- **Batch evaluation**: pairwise matrices (`pairwise`, `pdist`), cross-distance (`cdist`) and k-nearest-neighbour search (`top_k`), SIMD inside and Rayon across rows
- **Many metrics**: Minkowski family, Canberra, Bray–Curtis, χ², Tanimoto, Dice, Ruzicka, Hamming (incl. bit fingerprints), Levenshtein, LCS, DTW, KL / JS / Rényi / Hellinger / Bhattacharyya / Wasserstein, Spearman, Kendall τ-b, Lin's CCC, MAE / MSE / RMSE / MAPE / sMAPE / R²
- **Mass spectrometry**: m/z-tolerant (Da or ppm) greedy peak matching, modified spectral cosine and peak-matched entropy similarity
- **Flash entropy search**: indexed library search with the spectral entropy similarity (Li & Fiehn 2023): ~2000× faster than brute force on a 100k-spectrum library, exact on cleaned spectra
- **Batch library search**: precursor-filtered identity search with entropy or modified-cosine scoring, min-peaks / min-score thresholds, top-k, all-vs-all similarity matrices (molecular networking)
- **Python bindings** (`python/`, PyO3 + NumPy): vector metrics, `cdist` / `pdist` / `top_k`, divergences, Flash entropy and library search, GIL released
- **Vectorised `ln` / `exp` / `pow` / `sqrt`** (≤ 2 ulp): SIMD Shannon entropy, KL, cross-entropy, Jensen–Shannon, Bhattacharyya, Hellinger and spectral entropy weighting
- **Semantically Correct Trait System**: Separate traits for different types of calculations
- **Zero-Cost Abstractions**: Trait calls compile to direct function calls
- **Extensive Metric Coverage**: Distance, similarity, correlation, and entropy measures  
- **Spectral Data Support**: Specialized functions for mass spectrometry and signal processing
- **Performance Optimizations**: SIMD vectorization, Rayon multi-threaded parallelism, and FFT-based algorithms
- **Feature Gates**: Optional dependencies for parallel and FFT features

## Architecture

The library is organized into three main trait categories:

### 1. `Similarity<InputType, OutputType>` 
For comparing **multiple entities** and computing similarity or distance metrics:
- Cosine similarity/distance
- Euclidean distance  
- Pearson correlation distance
- Jaccard index for sets
- Cross-correlation and time shift detection
- Hit rate and overshoot rate for predictions
- Entropy similarity between spectra

### 2. `EntropyMeasure<InputType, OutputType>`
For analyzing **single entities** with information-theoretic measures:
- Shannon entropy
- Tsallis entropy with parameter q
- Both standard and optimized implementations

### 3. `DataTransform<InputType, OutputType>`
For **preprocessing and transforming** data:
- Weight factor transformation for spectral data
- Optimized implementations for large datasets

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
similarity = "0.3.0"
```

### Trait-Based API Examples

```rust
use similarity::*;
use similarity::similarity_traits::*;
use similarity::entropy_traits::*;
use similarity::transform_traits::*;
use std::collections::HashSet;

// Similarity between vectors
let a = [1.0, 2.0, 3.0];
let b = [2.0, 4.0, 6.0];
let cosine_sim = CosineSimilarity::similarity((&a, &b));
let euclidean_dist = EuclideanDistance::similarity((&a, &b));

// Set similarity
let mut set1 = HashSet::new();
set1.extend([1, 2, 3]);
let mut set2 = HashSet::new(); 
set2.extend([2, 3, 4]);
let jaccard = JaccardIndex::similarity((&set1, &set2));

// Entropy of spectral data
let spectrum = Spectrum::from_peaks(vec![
    Peak { mz: 100.0, intensity: 0.6 },
    Peak { mz: 200.0, intensity: 0.4 },
]);
let shannon_entropy = ShannonEntropy::entropy(&spectrum);
let tsallis_entropy = TsallisEntropy::entropy((&spectrum, 2.0));

// Data transformation
let mzs = [100.0, 200.0, 300.0];
let intensities = [0.5, 0.3, 0.2];
let transformed = WeightFactorTransformation::transform((&mzs, &intensities, 0.5, 2.0));
```

### Functional API (Still Available)

```rust
use similarity::*;

// All original functions remain available
let cosine_sim = cosine_similarity(&a, &b);
let euclidean_dist = euclidean_distance(&a, &b);
let entropy = calculate_entropy(&spectrum);
```

## Metric catalogue

| Module | Contents |
|---|---|
| `simd` | `*_simd` functions, generic over `f32`/`f64`: `ln`/`exp`/`pow`/`sqrt`, Shannon entropy, KL, cross-entropy, JS, Bhattacharyya, Hellinger, total variation, spectral entropy weights and similarity, dot, cosine, angular, Tanimoto, Dice, (weighted) Euclidean/Manhattan, Chebyshev, Canberra, Bray–Curtis, χ², histogram intersection, Ruzicka, Hamming, Pearson, covariance, MSE/RMSE/MAE/R², sum/mean/variance/std/min-max/norms, in-place scale/AXPY/clamp/L1-L2-min-max-z-score normalisation, cross-correlation (+ parallel), time-shift |
| `batch` | `Metric` enum, `pairwise`, `pdist`, `cdist`, `one_to_many`, `top_k`, `top_k_many`, `rows_from_flat` |
| `distance_metrics` | Generic `Float` versions of the vector distances, Minkowski-p, diagonal Mahalanobis, Hamming / Tanimoto on packed `u64` fingerprints, Levenshtein, LCS, DTW with Sakoe–Chiba band |
| `divergences` | Shannon & cross entropy, KL, JS divergence/distance, Rényi-α, Hellinger, Bhattacharyya, total variation, 1-D Wasserstein |
| `statistics` | Ranks, Pearson, Spearman, Kendall τ-b, Lin's concordance, MAE, MSE, RMSE, MAPE, sMAPE, R² |
| `set_metrics` | Dice, overlap, Tversky, Jaccard distance on `HashSet`; merge-based Jaccard/Dice on sorted slices; MinHash estimate |
| `flash_entropy` | `FlashEntropyIndex` (build, `search`, `search_top_k`, `search_many`), `FlashEntropyOptions`, `preprocess` |
| `spectral_entropy` | `Spectrum` helpers, Shannon/Tsallis entropy, entropy similarity, `MzTolerance`, `match_peaks`, `spectral_cosine_similarity`, `calculate_entropy_similarity_matched` |
| `similarity_metrics` | Original generic cosine/Euclidean/Pearson/Jaccard/hit-rate/cross-correlation (scalar, parallel, FFT) |

### SIMD in one line, for either precision

```rust
use similarity::simd::*;

let a: Vec<f32> = vec![0.1, 0.7, 0.2, 0.0];
let b: Vec<f32> = vec![0.2, 0.6, 0.1, 0.1];
let cos = cosine_similarity_simd(&a, &b).unwrap();   // f32 in, f32 out
let bc  = bray_curtis_distance_simd(&a, &b).unwrap();
let mut x = a.clone();
zscore_simd(&mut x);                                 // in-place standardisation
```

### Nearest-neighbour search and distance matrices

```rust
use similarity::batch::{pairwise, top_k, Metric};

let db: Vec<Vec<f32>> = load_embeddings();
let hits = top_k(&query, &db, 10, Metric::Cosine)?;  // best first
let dm   = pairwise(&db[..1000], Metric::Euclidean)?;
println!("{}", dm.get(3, 7));
```

### Spectral library matching with m/z tolerance

```rust
use similarity::{Spectrum, MzTolerance, spectral_cosine_similarity,
                 calculate_entropy_similarity_matched};

let tol = MzTolerance::Ppm(10.0);
let (cos, n_matched) = spectral_cosine_similarity(&query, &reference, tol, 0.0, 0.5);
let entropy = calculate_entropy_similarity_matched(&query, &reference, tol);
```

`calculate_entropy_similarity` (the original function) pairs peaks by index and is deprecated since 0.3.0.
Use `calculate_entropy_similarity_matched` when the two spectra do not have the same peak list.

### Flash entropy library search

```rust
use similarity::flash_entropy::{FlashEntropyIndex, FlashEntropyOptions};

let index = FlashEntropyIndex::new(&library, FlashEntropyOptions::absolute(0.01));
let hits = index.search_top_k(&query, 10);          // best entropy similarity first
let all  = index.search_many(&queries, 5);           // parallel over queries
```

Spectra are cleaned (peaks closer than 2 × tolerance merged, optional noise
filter and top-N), normalised and entropy-weighted. Scores equal
`calculate_entropy_similarity_matched` on the cleaned spectra.

### Batch library search with precursor filtering

```rust
use similarity::library_search::{LibrarySearchOptions, SpectralLibrary};
use similarity::MzTolerance;

let opts = LibrarySearchOptions::cosine(MzTolerance::Ppm(10.0))       // or ::entropy(..)
    .with_precursor_tolerance(MzTolerance::Absolute(0.01))
    .with_min_matched_peaks(3);
let lib = SpectralLibrary::with_precursors(&library, &precursor_mz, opts)?;
let hits = lib.search(&query, Some(query_precursor), 10);          // Vec<LibraryHit>
let all = lib.search_many(&queries, Some(&query_precursors), 10)?;  // parallel
let network = lib.similarity_matrix();                              // n × n, symmetric
```

Library spectra are cleaned once, so each query–library pair is scored with
one linear merge. Use `FlashEntropyIndex` for open (no precursor) entropy
search over large libraries.

### Python

```bash
cd python && uv venv && uv pip install maturin numpy pytest
uv run maturin develop --release && uv run pytest tests
```

```python
import similarity_rs as sr
d = sr.cdist(x, y, "cosine_distance")                 # float32 or float64
index = sr.FlashEntropyIndex([(mz, intensity), ...], tolerance=0.01)
idx, scores = index.search_top_k(q_mz, q_int, 10)
lib = sr.SpectralLibrary(spectra, precursor_mz=prec, metric="cosine", precursor_tolerance=0.01)
```

See [`python/README.md`](python/README.md).

## Performance Features

### Optional Dependencies

```toml
[dependencies]
similarity = { version = "0.3.0", features = ["parallel", "fft"] }
```

- **`parallel`**: Enables Rayon-based parallel processing for large datasets
- **`fft`**: Enables FFT-based optimizations for cross-correlation and convolution
- **`avx512`** (off by default): compiles AVX-512 kernels, used at runtime when the CPU supports them. Needs Rust ≥ 1.89.

Minimum supported Rust version: **1.83**.

`similarity::simd::Isa::detected()` reports which instruction set the SIMD kernels use on the current machine.

### Performance Comparison

The trait-based API has **zero performance overhead** compared to direct function calls:

```rust
// These are equivalent in performance:
let result1 = cosine_similarity(&a, &b);
let result2 = CosineSimilarity::similarity((&a, &b));
```

### Performance & Optimization Guide (Scalar vs SIMD vs Parallel)

The crate offers three tiers of execution depending on your data size and hardware capabilities:

1. **Generic / Scalar (`cosine_similarity`, `CosineSimilarity`)**:
   - Generic over numeric types (`T: Copy + Num`, etc.).
   - Best when vectors are extremely small (< 16 elements) where function dispatch overhead dominates.

2. **SIMD with Pulp (`*_simd`, `*Optimized` traits, `batch`)**:
   - Portable, safe runtime dispatch (AVX-512, AVX2+FMA, NEON). Generic over `f32` and `f64`.
   - 8 accumulators on 32-register ISAs (NEON, AVX-512), 4 on AVX2; FMA; all sums of a metric in one pass.
   - `f32` processes twice as many lanes as `f64` (about 2× faster at large sizes).
   - Zero threading overhead; deterministic.

   Measured on Apple Silicon (NEON, `f64`, `cargo bench --bench extended_benchmarks`):

   | Kernel (n = 1024) | Scalar | SIMD | Speed-up |
   |---|---|---|---|
   | dot product | 1.12 µs | 0.17 µs | 6.5× |
   | cosine similarity | 1.25 µs | 0.23 µs | 5.4× |
   | Tanimoto | 1.22 µs | 0.23 µs | 5.3× |
   | Ruzicka | 1.19 µs | 0.25 µs | 4.7× |
   | variance | 2.38 µs | 0.32 µs | 7.5× |
   | Bray–Curtis | 1.19 µs | 0.46 µs | 2.6× |
   | Canberra (division-bound) | 1.24 µs | 0.70 µs | 1.8× |
   | Chebyshev | 0.33 µs | 0.25 µs | 1.3× |

   | Information measures (n = 65 536) | Scalar | SIMD | Speed-up |
   |---|---|---|---|
   | Hellinger | 324 µs | 45 µs | 7.2× |
   | Jensen–Shannon | 1.54 ms | 0.73 ms | 2.1× |
   | KL divergence | 778 µs | 502 µs | 1.5× |
   | Shannon entropy | 363 µs | 231 µs | 1.6× |

   NEON has only two `f64` lanes and Apple's scalar `log` is very fast, so the gains for the `ln`-based measures should be larger on AVX2/AVX-512 (4/8 lanes).

   | Library search (100 000 spectra × 30 peaks) | Time |
   |---|---|
   | brute-force `calculate_entropy_similarity_matched` | 164 ms |
   | `FlashEntropyIndex::search` | 0.075 ms |
   | index build (one-off) | 59 ms |
   | `SpectralLibrary::search`, all spectra (entropy) | 40 ms |
   | `SpectralLibrary::search`, 10 ppm precursor filter | 0.0009 ms |

   | Batch | Naive loop | `batch` | Speed-up |
   |---|---|---|---|
   | 1024 × 1024 cosine matrix, d = 128 (`f32`) | 130 ms | 3.0 ms | 44× |
   | top-10 of 100 000 × 64 (`f32`, Euclidean) | – | 0.81 ms | – |

3. **Parallel with Rayon (`cosine_similarity_parallel`, `CosineSimilarityParallel`)**:
   - Multi-threaded CPU parallelization across all available CPU cores.
   - **Sweet spot: > 100,000 to 200,000+ elements.**
   - For small to medium arrays (< 50,000 elements), thread pool scheduling overhead makes parallel processing significantly slower than single-threaded SIMD.

#### When to use what:

| Vector Length ($N$) | Recommended Tier | Reasoning |
|---|---|---|
| $N \le 2$ (Dot Product only) | **Generic / Scalar** | For 1–2 elements, scalar multiply/accumulate avoids loop setup (~1.6 ns vs ~2.0 ns) |
| $N \ge 2$ (Cosine / Distance) or $N \ge 4$ (Dot Product) up to $\sim 100{,}000$ | **SIMD (`pulp`)** | **Consistently faster across the board** (1.3× to 7.5× speedup); SIMD dispatch overhead on modern hardware is under 1 ns |
| $N \ge 100{,}000$ | **Parallel (`rayon`)** or **SIMD** | Workload is large enough to amortize multi-core thread synchronization |

For large datasets, use the optimized trait variants:

```rust
let large_a: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
let large_b: Vec<f64> = (0..10_000).map(|i| i as f64 * 1.1).collect();

// 2-3x faster for large vectors with SIMD:
let result = CosineSimilarityOptimized::similarity((&large_a, &large_b));

// For huge vectors (100k+), parallel processing:
let result = CosineSimilarityParallel::similarity((&large_a, &large_b));
```

## Examples

Run the comprehensive demo:

```bash
cargo run --example trait_demo
```

This demonstrates all available trait implementations with:
- Similarity and distance metrics
- Entropy measures for spectral data
- Data transformations
- Performance comparisons
- FFT-optimized operations

## API Documentation

### Trait Definitions

```rust
// For comparing two entities
pub trait Similarity<InputType, OutputType> {
    fn similarity(input: InputType) -> OutputType;
}

// For analyzing single entities  
pub trait EntropyMeasure<InputType, OutputType> {
    fn entropy(input: InputType) -> OutputType;
}

// For transforming data
pub trait DataTransform<InputType, OutputType> {
    fn transform(input: InputType) -> OutputType;
}
```

### Available Implementations

**Similarity Traits:**
- `CosineSimilarity`, `CosineSimilarityOptimized`, `CosineSimilarityParallel`
- `CosineDistance`, `CosineDistanceOptimized`, `CosineDistanceParallel`  
- `EuclideanDistance`, `SquaredEuclideanDistance`
- `PearsonCorrelationDistance`, `PearsonCorrelationDistanceOptimized`, `PearsonCorrelationDistanceParallel`
- `JaccardIndex`
- `HitRate`, `OvershootRate`
- `CrossCorrelationOptimized`, `CrossCorrelationParallel`, `CrossCorrelationFFTOptimized`
- `TimeShiftFinder`, `TimeShiftFinderFFT`
- `EntropySimilarity`, `EntropySimilarityMatched`, `SpectralCosine`
- SIMD (generic `f32`/`f64`): `ChebyshevDistance`, `CanberraDistance`, `BrayCurtisDistance`, `ChiSquareDistance`, `TanimotoSimilarity`, `DiceSimilarity`, `RuzickaSimilarity`, `HistogramIntersection`, `AngularDistance`, `PearsonCorrelation`, `DotProductSimd`, `ManhattanDistanceOptimized`
- `MinkowskiDistance`, `DtwDistance`, `LevenshteinDistance`
- `KullbackLeiblerDivergence`, `JensenShannonDivergence`, `JensenShannonDistance`, `HellingerDistance`, `BhattacharyyaDistance`, `TotalVariationDistance`
- `SpearmanCorrelation`, `KendallTau`, `DiceCoefficient`, `OverlapCoefficient`

**Entropy Traits:**
- `ShannonEntropy`, `ShannonEntropyOptimized`
- `TsallisEntropy`, `TsallisEntropyOptimized`

**Transform Traits:**
- `WeightFactorTransformation`, `WeightFactorTransformationOptimized`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome. Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Related Projects

- [`nalgebra`](https://crates.io/crates/nalgebra): Linear algebra library
- [`ndarray`](https://crates.io/crates/ndarray): N-dimensional arrays
- [`rayon`](https://crates.io/crates/rayon): Parallel iterators
- [`rustfft`](https://crates.io/crates/rustfft): FFT implementation

