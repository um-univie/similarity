# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.4.0] - unreleased

### Added
- `sparse` module: `SparseVector` with dot, cosine, Euclidean, Manhattan,
  Jaccard, Tanimoto and Ruzicka; `SparseIndex` (inverted index) with
  `scores`, `top_k` and `top_k_many`; `sparse_cdist`.
- Property-based tests (proptest) for metric axioms, SIMD against scalar
  results, tiled against per-pair `cdist`, sparse against dense, and Flash
  entropy against `SpectralLibrary`.

### Changed
- `cdist`, `pairwise`, `pdist` and `top_k_many` compute dot product, cosine
  and Euclidean metrics with a register-blocked tile kernel, 2 to 3 times
  faster on large inputs. `pairwise` results are now exactly symmetric.
- Reworded the crate description and the README introduction.

## [0.3.0] - 2026-11-26

### Added
- Portable SIMD (pulp 0.22) for `f32` and `f64` with runtime dispatch
  (scalar, AVX2+FMA, AVX-512 via the `avx512` feature, NEON): statistics,
  transforms, inner-product / distance metrics, correlation and error metrics
  (`*_simd` functions, `SimdFloat`, `Isa::detected()`).
- Vectorised `ln` / `exp` (≤ 2 ulp), `sqrt`, `pow`; SIMD Shannon entropy, KL,
  cross-entropy, Jensen–Shannon, Bhattacharyya, Hellinger, total variation and
  spectral entropy weighting.
- `batch`: `pairwise`, `pdist`, `cdist`, `one_to_many`, `top_k`, `top_k_many`.
- `distance_metrics`, `divergences`, `statistics`, `set_metrics` modules
  (Minkowski family, Canberra, Bray–Curtis, χ², Tanimoto, Dice, Hamming,
  Levenshtein, DTW, Rényi, Wasserstein, Spearman, Kendall τ-b, CCC, MAPE, …).
- Mass spectrometry: `MzTolerance` (Da / ppm), greedy `match_peaks(_by)`,
  `spectral_cosine_similarity`, `calculate_entropy_similarity_matched`.
- `flash_entropy::FlashEntropyIndex`: exact indexed entropy-similarity
  library search (~2000× faster than brute force on 100k spectra).
- `library_search::SpectralLibrary`: batch search with entropy / unweighted
  entropy / modified cosine, precursor m/z filtering, min matched peaks,
  min score, top-k, parallel `search_many` and `similarity_matrix`.
- Python bindings (`python/`, PyO3 + NumPy, package `similarity-rs`).
- GitHub Actions CI on Linux x86/ARM, macOS and Windows; MSRV check.

### Changed
- Minimum supported Rust version is now 1.83 (the `avx512` feature needs 1.89).
- Generic `*Optimized` trait implementations work for both `f32` and `f64`.
- Empty inputs and NaN handling fixed in several `similarity_metrics` functions.

### Deprecated
- `calculate_entropy_similarity`: pairs peaks by index; use
  `calculate_entropy_similarity_matched` or `FlashEntropyIndex`.

### Removed
- The unused public `Sqrt` trait.

## [0.2.0] - 2025-07-02

- Trait-based API (`Similarity`, `EntropyMeasure`, `DataTransform`),
  Jaccard index, benchmarks.
