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

- `accelerate` feature (on by default): dense dot product and cosine
  matrices use Apple's Accelerate framework on macOS (only together with
  `parallel`, since Accelerate starts its own threads). No effect
  elsewhere. `batch::dense_backend()` reports the backend in use.
- `batch::cdist_into` and `batch::pairwise_into` write into a caller-provided
  buffer, so repeated calls can reuse one allocation.
- `benchmarks/gemm-compare` and a CI job that times the dense kernel against
  the `gemm` and `matrixmultiply` crates on x86.

### Changed
- `cdist`, `pairwise`, `pdist` and `top_k_many` compute dot product, cosine
  and Euclidean metrics with a GEMM-style kernel (packed operands, depth and
  column blocking), 2 to 4 times faster than before on large inputs.
  `pairwise` results are now exactly symmetric.
- `pairwise` and `pdist` compute only the upper triangle (like BLAS SYRK,
  on Accelerate through `ssyrk`/`dsyrk`), about 2 times faster than
  `cdist(a, a)` for long vectors. `pdist` no longer fills the lower half.
- Accelerate calls are split into blocks of 2048 rows; a single call on
  large outputs with short vectors was up to 10 times slower.
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
