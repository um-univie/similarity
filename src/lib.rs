//! Similarity, distance, divergence and correlation metrics for vectors,
//! sets, probability distributions and mass spectra, with scalar generic,
//! parallel (rayon) and portable-SIMD (pulp) implementations.
//!
//! See the [README](https://github.com/um-univie/similarity) for an overview.

pub mod similarity_metrics;
pub mod spectral_entropy;
pub mod simd;
pub mod batch;
pub mod distance_metrics;
pub mod divergences;
pub mod statistics;
pub mod set_metrics;
pub mod flash_entropy;
pub mod library_search;

// New trait system with semantic separation
pub mod traits;
pub mod similarity_traits;
pub mod entropy_traits;
pub mod transform_traits;

// Re-export everything from similarity_metrics
pub use similarity_metrics::*;

// Re-export everything from spectral_entropy
pub use spectral_entropy::*;

// Re-export the scalar metric modules
pub use distance_metrics::*;
pub use divergences::*;
pub use statistics::*;
pub use set_metrics::*;

// Re-export SIMD-accelerated functions (all have _simd suffix — no conflicts)
pub use simd::*;

// Re-export trait definitions
pub use traits::{Similarity, EntropyMeasure, DataTransform};
