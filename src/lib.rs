pub mod similarity_metrics;
pub mod spectral_entropy;

// New trait system with semantic separation
pub mod traits;
pub mod similarity_traits;
pub mod entropy_traits;
pub mod transform_traits;

// Re-export everything from similarity_metrics
pub use similarity_metrics::*;

// Re-export everything from spectral_entropy  
pub use spectral_entropy::*;

// Re-export trait definitions
pub use traits::{Similarity, EntropyMeasure, DataTransform};