"""SIMD-accelerated similarity metrics and spectral library search.

Python bindings for the Rust crate `similarity`
(https://github.com/um-univie/similarity).
"""

from ._core import (
    FlashEntropyIndex,
    SpectralLibrary,
    __version__,
    bhattacharyya_distance,
    cdist,
    cross_entropy,
    distance,
    entropy_similarity,
    hellinger_distance,
    isa,
    js_distance,
    js_divergence,
    kl_divergence,
    pdist,
    shannon_entropy,
    spectral_cosine,
    top_k,
    total_variation_distance,
)

__all__ = [
    "FlashEntropyIndex",
    "SpectralLibrary",
    "__version__",
    "bhattacharyya_distance",
    "cdist",
    "cross_entropy",
    "distance",
    "entropy_similarity",
    "hellinger_distance",
    "isa",
    "js_distance",
    "js_divergence",
    "kl_divergence",
    "pdist",
    "shannon_entropy",
    "spectral_cosine",
    "top_k",
    "total_variation_distance",
]
