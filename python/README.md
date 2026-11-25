# similarity-rs (Python)

Python bindings for the Rust crate [`similarity`](https://github.com/um-univie/similarity):
SIMD-accelerated vector metrics (AVX2 / AVX-512 / NEON picked at runtime),
divergences, and mass-spectrometry library search (Flash entropy index,
precursor-filtered batch search with entropy or cosine similarity).

## Build

```bash
cd python
uv venv && uv pip install maturin numpy pytest
uv run maturin develop --release
uv run pytest tests
```

## Usage

```python
import numpy as np
import similarity_rs as sr

x = np.random.rand(1000, 128).astype(np.float32)
d = sr.cdist(x[:10], x, "cosine_distance")     # (10, 1000), float32
idx, scores = sr.top_k(x[0], x, 5, "cosine")
sr.js_divergence(np.array([1.0, 2.0]), np.array([2.0, 1.0]))

# spectra are (mz, intensity) float64 array pairs
library = [(np.array([100.0, 150.0, 200.0]), np.array([1.0, 3.0, 5.0])), ...]
index = sr.FlashEntropyIndex(library, tolerance=0.02)
idx, scores = index.search_top_k(mz, intensity, k=10)

lib = sr.SpectralLibrary(library, precursor_mz=precursors, metric="cosine",
                         tolerance=10, ppm=True, precursor_tolerance=0.01)
hits = lib.search(mz, intensity, precursor_mz=250.1, k=5)  # [(index, score, matched_peaks)]
network = lib.similarity_matrix()
```

All heavy calls release the GIL and use all cores (rayon).
`sr.isa()` reports the instruction set in use.
