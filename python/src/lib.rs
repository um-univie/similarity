//! Python bindings for the `similarity` crate (module `similarity_rs._core`).

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use similarity::batch::{self, BatchError, Metric, Neighbor};
use similarity::flash_entropy::{FlashEntropyIndex as RsFlash, FlashEntropyOptions};
use similarity::library_search::{
    LibraryHit, LibrarySearchOptions, SpectralLibrary as RsLibrary, SpectralMetric,
};
use similarity::simd::{self, Isa, SimdFloat};
use similarity::{MzTolerance, Spectrum};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn err(e: BatchError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn parse_metric(name: &str) -> PyResult<Metric> {
    Ok(match name.to_ascii_lowercase().replace(['-', ' '], "_").as_str() {
        "dot" | "inner_product" => Metric::Dot,
        "cosine" | "cosine_similarity" => Metric::Cosine,
        "cosine_distance" => Metric::CosineDistance,
        "euclidean" | "l2" => Metric::Euclidean,
        "sqeuclidean" | "squared_euclidean" => Metric::SquaredEuclidean,
        "manhattan" | "cityblock" | "l1" => Metric::Manhattan,
        "chebyshev" | "linf" => Metric::Chebyshev,
        "canberra" => Metric::Canberra,
        "braycurtis" | "bray_curtis" => Metric::BrayCurtis,
        "chi_square" | "chisquare" => Metric::ChiSquare,
        "tanimoto" => Metric::Tanimoto,
        "ruzicka" | "weighted_jaccard" => Metric::Ruzicka,
        "pearson" | "correlation" => Metric::Pearson,
        other => return Err(PyValueError::new_err(format!("unknown metric {other:?}"))),
    })
}

fn tolerance(value: f64, ppm: bool) -> MzTolerance<f64> {
    if ppm {
        MzTolerance::Ppm(value)
    } else {
        MzTolerance::Absolute(value)
    }
}

fn spectrum(mz: &PyReadonlyArray1<'_, f64>, intensity: &PyReadonlyArray1<'_, f64>) -> PyResult<Spectrum<f64>> {
    let (m, i) = (mz.as_array(), intensity.as_array());
    if m.len() != i.len() {
        return Err(PyValueError::new_err("mz and intensity must have the same length"));
    }
    Ok(Spectrum::from_arrays(&m.to_vec(), &i.to_vec()).expect("lengths checked"))
}

fn spectra(list: &[(PyReadonlyArray1<'_, f64>, PyReadonlyArray1<'_, f64>)]) -> PyResult<Vec<Spectrum<f64>>> {
    list.iter().map(|(m, i)| spectrum(m, i)).collect()
}

fn slice<'a, E: numpy::Element>(a: &'a PyReadonlyArray1<'_, E>) -> PyResult<&'a [E]> {
    a.as_slice().map_err(|_| PyValueError::new_err("array must be C-contiguous"))
}

fn slice2<'a, E: numpy::Element>(a: &'a PyReadonlyArray2<'_, E>) -> PyResult<(&'a [E], usize)> {
    let dim = a.as_array().ncols();
    let s = a.as_slice().map_err(|_| PyValueError::new_err("array must be C-contiguous"))?;
    Ok((s, dim))
}

fn rows<E>(data: &[E], dim: usize) -> PyResult<Vec<&[E]>> {
    if dim == 0 {
        return Err(PyValueError::new_err("rows must have at least one column"));
    }
    batch::rows_from_flat(data, dim).map_err(err)
}

fn neighbors_to_py<'py, E: SimdFloat + numpy::Element>(
    py: Python<'py>,
    hits: Vec<Neighbor<E>>,
) -> (Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<E>>) {
    let idx: Vec<usize> = hits.iter().map(|h| h.index).collect();
    let sc: Vec<E> = hits.iter().map(|h| h.score).collect();
    (idx.into_pyarray(py), sc.into_pyarray(py))
}

fn matrix<'py, E: numpy::Element>(py: Python<'py>, r: usize, c: usize, data: Vec<E>) -> Bound<'py, PyArray2<E>> {
    Array2::from_shape_vec((r, c), data).expect("shape").into_pyarray(py)
}

/// 2-D float32 or float64 array.
#[derive(FromPyObject)]
enum Mat<'py> {
    F64(PyReadonlyArray2<'py, f64>),
    F32(PyReadonlyArray2<'py, f32>),
}

/// 1-D float32 or float64 array.
#[derive(FromPyObject)]
enum Vector<'py> {
    F64(PyReadonlyArray1<'py, f64>),
    F32(PyReadonlyArray1<'py, f32>),
}

// ---------------------------------------------------------------------------
// vector metrics
// ---------------------------------------------------------------------------

/// SIMD instruction set used on this machine ("Scalar", "Avx2", "Avx512", "Neon").
#[pyfunction]
fn isa() -> String {
    format!("{:?}", Isa::detected())
}

/// Backend for dense dot product and cosine matrices ("accelerate" or "simd").
#[pyfunction]
fn dense_backend() -> &'static str {
    batch::dense_backend()
}

/// Metric between two 1-D arrays of the same dtype (float32/float64).
/// Returns ``nan`` if undefined (e.g. cosine of a zero vector).
#[pyfunction]
#[pyo3(signature = (a, b, metric = "cosine"))]
fn distance(a: Vector<'_>, b: Vector<'_>, metric: &str) -> PyResult<f64> {
    let m = parse_metric(metric)?;
    let check = |la: usize, lb: usize| {
        if la == lb {
            Ok(())
        } else {
            Err(PyValueError::new_err("a and b must have the same length"))
        }
    };
    match (a, b) {
        (Vector::F64(a), Vector::F64(b)) => {
            let (a, b) = (slice(&a)?, slice(&b)?);
            check(a.len(), b.len())?;
            Ok(m.eval(a, b).unwrap_or(f64::NAN))
        }
        (Vector::F32(a), Vector::F32(b)) => {
            let (a, b) = (slice(&a)?, slice(&b)?);
            check(a.len(), b.len())?;
            Ok(m.eval(a, b).map_or(f64::NAN, f64::from))
        }
        _ => Err(PyValueError::new_err("a and b must have the same dtype")),
    }
}

/// All-pairs metric between the rows of ``a`` (n×d) and ``b`` (m×d): an n×m array.
#[pyfunction]
#[pyo3(signature = (a, b, metric = "euclidean"))]
fn cdist<'py>(py: Python<'py>, a: Mat<'py>, b: Mat<'py>, metric: &str) -> PyResult<Bound<'py, PyAny>> {
    let m = parse_metric(metric)?;
    macro_rules! run {
        ($a:expr, $b:expr) => {{
            let ((da, ka), (db, kb)) = (slice2(&$a)?, slice2(&$b)?);
            if ka != kb {
                return Err(PyValueError::new_err("a and b must have the same number of columns"));
            }
            let (ra, rb) = (rows(da, ka)?, rows(db, kb)?);
            let dm = py.detach(|| batch::cdist(&ra, &rb, m)).map_err(err)?;
            Ok(matrix(py, dm.rows, dm.cols, dm.data).into_any())
        }};
    }
    match (a, b) {
        (Mat::F64(a), Mat::F64(b)) => run!(a, b),
        (Mat::F32(a), Mat::F32(b)) => run!(a, b),
        _ => Err(PyValueError::new_err("a and b must have the same dtype")),
    }
}

/// Condensed pairwise metric of the rows of ``x`` (like ``scipy.spatial.distance.pdist``).
#[pyfunction]
#[pyo3(signature = (x, metric = "euclidean"))]
fn pdist<'py>(py: Python<'py>, x: Mat<'py>, metric: &str) -> PyResult<Bound<'py, PyAny>> {
    let m = parse_metric(metric)?;
    macro_rules! run {
        ($x:expr) => {{
            let (d, k) = slice2(&$x)?;
            let r = rows(d, k)?;
            let v = py.detach(|| batch::pdist(&r, m)).map_err(err)?;
            Ok(v.into_pyarray(py).into_any())
        }};
    }
    match x {
        Mat::F64(x) => run!(x),
        Mat::F32(x) => run!(x),
    }
}

/// The ``k`` rows of ``x`` closest to ``query`` (largest similarity or
/// smallest distance): ``(indices, scores)``.
#[pyfunction]
#[pyo3(signature = (query, x, k, metric = "cosine"))]
fn top_k<'py>(py: Python<'py>, query: Vector<'py>, x: Mat<'py>, k: usize, metric: &str) -> PyResult<Bound<'py, PyAny>> {
    let m = parse_metric(metric)?;
    macro_rules! run {
        ($q:expr, $x:expr) => {{
            let q = slice(&$q)?;
            let (d, dim) = slice2(&$x)?;
            let r = rows(d, dim)?;
            let hits = py.detach(|| batch::top_k(q, &r, k, m)).map_err(err)?;
            Ok(neighbors_to_py(py, hits).into_pyobject(py)?.into_any())
        }};
    }
    match (query, x) {
        (Vector::F64(q), Mat::F64(x)) => run!(q, x),
        (Vector::F32(q), Mat::F32(x)) => run!(q, x),
        _ => Err(PyValueError::new_err("query and x must have the same dtype")),
    }
}

// ---------------------------------------------------------------------------
// divergences
// ---------------------------------------------------------------------------

macro_rules! divergence {
    ($name:ident, $f:path, $doc:literal) => {
        #[doc = $doc]
        #[pyfunction]
        fn $name(p: PyReadonlyArray1<'_, f64>, q: PyReadonlyArray1<'_, f64>) -> PyResult<Option<f64>> {
            Ok($f(slice(&p)?, slice(&q)?))
        }
    };
}

divergence!(kl_divergence, simd::kl_divergence_simd, "Kullback–Leibler divergence D(p‖q) in nats (inputs are normalised). ``None`` if invalid.");
divergence!(cross_entropy, simd::cross_entropy_simd, "Cross-entropy H(p, q) in nats. ``None`` if invalid.");
divergence!(js_divergence, simd::js_divergence_simd, "Jensen–Shannon divergence in nats. ``None`` if invalid.");
divergence!(js_distance, simd::js_distance_simd, "Jensen–Shannon distance √(JSD / ln 2) in [0, 1] (base-2, a metric). ``None`` if invalid.");
divergence!(hellinger_distance, simd::hellinger_distance_simd, "Hellinger distance. ``None`` if invalid.");
divergence!(bhattacharyya_distance, simd::bhattacharyya_distance_simd, "Bhattacharyya distance. ``None`` if invalid.");
divergence!(total_variation_distance, simd::total_variation_distance_simd, "Total variation distance. ``None`` if invalid.");

/// Shannon entropy (nats) of a non-negative vector (normalised internally).
#[pyfunction]
fn shannon_entropy(x: PyReadonlyArray1<'_, f64>) -> PyResult<Option<f64>> {
    Ok(simd::shannon_entropy_simd(slice(&x)?))
}

// ---------------------------------------------------------------------------
// spectra
// ---------------------------------------------------------------------------

/// Entropy similarity of two spectra with m/z-tolerant greedy matching.
#[pyfunction]
#[pyo3(signature = (mz_a, intensity_a, mz_b, intensity_b, tolerance = 0.01, ppm = false))]
fn entropy_similarity(
    mz_a: PyReadonlyArray1<'_, f64>,
    intensity_a: PyReadonlyArray1<'_, f64>,
    mz_b: PyReadonlyArray1<'_, f64>,
    intensity_b: PyReadonlyArray1<'_, f64>,
    tolerance: f64,
    ppm: bool,
) -> PyResult<f64> {
    let (a, b) = (spectrum(&mz_a, &intensity_a)?, spectrum(&mz_b, &intensity_b)?);
    Ok(similarity::calculate_entropy_similarity_matched(&a, &b, self::tolerance(tolerance, ppm)))
}

/// Modified cosine of two spectra with greedy matching: ``(score, matched_peaks)``.
#[pyfunction]
#[pyo3(signature = (mz_a, intensity_a, mz_b, intensity_b, tolerance = 0.01, ppm = false, mz_power = 0.0, intensity_power = 1.0))]
#[allow(clippy::too_many_arguments)]
fn spectral_cosine(
    mz_a: PyReadonlyArray1<'_, f64>,
    intensity_a: PyReadonlyArray1<'_, f64>,
    mz_b: PyReadonlyArray1<'_, f64>,
    intensity_b: PyReadonlyArray1<'_, f64>,
    tolerance: f64,
    ppm: bool,
    mz_power: f64,
    intensity_power: f64,
) -> PyResult<(f64, usize)> {
    let (a, b) = (spectrum(&mz_a, &intensity_a)?, spectrum(&mz_b, &intensity_b)?);
    Ok(similarity::spectral_cosine_similarity(&a, &b, self::tolerance(tolerance, ppm), mz_power, intensity_power))
}

type PySpectrum<'py> = (PyReadonlyArray1<'py, f64>, PyReadonlyArray1<'py, f64>);

/// ``(indices, scores)`` pair returned by top-k searches.
type PyHits<'py> = (Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<f64>>);

/// Flash entropy index over a spectral library.
///
/// ``spectra`` is a list of ``(mz, intensity)`` float64 array pairs.
#[pyclass(module = "similarity_rs._core", frozen)]
struct FlashEntropyIndex {
    inner: RsFlash<f64>,
}

#[pymethods]
impl FlashEntropyIndex {
    #[new]
    #[pyo3(signature = (spectra, tolerance = 0.01, ppm = false, weighted = true, noise_fraction = 0.0, max_peaks = None))]
    fn new(
        py: Python<'_>,
        spectra: Vec<PySpectrum<'_>>,
        tolerance: f64,
        ppm: bool,
        weighted: bool,
        noise_fraction: f64,
        max_peaks: Option<usize>,
    ) -> PyResult<Self> {
        let lib = self::spectra(&spectra)?;
        let opts = FlashEntropyOptions { tolerance: self::tolerance(tolerance, ppm), weighted, noise_fraction, max_peaks };
        Ok(Self { inner: py.detach(|| RsFlash::new(&lib, opts)) })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Total number of indexed peaks after cleaning.
    #[getter]
    fn n_peaks(&self) -> usize {
        self.inner.n_peaks()
    }

    /// Scores against every library spectrum (library order).
    fn search<'py>(
        &self,
        py: Python<'py>,
        mz: PyReadonlyArray1<'py, f64>,
        intensity: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let q = spectrum(&mz, &intensity)?;
        Ok(py.detach(|| self.inner.search(&q)).into_pyarray(py))
    }

    /// Best ``k`` hits: ``(indices, scores)``.
    fn search_top_k<'py>(
        &self,
        py: Python<'py>,
        mz: PyReadonlyArray1<'py, f64>,
        intensity: PyReadonlyArray1<'py, f64>,
        k: usize,
    ) -> PyResult<PyHits<'py>> {
        let q = spectrum(&mz, &intensity)?;
        Ok(neighbors_to_py(py, py.detach(|| self.inner.search_top_k(&q, k))))
    }

    /// ``search_top_k`` for a list of ``(mz, intensity)`` queries, in parallel.
    fn search_many<'py>(
        &self,
        py: Python<'py>,
        queries: Vec<PySpectrum<'py>>,
        k: usize,
    ) -> PyResult<Vec<PyHits<'py>>> {
        let qs = spectra(&queries)?;
        let res = py.detach(|| self.inner.search_many(&qs, k));
        Ok(res.into_iter().map(|h| neighbors_to_py(py, h)).collect())
    }
}

/// Batch spectral library search with optional precursor filtering.
///
/// ``metric`` is ``"entropy"``, ``"unweighted_entropy"`` or ``"cosine"``.
#[pyclass(module = "similarity_rs._core", frozen)]
struct SpectralLibrary {
    inner: RsLibrary<f64>,
}

fn hits_to_py(hits: Vec<LibraryHit<f64>>) -> Vec<(usize, f64, usize)> {
    hits.into_iter().map(|h| (h.index, h.score, h.matched_peaks)).collect()
}

#[pymethods]
impl SpectralLibrary {
    #[new]
    #[pyo3(signature = (
        spectra, precursor_mz = None, metric = "entropy", tolerance = 0.01, ppm = false,
        precursor_tolerance = None, precursor_ppm = false, mz_power = 0.0, intensity_power = 1.0,
        min_matched_peaks = 1, min_score = 0.0, noise_fraction = 0.0, max_peaks = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        spectra: Vec<PySpectrum<'_>>,
        precursor_mz: Option<PyReadonlyArray1<'_, f64>>,
        metric: &str,
        tolerance: f64,
        ppm: bool,
        precursor_tolerance: Option<f64>,
        precursor_ppm: bool,
        mz_power: f64,
        intensity_power: f64,
        min_matched_peaks: usize,
        min_score: f64,
        noise_fraction: f64,
        max_peaks: Option<usize>,
    ) -> PyResult<Self> {
        let metric = match metric.to_ascii_lowercase().as_str() {
            "entropy" => SpectralMetric::Entropy,
            "unweighted_entropy" => SpectralMetric::UnweightedEntropy,
            "cosine" => SpectralMetric::Cosine { mz_power, intensity_power },
            other => return Err(PyValueError::new_err(format!("unknown spectral metric {other:?}"))),
        };
        let opts = LibrarySearchOptions {
            metric,
            fragment_tolerance: self::tolerance(tolerance, ppm),
            precursor_tolerance: precursor_tolerance.map(|t| self::tolerance(t, precursor_ppm)),
            noise_fraction,
            max_peaks,
            min_matched_peaks,
            min_score,
        };
        let lib = self::spectra(&spectra)?;
        let inner = match precursor_mz {
            Some(p) => {
                let p = p.as_array().to_vec();
                py.detach(|| RsLibrary::with_precursors(&lib, &p, opts)).map_err(err)?
            }
            None => py.detach(|| RsLibrary::new(&lib, opts)),
        };
        Ok(Self { inner })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Hits as a list of ``(index, score, matched_peaks)``, best first
    /// (all passing hits if ``k == 0``).
    #[pyo3(signature = (mz, intensity, precursor_mz = None, k = 0))]
    fn search(
        &self,
        py: Python<'_>,
        mz: PyReadonlyArray1<'_, f64>,
        intensity: PyReadonlyArray1<'_, f64>,
        precursor_mz: Option<f64>,
        k: usize,
    ) -> PyResult<Vec<(usize, f64, usize)>> {
        let q = spectrum(&mz, &intensity)?;
        Ok(hits_to_py(py.detach(|| self.inner.search(&q, precursor_mz, k))))
    }

    /// ``search`` for a list of ``(mz, intensity)`` queries, in parallel.
    #[pyo3(signature = (queries, precursor_mz = None, k = 0))]
    fn search_many(
        &self,
        py: Python<'_>,
        queries: Vec<PySpectrum<'_>>,
        precursor_mz: Option<PyReadonlyArray1<'_, f64>>,
        k: usize,
    ) -> PyResult<Vec<Vec<(usize, f64, usize)>>> {
        let qs = spectra(&queries)?;
        let p = precursor_mz.map(|p| p.as_array().to_vec());
        let res = py.detach(|| self.inner.search_many(&qs, p.as_deref(), k)).map_err(err)?;
        Ok(res.into_iter().map(hits_to_py).collect())
    }

    /// Symmetric n×n similarity matrix of the library spectra.
    fn similarity_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.len();
        let m = py.detach(|| self.inner.similarity_matrix());
        matrix(py, n, n, m)
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(isa, m)?)?;
    m.add_function(wrap_pyfunction!(dense_backend, m)?)?;
    m.add_function(wrap_pyfunction!(distance, m)?)?;
    m.add_function(wrap_pyfunction!(cdist, m)?)?;
    m.add_function(wrap_pyfunction!(pdist, m)?)?;
    m.add_function(wrap_pyfunction!(top_k, m)?)?;
    m.add_function(wrap_pyfunction!(shannon_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(kl_divergence, m)?)?;
    m.add_function(wrap_pyfunction!(cross_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(js_divergence, m)?)?;
    m.add_function(wrap_pyfunction!(js_distance, m)?)?;
    m.add_function(wrap_pyfunction!(hellinger_distance, m)?)?;
    m.add_function(wrap_pyfunction!(bhattacharyya_distance, m)?)?;
    m.add_function(wrap_pyfunction!(total_variation_distance, m)?)?;
    m.add_function(wrap_pyfunction!(entropy_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(spectral_cosine, m)?)?;
    m.add_class::<FlashEntropyIndex>()?;
    m.add_class::<SpectralLibrary>()?;
    Ok(())
}
