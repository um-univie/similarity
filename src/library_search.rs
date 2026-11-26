//! # Batch spectral library search
//!
//! Scores many query spectra against a spectral library with a choice of
//! similarity ([`SpectralMetric`]), optional **precursor m/z filtering**,
//! minimum-matched-peak and minimum-score thresholds, and top-k selection.
//!
//! Library spectra are cleaned once with [`flash_entropy::preprocess`]
//! (peaks closer than `2 × tolerance` merged, optional noise filter and
//! top-N) and stored in flat m/z-sorted arrays. Because cleaned peaks are
//! more than one tolerance window apart, every query peak matches at most
//! one library peak, so a single linear merge per candidate gives the exact
//! one-to-one matching, with no candidate sorting as in greedy matching.
//!
//! Use this for **identity search** (precursor-filtered) and for
//! all-vs-all similarity matrices ([`SpectralLibrary::similarity_matrix`],
//! e.g. for molecular networking). For open search over very large
//! libraries with the entropy similarity, [`FlashEntropyIndex`] is faster
//! because it only visits library peaks inside the query's m/z windows.
//!
//! [`FlashEntropyIndex`]: crate::flash_entropy::FlashEntropyIndex
//! [`flash_entropy::preprocess`]: crate::flash_entropy::preprocess
//!
//! ```
//! use similarity::library_search::{LibrarySearchOptions, SpectralLibrary};
//! use similarity::{MzTolerance, Spectrum};
//!
//! let library = vec![
//!     Spectrum::from_arrays(&[100.0, 150.0, 200.0], &[1.0, 3.0, 5.0]).unwrap(),
//!     Spectrum::from_arrays(&[100.0, 150.0, 200.0], &[5.0, 3.0, 1.0]).unwrap(),
//! ];
//! let precursors = [250.0, 400.0];
//! let opts = LibrarySearchOptions::entropy(MzTolerance::Absolute(0.02))
//!     .with_precursor_tolerance(MzTolerance::Ppm(10.0));
//! let lib = SpectralLibrary::with_precursors(&library, &precursors, opts).unwrap();
//!
//! let query = Spectrum::from_arrays(&[100.005, 150.0, 200.01], &[1.0, 3.0, 5.0]).unwrap();
//! // Only library entry 0 has a precursor within 10 ppm of 250.001.
//! let hits = lib.search(&query, Some(250.001), 5);
//! assert_eq!(hits.len(), 1);
//! assert_eq!(hits[0].index, 0);
//! assert_eq!(hits[0].matched_peaks, 3);
//! assert!((hits[0].score - 1.0_f64).abs() < 1e-10);
//! ```

use crate::batch::BatchError;
use crate::flash_entropy::{preprocess, FlashEntropyOptions};
use crate::simd::{self, SimdFloat};
use crate::{MzTolerance, Spectrum};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Similarity used to score a query against library spectra.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpectralMetric<E> {
    /// Spectral entropy similarity with entropy-based intensity weighting
    /// (Li et al. 2021), the same score as [`FlashEntropyIndex`](crate::flash_entropy::FlashEntropyIndex).
    Entropy,
    /// Entropy similarity without intensity weighting.
    UnweightedEntropy,
    /// Modified cosine with peak weights `mz^mz_power · intensity^intensity_power`
    /// (`(0, 1)` is the plain cosine, `(0, 0.5)` square-root scaling).
    Cosine { mz_power: E, intensity_power: E },
}

/// Options for [`SpectralLibrary`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LibrarySearchOptions<E> {
    /// Similarity measure.
    pub metric: SpectralMetric<E>,
    /// Fragment m/z tolerance for peak matching (and, doubled, peak merging).
    pub fragment_tolerance: MzTolerance<E>,
    /// If set, and both the library and the query have precursor m/z values,
    /// only library spectra within this precursor tolerance are scored.
    pub precursor_tolerance: Option<MzTolerance<E>>,
    /// Drop peaks below this fraction of the base peak (`0` keeps all).
    pub noise_fraction: E,
    /// Keep only the most intense `max_peaks` peaks after merging.
    pub max_peaks: Option<usize>,
    /// Hits need at least this many matched peaks (default `1`).
    pub min_matched_peaks: usize,
    /// Hits need at least this score (default `0`).
    pub min_score: E,
}

impl<E: SimdFloat> LibrarySearchOptions<E> {
    fn with_metric(metric: SpectralMetric<E>, fragment_tolerance: MzTolerance<E>) -> Self {
        Self {
            metric,
            fragment_tolerance,
            precursor_tolerance: None,
            noise_fraction: E::zero(),
            max_peaks: None,
            min_matched_peaks: 1,
            min_score: E::zero(),
        }
    }

    /// Weighted entropy similarity, no precursor filter.
    pub fn entropy(fragment_tolerance: MzTolerance<E>) -> Self {
        Self::with_metric(SpectralMetric::Entropy, fragment_tolerance)
    }

    /// Plain cosine (`mz_power = 0`, `intensity_power = 1`), no precursor filter.
    pub fn cosine(fragment_tolerance: MzTolerance<E>) -> Self {
        Self::with_metric(
            SpectralMetric::Cosine { mz_power: E::zero(), intensity_power: E::one() },
            fragment_tolerance,
        )
    }

    /// Sets the precursor tolerance.
    pub fn with_precursor_tolerance(mut self, tolerance: MzTolerance<E>) -> Self {
        self.precursor_tolerance = Some(tolerance);
        self
    }

    /// Sets the minimum number of matched peaks.
    pub fn with_min_matched_peaks(mut self, n: usize) -> Self {
        self.min_matched_peaks = n;
        self
    }

    /// Sets the minimum score.
    pub fn with_min_score(mut self, score: E) -> Self {
        self.min_score = score;
        self
    }

    fn cleaning(&self) -> FlashEntropyOptions<E> {
        FlashEntropyOptions {
            tolerance: self.fragment_tolerance,
            weighted: matches!(self.metric, SpectralMetric::Entropy),
            noise_fraction: self.noise_fraction,
            max_peaks: self.max_peaks,
        }
    }
}

/// One library match.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LibraryHit<E> {
    /// Index of the library spectrum.
    pub index: usize,
    /// Similarity score in `[0, 1]`.
    pub score: E,
    /// Number of matched peaks (after cleaning).
    pub matched_peaks: usize,
}

/// A preprocessed spectral library for batch searching.
#[derive(Debug, Clone)]
pub struct SpectralLibrary<E: SimdFloat> {
    opts: LibrarySearchOptions<E>,
    /// `offsets[i]..offsets[i + 1]` are the peaks of spectrum `i`.
    offsets: Vec<usize>,
    mz: Vec<E>,
    weight: Vec<E>,
    /// `(precursor_mz, index)` sorted by m/z, NaN entries removed.
    precursors: Option<Vec<(E, usize)>>,
}

impl<E: SimdFloat> SpectralLibrary<E> {
    /// Builds a library without precursor information (cleaning runs in
    /// parallel with the `parallel` feature).
    pub fn new(spectra: &[Spectrum<E>], opts: LibrarySearchOptions<E>) -> Self {
        let prepared: Vec<(Vec<E>, Vec<E>)> = par_map(spectra, |s| prepare(s, &opts));
        let mut offsets = Vec::with_capacity(spectra.len() + 1);
        offsets.push(0);
        let total = prepared.iter().map(|p| p.0.len()).sum();
        let (mut mz, mut weight) = (Vec::with_capacity(total), Vec::with_capacity(total));
        for (m, w) in prepared {
            mz.extend(m);
            weight.extend(w);
            offsets.push(mz.len());
        }
        Self { opts, offsets, mz, weight, precursors: None }
    }

    /// Builds a library with one precursor m/z per spectrum. Fails if the
    /// lengths differ. Entries with a NaN precursor never pass the
    /// precursor filter.
    pub fn with_precursors(
        spectra: &[Spectrum<E>],
        precursor_mz: &[E],
        opts: LibrarySearchOptions<E>,
    ) -> Result<Self, BatchError> {
        if spectra.len() != precursor_mz.len() {
            return Err(BatchError::InvalidArgument("precursor_mz must have one value per spectrum"));
        }
        let mut lib = Self::new(spectra, opts);
        let mut p: Vec<(E, usize)> =
            precursor_mz.iter().copied().enumerate().filter(|(_, m)| !m.is_nan()).map(|(i, m)| (m, i)).collect();
        p.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
        lib.precursors = Some(p);
        Ok(lib)
    }

    /// Number of library spectra.
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// `true` if the library is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Options the library was built with.
    pub fn options(&self) -> &LibrarySearchOptions<E> {
        &self.opts
    }

    fn spectrum(&self, i: usize) -> (&[E], &[E]) {
        let r = self.offsets[i]..self.offsets[i + 1];
        (&self.mz[r.clone()], &self.weight[r])
    }

    /// Library indices passing the precursor filter, or `None` for "all".
    fn candidates(&self, precursor_mz: Option<E>) -> Option<Vec<usize>> {
        let (tol, lib, q) = match (self.opts.precursor_tolerance, &self.precursors, precursor_mz) {
            (Some(t), Some(l), Some(q)) if !q.is_nan() => (t, l, q),
            _ => return None,
        };
        let w = tol.window(q);
        let lo = lib.partition_point(|p| p.0 < q - w);
        Some(
            lib[lo..]
                .iter()
                .take_while(|p| p.0 <= q + w)
                .filter(|p| tol.matches(q, p.0))
                .map(|p| p.1)
                .collect(),
        )
    }

    /// Similarity of `query` to every library spectrum, in library order,
    /// as `(score, matched_peaks)`. Ignores the precursor filter and the
    /// score/peak thresholds.
    pub fn score_all(&self, query: &Spectrum<E>) -> Vec<(E, usize)> {
        let (qm, qw) = prepare(query, &self.opts);
        (0..self.len())
            .map(|i| {
                let (lm, lw) = self.spectrum(i);
                score_pair(&self.opts, &qm, &qw, lm, lw)
            })
            .collect()
    }

    /// The best `k` hits for `query` (all passing hits if `k == 0`), best
    /// first, ties by index. `precursor_mz` enables the precursor filter if
    /// the library was built with precursors and the options set
    /// [`precursor_tolerance`](LibrarySearchOptions::precursor_tolerance).
    pub fn search(&self, query: &Spectrum<E>, precursor_mz: Option<E>, k: usize) -> Vec<LibraryHit<E>> {
        let (qm, qw) = prepare(query, &self.opts);
        let mut hits = Vec::new();
        let mut push = |i: usize| {
            let (lm, lw) = self.spectrum(i);
            let (score, matched_peaks) = score_pair(&self.opts, &qm, &qw, lm, lw);
            if matched_peaks >= self.opts.min_matched_peaks && score >= self.opts.min_score {
                hits.push(LibraryHit { index: i, score, matched_peaks });
            }
        };
        match self.candidates(precursor_mz) {
            Some(c) => c.into_iter().for_each(&mut push),
            None => (0..self.len()).for_each(&mut push),
        }
        let cmp = |a: &LibraryHit<E>, b: &LibraryHit<E>| {
            b.score.partial_cmp(&a.score).unwrap_or(core::cmp::Ordering::Equal).then(a.index.cmp(&b.index))
        };
        if k > 0 && k < hits.len() {
            hits.select_nth_unstable_by(k, cmp);
            hits.truncate(k);
        }
        hits.sort_by(cmp);
        hits
    }

    /// [`search`](Self::search) for many queries (parallel with the
    /// `parallel` feature). `precursor_mz`, if given, must have one value
    /// per query.
    pub fn search_many(
        &self,
        queries: &[Spectrum<E>],
        precursor_mz: Option<&[E]>,
        k: usize,
    ) -> Result<Vec<Vec<LibraryHit<E>>>, BatchError> {
        if let Some(p) = precursor_mz {
            if p.len() != queries.len() {
                return Err(BatchError::InvalidArgument("precursor_mz must have one value per query"));
            }
        }
        let idx: Vec<usize> = (0..queries.len()).collect();
        Ok(par_map(&idx, |&i| self.search(&queries[i], precursor_mz.map(|p| p[i]), k)))
    }

    /// Dense, symmetric `len × len` matrix (row-major) of the similarity
    /// between all library spectra, ignoring precursors and thresholds.
    /// Entry `(i, j)` with `i ≤ j` is the score of spectrum `i` as the query
    /// against spectrum `j`; the lower triangle is mirrored. The diagonal is
    /// `1` for every non-empty spectrum.
    pub fn similarity_matrix(&self) -> Vec<E> {
        let n = self.len();
        let rows: Vec<usize> = (0..n).collect();
        let upper: Vec<Vec<E>> = par_map(&rows, |&i| {
            let (am, aw) = self.spectrum(i);
            (i..n)
                .map(|j| {
                    let (bm, bw) = self.spectrum(j);
                    score_pair(&self.opts, am, aw, bm, bw).0
                })
                .collect()
        });
        let mut out = vec![E::zero(); n * n];
        for (i, row) in upper.iter().enumerate() {
            for (d, &s) in row.iter().enumerate() {
                out[i * n + i + d] = s;
                out[(i + d) * n + i] = s;
            }
        }
        out
    }
}

fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync + Send) -> Vec<R> {
    #[cfg(feature = "parallel")]
    {
        items.par_iter().map(f).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        items.iter().map(f).collect()
    }
}

/// Cleaned, m/z-sorted peaks and their metric-specific weights.
fn prepare<E: SimdFloat>(spectrum: &Spectrum<E>, opts: &LibrarySearchOptions<E>) -> (Vec<E>, Vec<E>) {
    let cleaned = preprocess(spectrum, &opts.cleaning());
    let mz = cleaned.mzs();
    let mut w = cleaned.intensities();
    if let SpectralMetric::Cosine { mz_power, intensity_power } = opts.metric {
        if intensity_power != E::one() {
            simd::pow_simd(&mut w, intensity_power);
        }
        if mz_power != E::zero() {
            for (x, &m) in w.iter_mut().zip(&mz) {
                *x = *x * m.powf(mz_power);
            }
        }
        if !simd::normalize_l2_simd(&mut w) {
            return (Vec::new(), Vec::new());
        }
    }
    (mz, w)
}

/// Linear merge of two cleaned spectra: `(score, matched_peaks)`.
fn score_pair<E: SimdFloat>(opts: &LibrarySearchOptions<E>, qm: &[E], qw: &[E], lm: &[E], lw: &[E]) -> (E, usize) {
    let tol = FlashEntropyOptions { tolerance: opts.fragment_tolerance, ..FlashEntropyOptions::absolute(E::zero()) };
    let cosine = matches!(opts.metric, SpectralMetric::Cosine { .. });
    let xlnx = |x: E| x * x.ln();
    let (mut i, mut j, mut n) = (0, 0, 0);
    let mut acc = E::zero();
    while i < qm.len() && j < lm.len() {
        let (a, b) = (qm[i], lm[j]);
        let hw = tol.half_window(a);
        if b > a + hw {
            i += 1;
        } else if b < a - hw {
            j += 1;
        } else {
            let (x, y) = (qw[i], lw[j]);
            acc = acc + if cosine { x * y } else { xlnx(x + y) - xlnx(x) - xlnx(y) };
            n += 1;
            i += 1;
            j += 1;
        }
    }
    if !cosine {
        acc = acc / E::from_f64_(4.0f64.ln());
    }
    (acc.max(E::zero()).min(E::one()), n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_entropy::FlashEntropyIndex;
    use rand::Rng;

    fn random_spectrum(rng: &mut impl Rng, base: &[f64]) -> Spectrum<f64> {
        let (mut mz, mut it) = (Vec::new(), Vec::new());
        for &m in base {
            if rng.random_bool(0.6) {
                mz.push(m + rng.random_range(-0.004..0.004));
                it.push(rng.random_range(0.1..100.0));
            }
        }
        for _ in 0..rng.random_range(0..8) {
            mz.push(rng.random_range(50.0..500.0));
            it.push(rng.random_range(0.1..100.0));
        }
        Spectrum::from_arrays(&mz, &it).unwrap()
    }

    fn setup() -> (Vec<Spectrum<f64>>, Vec<f64>, Vec<f64>) {
        let mut rng = rand::rng();
        let base: Vec<f64> = (0..40).map(|_| rng.random_range(50.0..500.0)).collect();
        let lib: Vec<Spectrum<f64>> = (0..150).map(|_| random_spectrum(&mut rng, &base)).collect();
        let prec: Vec<f64> = (0..150).map(|_| rng.random_range(100.0..110.0)).collect();
        (lib, prec, base)
    }

    #[test]
    fn entropy_matches_flash_index() {
        let (lib, _, base) = setup();
        let mut rng = rand::rng();
        for tol in [MzTolerance::Absolute(0.01), MzTolerance::Ppm(20.0)] {
            for weighted in [true, false] {
                let mut opts = LibrarySearchOptions::entropy(tol);
                if !weighted {
                    opts.metric = SpectralMetric::UnweightedEntropy;
                }
                let sl = SpectralLibrary::new(&lib, opts);
                let flash = FlashEntropyIndex::new(&lib, FlashEntropyOptions { weighted, ..opts.cleaning() });
                for _ in 0..10 {
                    let q = random_spectrum(&mut rng, &base);
                    let expected = flash.search(&q);
                    for (i, (s, _)) in sl.score_all(&q).into_iter().enumerate() {
                        assert!((s - expected[i]).abs() < 1e-10, "{tol:?} {weighted} lib {i}: {s} vs {}", expected[i]);
                    }
                }
            }
        }
    }

    #[test]
    fn cosine_matches_greedy_on_cleaned_spectra() {
        let (lib, _, base) = setup();
        let mut rng = rand::rng();
        let tol = MzTolerance::Absolute(0.01);
        for (mp, ip) in [(0.0, 1.0), (0.0, 0.5), (1.0, 0.5)] {
            let opts = LibrarySearchOptions {
                metric: SpectralMetric::Cosine { mz_power: mp, intensity_power: ip },
                ..LibrarySearchOptions::cosine(tol)
            };
            let clean: Vec<Spectrum<f64>> = lib.iter().map(|s| preprocess(s, &opts.cleaning())).collect();
            let sl = SpectralLibrary::new(&lib, opts);
            for _ in 0..10 {
                let q = random_spectrum(&mut rng, &base);
                let qc = preprocess(&q, &opts.cleaning());
                for (i, (s, n)) in sl.score_all(&q).into_iter().enumerate() {
                    let (e, en) = crate::spectral_cosine_similarity(&qc, &clean[i], tol, mp, ip);
                    assert!((s - e).abs() < 1e-10, "lib {i}: {s} vs {e}");
                    assert_eq!(n, en);
                }
            }
        }
    }

    #[test]
    fn precursor_filter_and_thresholds() {
        let (lib, prec, base) = setup();
        let mut rng = rand::rng();
        let tol = MzTolerance::Absolute(0.5);
        let opts = LibrarySearchOptions::entropy(MzTolerance::Absolute(0.01))
            .with_precursor_tolerance(tol)
            .with_min_matched_peaks(3)
            .with_min_score(0.1);
        let sl = SpectralLibrary::with_precursors(&lib, &prec, opts).unwrap();
        let queries: Vec<Spectrum<f64>> = (0..12).map(|_| random_spectrum(&mut rng, &base)).collect();
        let qprec: Vec<f64> = (0..12).map(|_| rng.random_range(100.0..110.0)).collect();
        let many = sl.search_many(&queries, Some(&qprec), 0).unwrap();
        for (qi, q) in queries.iter().enumerate() {
            let all = sl.score_all(q);
            let mut expected: Vec<usize> = (0..lib.len())
                .filter(|&i| (prec[i] - qprec[qi]).abs() <= 0.5 && all[i].1 >= 3 && all[i].0 >= 0.1)
                .collect();
            let mut got: Vec<usize> = many[qi].iter().map(|h| h.index).collect();
            assert!(many[qi].windows(2).all(|w| w[0].score >= w[1].score));
            got.sort_unstable();
            expected.sort_unstable();
            assert_eq!(got, expected);
            let top2 = sl.search(q, Some(qprec[qi]), 2);
            assert_eq!(top2.as_slice(), &many[qi][..many[qi].len().min(2)]);
        }
        // no query precursor => no filter
        assert!(sl.search(&lib[0], None, 0).len() >= sl.search(&lib[0], Some(prec[0]), 0).len());
        assert!(SpectralLibrary::with_precursors(&lib, &prec[1..], opts).is_err());
        assert!(sl.search_many(&queries, Some(&qprec[1..]), 1).is_err());
    }

    #[test]
    fn similarity_matrix_is_symmetric() {
        let (lib, _, _) = setup();
        let sl = SpectralLibrary::new(&lib[..30], LibrarySearchOptions::cosine(MzTolerance::Ppm(20.0)));
        let m = sl.similarity_matrix();
        for i in 0..30 {
            assert!((m[i * 30 + i] - 1.0).abs() < 1e-12 || lib[i].is_empty());
            for j in 0..30 {
                assert_eq!(m[i * 30 + j], m[j * 30 + i]);
            }
            // With a ppm tolerance the window is centred on the query peak, so
            // score(i, j) and score(j, i) can differ at the window edge; the
            // matrix holds score(i, j) for i <= j.
            let row = sl.score_all(&lib[i]);
            for j in i..30 {
                assert!((row[j].0 - m[i * 30 + j]).abs() < 1e-12);
            }
        }
        let empty = SpectralLibrary::<f32>::new(&[], LibrarySearchOptions::entropy(MzTolerance::Absolute(0.01)));
        assert!(empty.is_empty() && empty.similarity_matrix().is_empty());
    }
}
