//! # Flash entropy search
//!
//! Indexed spectral library search with the spectral entropy similarity
//! (Li & Fiehn, *Nature Methods* 20, 1530–1536, 2023).
//!
//! ## Why it is fast
//!
//! For two cleaned spectra whose peak intensities each sum to one, the
//! entropy similarity reduces to a sum over **matched peaks only**:
//!
//! ```text
//! S(A, B) = Σ_matched [ (a+b)·ln(a+b) − a·ln a − b·ln b ] / ln 4
//! ```
//!
//! Unmatched peaks contribute nothing, so a query only has to visit library
//! peaks whose m/z lies within the tolerance window of one of its own
//! peaks. All library peaks are stored in one array sorted by m/z; each
//! query peak locates its window by binary search, the per-peak gains are
//! evaluated with the SIMD `ln` kernel and scattered into the score of the
//! spectrum each peak belongs to.
//!
//! ## Exactness
//!
//! Spectra are cleaned so that peaks inside one spectrum are more than
//! `2 × tolerance` apart (closer peaks are merged). Then every query peak
//! matches at most one peak per library spectrum and vice versa, and the
//! scores equal [`crate::calculate_entropy_similarity_matched`] on the
//! cleaned spectra exactly (up to rounding).
//!
//! ```
//! use similarity::flash_entropy::{FlashEntropyIndex, FlashEntropyOptions};
//! use similarity::Spectrum;
//!
//! let library = vec![
//!     Spectrum::from_arrays(&[100.0, 150.0, 200.0], &[1.0, 3.0, 5.0]).unwrap(),
//!     Spectrum::from_arrays(&[120.0, 180.0], &[2.0, 2.0]).unwrap(),
//! ];
//! let index = FlashEntropyIndex::new(&library, FlashEntropyOptions::absolute(0.02));
//! let query = Spectrum::from_arrays(&[100.005, 150.0, 200.01], &[1.0, 3.0, 5.0]).unwrap();
//! let hits = index.search_top_k(&query, 1);
//! assert_eq!(hits[0].index, 0);
//! assert!((hits[0].score - 1.0_f64).abs() < 1e-10);
//! ```

use crate::batch::Neighbor;
use crate::simd::{self, SimdFloat};
use crate::{MzTolerance, Peak, Spectrum};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Preprocessing and matching options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlashEntropyOptions<E> {
    /// m/z tolerance used for matching (and, doubled, for peak merging).
    pub tolerance: MzTolerance<E>,
    /// Apply the entropy-based intensity weighting (the standard "entropy
    /// similarity"). `false` gives the unweighted variant.
    pub weighted: bool,
    /// Drop peaks below this fraction of the base peak (`0` keeps all).
    pub noise_fraction: E,
    /// Keep only the most intense `max_peaks` peaks after merging.
    pub max_peaks: Option<usize>,
}

impl<E: SimdFloat> FlashEntropyOptions<E> {
    /// Weighted entropy similarity with an absolute tolerance in Da and
    /// no noise filter.
    pub fn absolute(tolerance_da: E) -> Self {
        Self {
            tolerance: MzTolerance::Absolute(tolerance_da),
            weighted: true,
            noise_fraction: E::zero(),
            max_peaks: None,
        }
    }

    /// Weighted entropy similarity with a ppm tolerance.
    pub fn ppm(ppm: E) -> Self {
        Self {
            tolerance: MzTolerance::Ppm(ppm),
            ..Self::absolute(E::zero())
        }
    }

    #[inline]
    fn half_window(&self, mz: E) -> E {
        match self.tolerance {
            MzTolerance::Absolute(t) => t,
            MzTolerance::Ppm(p) => mz.abs() * p * E::from_f64_(1e-6),
        }
    }
}

/// Cleans and normalises one spectrum: drops non-positive/non-finite peaks,
/// applies the noise filter, merges peaks closer than `2 × tolerance`
/// (intensity-weighted m/z), keeps the top `max_peaks`, normalises to unit
/// total intensity and optionally applies the entropy weighting.
///
/// Returned peaks are sorted by m/z. An empty result means the spectrum
/// cannot match anything.
pub fn preprocess<E: SimdFloat>(spectrum: &Spectrum<E>, opts: &FlashEntropyOptions<E>) -> Spectrum<E> {
    let mut peaks: Vec<Peak<E>> = spectrum
        .peaks
        .iter()
        .copied()
        .filter(|p| p.intensity > E::zero() && p.intensity.is_finite() && p.mz.is_finite())
        .collect();
    if peaks.is_empty() {
        return Spectrum::new();
    }
    if opts.noise_fraction > E::zero() {
        let max = peaks.iter().fold(E::zero(), |m, p| m.max(p.intensity));
        let thr = max * opts.noise_fraction;
        peaks.retain(|p| p.intensity >= thr);
    }
    peaks.sort_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap_or(core::cmp::Ordering::Equal));

    // Merge chains of peaks with gaps <= 2·tolerance.
    let two = E::from_f64_(2.0);
    let mut merged: Vec<Peak<E>> = Vec::with_capacity(peaks.len());
    let (mut wsum, mut isum, mut last_mz) = (E::zero(), E::zero(), E::zero());
    for (k, p) in peaks.iter().enumerate() {
        if k > 0 && p.mz - last_mz > two * opts.half_window(p.mz) {
            merged.push(Peak { mz: wsum / isum, intensity: isum });
            wsum = E::zero();
            isum = E::zero();
        }
        wsum = wsum + p.mz * p.intensity;
        isum = isum + p.intensity;
        last_mz = p.mz;
    }
    merged.push(Peak { mz: wsum / isum, intensity: isum });

    let mut out = Spectrum::from_peaks(merged);
    if let Some(n) = opts.max_peaks {
        out.keep_top_n(n);
    }
    let ints = out.intensities();
    let weights = if opts.weighted {
        match simd::spectral_entropy_weights_simd(&ints) {
            Some((_, w)) => w,
            None => return Spectrum::new(),
        }
    } else {
        let mut w = ints;
        if !simd::normalize_sum_simd(&mut w) {
            return Spectrum::new();
        }
        w
    };
    for (p, w) in out.peaks.iter_mut().zip(weights) {
        p.intensity = w;
    }
    out
}

/// An m/z-sorted index over all peaks of a spectral library.
#[derive(Debug, Clone)]
pub struct FlashEntropyIndex<E: SimdFloat> {
    opts: FlashEntropyOptions<E>,
    mz: Vec<E>,
    intensity: Vec<E>,
    /// intensity · ln(intensity), precomputed
    ilni: Vec<E>,
    spectrum_of: Vec<u32>,
    n_spectra: usize,
}

impl<E: SimdFloat> FlashEntropyIndex<E> {
    /// Builds the index (preprocessing runs in parallel with the `parallel`
    /// feature).
    ///
    /// # Panics
    /// If the library has more than `u32::MAX` spectra.
    pub fn new(library: &[Spectrum<E>], opts: FlashEntropyOptions<E>) -> Self {
        assert!(library.len() <= u32::MAX as usize, "library too large");
        #[cfg(feature = "parallel")]
        let cleaned: Vec<Spectrum<E>> = library.par_iter().map(|s| preprocess(s, &opts)).collect();
        #[cfg(not(feature = "parallel"))]
        let cleaned: Vec<Spectrum<E>> = library.iter().map(|s| preprocess(s, &opts)).collect();

        let mut all: Vec<(E, E, u32)> = cleaned
            .iter()
            .enumerate()
            .flat_map(|(i, s)| s.peaks.iter().map(move |p| (p.mz, p.intensity, i as u32)))
            .collect();
        #[cfg(feature = "parallel")]
        all.par_sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
        #[cfg(not(feature = "parallel"))]
        all.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));

        let mz = all.iter().map(|p| p.0).collect();
        let intensity: Vec<E> = all.iter().map(|p| p.1).collect();
        let spectrum_of = all.iter().map(|p| p.2).collect();
        let mut ilni = intensity.clone();
        simd::ln_simd(&mut ilni);
        let ilni = simd::mul_simd(&ilni, &intensity).unwrap_or_default();
        Self {
            opts,
            mz,
            intensity,
            ilni,
            spectrum_of,
            n_spectra: library.len(),
        }
    }

    /// Number of library spectra.
    pub fn len(&self) -> usize {
        self.n_spectra
    }

    /// `true` if the library is empty.
    pub fn is_empty(&self) -> bool {
        self.n_spectra == 0
    }

    /// Total number of indexed (cleaned) peaks.
    pub fn n_peaks(&self) -> usize {
        self.mz.len()
    }

    /// Options the index was built with.
    pub fn options(&self) -> &FlashEntropyOptions<E> {
        &self.opts
    }

    /// Entropy similarity of `query` to every library spectrum, in library
    /// order, each in `[0, 1]`.
    pub fn search(&self, query: &Spectrum<E>) -> Vec<E> {
        let q = preprocess(query, &self.opts);
        let mut scores = vec![E::zero(); self.n_spectra];
        let mut buf: Vec<E> = Vec::new();
        for p in &q.peaks {
            let w = self.opts.half_window(p.mz);
            let lo = self.mz.partition_point(|&m| m < p.mz - w);
            let hi = self.mz.partition_point(|&m| m <= p.mz + w);
            if lo >= hi {
                continue;
            }
            let n = hi - lo;
            buf.clear();
            buf.resize(n, E::zero());
            simd::entropy_gain_simd(p.intensity, &self.intensity[lo..hi], &self.ilni[lo..hi], &mut buf);
            for (g, &s) in buf.iter().zip(&self.spectrum_of[lo..hi]) {
                scores[s as usize] = scores[s as usize] + *g;
            }
        }
        let inv_ln4 = E::one() / E::from_f64_(4.0f64.ln());
        simd::scale_simd(&mut scores, inv_ln4);
        simd::clamp_simd(&mut scores, E::zero(), E::one());
        scores
    }

    /// The `k` best-scoring library spectra (score > 0), best first.
    pub fn search_top_k(&self, query: &Spectrum<E>, k: usize) -> Vec<Neighbor<E>> {
        top_k_scores(self.search(query), k)
    }

    /// [`search_top_k`](Self::search_top_k) for many queries (parallel with
    /// the `parallel` feature).
    pub fn search_many(&self, queries: &[Spectrum<E>], k: usize) -> Vec<Vec<Neighbor<E>>> {
        #[cfg(feature = "parallel")]
        {
            queries.par_iter().map(|q| self.search_top_k(q, k)).collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            queries.iter().map(|q| self.search_top_k(q, k)).collect()
        }
    }
}

fn top_k_scores<E: SimdFloat>(scores: Vec<E>, k: usize) -> Vec<Neighbor<E>> {
    let mut hits: Vec<Neighbor<E>> = scores
        .into_iter()
        .enumerate()
        .filter(|(_, s)| *s > E::zero())
        .map(|(index, score)| Neighbor { index, score })
        .collect();
    let cmp = |a: &Neighbor<E>, b: &Neighbor<E>| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(a.index.cmp(&b.index))
    };
    if k == 0 {
        return Vec::new();
    }
    if k < hits.len() {
        hits.select_nth_unstable_by(k, cmp);
        hits.truncate(k);
    }
    hits.sort_by(cmp);
    hits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calculate_entropy_similarity_matched;
    use rand::Rng;

    fn random_spectrum(rng: &mut impl Rng, base: &[f64]) -> Spectrum<f64> {
        // reuse some base m/z values (with jitter) plus random extras
        let mut mz = Vec::new();
        let mut it = Vec::new();
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

    #[test]
    fn matches_bruteforce_matched_entropy() {
        let mut rng = rand::rng();
        let base: Vec<f64> = (0..40).map(|_| rng.random_range(50.0..500.0)).collect();
        let library: Vec<Spectrum<f64>> = (0..200).map(|_| random_spectrum(&mut rng, &base)).collect();
        for opts in [FlashEntropyOptions::absolute(0.01), FlashEntropyOptions { weighted: false, ..FlashEntropyOptions::absolute(0.01) }] {
            let index = FlashEntropyIndex::new(&library, opts);
            for _ in 0..20 {
                let query = random_spectrum(&mut rng, &base);
                let scores = index.search(&query);
                let q = preprocess(&query, &opts);
                for (i, lib) in library.iter().enumerate() {
                    let l = preprocess(lib, &opts);
                    // preprocess() already applied weighting/normalisation, so the
                    // reference is the plain matched-peak entropy formula.
                    let expected = if q.is_empty() || l.is_empty() {
                        0.0
                    } else {
                        brute_force_unweighted(&q, &l, opts.tolerance)
                    };
                    assert!((scores[i] - expected).abs() < 1e-10, "lib {i}: {} vs {expected}", scores[i]);
                }
            }
        }
    }

    /// Entropy similarity of two unit-sum, cleaned spectra with greedy matching
    /// and no further weighting.
    fn brute_force_unweighted(a: &Spectrum<f64>, b: &Spectrum<f64>, tol: MzTolerance<f64>) -> f64 {
        let pairs = crate::match_peaks(a, b, tol);
        let f = |x: f64| if x > 0.0 { x * x.ln() } else { 0.0 };
        let s: f64 = pairs
            .iter()
            .map(|&(i, j)| {
                let (x, y) = (a.peaks[i].intensity, b.peaks[j].intensity);
                f(x + y) - f(x) - f(y)
            })
            .sum();
        (s / 4f64.ln()).clamp(0.0, 1.0)
    }

    #[test]
    fn agrees_with_matched_entropy_similarity() {
        // On already-cleaned spectra the weighted flash score equals the
        // public matched entropy similarity.
        let mut rng = rand::rng();
        let base: Vec<f64> = (0..30).map(|_| rng.random_range(50.0..500.0)).collect();
        let opts = FlashEntropyOptions { weighted: false, ..FlashEntropyOptions::absolute(0.01) };
        let library: Vec<Spectrum<f64>> = (0..50).map(|_| preprocess(&random_spectrum(&mut rng, &base), &opts)).collect();
        let weighted = FlashEntropyOptions::absolute(0.01);
        let index = FlashEntropyIndex::new(&library, weighted);
        for _ in 0..10 {
            let query = preprocess(&random_spectrum(&mut rng, &base), &opts);
            let scores = index.search(&query);
            for (i, lib) in library.iter().enumerate() {
                let expected = calculate_entropy_similarity_matched(&query, lib, weighted.tolerance);
                assert!((scores[i] - expected).abs() < 1e-9, "lib {i}: {} vs {expected}", scores[i]);
            }
        }
    }

    #[test]
    fn identity_and_top_k() {
        let lib = vec![
            Spectrum::from_arrays(&[100.0_f64, 200.0, 300.0], &[1.0, 2.0, 3.0]).unwrap(),
            Spectrum::from_arrays(&[100.0, 250.0], &[5.0, 1.0]).unwrap(),
            Spectrum::from_arrays(&[400.0], &[1.0]).unwrap(),
            Spectrum::new(),
        ];
        let idx = FlashEntropyIndex::new(&lib, FlashEntropyOptions::ppm(20.0));
        assert_eq!(idx.len(), 4);
        assert_eq!(idx.n_peaks(), 6);
        let s = idx.search(&lib[0]);
        assert!((s[0] - 1.0).abs() < 1e-12);
        assert!(s[1] > 0.0 && s[1] < 1.0);
        assert_eq!(s[2], 0.0);
        assert_eq!(s[3], 0.0);
        let hits = idx.search_top_k(&lib[0], 10);
        assert_eq!(hits.iter().map(|h| h.index).collect::<Vec<_>>(), vec![0, 1]);
        let many = idx.search_many(&lib, 1);
        assert_eq!(many[2][0].index, 2);
        assert!(many[3].is_empty());
        assert!(idx.search(&Spectrum::new()).iter().all(|&x| x == 0.0));
    }

    #[test]
    fn preprocessing_merges_close_peaks() {
        let s = Spectrum::from_arrays(&[100.0_f64, 100.01, 100.03, 200.0, 0.0], &[1.0, 1.0, 2.0, 4.0, -1.0]).unwrap();
        let opts = FlashEntropyOptions { weighted: false, ..FlashEntropyOptions::absolute(0.01) };
        let p = preprocess(&s, &opts);
        assert_eq!(p.len(), 2); // 100.00/100.01/100.03 chain-merged (gaps 0.01, 0.02 <= 0.02)
        assert!((p.peaks[0].intensity - 0.5).abs() < 1e-12);
        assert!((p.peaks[0].mz - (100.0 + 100.01 + 2.0 * 100.03) / 4.0).abs() < 1e-9);
        let f32_lib = vec![Spectrum::from_arrays(&[100.0f32, 200.0], &[1.0, 1.0]).unwrap()];
        let idx = FlashEntropyIndex::new(&f32_lib, FlashEntropyOptions::absolute(0.01f32));
        assert!((idx.search(&f32_lib[0])[0] - 1.0).abs() < 1e-5);
    }
}
