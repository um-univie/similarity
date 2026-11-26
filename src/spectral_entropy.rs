use num_traits::{Float, FromPrimitive, ToPrimitive};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

const WEIGHT_START: f64 = 0.25;
const ENTROPY_CUTOFF: f64 = 3.0;
const WEIGHT_SLOPE: f64 = (1.0 - WEIGHT_START) / ENTROPY_CUTOFF;
const BLOCK_SIZE: usize = 64; // Cache-friendly block size

/// Represents a spectral peak with mass-to-charge ratio and intensity
#[derive(Debug, Clone, Copy)]
pub struct Peak<T: Float> {
    pub mz: T,
    pub intensity: T,
}

/// Represents a spectrum as a collection of peaks
#[derive(Debug, Clone)]
pub struct Spectrum<T: Float> {
    pub peaks: Vec<Peak<T>>,
}

impl<T: Float + std::iter::Sum<T> + FromPrimitive + ToPrimitive> Default for Spectrum<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + std::iter::Sum<T> + FromPrimitive + ToPrimitive> Spectrum<T> {
    /// Creates a new empty spectrum
    pub fn new() -> Self {
        Self { peaks: Vec::new() }
    }

    /// Creates a spectrum from a vector of peaks
    pub fn from_peaks(peaks: Vec<Peak<T>>) -> Self {
        Self { peaks }
    }

    /// Creates a spectrum from separate m/z and intensity vectors
    pub fn from_arrays(mzs: &[T], intensities: &[T]) -> Option<Self> {
        if mzs.len() != intensities.len() {
            return None;
        }

        let peaks = mzs.iter()
            .zip(intensities.iter())
            .map(|(&mz, &intensity)| Peak { mz, intensity })
            .collect();

        Some(Self { peaks })
    }

    /// Number of peaks.
    pub fn len(&self) -> usize {
        self.peaks.len()
    }

    /// `true` if the spectrum has no peaks.
    pub fn is_empty(&self) -> bool {
        self.peaks.is_empty()
    }

    /// m/z values of all peaks.
    pub fn mzs(&self) -> Vec<T> {
        self.peaks.iter().map(|p| p.mz).collect()
    }

    /// Intensities of all peaks.
    pub fn intensities(&self) -> Vec<T> {
        self.peaks.iter().map(|p| p.intensity).collect()
    }

    /// Sorts peaks by ascending m/z (NaN m/z values go last).
    pub fn sort_by_mz(&mut self) {
        self.peaks.sort_by(|a, b| {
            a.mz.partial_cmp(&b.mz)
                .unwrap_or_else(|| a.mz.is_nan().cmp(&b.mz.is_nan()))
        });
    }

    /// Sum of all intensities.
    pub fn total_intensity(&self) -> T {
        self.peaks.iter().map(|p| p.intensity).sum()
    }

    /// The most intense peak, if any.
    pub fn base_peak(&self) -> Option<Peak<T>> {
        self.peaks
            .iter()
            .copied()
            .fold(None, |best: Option<Peak<T>>, p| match best {
                Some(b) if b.intensity >= p.intensity => Some(b),
                _ => Some(p),
            })
    }

    /// Scales intensities so the base peak has intensity 1.
    pub fn normalize_to_max(&mut self) {
        if let Some(bp) = self.base_peak() {
            if bp.intensity > T::zero() {
                for p in &mut self.peaks {
                    p.intensity = p.intensity / bp.intensity;
                }
            }
        }
    }

    /// Keeps only the `n` most intense peaks (order by m/z is preserved).
    pub fn keep_top_n(&mut self, n: usize) {
        if self.peaks.len() <= n {
            return;
        }
        let mut idx: Vec<usize> = (0..self.peaks.len()).collect();
        idx.sort_by(|&i, &j| {
            self.peaks[j]
                .intensity
                .partial_cmp(&self.peaks[i].intensity)
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        let mut keep = vec![false; self.peaks.len()];
        for &i in &idx[..n] {
            keep[i] = true;
        }
        let mut k = keep.into_iter();
        self.peaks.retain(|_| k.next().unwrap_or(false));
    }

    /// Keeps only peaks with `lo <= mz <= hi`.
    pub fn filter_mz_range(&mut self, lo: T, hi: T) {
        self.peaks.retain(|p| p.mz >= lo && p.mz <= hi);
    }

    /// Removes peaks whose intensity is below `fraction` of the base peak.
    pub fn remove_relative_noise(&mut self, fraction: T) {
        if let Some(bp) = self.base_peak() {
            let threshold = bp.intensity * fraction;
            self.peaks.retain(|p| p.intensity >= threshold);
        }
    }

    /// Normalizes the spectrum by dividing all intensities by the sum
    pub fn normalize(&mut self) {
        let sum: T = self.peaks.iter().map(|p| p.intensity).sum();
        if sum != T::zero() {
            for peak in &mut self.peaks {
                peak.intensity = peak.intensity / sum;
            }
        }
    }

    /// Removes peaks with intensity below the threshold
    pub fn remove_noise(&mut self, threshold: T) {
        self.peaks.retain(|peak| peak.intensity >= threshold);
    }

    /// Centers peaks within the given tolerance
    pub fn centroid(&mut self, tolerance: T) {
        if self.peaks.is_empty() {
            return;
        }

        // Sort peaks by m/z
        self.sort_by_mz();

        let mut new_peaks = Vec::new();
        let mut current_peak = self.peaks[0];
        let mut current_sum = current_peak.intensity;
        let mut weighted_sum = current_peak.mz * current_peak.intensity;

        for peak in self.peaks.iter().skip(1) {
            if (peak.mz - current_peak.mz).abs() <= tolerance {
                current_sum = current_sum + peak.intensity;
                weighted_sum = weighted_sum + peak.mz * peak.intensity;
            } else {
                if current_sum != T::zero() {
                    new_peaks.push(Peak {
                        mz: weighted_sum / current_sum,
                        intensity: current_sum,
                    });
                }
                current_peak = *peak;
                current_sum = peak.intensity;
                weighted_sum = peak.mz * peak.intensity;
            }
        }

        // Add the last peak
        if current_sum != T::zero() {
            new_peaks.push(Peak {
                mz: weighted_sum / current_sum,
                intensity: current_sum,
            });
        }

        self.peaks = new_peaks;
    }

    /// Applies weight factor transformation to the spectrum
    pub fn apply_weight_factor(&mut self, wf_mz: T, wf_int: T) {
        let transformed_intensities: Vec<T> = self.peaks.iter()
            .map(|peak| peak.mz.powf(wf_mz) * peak.intensity.powf(wf_int))
            .collect();
        
        for (peak, new_intensity) in self.peaks.iter_mut().zip(transformed_intensities) {
            peak.intensity = new_intensity;
        }
    }

    /// Gets the entropy and weighted intensity of the spectrum
    pub fn get_entropy_and_weighted_intensity(&self) -> (T, Vec<T>) {
        let entropy = calculate_entropy(self);
        let sum: T = self.peaks.iter().map(|p| p.intensity).sum();
        
        if sum == T::zero() {
            return (entropy, self.peaks.iter().map(|p| p.intensity).collect());
        }

        let entropy_f64: f64 = entropy.to_f64().unwrap();
        if entropy_f64 < ENTROPY_CUTOFF {
            let weight = T::from_f64(WEIGHT_START + WEIGHT_SLOPE * entropy_f64).unwrap();
            let weighted_intensities: Vec<T> = self.peaks.iter()
                .map(|p| p.intensity.powf(weight))
                .collect();
            
            let weighted_sum: T = weighted_intensities.iter().copied().sum();
            let normalized_intensities: Vec<T> = weighted_intensities.iter()
                .map(|&i| i / weighted_sum)
                .collect();
            
            (calculate_entropy_from_intensities(&normalized_intensities), normalized_intensities)
        } else {
            (entropy, self.peaks.iter().map(|p| p.intensity).collect())
        }
    }
}

/// Calculates entropy from a vector of intensities, matching scipy.stats.entropy behavior
fn calculate_entropy_from_intensities<T: Float + std::iter::Sum<T>>(intensities: &[T]) -> T {
    let mut entropy = T::zero();
    let sum: T = intensities.iter().copied().sum();

    if sum != T::zero() {
        for &intensity in intensities {
            let p = intensity / sum;
            if p != T::zero() {
                entropy = entropy - p * p.ln();
            }
        }
    }

    entropy
}

/// Calculates relative entropy (Kullback-Leibler divergence) between two distributions
pub fn calculate_relative_entropy<T: Float + std::iter::Sum<T>>(p: &[T], q: &[T]) -> T {
    let mut entropy = T::zero();
    let p_sum: T = p.iter().copied().sum();
    let q_sum: T = q.iter().copied().sum();

    if p_sum != T::zero() && q_sum != T::zero() {
        for (&pi, &qi) in p.iter().zip(q.iter()) {
            let p_norm = pi / p_sum;
            let q_norm = qi / q_sum;
            if p_norm != T::zero() && q_norm != T::zero() {
                entropy = entropy + p_norm * (p_norm / q_norm).ln();
            }
        }
    }

    entropy
}

/// Calculates the Shannon entropy of a spectrum
pub fn calculate_entropy<T: Float + std::iter::Sum<T>>(spectrum: &Spectrum<T>) -> T {
    let intensities: Vec<T> = spectrum.peaks.iter().map(|p| p.intensity).collect();
    calculate_entropy_from_intensities(&intensities)
}

/// Calculates the Tsallis entropy of a spectrum
pub fn calculate_tsallis_entropy<T: Float + std::iter::Sum<T>>(spectrum: &Spectrum<T>, q: T) -> T {
    if q == T::one() {
        return calculate_entropy(spectrum);
    }

    let mut entropy = T::zero();
    let sum: T = spectrum.peaks.iter().map(|p| p.intensity).sum();

    if sum != T::zero() {
        for peak in &spectrum.peaks {
            let p = peak.intensity / sum;
            if p != T::zero() {
                entropy = entropy + p.powf(q);
            }
        }
        entropy = (entropy - T::one()) / (T::one() - q);
    }

    entropy
}

/// Calculates the entropy similarity between two spectra, pairing peaks
/// **by index** (peak `i` of one spectrum with peak `i` of the other).
///
/// This is only meaningful if both spectra share the same m/z grid. Use
/// [`calculate_entropy_similarity_matched`] (m/z-tolerant matching) or
/// [`crate::flash_entropy::FlashEntropyIndex`] for real spectra.
#[deprecated(
    since = "0.3.0",
    note = "pairs peaks by index; use `calculate_entropy_similarity_matched` or `FlashEntropyIndex`"
)]
pub fn calculate_entropy_similarity<T: Float + std::iter::Sum<T> + FromPrimitive + ToPrimitive>(
    spectrum1: &Spectrum<T>,
    spectrum2: &Spectrum<T>
) -> T {
    let (entropy1, weighted1) = spectrum1.get_entropy_and_weighted_intensity();
    let (entropy2, weighted2) = spectrum2.get_entropy_and_weighted_intensity();
    
    // Merge the weighted intensities
    let merged: Vec<T> = weighted1.iter()
        .zip(weighted2.iter())
        .map(|(&a, &b)| a + b)
        .collect();
    
    let entropy_merged = calculate_entropy_from_intensities(&merged);
    
    T::one() - (T::from_f64(2.0).unwrap() * entropy_merged - entropy1 - entropy2) / T::from_f64(4.0).unwrap().ln()
}

/// Applies weight factor transformation to m/z and intensity arrays
pub fn weight_factor_transformation<T: Float>(mzs: &[T], ints: &[T], wf_mz: T, wf_int: T) -> Vec<T> {
    mzs.iter()
        .zip(ints.iter())
        .map(|(&mz, &intensity)| mz.powf(wf_mz) * intensity.powf(wf_int))
        .collect()
}



/// Optimized Tsallis entropy calculation
#[cfg(feature = "parallel")]
pub fn calculate_tsallis_entropy_optimized<T: Float + std::iter::Sum<T> + Send + Sync>(
    spectrum: &Spectrum<T>,
    q: T
) -> T {
    if q == T::one() {
        return calculate_entropy(spectrum);
    }

    let sum: T = spectrum.peaks.iter().map(|p| p.intensity).sum();

    if sum != T::zero() {
        let entropy: T = spectrum.peaks.par_chunks(BLOCK_SIZE)
            .map(|chunk| {
                let mut block_entropy = T::zero();
                for peak in chunk {
                    let p = peak.intensity / sum;
                    if p != T::zero() {
                        block_entropy = block_entropy + p.powf(q);
                    }
                }
                block_entropy
            })
            .sum();
        
        (entropy - T::one()) / (T::one() - q)
    } else {
        T::zero()
    }
}


// ============================================================================
// m/z-tolerant peak matching and matched spectral similarities
// ============================================================================

/// Tolerance used to decide whether two peaks have the same m/z.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MzTolerance<T> {
    /// Absolute tolerance in Da (Th).
    Absolute(T),
    /// Relative tolerance in parts per million of the larger m/z.
    Ppm(T),
}

impl<T: Float> MzTolerance<T> {
    #[inline]
    pub(crate) fn matches(self, a: T, b: T) -> bool {
        let d = (a - b).abs();
        match self {
            MzTolerance::Absolute(t) => d <= t,
            MzTolerance::Ppm(ppm) => d <= a.abs().max(b.abs()) * ppm * T::from(1e-6).unwrap(),
        }
    }

    /// Upper bound of the absolute tolerance window around `mz`.
    #[inline]
    pub(crate) fn window(self, mz: T) -> T {
        match self {
            MzTolerance::Absolute(t) => t,
            // the larger m/z of a matching pair is at most mz·(1+ε)/(1−ε); use a safe bound
            MzTolerance::Ppm(ppm) => {
                let e = ppm * T::from(1e-6).unwrap();
                mz.abs() * e / (T::one() - e).max(T::epsilon())
            }
        }
    }
}

fn sorted_indices<T: Float>(s: &Spectrum<T>) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..s.peaks.len()).collect();
    idx.sort_by(|&i, &j| {
        s.peaks[i]
            .mz
            .partial_cmp(&s.peaks[j].mz)
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    idx
}

/// Greedy one-to-one peak matching within an m/z tolerance.
///
/// All candidate pairs within tolerance are scored with `score(i, j)` and
/// assigned greedily from the highest score down, so every peak is matched
/// at most once (the "greedy" strategy used by matchms' CosineGreedy).
/// Returns pairs `(index_in_a, index_in_b)`. Inputs need not be sorted.
pub fn match_peaks_by<T: Float, F: Fn(usize, usize) -> T>(
    a: &Spectrum<T>,
    b: &Spectrum<T>,
    tolerance: MzTolerance<T>,
    score: F,
) -> Vec<(usize, usize)> {
    let ib = sorted_indices(b);
    let bmz: Vec<T> = ib.iter().map(|&j| b.peaks[j].mz).collect();
    let mut cand: Vec<(T, usize, usize)> = Vec::new();
    for (i, pa) in a.peaks.iter().enumerate() {
        let w = tolerance.window(pa.mz);
        let lo = bmz.partition_point(|&m| m < pa.mz - w);
        for k in lo..bmz.len() {
            if bmz[k] > pa.mz + w {
                break;
            }
            let j = ib[k];
            if tolerance.matches(pa.mz, b.peaks[j].mz) {
                cand.push((score(i, j), i, j));
            }
        }
    }
    cand.sort_by(|x, y| {
        y.0.partial_cmp(&x.0)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(x.1.cmp(&y.1))
            .then(x.2.cmp(&y.2))
    });
    let mut used_a = vec![false; a.peaks.len()];
    let mut used_b = vec![false; b.peaks.len()];
    let mut out = Vec::new();
    for (_, i, j) in cand {
        if !used_a[i] && !used_b[j] {
            used_a[i] = true;
            used_b[j] = true;
            out.push((i, j));
        }
    }
    out
}

/// Greedy peak matching scored by intensity product (see [`match_peaks_by`]).
///
/// ```
/// use similarity::{Spectrum, match_peaks, MzTolerance};
/// let a = Spectrum::from_arrays(&[100.0, 200.0], &[1.0, 1.0]).unwrap();
/// let b = Spectrum::from_arrays(&[100.004, 300.0], &[1.0, 1.0]).unwrap();
/// assert_eq!(match_peaks(&a, &b, MzTolerance::Absolute(0.01)), vec![(0, 0)]);
/// ```
pub fn match_peaks<T: Float>(a: &Spectrum<T>, b: &Spectrum<T>, tolerance: MzTolerance<T>) -> Vec<(usize, usize)> {
    match_peaks_by(a, b, tolerance, |i, j| a.peaks[i].intensity * b.peaks[j].intensity)
}

/// Modified spectral cosine score with greedy peak matching.
///
/// Peak weights are `mz^mz_power · intensity^intensity_power`
/// (`(0, 1)` is the plain cosine; `(0, 0.5)` square-root scaling is
/// common for library search). Returns `(score, matched_peaks)`; the score
/// is `0` if either spectrum has no positive weight.
///
/// ```
/// use similarity::{Spectrum, spectral_cosine_similarity, MzTolerance};
/// let a = Spectrum::from_arrays(&[100.0, 150.0, 200.0], &[10.0, 5.0, 1.0]).unwrap();
/// let b = Spectrum::from_arrays(&[100.002, 150.001, 250.0], &[10.0, 5.0, 1.0]).unwrap();
/// let (score, n) = spectral_cosine_similarity(&a, &b, MzTolerance::Absolute(0.01), 0.0, 1.0);
/// assert_eq!(n, 2);
/// assert!(score > 0.98 && score < 1.0);
/// ```
pub fn spectral_cosine_similarity<T: Float>(
    a: &Spectrum<T>,
    b: &Spectrum<T>,
    tolerance: MzTolerance<T>,
    mz_power: T,
    intensity_power: T,
) -> (T, usize) {
    let weight = |p: &Peak<T>| {
        let i = if intensity_power == T::one() { p.intensity } else { p.intensity.max(T::zero()).powf(intensity_power) };
        if mz_power == T::zero() { i } else { i * p.mz.powf(mz_power) }
    };
    let wa: Vec<T> = a.peaks.iter().map(weight).collect();
    let wb: Vec<T> = b.peaks.iter().map(weight).collect();
    let na = wa.iter().fold(T::zero(), |s, &x| s + x * x).sqrt();
    let nb = wb.iter().fold(T::zero(), |s, &x| s + x * x).sqrt();
    if na == T::zero() || nb == T::zero() {
        return (T::zero(), 0);
    }
    let pairs = match_peaks_by(a, b, tolerance, |i, j| wa[i] * wb[j]);
    let dot = pairs.iter().fold(T::zero(), |s, &(i, j)| s + wa[i] * wb[j]);
    ((dot / (na * nb)).min(T::one()), pairs.len())
}

/// Spectral entropy similarity with m/z-tolerant peak matching.
///
/// Unlike [`calculate_entropy_similarity`], which pairs peaks by position,
/// this version first applies the entropy-based intensity weighting to each
/// spectrum, then matches peaks within `tolerance` (greedy, by weighted
/// intensity product) and merges matched peaks. Unmatched peaks keep their
/// own intensity. Returns a score in `[0, 1]`; `0` if either spectrum has
/// zero total intensity.
///
/// Reference: Li et al., *Nature Methods* 18, 1524–1531 (2021).
///
/// ```
/// use similarity::{Spectrum, calculate_entropy_similarity_matched, MzTolerance};
/// let a = Spectrum::from_arrays(&[100.0, 200.0], &[1.0, 1.0]).unwrap();
/// let b = Spectrum::from_arrays(&[200.0, 100.0], &[1.0, 1.0]).unwrap();
/// let s = calculate_entropy_similarity_matched(&a, &b, MzTolerance::Absolute(0.01));
/// assert!((s - 1.0_f64).abs() < 1e-12);
/// ```
pub fn calculate_entropy_similarity_matched<T: Float + std::iter::Sum<T> + FromPrimitive + ToPrimitive>(
    a: &Spectrum<T>,
    b: &Spectrum<T>,
    tolerance: MzTolerance<T>,
) -> T {
    if a.total_intensity() <= T::zero() || b.total_intensity() <= T::zero() {
        return T::zero();
    }
    let (ea, wa) = a.get_entropy_and_weighted_intensity();
    let (eb, wb) = b.get_entropy_and_weighted_intensity();
    let sa: T = wa.iter().copied().sum();
    let sb: T = wb.iter().copied().sum();
    let wa: Vec<T> = wa.iter().map(|&x| x / sa).collect();
    let wb: Vec<T> = wb.iter().map(|&x| x / sb).collect();

    let pairs = match_peaks_by(a, b, tolerance, |i, j| wa[i] * wb[j]);
    let mut matched_a = vec![false; wa.len()];
    let mut matched_b = vec![false; wb.len()];
    let mut merged: Vec<T> = Vec::with_capacity(wa.len() + wb.len());
    for &(i, j) in &pairs {
        matched_a[i] = true;
        matched_b[j] = true;
        merged.push(wa[i] + wb[j]);
    }
    merged.extend(wa.iter().zip(&matched_a).filter(|(_, &m)| !m).map(|(&x, _)| x));
    merged.extend(wb.iter().zip(&matched_b).filter(|(_, &m)| !m).map(|(&x, _)| x));

    let em = calculate_entropy_from_intensities(&merged);
    let ln4 = T::from_f64(4.0).unwrap().ln();
    let two = T::from_f64(2.0).unwrap();
    let s = T::one() - (two * em - ea - eb) / ln4;
    s.max(T::zero()).min(T::one())
}


#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_spectrum_creation() {
        let mzs = vec![100.0, 200.0, 300.0];
        let intensities = vec![1.0, 2.0, 3.0];
        
        let spectrum = Spectrum::from_arrays(&mzs, &intensities).unwrap();
        assert_eq!(spectrum.peaks.len(), 3);
        assert_relative_eq!(spectrum.peaks[0].mz, 100.0);
        assert_relative_eq!(spectrum.peaks[0].intensity, 1.0);
    }

    #[test]
    fn test_spectrum_normalization() {
        let mzs = vec![100.0, 200.0, 300.0];
        let intensities = vec![1.0, 2.0, 3.0];
        
        let mut spectrum = Spectrum::from_arrays(&mzs, &intensities).unwrap();
        spectrum.normalize();
        
        let sum: f64 = spectrum.peaks.iter().map(|p| p.intensity).sum();
        assert_relative_eq!(sum, 1.0);
    }

    #[test]
    fn test_entropy_calculation() {
        // Test case from scipy.stats.entropy documentation
        let p = vec![0.1, 0.2, 0.3, 0.4];
        let entropy = calculate_entropy_from_intensities(&p);
        assert_relative_eq!(entropy, 1.2798542258336676, epsilon = 1e-10);
    }

    #[test]
    fn test_tsallis_entropy() {
        let mzs = vec![100.0, 200.0, 300.0];
        let intensities = vec![0.25, 0.25, 0.5];
        
        let spectrum = Spectrum::from_arrays(&mzs, &intensities).unwrap();
        let entropy = calculate_tsallis_entropy(&spectrum, 2.0);
        
        // For q=2, Tsallis entropy should be different from Shannon entropy
        assert!(entropy != calculate_entropy(&spectrum));
    }

    #[test]
    fn test_empty_spectra() {
        let spec_query: Spectrum<f64> = Spectrum::new();
        let spec_reference: Spectrum<f64> = Spectrum::new();
        
        let similarity = calculate_entropy_similarity(&spec_query, &spec_reference);
        assert_relative_eq!(similarity, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_single_peak_spectra() {
        let spec_query = Spectrum::from_arrays(&[200.0], &[100.0]).unwrap();
        let spec_reference = Spectrum::from_arrays(&[201.0], &[100.0]).unwrap();
        
        let similarity = calculate_entropy_similarity(&spec_query, &spec_reference);
        assert!(similarity > 0.0);
    }

    #[test]
    fn test_spectra_with_zero_intensities() {
        let spec_query = Spectrum::from_arrays(&[100.0, 200.0, 300.0], &[0.0, 80.0, 20.0]).unwrap();
        let spec_reference = Spectrum::from_arrays(&[100.0, 200.0, 300.0], &[30.0, 0.0, 70.0]).unwrap();
        
        let similarity = calculate_entropy_similarity(&spec_query, &spec_reference);
        assert!(similarity > 0.0);
    }

    #[test]
    fn test_real_world_spectra() {
        let spec_query = Spectrum::from_arrays(
            &[124.0869, 148.9238, 156.015, 186.0342, 279.0911, 300.0],
            &[0.32267, 20.5, 0.222153, 40.0, 50.0, 10.0]
        ).unwrap();
        
        let spec_reference = Spectrum::from_arrays(
            &[124.0869, 148.9238, 156.011, 279.0912, 289.0911],
            &[0.32267, 20.5, 0.222153, 50.0, 50.0]
        ).unwrap();
        
        let similarity = calculate_entropy_similarity(&spec_query, &spec_reference);
        assert!(similarity > 0.0);
    }

    #[test]
    fn test_weight_factor_transformation() {
        let mzs = vec![69.071, 86.066, 86.0969];
        let ints = vec![7.917962, 1.021589, 100.0];
        
        let transformed = weight_factor_transformation(&mzs, &ints, 0.5, 1.5);
        
        // Verify the transformation
        for (i, (&mz, &intensity)) in mzs.iter().zip(ints.iter()).enumerate() {
            let expected = mz.powf(0.5) * intensity.powf(1.5);
            assert_relative_eq!(transformed[i], expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_spectrum_weight_factor_transformation() {
        let mut spectrum = Spectrum::from_arrays(
            &[69.071, 86.066, 86.0969],
            &[7.917962, 1.021589, 100.0]
        ).unwrap();
        
        let original_intensities: Vec<f64> = spectrum.peaks.iter().map(|p| p.intensity).collect();
        spectrum.apply_weight_factor(0.5, 1.5);
        
        // Verify the transformation
        for (i, peak) in spectrum.peaks.iter().enumerate() {
            let expected = peak.mz.powf(0.5) * original_intensities[i].powf(1.5);
            assert_relative_eq!(peak.intensity, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_complex_spectra() {
        let spec_query = Spectrum::from_arrays(
            &[192.1383, 136.0757, 121.0648, 150.0914, 119.0492, 91.0540, 135.0679, 57.0697, 211.0542],
            &[999.00, 178.22, 82.82, 38.46, 35.76, 16.58, 14.99, 8.09, 5.19]
        ).unwrap();
        
        let spec_reference = Spectrum::from_arrays(
            &[91.0540, 136.0757, 119.0491, 121.0648, 192.1384, 135.0679, 57.0697, 150.0914, 211.0543],
            &[999.00, 738.56, 605.09, 477.12, 270.73, 195.10, 147.95, 41.86, 37.86]
        ).unwrap();
        
        let similarity = calculate_entropy_similarity(&spec_query, &spec_reference);
        assert!(similarity > 0.0);
    }

    #[test]
    fn test_relative_entropy() {
        // Test case for Kullback-Leibler divergence
        let p = vec![0.1, 0.2, 0.3, 0.4];
        let q = vec![0.2, 0.2, 0.2, 0.4];
        let entropy = calculate_relative_entropy(&p, &q);
        assert!(entropy > 0.0);
    }

    #[test]
    fn test_entropy_normalization() {
        // Test that unnormalized probabilities are handled correctly
        let p = vec![1.0, 2.0, 3.0, 4.0];
        let entropy = calculate_entropy_from_intensities(&p);
        assert_relative_eq!(entropy, 1.2798542258336676, epsilon = 1e-10);
    }

    #[test]
    fn test_zero_entropy() {
        // Test case with zero probabilities
        let p = vec![1.0, 0.0, 0.0];
        let entropy = calculate_entropy_from_intensities(&p);
        assert_relative_eq!(entropy, 0.0, epsilon = 1e-10);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_optimized_tsallis_entropy() {
        let mzs = vec![100.0, 200.0, 300.0];
        let intensities = vec![0.25, 0.25, 0.5];
        
        let spectrum = Spectrum::from_arrays(&mzs, &intensities).unwrap();
        let entropy = calculate_tsallis_entropy(&spectrum, 2.0);
        let entropy_opt = calculate_tsallis_entropy_optimized(&spectrum, 2.0);
        
        assert_relative_eq!(entropy, entropy_opt, epsilon = 1e-10);
    }

    #[test]
    fn test_spectrum_helpers() {
        let mut s = Spectrum::from_arrays(&[300.0, 100.0, 200.0, 150.0], &[5.0, 50.0, 1.0, 20.0]).unwrap();
        assert_eq!(s.len(), 4);
        assert_relative_eq!(s.total_intensity(), 76.0);
        assert_relative_eq!(s.base_peak().unwrap().mz, 100.0);
        s.sort_by_mz();
        assert_eq!(s.mzs(), vec![100.0, 150.0, 200.0, 300.0]);
        let mut t = s.clone();
        t.keep_top_n(2);
        assert_eq!(t.mzs(), vec![100.0, 150.0]);
        let mut t = s.clone();
        t.remove_relative_noise(0.1);
        assert_eq!(t.mzs(), vec![100.0, 150.0, 300.0]);
        let mut t = s.clone();
        t.filter_mz_range(120.0, 250.0);
        assert_eq!(t.mzs(), vec![150.0, 200.0]);
        s.normalize_to_max();
        assert_relative_eq!(s.base_peak().unwrap().intensity, 1.0);
        assert!(Spectrum::<f64>::new().base_peak().is_none());
    }

    #[test]
    fn test_match_peaks_greedy_and_ppm() {
        // one peak in a sits between two in b; the stronger pair wins
        let a = Spectrum::from_arrays(&[100.0, 100.02], &[1.0, 10.0]).unwrap();
        let b = Spectrum::from_arrays(&[100.01], &[5.0]).unwrap();
        assert_eq!(match_peaks(&a, &b, MzTolerance::Absolute(0.02)), vec![(1, 0)]);
        // 10 ppm at m/z 1000 is 0.01 Da
        let a = Spectrum::from_arrays(&[1000.0], &[1.0]).unwrap();
        let b = Spectrum::from_arrays(&[1000.009], &[1.0]).unwrap();
        let c = Spectrum::from_arrays(&[1000.02], &[1.0]).unwrap();
        assert_eq!(match_peaks(&a, &b, MzTolerance::Ppm(10.0)).len(), 1);
        assert_eq!(match_peaks(&a, &c, MzTolerance::Ppm(10.0)).len(), 0);
    }

    #[test]
    fn test_spectral_cosine() {
        let a = Spectrum::from_arrays(&[100.0, 200.0, 300.0], &[3.0, 4.0, 0.0]).unwrap();
        let (s, n) = spectral_cosine_similarity(&a, &a, MzTolerance::Absolute(0.01), 0.0, 1.0);
        assert_relative_eq!(s, 1.0, epsilon = 1e-12);
        assert_eq!(n, 3);
        let b = Spectrum::from_arrays(&[100.0, 250.0], &[3.0, 4.0]).unwrap();
        let (s, n) = spectral_cosine_similarity(&a, &b, MzTolerance::Absolute(0.01), 0.0, 1.0);
        assert_relative_eq!(s, 9.0 / 25.0, epsilon = 1e-12);
        assert_eq!(n, 1);
        let empty = Spectrum::<f64>::new();
        assert_eq!(spectral_cosine_similarity(&a, &empty, MzTolerance::Absolute(0.01), 0.0, 1.0), (0.0, 0));
    }

    #[test]
    fn test_entropy_similarity_matched() {
        let a = Spectrum::from_arrays(
            &[124.0869, 148.9238, 156.015, 186.0342, 279.0911, 300.0],
            &[0.32267, 20.5, 0.222153, 40.0, 50.0, 10.0],
        ).unwrap();
        let b = Spectrum::from_arrays(
            &[124.0869, 148.9238, 156.011, 279.0912, 289.0911],
            &[0.32267, 20.5, 0.222153, 50.0, 50.0],
        ).unwrap();
        let s = calculate_entropy_similarity_matched(&a, &b, MzTolerance::Absolute(0.01));
        assert!(s > 0.0 && s < 1.0);
        assert_relative_eq!(calculate_entropy_similarity_matched(&a, &a, MzTolerance::Absolute(0.01)), 1.0, epsilon = 1e-10);
        // disjoint spectra -> 0
        let c = Spectrum::from_arrays(&[500.0, 600.0], &[1.0, 1.0]).unwrap();
        assert_relative_eq!(calculate_entropy_similarity_matched(&a, &c, MzTolerance::Absolute(0.01)), 0.0, epsilon = 1e-10);
        let empty = Spectrum::<f64>::new();
        assert_eq!(calculate_entropy_similarity_matched(&a, &empty, MzTolerance::Absolute(0.01)), 0.0);
    }
}
