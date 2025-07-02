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
        self.peaks.sort_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap());

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

/// Calculates the entropy similarity between two spectra
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



#[cfg(test)]
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

    #[test]
    fn test_optimized_tsallis_entropy() {
        let mzs = vec![100.0, 200.0, 300.0];
        let intensities = vec![0.25, 0.25, 0.5];
        
        let spectrum = Spectrum::from_arrays(&mzs, &intensities).unwrap();
        let entropy = calculate_tsallis_entropy(&spectrum, 2.0);
        let entropy_opt = calculate_tsallis_entropy_optimized(&spectrum, 2.0);
        
        assert_relative_eq!(entropy, entropy_opt, epsilon = 1e-10);
    }
} 