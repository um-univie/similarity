use crate::traits::EntropyMeasure;
use crate::spectral_entropy::*;
use num_traits::Float;

// ============================================================================
// INFORMATION THEORY ENTROPY MEASURES  
// ============================================================================

/// Shannon entropy implementation using the EntropyMeasure trait
/// Input: reference to a spectrum, Output: entropy value
pub struct ShannonEntropy;

impl<T> EntropyMeasure<&crate::Spectrum<T>, T> for ShannonEntropy
where
    T: Float + std::iter::Sum<T> + num_traits::FromPrimitive + num_traits::ToPrimitive,
{
    /// Calculate Shannon entropy of a spectrum
    /// 
    /// # Arguments
    /// 
    /// * `input` - A reference to the spectrum to analyze
    /// 
    /// # Returns
    /// 
    /// The Shannon entropy value
    /// 
    /// # Examples
    /// 
    /// ```
    /// use similarity::{EntropyMeasure, entropy_traits::ShannonEntropy, Spectrum, Peak};
    /// 
    /// let peaks = vec![
    ///     Peak { mz: 100.0, intensity: 0.5 },
    ///     Peak { mz: 200.0, intensity: 0.5 },
    /// ];
    /// let spectrum = Spectrum::from_peaks(peaks);
    /// let entropy = ShannonEntropy::entropy(&spectrum);
    /// assert!(entropy > 0.0);
    /// ```
    fn entropy(input: &crate::Spectrum<T>) -> T {
        calculate_entropy(input)
    }
}



/// Tsallis entropy implementation using the EntropyMeasure trait
/// Input: tuple of (spectrum reference, q parameter), Output: entropy value
pub struct TsallisEntropy;

impl<T> EntropyMeasure<(&crate::Spectrum<T>, T), T> for TsallisEntropy
where
    T: Float + std::iter::Sum<T> + num_traits::FromPrimitive + num_traits::ToPrimitive,
{
    /// Calculate Tsallis entropy of a spectrum with parameter q
    /// 
    /// # Arguments
    /// 
    /// * `input` - A tuple containing (spectrum reference, q parameter)
    /// 
    /// # Returns
    /// 
    /// The Tsallis entropy value
    /// 
    /// # Examples
    /// 
    /// ```
    /// use similarity::{EntropyMeasure, entropy_traits::TsallisEntropy, Spectrum, Peak};
    /// 
    /// let peaks = vec![Peak { mz: 100.0, intensity: 1.0 }];
    /// let spectrum = Spectrum::from_peaks(peaks);
    /// let q = 2.0;
    /// let entropy = TsallisEntropy::entropy((&spectrum, q));
    /// assert!(entropy >= 0.0);
    /// ```
    fn entropy(input: (&crate::Spectrum<T>, T)) -> T {
        calculate_tsallis_entropy(input.0, input.1)
    }
}

/// Optimized Tsallis entropy implementation
#[cfg(feature = "parallel")]
pub struct TsallisEntropyOptimized;

#[cfg(feature = "parallel")]
impl<T> EntropyMeasure<(&crate::Spectrum<T>, T), T> for TsallisEntropyOptimized
where
    T: Float + std::iter::Sum<T> + num_traits::FromPrimitive + num_traits::ToPrimitive + Send + Sync,
{
    /// Calculate optimized Tsallis entropy of a spectrum using parallel processing
    fn entropy(input: (&crate::Spectrum<T>, T)) -> T {
        calculate_tsallis_entropy_optimized(input.0, input.1)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Spectrum, Peak};

    #[test]
    fn test_shannon_entropy_trait() {
        let peaks = vec![
            Peak { mz: 100.0, intensity: 0.5 },
            Peak { mz: 200.0, intensity: 0.5 },
        ];
        let spectrum = Spectrum::from_peaks(peaks);
        let entropy = ShannonEntropy::entropy(&spectrum);
        assert!(entropy > 0.0);
    }

    #[test]
    fn test_tsallis_entropy_trait() {
        let peaks = vec![Peak { mz: 100.0, intensity: 1.0 }];
        let spectrum = Spectrum::from_peaks(peaks);
        let q = 2.0;
        let entropy = TsallisEntropy::entropy((&spectrum, q));
        assert!(entropy >= 0.0);
    }
} 