use crate::traits::Similarity;
use crate::similarity_metrics::*;
use num_traits::{Num, Float};
use std::collections::HashSet;

// ============================================================================
// PAIRWISE SIMILARITY AND DISTANCE METRICS
// ============================================================================

/// Cosine similarity implementation using the Similarity trait
/// Input: tuple of two slices, Output: Option<f64>
pub struct CosineSimilarity;

impl<T> Similarity<(&[T], &[T]), Option<f64>> for CosineSimilarity
where
    T: Copy + Num,
    f64: From<T>,
{
    /// Calculate cosine similarity between two slices
    /// 
    /// # Arguments
    /// 
    /// * `input` - A tuple containing two slices to compare
    /// 
    /// # Returns
    /// 
    /// Some(similarity) if slices are same length and computation is valid, None otherwise
    /// 
    /// # Examples
    /// 
    /// ```
    /// use similarity::{Similarity, similarity_traits::CosineSimilarity};
    /// 
    /// let a = [1.0, 2.0, 3.0];
    /// let b = [1.0, 2.0, 3.0];
    /// let result = CosineSimilarity::similarity((&a, &b));
    /// assert_eq!(result, Some(1.0));
    /// ```
    fn similarity(input: (&[T], &[T])) -> Option<f64> {
        cosine_similarity(input.0, input.1)
    }
}



/// Parallel cosine similarity implementation
#[cfg(feature = "parallel")]
pub struct CosineSimilarityParallel;

#[cfg(feature = "parallel")]
impl<T> Similarity<(&[T], &[T]), Option<f64>> for CosineSimilarityParallel
where
    T: Copy + Num + Send + Sync,
    f64: From<T>,
{
    /// Calculate parallel cosine similarity between two slices
    fn similarity(input: (&[T], &[T])) -> Option<f64> {
        cosine_similarity_parallel(input.0, input.1)
    }
}

/// Cosine distance implementation using the Similarity trait
pub struct CosineDistance;

impl<T> Similarity<(&[T], &[T]), Option<f64>> for CosineDistance
where
    T: Copy + Num,
    f64: From<T>,
{
    /// Calculate cosine distance between two slices
    /// 
    /// # Examples
    /// 
    /// ```
    /// use similarity::{Similarity, similarity_traits::CosineDistance};
    /// 
    /// let a = [1.0, 0.0];
    /// let b = [0.0, 1.0];
    /// let result = CosineDistance::similarity((&a, &b));
    /// assert_eq!(result, Some(1.0)); // Perpendicular vectors have distance 1
    /// ```
    fn similarity(input: (&[T], &[T])) -> Option<f64> {
        cosine_distance(input.0, input.1)
    }
}



/// Parallel cosine distance implementation
#[cfg(feature = "parallel")]
pub struct CosineDistanceParallel;

#[cfg(feature = "parallel")]
impl<T> Similarity<(&[T], &[T]), Option<f64>> for CosineDistanceParallel
where
    T: Copy + Num + Send + Sync,
    f64: From<T>,
{
    /// Calculate parallel cosine distance between two slices
    fn similarity(input: (&[T], &[T])) -> Option<f64> {
        cosine_distance_parallel(input.0, input.1)
    }
}

/// Euclidean distance implementation using the Similarity trait
pub struct EuclideanDistance;

impl<T> Similarity<(&[T], &[T]), Option<T>> for EuclideanDistance
where
    T: std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Float + Copy + std::iter::Sum<T>,
{
    /// Calculate Euclidean distance between two slices
    /// 
    /// # Examples
    /// 
    /// ```
    /// use similarity::{Similarity, similarity_traits::EuclideanDistance};
    /// 
    /// let a = [0.0, 0.0];
    /// let b = [3.0, 4.0];
    /// let result = EuclideanDistance::similarity((&a, &b));
    /// assert_eq!(result, Some(5.0));
    /// ```
    fn similarity(input: (&[T], &[T])) -> Option<T> {
        euclidean_distance(input.0, input.1)
    }
}

/// Squared Euclidean distance implementation
pub struct SquaredEuclideanDistance;

impl<T> Similarity<(&[T], &[T]), Option<T>> for SquaredEuclideanDistance
where
    T: std::ops::Mul<Output = T> + std::ops::Sub<Output = T> + Copy + std::iter::Sum<T>,
{
    /// Calculate squared Euclidean distance between two slices
    /// 
    /// # Examples
    /// 
    /// ```
    /// use similarity::{Similarity, similarity_traits::SquaredEuclideanDistance};
    /// 
    /// let a = [0.0, 0.0];
    /// let b = [3.0, 4.0];
    /// let result = SquaredEuclideanDistance::similarity((&a, &b));
    /// assert_eq!(result, Some(25.0));
    /// ```
    fn similarity(input: (&[T], &[T])) -> Option<T> {
        squared_euclidean_distance(input.0, input.1)
    }
}

/// Pearson correlation distance implementation
pub struct PearsonCorrelationDistance;

impl<T> Similarity<(&[T], &[T]), Option<f64>> for PearsonCorrelationDistance
where
    T: Copy + Num + Float,
{
    /// Calculate Pearson correlation distance between two slices
    /// 
    /// # Examples
    /// 
    /// ```
    /// use similarity::{Similarity, similarity_traits::PearsonCorrelationDistance};
    /// 
    /// let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let y = [2.0, 4.0, 6.0, 8.0, 10.0];
    /// let result = PearsonCorrelationDistance::similarity((&x, &y));
    /// assert!(result.is_some());
    /// ```
    fn similarity(input: (&[T], &[T])) -> Option<f64> {
        pearson_correlation_distance(input.0, input.1)
    }
}



/// Parallel Pearson correlation distance implementation
#[cfg(feature = "parallel")]
pub struct PearsonCorrelationDistanceParallel;

#[cfg(feature = "parallel")]
impl<T> Similarity<(&[T], &[T]), Option<f64>> for PearsonCorrelationDistanceParallel
where
    T: Copy + Num + Float + Send + Sync,
{
    /// Calculate parallel Pearson correlation distance between two slices
    fn similarity(input: (&[T], &[T])) -> Option<f64> {
        pearson_correlation_distance_parallel(input.0, input.1)
    }
}

/// Jaccard index implementation using the Similarity trait
pub struct JaccardIndex;

impl<T> Similarity<(&HashSet<T>, &HashSet<T>), f64> for JaccardIndex
where
    T: Eq + std::hash::Hash,
{
    /// Calculate Jaccard index between two sets
    /// 
    /// # Examples
    /// 
    /// ```
    /// use similarity::{Similarity, similarity_traits::JaccardIndex};
    /// use std::collections::HashSet;
    /// 
    /// let mut set1 = HashSet::new();
    /// set1.insert(1);
    /// set1.insert(2);
    /// set1.insert(3);
    /// 
    /// let mut set2 = HashSet::new();
    /// set2.insert(2);
    /// set2.insert(3);
    /// set2.insert(4);
    /// 
    /// let result = JaccardIndex::similarity((&set1, &set2));
    /// assert_eq!(result, 0.5); // 2 intersect / 4 union
    /// ```
    fn similarity(input: (&HashSet<T>, &HashSet<T>)) -> f64 {
        jaccard_index(input.0, input.1)
    }
}

// ============================================================================
// PREDICTION ACCURACY METRICS
// ============================================================================

/// Hit rate implementation using the Similarity trait
/// Input: tuple of (actual slice, predicted slice, tolerance)
pub struct HitRate;

impl<T> Similarity<(&[T], &[T], T), Option<f64>> for HitRate
where
    T: std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + Copy
        + std::iter::Sum<T>
        + std::cmp::PartialOrd
        + num_traits::Signed,
{
    /// Calculate hit rate between actual and predicted values with given tolerance
    /// 
    /// # Examples
    /// 
    /// ```
    /// use similarity::{Similarity, similarity_traits::HitRate};
    /// 
    /// let actual = [1.0, 2.0, 3.0];
    /// let predicted = [1.1, 1.9, 3.2];
    /// let result = HitRate::similarity((&actual, &predicted, 0.3));
    /// assert!(result.is_some());
    /// ```
    fn similarity(input: (&[T], &[T], T)) -> Option<f64> {
        hit_rate(input.0, input.1, input.2)
    }
}

/// Overshoot rate implementation using the Similarity trait
/// Input: tuple of (actual slice, predicted slice, tolerance)
pub struct OvershootRate;

impl<T> Similarity<(&[T], &[T], T), Option<f64>> for OvershootRate
where
    T: std::ops::Sub<Output = T> + Copy + std::cmp::PartialOrd + num_traits::Signed,
{
    /// Calculate overshoot rate between actual and predicted values with given tolerance
    /// 
    /// # Examples
    /// 
    /// ```
    /// use similarity::{Similarity, similarity_traits::OvershootRate};
    /// 
    /// let actual = [1.0, 2.0, 3.0];
    /// let predicted = [1.5, 2.5, 3.5];
    /// let result = OvershootRate::similarity((&actual, &predicted, 0.3));
    /// assert!(result.is_some());
    /// ```
    fn similarity(input: (&[T], &[T], T)) -> Option<f64> {
        overshoot_rate(input.0, input.1, input.2)
    }
}

// ============================================================================
// CROSS-CORRELATION AND TIME SERIES SIMILARITY
// ============================================================================

/// Cross-correlation optimized implementation
pub struct CrossCorrelationOptimized;

impl Similarity<(&[f64], &[f64]), Vec<f64>> for CrossCorrelationOptimized {
    /// Calculate optimized cross-correlation between two arrays
    /// 
    /// # Examples
    /// 
    /// ```
    /// use similarity::{Similarity, similarity_traits::CrossCorrelationOptimized};
    /// 
    /// let x = [1.0, 2.0, 3.0];
    /// let y = [1.0, 2.0, 3.0];
    /// let result = CrossCorrelationOptimized::similarity((&x, &y));
    /// assert!(!result.is_empty());
    /// ```
    fn similarity(input: (&[f64], &[f64])) -> Vec<f64> {
        cross_correlate(input.0, input.1)
    }
}

/// Cross-correlation parallel implementation
#[cfg(feature = "parallel")]
pub struct CrossCorrelationParallel;

#[cfg(feature = "parallel")]
impl Similarity<(&[f64], &[f64]), Vec<f64>> for CrossCorrelationParallel {
    /// Calculate parallel cross-correlation between two arrays
    fn similarity(input: (&[f64], &[f64])) -> Vec<f64> {
        cross_correlate_parallel(input.0, input.1)
    }
}

/// Cross-correlation FFT optimized implementation  
#[cfg(feature = "fft")]
pub struct CrossCorrelationFFT;

#[cfg(feature = "fft")]
impl Similarity<(&[f64], &[f64]), Vec<f64>> for CrossCorrelationFFT {
    /// Calculate FFT-optimized cross-correlation between two arrays
    fn similarity(input: (&[f64], &[f64])) -> Vec<f64> {
        cross_correlate_fft(input.0, input.1)
    }
}

/// Time shift finder implementation
pub struct TimeShiftFinder;

impl Similarity<(&[f64], &[f64]), Option<usize>> for TimeShiftFinder {
    /// Find time shift between two arrays
    /// 
    /// # Examples
    /// 
    /// ```
    /// use similarity::{Similarity, similarity_traits::TimeShiftFinder};
    /// 
    /// let x = [0.0, 1.0, 2.0];
    /// let y = [1.0, 2.0, 0.0];
    /// let result = TimeShiftFinder::similarity((&x, &y));
    /// assert!(result.is_some());
    /// ```
    fn similarity(input: (&[f64], &[f64])) -> Option<usize> {
        find_time_shift(input.0, input.1)
    }
}

/// Time shift finder FFT implementation
#[cfg(feature = "fft")]
pub struct TimeShiftFinderFFT;

#[cfg(feature = "fft")]
impl Similarity<(&[f64], &[f64]), Option<usize>> for TimeShiftFinderFFT {
    /// Find time shift between two arrays using FFT method
    fn similarity(input: (&[f64], &[f64])) -> Option<usize> {
        find_time_shift_fft(input.0, input.1)
    }
}

// ============================================================================
// SPECTRAL SIMILARITY METRICS
// ============================================================================

/// Entropy similarity calculator between two spectra
pub struct EntropySimilarity;

impl<T> Similarity<(&crate::Spectrum<T>, &crate::Spectrum<T>), T> for EntropySimilarity
where
    T: Float + std::iter::Sum<T> + num_traits::FromPrimitive + num_traits::ToPrimitive,
{
    /// Calculate entropy similarity between two spectra
    /// 
    /// # Examples
    /// 
    /// ```
    /// use similarity::{Similarity, similarity_traits::EntropySimilarity, Spectrum, Peak};
    /// 
    /// let peaks1 = vec![Peak { mz: 100.0, intensity: 1.0 }];
    /// let peaks2 = vec![Peak { mz: 100.0, intensity: 1.0 }];
    /// let spectrum1 = Spectrum::from_peaks(peaks1);
    /// let spectrum2 = Spectrum::from_peaks(peaks2);
    /// let similarity = EntropySimilarity::similarity((&spectrum1, &spectrum2));
    /// assert!(similarity >= 0.0);
    /// ```
    fn similarity(input: (&crate::Spectrum<T>, &crate::Spectrum<T>)) -> T {
        crate::spectral_entropy::calculate_entropy_similarity(input.0, input.1)
    }
}



// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_cosine_similarity_trait() {
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 2.0, 3.0];
        let result = CosineSimilarity::similarity((&a, &b));
        assert_eq!(result, Some(1.0));
    }

    #[test]
    fn test_cosine_distance_trait() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        let result = CosineDistance::similarity((&a, &b));
        assert_eq!(result, Some(1.0));
    }

    #[test]
    fn test_euclidean_distance_trait() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        let result = EuclideanDistance::similarity((&a, &b));
        assert_eq!(result, Some(5.0));
    }

    #[test]
    fn test_jaccard_index_trait() {
        let mut set1 = HashSet::new();
        set1.insert(1);
        set1.insert(2);
        set1.insert(3);

        let mut set2 = HashSet::new();
        set2.insert(2);
        set2.insert(3);
        set2.insert(4);

        let result = JaccardIndex::similarity((&set1, &set2));
        assert_eq!(result, 0.5);
    }

    #[test]
    fn test_entropy_similarity_trait() {
        use crate::{Spectrum, Peak};
        
        let peaks1 = vec![Peak { mz: 100.0, intensity: 1.0 }];
        let peaks2 = vec![Peak { mz: 100.0, intensity: 1.0 }];
        let spectrum1 = Spectrum::from_peaks(peaks1);
        let spectrum2 = Spectrum::from_peaks(peaks2);
        let similarity = EntropySimilarity::similarity((&spectrum1, &spectrum2));
        assert!(similarity >= 0.0);
    }
} 