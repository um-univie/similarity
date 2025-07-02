use crate::traits::DataTransform;
use crate::spectral_entropy::*;

// ============================================================================
// DATA TRANSFORMATION OPERATIONS
// ============================================================================

/// Weight factor transformation implementation using the DataTransform trait
/// Input: tuple of (mz slice, intensity slice, mz weight factor, intensity weight factor)
/// Output: Vec of transformed weight factors
pub struct WeightFactorTransformation;

impl<T> DataTransform<(&[T], &[T], T, T), Vec<T>> for WeightFactorTransformation
where
    T: num_traits::Float + std::iter::Sum<T> + num_traits::FromPrimitive + num_traits::ToPrimitive,
{
    /// Transform mz and intensity data using weight factors
    /// 
    /// # Arguments
    /// 
    /// * `input` - A tuple containing (mz slice, intensity slice, mz weight factor, intensity weight factor)
    /// 
    /// # Returns
    /// 
    /// A vector of transformed weight factors
    /// 
    /// # Examples
    /// 
    /// ```
    /// use similarity::{DataTransform, transform_traits::WeightFactorTransformation};
    /// 
    /// let mzs = [100.0, 200.0, 300.0];
    /// let intensities = [0.5, 0.3, 0.2];
    /// let wf_mz = 0.5;
    /// let wf_int = 0.5;
    /// let result = WeightFactorTransformation::transform((&mzs, &intensities, wf_mz, wf_int));
    /// assert_eq!(result.len(), 3);
    /// ```
    fn transform(input: (&[T], &[T], T, T)) -> Vec<T> {
        weight_factor_transformation(input.0, input.1, input.2, input.3)
    }
}

/// Optimized weight factor transformation implementation
#[cfg(feature = "parallel")]
pub struct WeightFactorTransformationOptimized;

#[cfg(feature = "parallel")]
impl<T> DataTransform<(&[T], &[T], T, T), Vec<T>> for WeightFactorTransformationOptimized
where
    T: num_traits::Float + std::iter::Sum<T> + num_traits::FromPrimitive + num_traits::ToPrimitive + Send + Sync,
{
    /// Transform mz and intensity data using weight factors with parallel processing
    fn transform(input: (&[T], &[T], T, T)) -> Vec<T> {
        // No separate optimized implementation available, using standard function
        weight_factor_transformation(input.0, input.1, input.2, input.3)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_factor_transformation_trait() {
        let mzs = [100.0, 200.0, 300.0];
        let intensities = [0.5, 0.3, 0.2];
        let wf_mz = 0.5;
        let wf_int = 0.5;
        let result = WeightFactorTransformation::transform((&mzs, &intensities, wf_mz, wf_int));
        assert_eq!(result.len(), 3);
        // Each transformed value should be > 0
        for value in result {
            assert!(value > 0.0);
        }
    }
} 