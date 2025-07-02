use num_traits::sign::abs;
use num_traits::{Num, Float};
use std::collections::HashSet;

#[cfg(feature = "fft")]
use rustfft::{FftPlanner, num_complex::Complex};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// This function calculates the cosine similarity between two slices. 
/// Geometrically, this is the cosine of the angle between two vectors.
/// An Option is returned that contains the cosine similarity if the slices are the same length and
/// the norm of the product of the slices is not zero. The return type is f64 as the cosine similarity
/// might be fractional even for integer slices.
///
/// # Arguments
/// * `slice_a` - A slice of values
/// * `slice_b` - A slice of values
///
/// # Example
/// ```
/// use crate::similarity::similarity_metrics::*;
/// let slice_a = [0.0, 0.0, 0.0, 0.0, 1.0];
/// let slice_b = [0.0, 0.0, 0.0, 0.0, 1.0];
/// let distance = cosine_similarity(&slice_a, &slice_b).unwrap();
/// assert_eq!(distance, 1.0);
/// let slice_a = [0, 0, 0, 0, 1];
/// let slice_b = [0, 0, 0, 1, 0];
/// let distance = cosine_similarity(&slice_a, &slice_b).unwrap();
/// assert_eq!(distance, 0.0);
/// ```
pub fn cosine_similarity<T>(slice_a: &[T], slice_b: &[T]) -> Option<f64>
where
    T: Copy + Num, f64: From<T>,
{
    if slice_a.len() != slice_b.len() {
        return None;
    }

    let mut dot_product = T::zero();
    let mut norm_a_squared = T::zero();
    let mut norm_b_squared = T::zero();

    for (a, b) in slice_a.iter().zip(slice_b.iter()) {
        let product = *a * *b;
        dot_product = dot_product + product;
        norm_a_squared = norm_a_squared + *a * *a;
        norm_b_squared = norm_b_squared + *b * *b;
    }

    let norm_a = (f64::from(norm_a_squared)).sqrt();
    let norm_b = (f64::from(norm_b_squared)).sqrt();
    let norm_product = norm_a * norm_b;
    if norm_product == 0.0 {
        None
    } else {
        Some(f64::from(dot_product) / norm_product)
    }
}

/// Block size for cache-friendly processing (used by Tsallis entropy optimization)
const BLOCK_SIZE: usize = 64;

/// Parallel version of cosine similarity using rayon
#[cfg(feature = "parallel")]
pub fn cosine_similarity_parallel<T>(slice_a: &[T], slice_b: &[T]) -> Option<f64>
where
    T: Copy + Num + Send + Sync, f64: From<T>,
{
    if slice_a.len() != slice_b.len() {
        return None;
    }

    let (dot_product, norm_a_squared, norm_b_squared) = slice_a.par_iter()
        .zip(slice_b.par_iter())
        .map(|(&a, &b)| {
            let product = a * b;
            (product, a * a, b * b)
        })
        .reduce(
            || (T::zero(), T::zero(), T::zero()),
            |(acc_dot, acc_a, acc_b), (dot, a, b)| {
                (acc_dot + dot, acc_a + a, acc_b + b)
            }
        );

    let norm_a = (f64::from(norm_a_squared)).sqrt();
    let norm_b = (f64::from(norm_b_squared)).sqrt();
    let norm_product = norm_a * norm_b;
    if norm_product == 0.0 {
        None
    } else {
        Some(f64::from(dot_product) / norm_product)
    }
}

/// This function calculates the cosine distance between two slices.
///
/// # Arguments
/// * `slice_a` - A slice of values
/// * `slice_b` - A slice of values
///
/// # Example
/// ```
/// use crate::similarity::similarity_metrics::*;
/// let slice_a = [0.0, 0.0, 0.0, 0.0, 1.0];
/// let slice_b = [0.1, 0.0, 0.0, 0.0, 1.0];
/// let distance = cosine_distance(&slice_a, &slice_b).unwrap();
/// assert_eq!(distance.round(), 0.0);
/// ```
pub fn cosine_distance<T>(slice_a: &[T], slice_b: &[T]) -> Option<f64> where T: Copy + Num, f64: From<T> {
    if slice_a.len() != slice_b.len() {
        None
    } else {
        let cosine_similarity = cosine_similarity(slice_a, slice_b)?;
        Some(1.0 - cosine_similarity)
    }
}



/// Parallel version of cosine distance
#[cfg(feature = "parallel")]
pub fn cosine_distance_parallel<T>(slice_a: &[T], slice_b: &[T]) -> Option<f64>
where
    T: Copy + Num + Send + Sync, f64: From<T>,
{
    if slice_a.len() != slice_b.len() {
        None
    } else {
        let cosine_similarity = cosine_similarity_parallel(slice_a, slice_b)?;
        Some(1.0 - cosine_similarity)
    }
}

/// This function calculates the euclidean distance between two slices.
///
/// Since sqrt is only available for floats, this function does not accept integers
/// We could avoid this by casting the integers to floats, but that could be lossy and we would
/// also have to do two conversions.
///
/// # Arguments
/// * `slice_a` - A slice of values
/// * `slice_b` - A slice of values
///
/// # Example
/// ```
/// use crate::similarity::similarity_metrics::*;
/// let slice_a = [0.0, 0.0, 0.0, 0.0, 1.0];
/// let slice_b = [0.0, 0.0, 0.0, 0.0, 0.0];
/// let distance = euclidean_distance(&slice_a, &slice_b).unwrap();
/// assert_eq!(distance, 1.0);
/// ```
pub fn euclidean_distance<T>(slice_a: &[T], slice_b: &[T]) -> Option<T>
where
    T: std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Float + Copy + std::iter::Sum<T>,
{
    if slice_a.len() != slice_b.len() {
        None
    } else {
        let sum_of_squares = squared_euclidean_distance(slice_a, slice_b);

        // This unwrap is safe because we know that the slices are the same length
        Some(sum_of_squares.unwrap().sqrt())
    }
}

/// This function calculates the squared euclidean distance between two slices.
///
/// # Arguments
/// * `slice_a` - A slice of values
/// * `slice_b` - A slice of values
///
/// # Example
/// ```
/// use crate::similarity::similarity_metrics::*;
/// let slice_a = [0.0, 0.0, 0.0, 0.0, 1.0];
/// let slice_b = [0.0, 0.0, 0.0, 0.0, 0.0];
/// let squared_distance = squared_euclidean_distance(&slice_a, &slice_b).unwrap();
/// assert_eq!(squared_distance, 1.0);
/// let slice_a = [0, 0, 0, 0, 0];
/// let slice_b = [0, 0, 0, 0, 0];
/// let squared_distance = squared_euclidean_distance(&slice_a, &slice_b).unwrap();
/// assert_eq!(squared_distance, 0);
/// ```
pub fn squared_euclidean_distance<T>(slice_a: &[T], slice_b: &[T]) -> Option<T>
where
    T: std::ops::Mul<Output = T> + std::ops::Sub<Output=T> + Copy + std::iter::Sum<T>,
{
    if slice_a.len() != slice_b.len() {
        None
    } else {
        Some(
            slice_a
                .iter()
                .zip(slice_b.iter())
                .map(|(a, b)| (*a - *b) * (*a - *b))
                .sum(),
        )
    }
}

/// This function calculates the hit rate, which is the number of times the predicted value is
/// within the tolerance (<=) of the actual value, divided by the total number of predictions.
///
/// # Arguments
/// * `actual` - A slice of actual values
/// * `predicted` - A slice of predicted values
/// * `tolerance` - The tolerance for the hit
///
/// # Example
/// ```
/// use crate::similarity::similarity_metrics::*;
/// let actual = [0, 0, 0, 0, 1, 0, 0];
/// let predicted = [0, 0, 0, 0, 0, 0, 0];
/// let hit_rate = hit_rate(&actual, &predicted, 1).unwrap();
/// assert_eq!(hit_rate, 1.0);
/// ```
pub fn hit_rate<T>(actual: &[T], predicted: &[T], tolerance: T) -> Option<f64>
where
    T: std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + Copy
        + std::iter::Sum<T>
        + std::cmp::PartialOrd
        + num_traits::Signed,
{
    if actual.len() != predicted.len() {
        None
    } else {
        let hits: usize = actual
            .iter()
            .zip(predicted.iter())
            .filter(|&(a, p)| abs(*a - *p) <= tolerance)
            .count();
        Some(hits as f64 / actual.len() as f64)
    }
}

/// This function calculates the overshoot rate, which is the number of times the predicted value
/// is greater than the actual value by more than the tolerance, divided by the total number of
/// predictions.
///
/// # Arguments
/// * `actual` - A slice of actual values
/// * `predicted` - A slice of predicted values
/// * `tolerance` - The tolerance for the overshoot
///
/// # Example
/// ```
/// use crate::similarity::similarity_metrics::*;
/// let actual = [0, 0, 0, 0, 1, 0, 0];
/// let predicted = [0, 0, 0, 0, 0, 0, 0];
/// let overshoot_rate = overshoot_rate(&actual, &predicted, 1).unwrap();
/// assert_eq!(overshoot_rate, 0.0);
/// ```
pub fn overshoot_rate<T>(actual: &[T], predicted: &[T], tolerance: T) -> Option<f64> where T: std::ops::Sub<Output = T> + Copy + std::cmp::PartialOrd + num_traits::Signed {
    if actual.len() != predicted.len() {
        None
    } else {
        let overshoots: usize = actual
            .iter()
            .zip(predicted.iter())
            .filter(|&(a, p)| *a > *p + tolerance)
            .count();
        Some(overshoots as f64 / actual.len() as f64)
    }
}

pub trait Sqrt {
    type Output;
    fn sqrt(self) -> Self::Output;
}

pub fn jaccard_index<T: Eq + std::hash::Hash>(set1: &HashSet<T>, set2: &HashSet<T>) -> f64 {
    let intersection = set1.intersection(set2).count();
    let union: usize = set1.union(set2).count();

    if union == 0 {
        // Handle potential division by zero if both sets or the union are empty
        return 0.0; 
    }

    intersection as f64 / union as f64
}

/// Optimized version of cross-correlation using block processing for better cache utilization
pub fn cross_correlate(x: &[f64], y: &[f64]) -> Vec<f64> {
    let x_length = x.len();
    let y_length = y.len();
    let max_lag = x_length + y_length - 1;
    let mut result = Vec::with_capacity(max_lag);

    for lag in 0..max_lag {
        let x_start = lag.saturating_sub(y_length - 1);
        let y_start = y_length.saturating_sub(lag + 1);
        
        let mut sum = 0.0;
        let mut i = 0;
        
        // Process in blocks for better cache utilization
        while i + BLOCK_SIZE <= x.len() - x_start && i + BLOCK_SIZE <= y.len() - y_start {
            let x_block = &x[x_start + i..x_start + i + BLOCK_SIZE];
            let y_block = &y[y_start + i..y_start + i + BLOCK_SIZE];
            
            // Process block
            for j in 0..BLOCK_SIZE {
                sum += x_block[j] * y_block[j];
            }
            
            i += BLOCK_SIZE;
        }
        
        // Handle remaining elements
        for j in i..(x.len() - x_start).min(y.len() - y_start) {
            sum += x[x_start + j] * y[y_start + j];
        }
        
        result.push(sum);
    }
    result
}

/// Parallel version of cross-correlation using rayon with block processing
#[cfg(feature = "parallel")]
pub fn cross_correlate_parallel(x: &[f64], y: &[f64]) -> Vec<f64> {
    let x_length = x.len();
    let y_length = y.len();
    let max_lag = x_length + y_length - 1;

    (0..max_lag)
        .into_par_iter()
        .map(|lag| {
            let x_start = lag.saturating_sub(y_length - 1);
            let y_start = y_length.saturating_sub(lag + 1);
            
            let mut sum = 0.0;
            let mut i = 0;
            
            // Process in blocks for better cache utilization
            while i + BLOCK_SIZE <= x.len() - x_start && i + BLOCK_SIZE <= y.len() - y_start {
                let x_block = &x[x_start + i..x_start + i + BLOCK_SIZE];
                let y_block = &y[y_start + i..y_start + i + BLOCK_SIZE];
                
                // Process block
                for j in 0..BLOCK_SIZE {
                    sum += x_block[j] * y_block[j];
                }
                
                i += BLOCK_SIZE;
            }
            
            // Handle remaining elements
            for j in i..(x.len() - x_start).min(y.len() - y_start) {
                sum += x[x_start + j] * y[y_start + j];
            }
            
            sum
        })
        .collect()
}

/// Optimized FFT-based cross-correlation with cached planner
#[cfg(feature = "fft")]
pub fn cross_correlate_fft(x: &[f64], y: &[f64]) -> Vec<f64> {
    let x_len = x.len();
    let y_len = y.len();
    let total_len = x_len + y_len - 1;
    let fft_size = total_len.next_power_of_two();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    let ifft = planner.plan_fft_inverse(fft_size);

    let mut x_padded = vec![Complex::new(0.0, 0.0); fft_size];
    let mut y_padded = vec![Complex::new(0.0, 0.0); fft_size];

    // More efficient copying using iterators
    x_padded[..x_len].iter_mut().zip(x.iter()).for_each(|(dst, &src)| {
        *dst = Complex::new(src, 0.0);
    });
    y_padded[..y_len].iter_mut().zip(y.iter()).for_each(|(dst, &src)| {
        *dst = Complex::new(src, 0.0);
    });

    fft.process(&mut x_padded);
    fft.process(&mut y_padded);

    // Multiplication of complex numbers
    // At very large sizes, parallelization should be considered
    x_padded.iter_mut().zip(&y_padded).for_each(|(x, y)| {
        *x *= y.conj();
    });

    ifft.process(&mut x_padded);

    x_padded.iter()
        .cycle()
        .skip(fft_size - y_len + 1)
        .take(total_len)
        .map(|&c| c.re / fft_size as f64)
        .collect()
}

pub fn find_time_shift(x: &[f64], y: &[f64]) -> Option<usize> {
    let cross_correlation = cross_correlate(x, y);
    let max_index = cross_correlation.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())?.0;
    Some(max_index)
}

#[cfg(feature = "fft")]
pub fn find_time_shift_fft(x: &[f64], y: &[f64]) -> Option<usize> {
    let cross_correlation = cross_correlate_fft(x, y);
    let max_index = cross_correlation.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())?.0;
    Some(max_index)
}

/// Calculates the Pearson correlation distance between two slices.
/// This is the negative of the Pearson correlation coefficient, which measures
/// the linear correlation between two variables.
///
/// # Arguments
/// * `slice_a` - A slice of values
/// * `slice_b` - A slice of values
///
/// # Example
/// ```
/// use crate::similarity::similarity_metrics::*;
/// let slice_a = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let slice_b = [2.0, 4.0, 5.0, 4.0, 5.0];
/// let distance = pearson_correlation_distance(&slice_a, &slice_b).unwrap();
/// assert!((distance - (-0.7745966692414834)).abs() < 1e-6);
/// ```
pub fn pearson_correlation_distance<T>(slice_a: &[T], slice_b: &[T]) -> Option<f64>
where
    T: Copy + Num + Float,
{
    if slice_a.len() != slice_b.len() {
        return None;
    }

    let n = slice_a.len() as f64;
    let sum_a: f64 = slice_a.iter().map(|&x| x.to_f64().unwrap()).sum();
    let sum_b: f64 = slice_b.iter().map(|&x| x.to_f64().unwrap()).sum();
    let mean_a = sum_a / n;
    let mean_b = sum_b / n;

    let mut numerator = 0.0;
    let mut sum_sq_a = 0.0;
    let mut sum_sq_b = 0.0;

    for (a, b) in slice_a.iter().zip(slice_b.iter()) {
        let a_f64 = a.to_f64().unwrap();
        let b_f64 = b.to_f64().unwrap();
        let diff_a = a_f64 - mean_a;
        let diff_b = b_f64 - mean_b;
        numerator += diff_a * diff_b;
        sum_sq_a += diff_a * diff_a;
        sum_sq_b += diff_b * diff_b;
    }

    let denominator = (sum_sq_a * sum_sq_b).sqrt();
    if denominator == 0.0 {
        if numerator == 0.0 {
            Some(0.0)
        } else {
            None
        }
    } else {
        Some(-numerator / denominator)
    }
}



/// Parallel version of Pearson correlation distance using rayon
#[cfg(feature = "parallel")]
pub fn pearson_correlation_distance_parallel<T>(slice_a: &[T], slice_b: &[T]) -> Option<f64>
where
    T: Copy + Num + Float + Send + Sync,
{
    if slice_a.len() != slice_b.len() {
        return None;
    }

    let n = slice_a.len() as f64;
    let sum_a: f64 = slice_a.par_iter().map(|&x| x.to_f64().unwrap()).sum();
    let sum_b: f64 = slice_b.par_iter().map(|&x| x.to_f64().unwrap()).sum();
    let mean_a = sum_a / n;
    let mean_b = sum_b / n;

    let (numerator, sum_sq_a, sum_sq_b) = slice_a.par_iter()
        .zip(slice_b.par_iter())
        .map(|(&a, &b)| {
            let a_f64 = a.to_f64().unwrap();
            let b_f64 = b.to_f64().unwrap();
            let diff_a = a_f64 - mean_a;
            let diff_b = b_f64 - mean_b;
            (diff_a * diff_b, diff_a * diff_a, diff_b * diff_b)
        })
        .reduce(
            || (0.0, 0.0, 0.0),
            |(acc_num, acc_sq_a, acc_sq_b), (num, sq_a, sq_b)| {
                (acc_num + num, acc_sq_a + sq_a, acc_sq_b + sq_b)
            }
        );

    let denominator = (sum_sq_a * sum_sq_b).sqrt();
    if denominator == 0.0 {
        if numerator == 0.0 {
            Some(0.0)
        } else {
            None
        }
    } else {
        Some(-numerator / denominator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_cross_correlation() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 1.0, 0.0, 1.0, 2.0];
        let expected = [2.0, 5.0, 8.0, 12.0, 18.0, 12.0, 10.0, 13.0, 10.0];

        let result = cross_correlate(&x, &y);
        println!("{:?}", result);

        assert_eq!(result.len(), expected.len());
        result.iter().zip(expected.iter()).for_each(|(&a, &b)| {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        });

        #[cfg(feature = "fft")]
        {
            let result = cross_correlate_fft(&x, &y);
            println!("{:?}", result);

            assert_eq!(result.len(), expected.len());
            result.iter().zip(expected.iter()).for_each(|(&a, &b)| {
                assert_relative_eq!(a, b, epsilon = 1e-10);
            });
        }
    }

    #[test]
    fn find_time_shift_test() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 1.0, 0.0, 1.0, 2.0];
        let expected = 4;

        let result = find_time_shift(&x, &y).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_pearson_correlation_distance() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [2.0, 4.0, 5.0, 4.0, 5.0];
        
        let result = pearson_correlation_distance(&a, &b).unwrap();
        assert_relative_eq!(result, -0.7745966692414834, epsilon = 1e-6);
        
        #[cfg(feature = "parallel")]
        {
            let result_par = pearson_correlation_distance_parallel(&a, &b).unwrap();
            assert_relative_eq!(result_par, -0.7745966692414834, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_pearson_correlation_distance_identical() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0];
        
        let result = pearson_correlation_distance(&a, &b).unwrap();
        assert_relative_eq!(result, -1.0, epsilon = 1e-10);
        

        
        #[cfg(feature = "parallel")]
        {
            let result_par = pearson_correlation_distance_parallel(&a, &b).unwrap();
            assert_relative_eq!(result_par, -1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_pearson_correlation_distance_opposite() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [5.0, 4.0, 3.0, 2.0, 1.0];
        
        let result = pearson_correlation_distance(&a, &b).unwrap();
        assert_relative_eq!(result, 1.0, epsilon = 1e-10);
        
        #[cfg(feature = "parallel")]
        {
            let result_par = pearson_correlation_distance_parallel(&a, &b).unwrap();
            assert_relative_eq!(result_par, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_pearson_correlation_distance_zero() {
        let a = [0.0, 0.0, 0.0, 0.0, 0.0];
        let b = [0.0, 0.0, 0.0, 0.0, 0.0];
        
        let result = pearson_correlation_distance(&a, &b).unwrap();
        assert_relative_eq!(result, 0.0, epsilon = 1e-10);
        
        #[cfg(feature = "parallel")]
        {
            let result_par = pearson_correlation_distance_parallel(&a, &b).unwrap();
            assert_relative_eq!(result_par, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_pearson_correlation_distance_different_lengths() {
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 2.0];
        
        assert!(pearson_correlation_distance(&a, &b).is_none());
        
        #[cfg(feature = "parallel")]
        {
            assert!(pearson_correlation_distance_parallel(&a, &b).is_none());
        }
    }
}
