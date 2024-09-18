use num_traits::sign::abs;
use num_traits::{Num, Float};
use std::collections::HashSet;
use rustfft::{FftPlanner, num_complex::Complex};

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

pub fn cross_correlate(x: &[f64], y: &[f64]) -> Vec<f64> {
    let x_length = x.len();
    let y_length = y.len();
    let max_lag = x_length + y_length - 1;

    (0..max_lag)
        .map(|lag| {
            let x_start = lag.saturating_sub(y_length - 1);
            let y_start = y_length.saturating_sub(lag + 1);

            let sum = x.iter()
                .skip(x_start)
                .zip(y.iter().skip(y_start))
                .map(|(&x, &y)| x * y)
                .sum();
            sum
        })
        .collect()
}

pub fn cross_correlate_fft(x: &[f64], y: &[f64]) -> Vec<f64> {
    let x_len = x.len();
    let y_len = y.len();
    let total_len = x_len + y_len - 1;
    // Find the next power of two that is greater than or equal to the total length
    // This often results in a more efficient FFT
    let fft_size = total_len.next_power_of_two();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    let ifft = planner.plan_fft_inverse(fft_size);

    // Pre-allocate and initialize vectors
    let mut x_padded = vec![Complex::new(0.0, 0.0); fft_size];
    let mut y_padded = vec![Complex::new(0.0, 0.0); fft_size];

    // Fill x_padded and y_padded
    x_padded[..x_len].copy_from_slice(&x.iter().map(|&x| Complex::new(x, 0.0)).collect::<Vec<_>>());
    y_padded[..y_len].copy_from_slice(&y.iter().map(|&y| Complex::new(y, 0.0)).collect::<Vec<_>>());

    fft.process(&mut x_padded);
    fft.process(&mut y_padded);

    for (x, y) in x_padded.iter_mut().zip(&y_padded) {
        *x *= y.conj();
    }

    ifft.process(&mut x_padded);

    x_padded.iter()
        .cycle()
        .skip(fft_size - y_len + 1)
        .take(total_len)
        .map(|&c| c.re / fft_size as f64)
        .collect()
}

// Helper function to test and compare both methods
pub fn compare_cross_correlation_methods(x: &[f64], y: &[f64]) {
    let time_domain_result = cross_correlate(x, y);
    let fft_result = cross_correlate_fft(x, y);

    println!("Time-domain result: {:?}", time_domain_result);
    println!("FFT-based result: {:?}", fft_result);

    // Compare results
    let max_diff = time_domain_result.iter().zip(&fft_result)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);

    println!("Maximum difference between methods: {}", max_diff);
}

pub fn find_time_shift(x: &[f64], y: &[f64]) -> Option<usize> {
    let cross_correlation = cross_correlate(x, y);
    let max_index = cross_correlation.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())?.0;
    Some(max_index)
}

pub fn find_time_shift_fft(x: &[f64], y: &[f64]) -> Option<usize> {
    let cross_correlation = cross_correlate_fft(x, y);
    let max_index = cross_correlation.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())?.0;
    Some(max_index)
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

        let result = cross_correlate_fft(&x, &y);
        println!("{:?}", result);

        assert_eq!(result.len(), expected.len());
        result.iter().zip(expected.iter()).for_each(|(&a, &b)| {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        });
    }

    #[test]
    fn find_time_shift_test() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 1.0, 0.0, 1.0, 2.0];
        let expected = 4;

        let result = find_time_shift(&x, &y).unwrap();
        assert_eq!(result, expected);
    }
}
