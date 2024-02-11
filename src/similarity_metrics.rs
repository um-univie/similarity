use num_traits::sign::abs;
use num_traits::{real::Real, Num, Float};
use std::collections::HashSet;

/// This function calculates the cosine similarity between two slices. 
/// Geometrically, this is the cosine of the angle between two vectors.
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

fn jaccard_index<T: Eq + std::hash::Hash>(set1: &HashSet<T>, set2: &HashSet<T>) -> f64 {
    let intersection = set1.intersection(set2).count();
    let union: usize = set1.union(set2).count();

    if union == 0 {
        // Handle potential division by zero if both sets are empty
        return 0.0; 
    }

    intersection as f64 / union as f64
}
