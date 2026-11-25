use super::*;
use approx::assert_relative_eq;
use rand::Rng;

// --- helpers ---

fn random_vec(n: usize) -> Vec<f64> {
    let mut rng = rand::rng();
    (0..n).map(|_| rng.random_range(-10.0..10.0)).collect()
}

/// Reference scalar dot product.
fn dot_ref(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Reference scalar cosine similarity.
fn cosine_ref(a: &[f64], b: &[f64]) -> Option<f64> {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    let d = na * nb;
    if d == 0.0 {
        None
    } else {
        Some(dot / d)
    }
}

/// Reference scalar squared Euclidean distance.
fn sq_euclidean_ref(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Reference scalar Manhattan distance.
fn manhattan_ref(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}

/// Reference Pearson correlation distance (negative r).
fn pearson_ref(a: &[f64], b: &[f64]) -> Option<f64> {
    let n = a.len() as f64;
    let ma: f64 = a.iter().sum::<f64>() / n;
    let mb: f64 = b.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for (x, y) in a.iter().zip(b) {
        let dx = x - ma;
        let dy = y - mb;
        num += dx * dy;
        da += dx * dx;
        db += dy * dy;
    }
    let denom = (da * db).sqrt();
    if denom == 0.0 {
        if num == 0.0 {
            Some(0.0)
        } else {
            None
        }
    } else {
        Some(-num / denom)
    }
}

// -----------------------------------------------------------------------
// dot_product_simd
// -----------------------------------------------------------------------

#[test]
fn dot_product_empty() {
    super::isa::for_each_isa(|_isa| {
        assert_eq!(dot_product_simd(&[], &[]), Some(0.0));
    });
}

#[test]
fn dot_product_single() {
    super::isa::for_each_isa(|_isa| {
        assert_eq!(dot_product_simd(&[3.0], &[4.0]), Some(12.0));
    });
}

#[test]
fn dot_product_known() {
    super::isa::for_each_isa(|_isa| {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [5.0, 4.0, 3.0, 2.0, 1.0];
        assert_relative_eq!(dot_product_simd(&a, &b).unwrap(), 35.0, epsilon = 1e-10);
    });
}

#[test]
fn dot_product_length_mismatch() {
    super::isa::for_each_isa(|_isa| {
        assert!(dot_product_simd(&[1.0], &[1.0, 2.0]).is_none());
    });
}

#[test]
fn dot_product_large_random() {
    super::isa::for_each_isa(|_isa| {
        let a = random_vec(10_003);
        let b = random_vec(10_003);
        let expected = dot_ref(&a, &b);
        assert_relative_eq!(
            dot_product_simd(&a, &b).unwrap(),
            expected,
            epsilon = 1e-6
        );
    });
}

// -----------------------------------------------------------------------
// cosine_similarity_simd
// -----------------------------------------------------------------------

#[test]
fn cosine_identical() {
    super::isa::for_each_isa(|_isa| {
        let a = [1.0, 2.0, 3.0];
        assert_relative_eq!(
            cosine_similarity_simd(&a, &a).unwrap(),
            1.0,
            epsilon = 1e-12
        );
    });
}

#[test]
fn cosine_orthogonal() {
    super::isa::for_each_isa(|_isa| {
        let a = [1.0, 0.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0, 0.0];
        assert_relative_eq!(
            cosine_similarity_simd(&a, &b).unwrap(),
            0.0,
            epsilon = 1e-12
        );
    });
}

#[test]
fn cosine_opposite() {
    super::isa::for_each_isa(|_isa| {
        let a = [1.0, 2.0, 3.0];
        let b = [-1.0, -2.0, -3.0];
        assert_relative_eq!(
            cosine_similarity_simd(&a, &b).unwrap(),
            -1.0,
            epsilon = 1e-12
        );
    });
}

#[test]
fn cosine_zero_norm() {
    super::isa::for_each_isa(|_isa| {
        let a = [0.0, 0.0];
        let b = [1.0, 2.0];
        assert!(cosine_similarity_simd(&a, &b).is_none());
    });
}

#[test]
fn cosine_large_random() {
    super::isa::for_each_isa(|_isa| {
        let a = random_vec(10_007);
        let b = random_vec(10_007);
        let expected = cosine_ref(&a, &b).unwrap();
        assert_relative_eq!(
            cosine_similarity_simd(&a, &b).unwrap(),
            expected,
            epsilon = 1e-10
        );
    });
}

// -----------------------------------------------------------------------
// cosine_distance_simd
// -----------------------------------------------------------------------

#[test]
fn cosine_distance_identical() {
    super::isa::for_each_isa(|_isa| {
        let a = [1.0, 2.0, 3.0];
        assert_relative_eq!(
            cosine_distance_simd(&a, &a).unwrap(),
            0.0,
            epsilon = 1e-12
        );
    });
}

// -----------------------------------------------------------------------
// euclidean_distance_simd
// -----------------------------------------------------------------------

#[test]
fn euclidean_zero() {
    super::isa::for_each_isa(|_isa| {
        let a = [1.0, 2.0, 3.0];
        assert_relative_eq!(
            euclidean_distance_simd(&a, &a).unwrap(),
            0.0,
            epsilon = 1e-12
        );
    });
}

#[test]
fn euclidean_known() {
    super::isa::for_each_isa(|_isa| {
        let a = [0.0, 0.0, 0.0, 0.0, 1.0];
        let b = [0.0, 0.0, 0.0, 0.0, 0.0];
        assert_relative_eq!(
            euclidean_distance_simd(&a, &b).unwrap(),
            1.0,
            epsilon = 1e-12
        );
    });
}

#[test]
fn euclidean_large_random() {
    super::isa::for_each_isa(|_isa| {
        let a = random_vec(10_001);
        let b = random_vec(10_001);
        let expected = sq_euclidean_ref(&a, &b).sqrt();
        assert_relative_eq!(
            euclidean_distance_simd(&a, &b).unwrap(),
            expected,
            epsilon = 1e-6
        );
    });
}

// -----------------------------------------------------------------------
// squared_euclidean_distance_simd
// -----------------------------------------------------------------------

#[test]
fn sq_euclidean_empty() {
    super::isa::for_each_isa(|_isa| {
        assert_eq!(squared_euclidean_distance_simd(&[], &[]), Some(0.0));
    });
}

#[test]
fn sq_euclidean_single() {
    super::isa::for_each_isa(|_isa| {
        assert_relative_eq!(
            squared_euclidean_distance_simd(&[3.0], &[1.0]).unwrap(),
            4.0,
            epsilon = 1e-12
        );
    });
}

#[test]
fn sq_euclidean_large_random() {
    super::isa::for_each_isa(|_isa| {
        let a = random_vec(9_999);
        let b = random_vec(9_999);
        let expected = sq_euclidean_ref(&a, &b);
        assert_relative_eq!(
            squared_euclidean_distance_simd(&a, &b).unwrap(),
            expected,
            epsilon = 1e-6
        );
    });
}

// -----------------------------------------------------------------------
// manhattan_distance_simd
// -----------------------------------------------------------------------

#[test]
fn manhattan_empty() {
    super::isa::for_each_isa(|_isa| {
        assert_eq!(manhattan_distance_simd(&[], &[]), Some(0.0));
    });
}

#[test]
fn manhattan_known() {
    super::isa::for_each_isa(|_isa| {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 6.0, 3.0];
        // |1-4| + |2-6| + |3-3| = 3 + 4 + 0 = 7
        assert_relative_eq!(
            manhattan_distance_simd(&a, &b).unwrap(),
            7.0,
            epsilon = 1e-12
        );
    });
}

#[test]
fn manhattan_large_random() {
    super::isa::for_each_isa(|_isa| {
        let a = random_vec(10_013);
        let b = random_vec(10_013);
        let expected = manhattan_ref(&a, &b);
        assert_relative_eq!(
            manhattan_distance_simd(&a, &b).unwrap(),
            expected,
            epsilon = 1e-6
        );
    });
}

// -----------------------------------------------------------------------
// pearson_correlation_distance_simd
// -----------------------------------------------------------------------

#[test]
fn pearson_identical() {
    super::isa::for_each_isa(|_isa| {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(
            pearson_correlation_distance_simd(&a, &a).unwrap(),
            -1.0,
            epsilon = 1e-10
        );
    });
}

#[test]
fn pearson_opposite() {
    super::isa::for_each_isa(|_isa| {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [5.0, 4.0, 3.0, 2.0, 1.0];
        assert_relative_eq!(
            pearson_correlation_distance_simd(&a, &b).unwrap(),
            1.0,
            epsilon = 1e-10
        );
    });
}

#[test]
fn pearson_zero_variance() {
    super::isa::for_each_isa(|_isa| {
        let a = [3.0; 10];
        let b = [7.0; 10];
        assert_relative_eq!(
            pearson_correlation_distance_simd(&a, &b).unwrap(),
            0.0,
            epsilon = 1e-10
        );
    });
}

#[test]
fn pearson_matches_generic() {
    super::isa::for_each_isa(|_isa| {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [2.0, 4.0, 5.0, 4.0, 5.0];
        let expected = crate::pearson_correlation_distance(&a, &b).unwrap();
        assert_relative_eq!(
            pearson_correlation_distance_simd(&a, &b).unwrap(),
            expected,
            epsilon = 1e-10
        );
    });
}

#[test]
fn pearson_large_random() {
    super::isa::for_each_isa(|_isa| {
        let a = random_vec(10_009);
        let b = random_vec(10_009);
        let expected = pearson_ref(&a, &b).unwrap();
        assert_relative_eq!(
            pearson_correlation_distance_simd(&a, &b).unwrap(),
            expected,
            epsilon = 1e-6
        );
    });
}

#[test]
fn pearson_length_mismatch() {
    super::isa::for_each_isa(|_isa| {
        assert!(pearson_correlation_distance_simd(&[1.0], &[1.0, 2.0]).is_none());
    });
}

// -----------------------------------------------------------------------
// cross_correlate_simd
// -----------------------------------------------------------------------

#[test]
fn cross_correlate_empty() {
    super::isa::for_each_isa(|_isa| {
        assert!(cross_correlate_simd(&[], &[1.0]).is_empty());
        assert!(cross_correlate_simd(&[1.0], &[]).is_empty());
    });
}

#[test]
fn cross_correlate_matches_generic() {
    super::isa::for_each_isa(|_isa| {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 1.0, 0.0, 1.0, 2.0];
        let expected = crate::cross_correlate(&x, &y);
        let result = cross_correlate_simd(&x, &y);
        assert_eq!(result.len(), expected.len());
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-10);
        }
    });
}

#[test]
fn cross_correlate_single() {
    super::isa::for_each_isa(|_isa| {
        let x = [3.0];
        let y = [4.0];
        let result = cross_correlate_simd(&x, &y);
        assert_eq!(result.len(), 1);
        assert_relative_eq!(result[0], 12.0, epsilon = 1e-12);
    });
}

#[test]
fn cross_correlate_random() {
    super::isa::for_each_isa(|_isa| {
        let x = random_vec(128);
        let y = random_vec(64);
        let expected = crate::cross_correlate(&x, &y);
        let result = cross_correlate_simd(&x, &y);
        assert_eq!(result.len(), expected.len());
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-6);
        }
    });
}

// -----------------------------------------------------------------------
// Consistency with the generic crate API
// -----------------------------------------------------------------------

#[test]
fn cosine_matches_generic() {
    super::isa::for_each_isa(|_isa| {
        let a = random_vec(1_000);
        let b = random_vec(1_000);
        let generic = crate::cosine_similarity(&a, &b).unwrap();
        let simd = cosine_similarity_simd(&a, &b).unwrap();
        assert_relative_eq!(simd, generic, epsilon = 1e-10);
    });
}

#[test]
fn sq_euclidean_matches_generic() {
    super::isa::for_each_isa(|_isa| {
        let a = random_vec(1_000);
        let b = random_vec(1_000);
        let generic: f64 = crate::squared_euclidean_distance(&a, &b).unwrap();
        let simd = squared_euclidean_distance_simd(&a, &b).unwrap();
        assert_relative_eq!(simd, generic, epsilon = 1e-6);
    });
}

#[test]
fn euclidean_matches_generic() {
    super::isa::for_each_isa(|_isa| {
        let a = random_vec(1_000);
        let b = random_vec(1_000);
        let generic: f64 = crate::euclidean_distance(&a, &b).unwrap();
        let simd = euclidean_distance_simd(&a, &b).unwrap();
        assert_relative_eq!(simd, generic, epsilon = 1e-6);
    });
}
