//! Rank correlations and regression / forecast error metrics.
//!
//! Generic over [`num_traits::Float`]. Functions return `None` for length
//! mismatches, empty input, or undefined results (e.g. constant input for
//! correlations).

use num_traits::Float;

fn check<T>(a: &[T], b: &[T]) -> bool {
    a.len() == b.len() && !a.is_empty()
}

/// Fractional ranks (1-based, ties get the average rank), as used by
/// Spearman's ρ. NaN values are ranked last.
///
/// ```
/// use similarity::statistics::rank;
/// assert_eq!(rank(&[10.0, 30.0, 20.0, 20.0]), vec![1.0, 4.0, 2.5, 2.5]);
/// ```
pub fn rank<T: Float>(x: &[T]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&i, &j| match (x[i].is_nan(), x[j].is_nan()) {
        (false, false) => x[i].partial_cmp(&x[j]).unwrap(),
        (a, b) => a.cmp(&b),
    });
    let mut ranks = vec![0.0; x.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && x[idx[j + 1]] == x[idx[i]] {
            j += 1;
        }
        let r = (i + j) as f64 / 2.0 + 1.0;
        for &k in &idx[i..=j] {
            ranks[k] = r;
        }
        i = j + 1;
    }
    ranks
}

fn pearson_f64(a: &[f64], b: &[f64]) -> Option<f64> {
    crate::simd::pearson_correlation_simd(a, b)
}

/// Pearson correlation coefficient *r*.
pub fn pearson_correlation<T: Float>(a: &[T], b: &[T]) -> Option<f64> {
    if !check(a, b) {
        return None;
    }
    let a: Vec<f64> = a.iter().map(|v| v.to_f64().unwrap()).collect();
    let b: Vec<f64> = b.iter().map(|v| v.to_f64().unwrap()).collect();
    pearson_f64(&a, &b)
}

/// Spearman rank correlation ρ (Pearson correlation of the fractional
/// ranks; handles ties correctly).
///
/// ```
/// use similarity::statistics::spearman_correlation;
/// // monotone but non-linear relationship
/// let r = spearman_correlation(&[1.0, 2.0, 3.0, 4.0], &[1.0, 8.0, 27.0, 64.0]).unwrap();
/// assert!((r - 1.0).abs() < 1e-12);
/// ```
pub fn spearman_correlation<T: Float>(a: &[T], b: &[T]) -> Option<f64> {
    if !check(a, b) {
        return None;
    }
    pearson_f64(&rank(a), &rank(b))
}

/// Kendall rank correlation τ-b (tie-corrected), `O(n²)`.
///
/// ```
/// use similarity::statistics::kendall_tau;
/// assert_eq!(kendall_tau(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]), Some(-1.0));
/// ```
pub fn kendall_tau<T: Float>(a: &[T], b: &[T]) -> Option<f64> {
    if !check(a, b) {
        return None;
    }
    let n = a.len();
    let (mut concordant, mut discordant, mut ties_a, mut ties_b) = (0i64, 0i64, 0i64, 0i64);
    for i in 0..n {
        for j in i + 1..n {
            let da = a[i] - a[j];
            let db = b[i] - b[j];
            let za = da == T::zero();
            let zb = db == T::zero();
            if za && zb {
                continue;
            } else if za {
                ties_a += 1;
            } else if zb {
                ties_b += 1;
            } else if (da > T::zero()) == (db > T::zero()) {
                concordant += 1;
            } else {
                discordant += 1;
            }
        }
    }
    let n1 = (concordant + discordant + ties_a) as f64;
    let n2 = (concordant + discordant + ties_b) as f64;
    let den = (n1 * n2).sqrt();
    (den != 0.0).then(|| (concordant - discordant) as f64 / den)
}

/// Concordance correlation coefficient (Lin's CCC), measuring agreement
/// rather than mere correlation.
pub fn concordance_correlation<T: Float>(a: &[T], b: &[T]) -> Option<f64> {
    if !check(a, b) {
        return None;
    }
    let n = a.len() as f64;
    let a: Vec<f64> = a.iter().map(|v| v.to_f64().unwrap()).collect();
    let b: Vec<f64> = b.iter().map(|v| v.to_f64().unwrap()).collect();
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(&b) {
        cov += (x - ma) * (y - mb);
        va += (x - ma) * (x - ma);
        vb += (y - mb) * (y - mb);
    }
    let den = va / n + vb / n + (ma - mb).powi(2);
    (den != 0.0).then(|| 2.0 * cov / n / den)
}

/// Mean absolute error.
pub fn mean_absolute_error<T: Float>(actual: &[T], predicted: &[T]) -> Option<T> {
    if !check(actual, predicted) {
        return None;
    }
    let s = actual.iter().zip(predicted).fold(T::zero(), |s, (&a, &p)| s + (a - p).abs());
    Some(s / T::from(actual.len()).unwrap())
}

/// Mean squared error.
pub fn mean_squared_error<T: Float>(actual: &[T], predicted: &[T]) -> Option<T> {
    if !check(actual, predicted) {
        return None;
    }
    let s = actual.iter().zip(predicted).fold(T::zero(), |s, (&a, &p)| s + (a - p) * (a - p));
    Some(s / T::from(actual.len()).unwrap())
}

/// Root mean squared error.
pub fn root_mean_squared_error<T: Float>(actual: &[T], predicted: &[T]) -> Option<T> {
    mean_squared_error(actual, predicted).map(Float::sqrt)
}

/// Mean absolute percentage error (as a fraction, not ×100). Pairs with
/// `actual == 0` are skipped; `None` if every pair is skipped.
pub fn mean_absolute_percentage_error<T: Float>(actual: &[T], predicted: &[T]) -> Option<T> {
    if !check(actual, predicted) {
        return None;
    }
    let (s, n) = actual
        .iter()
        .zip(predicted)
        .filter(|(&a, _)| a != T::zero())
        .fold((T::zero(), 0usize), |(s, n), (&a, &p)| (s + ((a - p) / a).abs(), n + 1));
    (n > 0).then(|| s / T::from(n).unwrap())
}

/// Symmetric mean absolute percentage error in `[0, 2]`. Pairs where both
/// values are zero count as 0 error.
pub fn symmetric_mean_absolute_percentage_error<T: Float>(actual: &[T], predicted: &[T]) -> Option<T> {
    if !check(actual, predicted) {
        return None;
    }
    let s = actual.iter().zip(predicted).fold(T::zero(), |s, (&a, &p)| {
        let d = a.abs() + p.abs();
        if d == T::zero() {
            s
        } else {
            s + (T::one() + T::one()) * (a - p).abs() / d
        }
    });
    Some(s / T::from(actual.len()).unwrap())
}

/// Coefficient of determination R². `None` if `actual` is constant.
pub fn r_squared<T: Float>(actual: &[T], predicted: &[T]) -> Option<T> {
    if !check(actual, predicted) {
        return None;
    }
    let n = T::from(actual.len()).unwrap();
    let mean = actual.iter().fold(T::zero(), |s, &a| s + a) / n;
    let ss_tot = actual.iter().fold(T::zero(), |s, &a| s + (a - mean) * (a - mean));
    if ss_tot == T::zero() {
        return None;
    }
    let ss_res = actual.iter().zip(predicted).fold(T::zero(), |s, (&a, &p)| s + (a - p) * (a - p));
    Some(T::one() - ss_res / ss_tot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn ranks_and_spearman() {
        assert_eq!(rank(&[3.0, 1.0, 2.0]), vec![3.0, 1.0, 2.0]);
        assert_eq!(rank(&[1.0, 1.0, 1.0]), vec![2.0, 2.0, 2.0]);
        assert_eq!(rank::<f64>(&[]), Vec::<f64>::new());
        // scipy.stats.spearmanr([1,2,3,4,5],[5,6,7,8,7]) -> 0.8207826816681233
        let r = spearman_correlation(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5.0, 6.0, 7.0, 8.0, 7.0]).unwrap();
        assert_relative_eq!(r, 0.8207826816681233, epsilon = 1e-12);
        assert_eq!(spearman_correlation(&[1.0, 1.0], &[1.0, 2.0]), None);
    }

    #[test]
    fn kendall() {
        // scipy.stats.kendalltau([12,2,1,12,2],[1,4,7,1,0]) -> -0.47140452079103173
        let t = kendall_tau(&[12.0, 2.0, 1.0, 12.0, 2.0], &[1.0, 4.0, 7.0, 1.0, 0.0]).unwrap();
        assert_relative_eq!(t, -0.47140452079103173, epsilon = 1e-12);
        assert_eq!(kendall_tau(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), Some(1.0));
    }

    #[test]
    fn pearson_and_ccc() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [2.0, 4.0, 5.0, 4.0, 5.0];
        assert_relative_eq!(pearson_correlation(&a, &b).unwrap(), 0.7745966692414834, epsilon = 1e-12);
        assert_relative_eq!(concordance_correlation(&a, &a).unwrap(), 1.0);
        let shifted: Vec<f64> = a.iter().map(|x| x + 10.0).collect();
        assert!(concordance_correlation(&a, &shifted).unwrap() < 0.2);
    }

    #[test]
    fn errors() {
        let a = [1.0, 2.0, 3.0, 0.0];
        let p = [1.5, 2.0, 2.0, 1.0];
        assert_relative_eq!(mean_absolute_error(&a, &p).unwrap(), 0.625);
        assert_relative_eq!(mean_squared_error(&a, &p).unwrap(), 0.5625);
        assert_relative_eq!(root_mean_squared_error(&a, &p).unwrap(), 0.75);
        assert_relative_eq!(mean_absolute_percentage_error(&a, &p).unwrap(), (0.5 + 0.0 + 1.0 / 3.0) / 3.0);
        let smape = symmetric_mean_absolute_percentage_error(&a, &p).unwrap();
        assert!(smape > 0.0 && smape < 2.0);
        assert_relative_eq!(r_squared(&a, &a).unwrap(), 1.0);
        assert_eq!(r_squared(&[1.0, 1.0], &[1.0, 2.0]), None);
        assert_eq!(mean_squared_error::<f64>(&[], &[]), None);
        let s = crate::simd::mean_squared_error_simd(&a, &p).unwrap();
        assert_relative_eq!(s, mean_squared_error(&a, &p).unwrap());
    }
}
