//! Generic (scalar) vector distances beyond the Euclidean/cosine family.
//!
//! All functions are generic over [`num_traits::Float`] and return `None`
//! when the input slices differ in length. For `f32`/`f64` slices the
//! `_simd` variants in [`crate::simd`] compute the same values faster.

use num_traits::Float;

#[inline]
fn pairs<'a, T: Copy>(a: &'a [T], b: &'a [T]) -> Option<impl Iterator<Item = (T, T)> + 'a> {
    (a.len() == b.len()).then(|| a.iter().copied().zip(b.iter().copied()))
}

/// Chebyshev (L∞) distance: max |aᵢ − bᵢ|. Zero for empty input.
///
/// ```
/// use similarity::distance_metrics::chebyshev_distance;
/// assert_eq!(chebyshev_distance(&[1.0, 5.0, 2.0], &[2.0, 1.0, 2.0]), Some(4.0));
/// ```
pub fn chebyshev_distance<T: Float>(a: &[T], b: &[T]) -> Option<T> {
    Some(pairs(a, b)?.fold(T::zero(), |m, (x, y)| m.max((x - y).abs())))
}

/// Minkowski distance of order `p`: (Σ |aᵢ − bᵢ|ᵖ)^(1/p).
///
/// `p = 1` is Manhattan, `p = 2` Euclidean, `p = ∞` Chebyshev. Returns
/// `None` for a length mismatch or `p < 1` (not a metric).
///
/// ```
/// use similarity::distance_metrics::minkowski_distance;
/// let d = minkowski_distance(&[0.0, 0.0], &[3.0, 4.0], 2.0).unwrap();
/// assert!((d - 5.0_f64).abs() < 1e-12);
/// ```
pub fn minkowski_distance<T: Float>(a: &[T], b: &[T], p: T) -> Option<T> {
    if p.is_nan() || p < T::one() {
        return None;
    }
    if p.is_infinite() {
        return chebyshev_distance(a, b);
    }
    let it = pairs(a, b)?;
    if p == T::one() {
        return Some(it.fold(T::zero(), |s, (x, y)| s + (x - y).abs()));
    }
    let two = T::one() + T::one();
    if p == two {
        return Some(it.fold(T::zero(), |s, (x, y)| s + (x - y) * (x - y)).sqrt());
    }
    Some(it.fold(T::zero(), |s, (x, y)| s + (x - y).abs().powf(p)).powf(p.recip()))
}

/// Manhattan (L1) distance Σ |aᵢ − bᵢ|.
pub fn manhattan_distance<T: Float>(a: &[T], b: &[T]) -> Option<T> {
    Some(pairs(a, b)?.fold(T::zero(), |s, (x, y)| s + (x - y).abs()))
}

/// Canberra distance Σ |aᵢ − bᵢ| / (|aᵢ| + |bᵢ|); `0/0` terms count as 0.
pub fn canberra_distance<T: Float>(a: &[T], b: &[T]) -> Option<T> {
    Some(pairs(a, b)?.fold(T::zero(), |s, (x, y)| {
        let d = x.abs() + y.abs();
        if d == T::zero() {
            s
        } else {
            s + (x - y).abs() / d
        }
    }))
}

/// Bray–Curtis dissimilarity Σ |aᵢ − bᵢ| / Σ |aᵢ + bᵢ|. `None` if the
/// denominator is zero.
pub fn bray_curtis_distance<T: Float>(a: &[T], b: &[T]) -> Option<T> {
    let (n, d) = pairs(a, b)?.fold((T::zero(), T::zero()), |(n, d), (x, y)| {
        (n + (x - y).abs(), d + (x + y).abs())
    });
    (d != T::zero()).then(|| n / d)
}

/// Chi-squared distance Σ (aᵢ − bᵢ)² / (aᵢ + bᵢ); `0/0` terms count as 0.
pub fn chi_square_distance<T: Float>(a: &[T], b: &[T]) -> Option<T> {
    Some(pairs(a, b)?.fold(T::zero(), |s, (x, y)| {
        let d = x + y;
        if d == T::zero() {
            s
        } else {
            s + (x - y) * (x - y) / d
        }
    }))
}

/// Continuous Tanimoto (extended Jaccard) similarity
/// `a·b / (‖a‖² + ‖b‖² − a·b)`. `None` if both vectors are zero.
pub fn tanimoto_similarity<T: Float>(a: &[T], b: &[T]) -> Option<T> {
    let (d, na, nb) = pairs(a, b)?.fold((T::zero(), T::zero(), T::zero()), |(d, na, nb), (x, y)| {
        (d + x * y, na + x * x, nb + y * y)
    });
    let den = na + nb - d;
    (den != T::zero()).then(|| d / den)
}

/// Ruzicka (weighted Jaccard) similarity Σ min / Σ max.
pub fn ruzicka_similarity<T: Float>(a: &[T], b: &[T]) -> Option<T> {
    let (mn, mx) = pairs(a, b)?.fold((T::zero(), T::zero()), |(mn, mx), (x, y)| {
        (mn + x.min(y), mx + x.max(y))
    });
    (mx != T::zero()).then(|| mn / mx)
}

/// Weighted Euclidean distance √(Σ wᵢ (aᵢ − bᵢ)²).
pub fn weighted_euclidean_distance<T: Float>(a: &[T], b: &[T], w: &[T]) -> Option<T> {
    if a.len() != w.len() {
        return None;
    }
    Some(
        pairs(a, b)?
            .zip(w.iter())
            .fold(T::zero(), |s, ((x, y), &wi)| s + wi * (x - y) * (x - y))
            .sqrt(),
    )
}

/// Mahalanobis distance with a **diagonal** covariance (per-feature
/// variances): √(Σ (aᵢ − bᵢ)² / σᵢ²). `None` if any variance is ≤ 0.
pub fn mahalanobis_diagonal_distance<T: Float>(a: &[T], b: &[T], variances: &[T]) -> Option<T> {
    if a.len() != variances.len() || variances.iter().any(|&v| v <= T::zero()) {
        return None;
    }
    Some(
        pairs(a, b)?
            .zip(variances.iter())
            .fold(T::zero(), |s, ((x, y), &v)| s + (x - y) * (x - y) / v)
            .sqrt(),
    )
}

/// Hamming distance: number of positions where the elements differ.
/// Works for any comparable element type (bytes, chars, booleans, …).
///
/// ```
/// use similarity::distance_metrics::hamming_distance;
/// assert_eq!(hamming_distance(b"karolin", b"kathrin"), Some(3));
/// ```
pub fn hamming_distance<T: PartialEq>(a: &[T], b: &[T]) -> Option<usize> {
    (a.len() == b.len()).then(|| a.iter().zip(b).filter(|(x, y)| x != y).count())
}

/// Bitwise Hamming distance between packed binary fingerprints
/// (popcount of XOR). Lengths must match.
///
/// ```
/// use similarity::distance_metrics::hamming_distance_bits;
/// assert_eq!(hamming_distance_bits(&[0b1011u64], &[0b0001u64]), Some(2));
/// ```
pub fn hamming_distance_bits(a: &[u64], b: &[u64]) -> Option<u32> {
    (a.len() == b.len()).then(|| a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum())
}

/// Tanimoto (Jaccard) similarity of packed binary fingerprints:
/// popcount(a & b) / popcount(a | b). Two empty fingerprints give `1.0`.
///
/// This is the standard similarity for chemical fingerprints (ECFP, MACCS).
///
/// ```
/// use similarity::distance_metrics::tanimoto_bits;
/// assert_eq!(tanimoto_bits(&[0b1110u64], &[0b0111u64]), Some(0.5));
/// ```
pub fn tanimoto_bits(a: &[u64], b: &[u64]) -> Option<f64> {
    if a.len() != b.len() {
        return None;
    }
    let (inter, union) = a.iter().zip(b).fold((0u64, 0u64), |(i, u), (x, y)| {
        (i + (x & y).count_ones() as u64, u + (x | y).count_ones() as u64)
    });
    Some(if union == 0 { 1.0 } else { inter as f64 / union as f64 })
}

/// Levenshtein edit distance (insertions, deletions, substitutions) between
/// two sequences, in `O(len(a)·len(b))` time and `O(min)` memory.
///
/// ```
/// use similarity::distance_metrics::levenshtein_distance;
/// assert_eq!(levenshtein_distance(b"kitten", b"sitting"), 3);
/// ```
pub fn levenshtein_distance<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    let (a, b) = if a.len() < b.len() { (b, a) } else { (a, b) };
    if b.is_empty() {
        return a.len();
    }
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, x) in a.iter().enumerate() {
        let mut diag = row[0];
        row[0] = i + 1;
        for (j, y) in b.iter().enumerate() {
            let above = row[j + 1];
            let cost = usize::from(x != y);
            row[j + 1] = (diag + cost).min(above + 1).min(row[j] + 1);
            diag = above;
        }
    }
    row[b.len()]
}

/// Normalised Levenshtein similarity `1 − d / max(len(a), len(b))` in
/// `[0, 1]`. Two empty sequences have similarity `1`.
pub fn levenshtein_similarity<T: PartialEq>(a: &[T], b: &[T]) -> f64 {
    let m = a.len().max(b.len());
    if m == 0 {
        1.0
    } else {
        1.0 - levenshtein_distance(a, b) as f64 / m as f64
    }
}

/// Length of the longest common subsequence.
///
/// ```
/// use similarity::distance_metrics::longest_common_subsequence;
/// assert_eq!(longest_common_subsequence(b"ABCBDAB", b"BDCABA"), 4);
/// ```
pub fn longest_common_subsequence<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    let mut row = vec![0usize; b.len() + 1];
    for x in a {
        let mut diag = 0;
        for (j, y) in b.iter().enumerate() {
            let above = row[j + 1];
            row[j + 1] = if x == y { diag + 1 } else { above.max(row[j]) };
            diag = above;
        }
    }
    row[b.len()]
}

/// Dynamic time warping distance with absolute-difference cost, optionally
/// constrained to a Sakoe–Chiba band of half-width `window`.
///
/// Returns `None` if either series is empty or the band cannot connect the
/// two ends (`window < |len(a) − len(b)|`).
///
/// ```
/// use similarity::distance_metrics::dtw_distance;
/// let a = [0.0, 1.0, 2.0, 3.0];
/// let b = [0.0, 0.0, 1.0, 2.0, 3.0];
/// assert_eq!(dtw_distance(&a, &b, None), Some(0.0));
/// ```
pub fn dtw_distance<T: Float>(a: &[T], b: &[T], window: Option<usize>) -> Option<T> {
    let (n, m) = (a.len(), b.len());
    if n == 0 || m == 0 {
        return None;
    }
    let w = window.unwrap_or(n.max(m));
    if w < n.abs_diff(m) {
        return None;
    }
    let inf = T::infinity();
    let mut prev = vec![inf; m + 1];
    let mut cur = vec![inf; m + 1];
    prev[0] = T::zero();
    for i in 1..=n {
        cur.iter_mut().for_each(|c| *c = inf);
        let lo = i.saturating_sub(w).max(1);
        let hi = (i + w).min(m);
        for j in lo..=hi {
            let cost = (a[i - 1] - b[j - 1]).abs();
            cur[j] = cost + prev[j].min(cur[j - 1]).min(prev[j - 1]);
        }
        core::mem::swap(&mut prev, &mut cur);
    }
    let d = prev[m];
    d.is_finite().then_some(d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd;
    use approx::assert_relative_eq;

    #[test]
    fn matches_simd() {
        let a = [0.5, 1.0, 0.0, 3.0, 2.5, 0.0, 7.0];
        let b = [1.5, 0.0, 0.0, 2.0, 2.5, 1.0, 3.0];
        assert_relative_eq!(chebyshev_distance(&a, &b).unwrap(), simd::chebyshev_distance_simd(&a, &b).unwrap());
        assert_relative_eq!(canberra_distance(&a, &b).unwrap(), simd::canberra_distance_simd(&a, &b).unwrap(), epsilon = 1e-12);
        assert_relative_eq!(bray_curtis_distance(&a, &b).unwrap(), simd::bray_curtis_distance_simd(&a, &b).unwrap(), epsilon = 1e-12);
        assert_relative_eq!(chi_square_distance(&a, &b).unwrap(), simd::chi_square_distance_simd(&a, &b).unwrap(), epsilon = 1e-12);
        assert_relative_eq!(tanimoto_similarity(&a, &b).unwrap(), simd::tanimoto_similarity_simd(&a, &b).unwrap(), epsilon = 1e-12);
        assert_relative_eq!(ruzicka_similarity(&a, &b).unwrap(), simd::ruzicka_similarity_simd(&a, &b).unwrap(), epsilon = 1e-12);
        assert_relative_eq!(manhattan_distance(&a, &b).unwrap(), simd::manhattan_distance_simd(&a, &b).unwrap(), epsilon = 1e-12);
    }

    #[test]
    fn minkowski_orders() {
        let a = [1.0, -2.0, 3.0];
        let b = [0.0, 2.0, 1.0];
        assert_relative_eq!(minkowski_distance(&a, &b, 1.0).unwrap(), 7.0);
        assert_relative_eq!(minkowski_distance(&a, &b, 2.0).unwrap(), 21.0_f64.sqrt());
        assert_relative_eq!(minkowski_distance(&a, &b, 3.0).unwrap(), 73.0_f64.cbrt(), epsilon = 1e-12);
        assert_relative_eq!(minkowski_distance(&a, &b, f64::INFINITY).unwrap(), 4.0);
        assert!(minkowski_distance(&a, &b, 0.5).is_none());
        assert!(minkowski_distance(&a, &b[..2], 2.0).is_none());
    }

    #[test]
    fn weighted_and_mahalanobis() {
        let a = [1.0, 2.0];
        let b = [3.0, 6.0];
        assert_relative_eq!(weighted_euclidean_distance(&a, &b, &[1.0, 0.25]).unwrap(), 8.0_f64.sqrt());
        assert_relative_eq!(mahalanobis_diagonal_distance(&a, &b, &[4.0, 16.0]).unwrap(), 2.0_f64.sqrt());
        assert!(mahalanobis_diagonal_distance(&a, &b, &[0.0, 1.0]).is_none());
    }

    #[test]
    fn sequences() {
        assert_eq!(levenshtein_distance(b"", b"abc"), 3);
        assert_eq!(levenshtein_distance(b"flaw", b"lawn"), 2);
        assert_eq!(levenshtein_distance(&[1, 2, 3], &[1, 2, 3]), 0);
        assert_relative_eq!(levenshtein_similarity(b"abcd", b"abed"), 0.75);
        assert_relative_eq!(levenshtein_similarity::<u8>(b"", b""), 1.0);
        assert_eq!(longest_common_subsequence(b"", b"abc"), 0);
        assert_eq!(hamming_distance(&[true, false], &[true, true]), Some(1));
        assert_eq!(hamming_distance(b"ab", b"abc"), None);
    }

    #[test]
    fn bit_fingerprints() {
        assert_eq!(tanimoto_bits(&[0, 0], &[0, 0]), Some(1.0));
        assert_eq!(tanimoto_bits(&[u64::MAX], &[0]), Some(0.0));
        assert_eq!(hamming_distance_bits(&[u64::MAX, 0], &[0, 0]), Some(64));
        assert_eq!(tanimoto_bits(&[1], &[1, 2]), None);
    }

    #[test]
    fn dtw() {
        assert_eq!(dtw_distance(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], Some(0)), Some(0.0));
        assert_eq!(dtw_distance(&[0.0, 0.0], &[1.0], None), Some(2.0));
        assert_eq!(dtw_distance(&[0.0, 1.0, 2.0], &[0.0], Some(1)), None);
        assert_eq!(dtw_distance::<f64>(&[], &[1.0], None), None);
        // band 1 vs unconstrained on a shifted signal
        let a = [0.0, 0.0, 1.0, 2.0, 1.0, 0.0];
        let b = [0.0, 1.0, 2.0, 1.0, 0.0, 0.0];
        assert_eq!(dtw_distance(&a, &b, None), Some(0.0));
        assert_eq!(dtw_distance(&a, &b, Some(1)), Some(0.0));
        assert!(dtw_distance(&a, &b, Some(0)).unwrap() > 0.0);
    }
}
