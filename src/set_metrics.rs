//! Similarity coefficients for sets.
//!
//! Works on [`HashSet`]s and, much faster for large collections, on
//! **sorted, deduplicated slices** (e.g. lists of feature or fragment IDs),
//! which are compared with a linear merge instead of hashing.

use std::collections::HashSet;
use std::hash::Hash;

/// Size of the intersection and of both sets.
fn counts<T: Eq + Hash>(a: &HashSet<T>, b: &HashSet<T>) -> (usize, usize, usize) {
    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    let inter = small.iter().filter(|x| large.contains(x)).count();
    (inter, a.len(), b.len())
}

/// Sørensen–Dice coefficient 2|A ∩ B| / (|A| + |B|). Two empty sets give `1.0`.
///
/// ```
/// use similarity::set_metrics::dice_coefficient;
/// use std::collections::HashSet;
/// let a: HashSet<_> = [1, 2, 3].into();
/// let b: HashSet<_> = [2, 3, 4].into();
/// assert!((dice_coefficient(&a, &b) - 2.0 / 3.0).abs() < 1e-12);
/// ```
pub fn dice_coefficient<T: Eq + Hash>(a: &HashSet<T>, b: &HashSet<T>) -> f64 {
    let (i, na, nb) = counts(a, b);
    if na + nb == 0 {
        1.0
    } else {
        2.0 * i as f64 / (na + nb) as f64
    }
}

/// Overlap (Szymkiewicz–Simpson) coefficient |A ∩ B| / min(|A|, |B|).
/// `1.0` if either set is empty and the other is too, `0.0` if only one is.
pub fn overlap_coefficient<T: Eq + Hash>(a: &HashSet<T>, b: &HashSet<T>) -> f64 {
    let (i, na, nb) = counts(a, b);
    match na.min(nb) {
        0 if na == nb => 1.0,
        0 => 0.0,
        m => i as f64 / m as f64,
    }
}

/// Tversky index |A ∩ B| / (|A ∩ B| + α|A \ B| + β|B \ A|).
///
/// α = β = 1 gives Jaccard, α = β = ½ gives Dice. Two empty sets give `1.0`.
pub fn tversky_index<T: Eq + Hash>(a: &HashSet<T>, b: &HashSet<T>, alpha: f64, beta: f64) -> f64 {
    let (i, na, nb) = counts(a, b);
    let den = i as f64 + alpha * (na - i) as f64 + beta * (nb - i) as f64;
    if den == 0.0 {
        1.0
    } else {
        i as f64 / den
    }
}

/// Jaccard distance 1 − |A ∩ B| / |A ∪ B| (0 for two empty sets).
pub fn jaccard_distance<T: Eq + Hash>(a: &HashSet<T>, b: &HashSet<T>) -> f64 {
    let (i, na, nb) = counts(a, b);
    let u = na + nb - i;
    if u == 0 {
        0.0
    } else {
        1.0 - i as f64 / u as f64
    }
}

/// Intersection size of two **sorted, deduplicated** slices by linear merge.
pub fn sorted_intersection_size<T: Ord>(a: &[T], b: &[T]) -> usize {
    let (mut i, mut j, mut n) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            core::cmp::Ordering::Less => i += 1,
            core::cmp::Ordering::Greater => j += 1,
            core::cmp::Ordering::Equal => {
                n += 1;
                i += 1;
                j += 1;
            }
        }
    }
    n
}

/// Jaccard index of two **sorted, deduplicated** slices (no hashing).
/// Two empty slices give `0.0`, matching [`crate::jaccard_index`].
///
/// ```
/// use similarity::set_metrics::jaccard_index_sorted;
/// assert_eq!(jaccard_index_sorted(&[1, 2, 3, 5], &[2, 3, 4, 5]), 0.6);
/// ```
pub fn jaccard_index_sorted<T: Ord>(a: &[T], b: &[T]) -> f64 {
    let i = sorted_intersection_size(a, b);
    let u = a.len() + b.len() - i;
    if u == 0 {
        0.0
    } else {
        i as f64 / u as f64
    }
}

/// Dice coefficient of two **sorted, deduplicated** slices.
pub fn dice_coefficient_sorted<T: Ord>(a: &[T], b: &[T]) -> f64 {
    let i = sorted_intersection_size(a, b);
    if a.is_empty() && b.is_empty() {
        1.0
    } else {
        2.0 * i as f64 / (a.len() + b.len()) as f64
    }
}

/// Estimated Jaccard similarity from two MinHash signatures of equal
/// length: the fraction of matching positions. `None` on length mismatch
/// or empty input.
pub fn minhash_similarity<T: PartialEq>(sig_a: &[T], sig_b: &[T]) -> Option<f64> {
    if sig_a.len() != sig_b.len() || sig_a.is_empty() {
        return None;
    }
    let eq = sig_a.iter().zip(sig_b).filter(|(a, b)| a == b).count();
    Some(eq as f64 / sig_a.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(v: &[i32]) -> HashSet<i32> {
        v.iter().copied().collect()
    }

    #[test]
    fn coefficients() {
        let a = set(&[1, 2, 3, 4]);
        let b = set(&[3, 4, 5]);
        assert!((dice_coefficient(&a, &b) - 4.0 / 7.0).abs() < 1e-12);
        assert!((overlap_coefficient(&a, &b) - 2.0 / 3.0).abs() < 1e-12);
        assert!((tversky_index(&a, &b, 1.0, 1.0) - crate::jaccard_index(&a, &b)).abs() < 1e-12);
        assert!((tversky_index(&a, &b, 0.5, 0.5) - dice_coefficient(&a, &b)).abs() < 1e-12);
        assert!((jaccard_distance(&a, &b) - 0.6).abs() < 1e-12);
        let e = set(&[]);
        assert_eq!(dice_coefficient(&e, &e), 1.0);
        assert_eq!(overlap_coefficient(&a, &e), 0.0);
        assert_eq!(jaccard_distance(&e, &e), 0.0);
    }

    #[test]
    fn sorted_matches_hash() {
        let a = [1, 3, 5, 7, 9, 11];
        let b = [2, 3, 4, 5, 11, 12, 13];
        assert_eq!(sorted_intersection_size(&a, &b), 3);
        let jh = crate::jaccard_index(&set(&a), &set(&b));
        assert!((jaccard_index_sorted(&a, &b) - jh).abs() < 1e-12);
        assert!((dice_coefficient_sorted(&a, &b) - dice_coefficient(&set(&a), &set(&b))).abs() < 1e-12);
        assert_eq!(jaccard_index_sorted::<i32>(&[], &[]), 0.0);
    }

    #[test]
    fn minhash() {
        assert_eq!(minhash_similarity(&[1, 2, 3, 4], &[1, 9, 3, 9]), Some(0.5));
        assert_eq!(minhash_similarity::<u8>(&[], &[]), None);
    }
}
