//! Divergences and distances between discrete probability distributions.
//!
//! Inputs are non-negative weight vectors of equal length. They are
//! **normalised internally** to sum to one (like `scipy.stats.entropy`), so
//! raw counts or intensities can be passed directly. Every function returns
//! `None` on a length mismatch, an empty input, a zero total, or a negative
//! weight. Logarithms are natural (nats) unless stated otherwise.

use num_traits::Float;

fn normalized<T: Float>(p: &[T]) -> Option<Vec<T>> {
    if p.is_empty() || p.iter().any(|&x| x < T::zero() || x.is_nan()) {
        return None;
    }
    let s = p.iter().fold(T::zero(), |a, &x| a + x);
    if s == T::zero() || !s.is_finite() {
        return None;
    }
    Some(p.iter().map(|&x| x / s).collect())
}

fn normalized_pair<T: Float>(p: &[T], q: &[T]) -> Option<(Vec<T>, Vec<T>)> {
    if p.len() != q.len() {
        return None;
    }
    Some((normalized(p)?, normalized(q)?))
}

/// Shannon entropy H(p) = −Σ pᵢ ln pᵢ.
///
/// ```
/// use similarity::divergences::shannon_entropy;
/// let h = shannon_entropy(&[1.0, 1.0]).unwrap();
/// assert!((h - 2f64.ln()).abs() < 1e-12);
/// ```
pub fn shannon_entropy<T: Float>(p: &[T]) -> Option<T> {
    let p = normalized(p)?;
    Some(p.iter().filter(|&&x| x > T::zero()).fold(T::zero(), |h, &x| h - x * x.ln()))
}

/// Cross-entropy H(p, q) = −Σ pᵢ ln qᵢ. Infinite if q has zero mass where
/// p has positive mass.
pub fn cross_entropy<T: Float>(p: &[T], q: &[T]) -> Option<T> {
    let (p, q) = normalized_pair(p, q)?;
    Some(p.iter().zip(&q).filter(|(&a, _)| a > T::zero()).fold(T::zero(), |h, (&a, &b)| h - a * b.ln()))
}

/// Kullback–Leibler divergence D_KL(p ‖ q) = Σ pᵢ ln(pᵢ / qᵢ).
///
/// Infinite if q has zero mass where p has positive mass. Not symmetric.
///
/// ```
/// use similarity::divergences::kl_divergence;
/// assert_eq!(kl_divergence(&[0.5, 0.5], &[0.5, 0.5]), Some(0.0));
/// assert!(kl_divergence(&[1.0_f64, 0.0], &[0.0, 1.0]).unwrap().is_infinite());
/// ```
pub fn kl_divergence<T: Float>(p: &[T], q: &[T]) -> Option<T> {
    let (p, q) = normalized_pair(p, q)?;
    let mut d = T::zero();
    for (&a, &b) in p.iter().zip(&q) {
        if a > T::zero() {
            if b == T::zero() {
                return Some(T::infinity());
            }
            d = d + a * (a / b).ln();
        }
    }
    Some(d.max(T::zero()))
}

/// Jensen–Shannon divergence ½ D_KL(p ‖ m) + ½ D_KL(q ‖ m), m = (p + q)/2.
///
/// Symmetric, always finite, bounded by ln 2.
pub fn js_divergence<T: Float>(p: &[T], q: &[T]) -> Option<T> {
    let (p, q) = normalized_pair(p, q)?;
    let half = T::from(0.5).unwrap();
    let mut d = T::zero();
    for (&a, &b) in p.iter().zip(&q) {
        let m = (a + b) * half;
        if a > T::zero() {
            d = d + a * (a / m).ln();
        }
        if b > T::zero() {
            d = d + b * (b / m).ln();
        }
    }
    Some((d * half).max(T::zero()))
}

/// Jensen–Shannon distance √(JSD / ln 2) — a proper metric in `[0, 1]`.
///
/// ```
/// use similarity::divergences::js_distance;
/// assert!((js_distance(&[1.0, 0.0], &[0.0, 1.0]).unwrap() - 1.0_f64).abs() < 1e-12);
/// ```
pub fn js_distance<T: Float>(p: &[T], q: &[T]) -> Option<T> {
    let ln2 = T::from(core::f64::consts::LN_2).unwrap();
    js_divergence(p, q).map(|d| (d / ln2).sqrt().min(T::one()))
}

/// Bhattacharyya coefficient Σ √(pᵢ qᵢ) ∈ [0, 1].
pub fn bhattacharyya_coefficient<T: Float>(p: &[T], q: &[T]) -> Option<T> {
    let (p, q) = normalized_pair(p, q)?;
    Some(p.iter().zip(&q).fold(T::zero(), |s, (&a, &b)| s + (a * b).sqrt()).min(T::one()))
}

/// Bhattacharyya distance −ln(BC). Infinite for disjoint supports.
pub fn bhattacharyya_distance<T: Float>(p: &[T], q: &[T]) -> Option<T> {
    bhattacharyya_coefficient(p, q).map(|bc| -bc.ln())
}

/// Hellinger distance √(1 − BC) ∈ [0, 1].
///
/// ```
/// use similarity::divergences::hellinger_distance;
/// assert_eq!(hellinger_distance(&[1.0, 0.0], &[0.0, 1.0]), Some(1.0));
/// ```
pub fn hellinger_distance<T: Float>(p: &[T], q: &[T]) -> Option<T> {
    bhattacharyya_coefficient(p, q).map(|bc| (T::one() - bc).max(T::zero()).sqrt())
}

/// Total variation distance ½ Σ |pᵢ − qᵢ| ∈ [0, 1].
pub fn total_variation_distance<T: Float>(p: &[T], q: &[T]) -> Option<T> {
    let (p, q) = normalized_pair(p, q)?;
    let half = T::from(0.5).unwrap();
    Some(p.iter().zip(&q).fold(T::zero(), |s, (&a, &b)| s + (a - b).abs()) * half)
}

/// Rényi divergence of order α ≥ 0, α ≠ 1:
/// 1/(α−1) · ln Σ pᵢ^α qᵢ^(1−α). Returns KL divergence for α = 1.
pub fn renyi_divergence<T: Float>(p: &[T], q: &[T], alpha: T) -> Option<T> {
    if alpha < T::zero() || alpha.is_nan() {
        return None;
    }
    if alpha == T::one() {
        return kl_divergence(p, q);
    }
    let (p, q) = normalized_pair(p, q)?;
    let mut s = T::zero();
    for (&a, &b) in p.iter().zip(&q) {
        if a > T::zero() {
            if b == T::zero() {
                if alpha > T::one() {
                    return Some(T::infinity());
                }
                continue;
            }
            s = s + a.powf(alpha) * b.powf(T::one() - alpha);
        }
    }
    Some((s.ln() / (alpha - T::one())).max(T::zero()))
}

/// 1-D Wasserstein (earth mover's) distance between two histograms defined
/// on the same equally spaced bins of width `bin_width`:
/// `bin_width · Σ |CDF_p(i) − CDF_q(i)|`.
///
/// ```
/// use similarity::divergences::wasserstein_1d;
/// // all mass moves by two bins
/// assert_eq!(wasserstein_1d(&[1.0, 0.0, 0.0], &[0.0, 0.0, 1.0], 1.0), Some(2.0));
/// ```
pub fn wasserstein_1d<T: Float>(p: &[T], q: &[T], bin_width: T) -> Option<T> {
    let (p, q) = normalized_pair(p, q)?;
    let (mut cp, mut cq, mut d) = (T::zero(), T::zero(), T::zero());
    for (&a, &b) in p.iter().zip(&q) {
        cp = cp + a;
        cq = cq + b;
        d = d + (cp - cq).abs();
    }
    Some(d * bin_width)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn entropies() {
        assert_relative_eq!(shannon_entropy(&[0.1, 0.2, 0.3, 0.4]).unwrap(), 1.2798542258336676, epsilon = 1e-12);
        assert_relative_eq!(shannon_entropy(&[1.0, 2.0, 3.0, 4.0]).unwrap(), 1.2798542258336676, epsilon = 1e-12);
        let p = [0.1, 0.4, 0.5];
        let q = [0.3, 0.3, 0.4];
        let h = shannon_entropy(&p).unwrap();
        assert_relative_eq!(cross_entropy(&p, &q).unwrap() - h, kl_divergence(&p, &q).unwrap(), epsilon = 1e-12);
    }

    #[test]
    fn kl_and_js() {
        // closed form: Σ p ln(p/q)
        let p = [0.1, 0.4, 0.5];
        let q = [0.3, 0.3, 0.4];
        assert_relative_eq!(kl_divergence(&p, &q).unwrap(), 0.1 * (1.0f64 / 3.0).ln() + 0.4 * (4.0f64 / 3.0).ln() + 0.5 * 1.25f64.ln(), epsilon = 1e-12);
        let js = js_divergence(&p, &q).unwrap();
        assert!(js > 0.0 && js < 2f64.ln());
        assert_relative_eq!(js, js_divergence(&q, &p).unwrap(), epsilon = 1e-15);
        assert_relative_eq!(js_divergence(&[1.0, 0.0], &[0.0, 1.0]).unwrap(), 2f64.ln(), epsilon = 1e-12);
        assert_eq!(kl_divergence(&[1.0], &[1.0, 2.0]), None);
        assert_eq!(kl_divergence(&[-1.0, 2.0], &[1.0, 2.0]), None);
        assert_eq!(kl_divergence(&[0.0, 0.0], &[1.0, 2.0]), None);
    }

    #[test]
    fn bounded_distances() {
        let p = [0.2, 0.5, 0.3];
        assert_relative_eq!(hellinger_distance(&p, &p).unwrap(), 0.0, epsilon = 1e-7);
        assert_relative_eq!(total_variation_distance(&p, &[0.3, 0.5, 0.2]).unwrap(), 0.1, epsilon = 1e-12);
        assert_relative_eq!(bhattacharyya_coefficient(&p, &p).unwrap(), 1.0, epsilon = 1e-12);
        assert!(bhattacharyya_distance(&[1.0, 0.0], &[0.0, 1.0]).unwrap().is_infinite());
        assert_relative_eq!(js_distance(&p, &p).unwrap(), 0.0, epsilon = 1e-7);
    }

    #[test]
    fn renyi() {
        let p = [0.1, 0.4, 0.5];
        let q = [0.3, 0.3, 0.4];
        assert_relative_eq!(renyi_divergence(&p, &q, 1.0).unwrap(), kl_divergence(&p, &q).unwrap());
        // Rényi divergence is non-decreasing in alpha and approaches KL near 1.
        let near = renyi_divergence(&p, &q, 1.0 + 1e-6).unwrap();
        assert_relative_eq!(near, kl_divergence(&p, &q).unwrap(), epsilon = 1e-5);
        assert!(renyi_divergence(&p, &q, 0.5).unwrap() <= renyi_divergence(&p, &q, 2.0).unwrap());
        assert!(renyi_divergence(&p, &q, -1.0).is_none());
    }

    #[test]
    fn wasserstein() {
        assert_relative_eq!(wasserstein_1d(&[0.5, 0.5, 0.0], &[0.0, 0.5, 0.5], 2.0).unwrap(), 2.0);
        assert_eq!(wasserstein_1d(&[1.0, 2.0], &[1.0, 2.0], 1.0), Some(0.0));
    }
}
