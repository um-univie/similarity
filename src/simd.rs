//! # Portable-SIMD similarity, distance and statistics kernels
//!
//! Every function in this module is generic over [`SimdFloat`] (implemented
//! for `f32` and `f64`) and is accelerated with the [`pulp`] crate. At the
//! first call, pulp detects the best instruction set of the running CPU
//! (AVX-512, AVX2+FMA, NEON, or a scalar fallback) and every later call
//! dispatches to code compiled for exactly that instruction set. No
//! `target-cpu` flags or `unsafe` code are required from callers.
//!
//! ## Design
//!
//! * Kernels are described once as a `Fold` (reduction) or `Map`
//!   (element-wise update) and run by a shared engine.
//! * Reductions keep **4 independent accumulators** so out-of-order cores
//!   keep several SIMD units busy at once.
//! * Multiplications and additions are **fused** (FMA) where the hardware
//!   supports it: better throughput and a single rounding step.
//! * Metrics that need several sums (cosine, Pearson, Bray–Curtis, …) get
//!   them all in **one pass** over memory.
//! * The final partial register is padded with neutral values instead of
//!   using a scalar loop, so short inputs stay vectorised.
//!
//! ## Conventions
//!
//! * Functions taking two slices return `None` when their lengths differ.
//! * Results are computed in the input precision. Use `f64` input when
//!   summing many values and accuracy matters.
//!
//! ```
//! use similarity::simd::*;
//!
//! let a = [1.0_f32, 2.0, 3.0, 4.0];
//! let b = [4.0_f32, 3.0, 2.0, 1.0];
//! assert_eq!(dot_product_simd(&a, &b), Some(20.0));
//! assert_eq!(manhattan_distance_simd(&a, &b), Some(8.0));
//! assert_eq!(chebyshev_distance_simd(&a, &b), Some(3.0));
//! ```

mod element;
mod engine;
mod isa;
mod kernels;
mod math;
mod tile;

pub use isa::Isa;
pub(crate) use tile::{all_pairs, TileOp};

pub use element::SimdFloat;

use engine::{fold, fold_in, map};
use kernels::*;
use num_traits::Float;
use pulp::{Simd, WithSimd};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[inline(always)]
fn same_len<E>(a: &[E], b: &[E]) -> bool {
    a.len() == b.len()
}

#[inline(always)]
fn from_usize<E: SimdFloat>(n: usize) -> E {
    E::from_usize(n).unwrap_or_else(E::infinity)
}

// ===========================================================================
// Single-vector statistics
// ===========================================================================

/// Sum of all elements.
///
/// ```
/// use similarity::simd::sum_simd;
/// assert_eq!(sum_simd(&[1.0, 2.0, 3.0]), 6.0);
/// assert_eq!(sum_simd::<f32>(&[]), 0.0);
/// ```
pub fn sum_simd<E: SimdFloat>(x: &[E]) -> E {
    fold(Sum, [x])
}

/// Arithmetic mean. Returns `None` for an empty slice.
///
/// ```
/// use similarity::simd::mean_simd;
/// assert_eq!(mean_simd(&[1.0, 2.0, 3.0, 6.0]), Some(3.0));
/// ```
pub fn mean_simd<E: SimdFloat>(x: &[E]) -> Option<E> {
    if x.is_empty() {
        return None;
    }
    Some(sum_simd(x) / from_usize(x.len()))
}

/// Variance using an exact two-pass algorithm.
///
/// `ddof` is the "delta degrees of freedom": `0` gives the population
/// variance, `1` the unbiased sample variance. Returns `None` if
/// `x.len() <= ddof`.
///
/// ```
/// use similarity::simd::variance_simd;
/// let x = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
/// assert_eq!(variance_simd(&x, 0), Some(4.0));
/// ```
pub fn variance_simd<E: SimdFloat>(x: &[E], ddof: usize) -> Option<E> {
    if x.len() <= ddof {
        return None;
    }
    let mean = mean_simd(x)?;
    let ss = fold(SumSqDev { mean }, [x]);
    Some(ss / from_usize(x.len() - ddof))
}

/// Standard deviation (`sqrt` of [`variance_simd`]).
pub fn std_dev_simd<E: SimdFloat>(x: &[E], ddof: usize) -> Option<E> {
    variance_simd(x, ddof).map(Float::sqrt)
}

/// Minimum and maximum in a single pass. Returns `None` for an empty slice.
/// NaN handling follows the platform `min`/`max` instructions.
///
/// ```
/// use similarity::simd::min_max_simd;
/// assert_eq!(min_max_simd(&[3.0, -1.0, 8.0, 2.0, 0.5]), Some((-1.0, 8.0)));
/// ```
pub fn min_max_simd<E: SimdFloat>(x: &[E]) -> Option<(E, E)> {
    let first = *x.first()?;
    Some(fold(MinMax { first }, [x]))
}

/// L1 norm: Σ |xᵢ|.
pub fn l1_norm_simd<E: SimdFloat>(x: &[E]) -> E {
    fold(SumAbs, [x])
}

/// Squared L2 norm: Σ xᵢ².
pub fn squared_l2_norm_simd<E: SimdFloat>(x: &[E]) -> E {
    fold(SumSq, [x])
}

/// L2 (Euclidean) norm: √(Σ xᵢ²).
///
/// ```
/// use similarity::simd::l2_norm_simd;
/// assert_eq!(l2_norm_simd(&[3.0, 4.0]), 5.0);
/// ```
pub fn l2_norm_simd<E: SimdFloat>(x: &[E]) -> E {
    squared_l2_norm_simd(x).sqrt()
}

// ===========================================================================
// In-place / element-wise transforms
// ===========================================================================

/// Multiplies every element by `factor` in place.
pub fn scale_simd<E: SimdFloat>(x: &mut [E], factor: E) {
    map(Scale(factor), x, [])
}

/// `y ← y + alpha · x` (BLAS `axpy`). Returns `None` on a length mismatch.
///
/// ```
/// use similarity::simd::axpy_simd;
/// let mut y = vec![1.0, 1.0, 1.0];
/// axpy_simd(&mut y, 2.0, &[1.0, 2.0, 3.0]).unwrap();
/// assert_eq!(y, [3.0, 5.0, 7.0]);
/// ```
pub fn axpy_simd<E: SimdFloat>(y: &mut [E], alpha: E, x: &[E]) -> Option<()> {
    if y.len() != x.len() {
        return None;
    }
    map(Axpy(alpha), y, [x]);
    Some(())
}

/// Element-wise sum `a + b` into a new vector.
pub fn add_simd<E: SimdFloat>(a: &[E], b: &[E]) -> Option<Vec<E>> {
    if !same_len(a, b) {
        return None;
    }
    let mut out = vec![E::zero(); a.len()];
    map(Add2, &mut out, [a, b]);
    Some(out)
}

/// Element-wise difference `a − b` into a new vector.
pub fn sub_simd<E: SimdFloat>(a: &[E], b: &[E]) -> Option<Vec<E>> {
    if !same_len(a, b) {
        return None;
    }
    let mut out = vec![E::zero(); a.len()];
    map(Sub2, &mut out, [a, b]);
    Some(out)
}

/// Element-wise (Hadamard) product `a ⊙ b` into a new vector.
pub fn mul_simd<E: SimdFloat>(a: &[E], b: &[E]) -> Option<Vec<E>> {
    if !same_len(a, b) {
        return None;
    }
    let mut out = vec![E::zero(); a.len()];
    map(Mul2, &mut out, [a, b]);
    Some(out)
}

/// Clamps every element into `[lo, hi]` in place.
pub fn clamp_simd<E: SimdFloat>(x: &mut [E], lo: E, hi: E) {
    map(Clamp { lo, hi }, x, [])
}

/// Scales `x` in place to unit L2 norm. Returns `false` (and leaves `x`
/// untouched) if the norm is zero.
///
/// ```
/// use similarity::simd::{normalize_l2_simd, l2_norm_simd};
/// let mut x = vec![3.0, 4.0];
/// assert!(normalize_l2_simd(&mut x));
/// assert!((l2_norm_simd(&x) - 1.0_f64).abs() < 1e-12);
/// ```
pub fn normalize_l2_simd<E: SimdFloat>(x: &mut [E]) -> bool {
    let n = l2_norm_simd(x);
    if n == E::zero() || !n.is_finite() {
        return false;
    }
    scale_simd(x, E::one() / n);
    true
}

/// Scales `x` in place so its elements sum to one (probability
/// normalisation). Returns `false` if the sum is zero.
pub fn normalize_sum_simd<E: SimdFloat>(x: &mut [E]) -> bool {
    let s = sum_simd(x);
    if s == E::zero() || !s.is_finite() {
        return false;
    }
    scale_simd(x, E::one() / s);
    true
}

/// Rescales `x` in place to the range `[0, 1]` (min–max normalisation).
/// Returns `false` if `x` is empty or constant.
pub fn normalize_min_max_simd<E: SimdFloat>(x: &mut [E]) -> bool {
    let Some((lo, hi)) = min_max_simd(x) else {
        return false;
    };
    let range = hi - lo;
    if range == E::zero() || !range.is_finite() {
        return false;
    }
    map(ShiftScale { shift: lo, scale: E::one() / range }, x, []);
    true
}

/// Standardises `x` in place to zero mean and unit (population) standard
/// deviation. Returns `false` if `x` is empty or constant.
///
/// ```
/// use similarity::simd::{zscore_simd, mean_simd};
/// let mut x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// assert!(zscore_simd(&mut x));
/// assert!(mean_simd(&x).unwrap().abs() < 1e-12);
/// ```
pub fn zscore_simd<E: SimdFloat>(x: &mut [E]) -> bool {
    let Some(mean) = mean_simd(x) else {
        return false;
    };
    let sd = (fold(SumSqDev { mean }, [&*x]) / from_usize(x.len())).sqrt();
    if sd == E::zero() || !sd.is_finite() {
        return false;
    }
    map(ShiftScale { shift: mean, scale: E::one() / sd }, x, []);
    true
}

// ===========================================================================
// Inner-product family
// ===========================================================================

/// Dot product Σ aᵢbᵢ.
///
/// ```
/// use similarity::simd::dot_product_simd;
/// assert_eq!(dot_product_simd(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), Some(32.0));
/// ```
pub fn dot_product_simd<E: SimdFloat>(a: &[E], b: &[E]) -> Option<E> {
    same_len(a, b).then(|| fold(Dot, [a, b]))
}

/// Cosine similarity `a·b / (‖a‖‖b‖)`, computed in a single pass.
///
/// Returns `None` on a length mismatch or if either vector has zero norm.
///
/// ```
/// use similarity::simd::cosine_similarity_simd;
/// let s = cosine_similarity_simd(&[1.0, 0.0], &[1.0, 1.0]).unwrap();
/// assert!((s - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
/// ```
pub fn cosine_similarity_simd<E: SimdFloat>(a: &[E], b: &[E]) -> Option<E> {
    if !same_len(a, b) {
        return None;
    }
    let (dot, na, nb) = fold(DotNorms, [a, b]);
    let denom = (na * nb).sqrt();
    if denom == E::zero() {
        None
    } else {
        Some(dot / denom)
    }
}

/// Cosine distance `1 − cosine_similarity`.
pub fn cosine_distance_simd<E: SimdFloat>(a: &[E], b: &[E]) -> Option<E> {
    cosine_similarity_simd(a, b).map(|s| E::one() - s)
}

/// Angular distance `arccos(cosine_similarity) / π`, a proper metric in
/// `[0, 1]`.
pub fn angular_distance_simd<E: SimdFloat>(a: &[E], b: &[E]) -> Option<E> {
    let c = cosine_similarity_simd(a, b)?;
    let c = c.max(-E::one()).min(E::one());
    Some(c.acos() / E::from_f64_(core::f64::consts::PI))
}

/// Continuous Tanimoto (extended Jaccard) similarity
/// `a·b / (‖a‖² + ‖b‖² − a·b)`.
///
/// Equals the Jaccard index for binary 0/1 vectors. Returns `None` on a
/// length mismatch or if both vectors are zero.
///
/// ```
/// use similarity::simd::tanimoto_similarity_simd;
/// // binary fingerprints: |A ∩ B| = 2, |A ∪ B| = 4
/// let a = [1.0, 1.0, 1.0, 0.0];
/// let b = [0.0, 1.0, 1.0, 1.0];
/// assert_eq!(tanimoto_similarity_simd(&a, &b), Some(0.5));
/// ```
pub fn tanimoto_similarity_simd<E: SimdFloat>(a: &[E], b: &[E]) -> Option<E> {
    if !same_len(a, b) {
        return None;
    }
    let (dot, na, nb) = fold(DotNorms, [a, b]);
    let denom = na + nb - dot;
    if denom == E::zero() {
        None
    } else {
        Some(dot / denom)
    }
}

/// Continuous Dice (Sørensen) similarity `2·a·b / (‖a‖² + ‖b‖²)`.
pub fn dice_similarity_simd<E: SimdFloat>(a: &[E], b: &[E]) -> Option<E> {
    if !same_len(a, b) {
        return None;
    }
    let (dot, na, nb) = fold(DotNorms, [a, b]);
    let denom = na + nb;
    if denom == E::zero() {
        None
    } else {
        Some((dot + dot) / denom)
    }
}

// ===========================================================================
// Minkowski family
// ===========================================================================

/// Squared Euclidean distance Σ (aᵢ − bᵢ)².
pub fn squared_euclidean_distance_simd<E: SimdFloat>(a: &[E], b: &[E]) -> Option<E> {
    same_len(a, b).then(|| fold(SqEuclid, [a, b]))
}

/// Euclidean (L2) distance.
///
/// ```
/// use similarity::simd::euclidean_distance_simd;
/// assert_eq!(euclidean_distance_simd(&[0.0, 0.0], &[3.0, 4.0]), Some(5.0));
/// ```
pub fn euclidean_distance_simd<E: SimdFloat>(a: &[E], b: &[E]) -> Option<E> {
    squared_euclidean_distance_simd(a, b).map(Float::sqrt)
}

/// Manhattan (L1, city-block) distance Σ |aᵢ − bᵢ|.
pub fn manhattan_distance_simd<E: SimdFloat>(a: &[E], b: &[E]) -> Option<E> {
    same_len(a, b).then(|| fold(Manhattan, [a, b]))
}

/// Chebyshev (L∞) distance max |aᵢ − bᵢ|. Zero for empty input.
pub fn chebyshev_distance_simd<E: SimdFloat>(a: &[E], b: &[E]) -> Option<E> {
    same_len(a, b).then(|| fold(Chebyshev, [a, b]))
}

/// Weighted squared Euclidean distance Σ wᵢ (aᵢ − bᵢ)².
pub fn weighted_squared_euclidean_distance_simd<E: SimdFloat>(
    a: &[E],
    b: &[E],
    w: &[E],
) -> Option<E> {
    (same_len(a, b) && same_len(a, w)).then(|| fold(WeightedSqEuclid, [a, b, w]))
}

/// Weighted Euclidean distance √(Σ wᵢ (aᵢ − bᵢ)²).
pub fn weighted_euclidean_distance_simd<E: SimdFloat>(a: &[E], b: &[E], w: &[E]) -> Option<E> {
    weighted_squared_euclidean_distance_simd(a, b, w).map(Float::sqrt)
}

/// Weighted Manhattan distance Σ wᵢ |aᵢ − bᵢ|.
pub fn weighted_manhattan_distance_simd<E: SimdFloat>(a: &[E], b: &[E], w: &[E]) -> Option<E> {
    (same_len(a, b) && same_len(a, w)).then(|| fold(WeightedManhattan, [a, b, w]))
}

// ===========================================================================
// Ratio-based distances (common in ecology, chemistry and spectroscopy)
// ===========================================================================

/// Canberra distance Σ |aᵢ − bᵢ| / (|aᵢ| + |bᵢ|); `0/0` terms count as 0.
///
/// ```
/// use similarity::simd::canberra_distance_simd;
/// assert_eq!(canberra_distance_simd(&[1.0, 0.0], &[3.0, 0.0]), Some(0.5));
/// ```
pub fn canberra_distance_simd<E: SimdFloat>(a: &[E], b: &[E]) -> Option<E> {
    same_len(a, b).then(|| fold(Canberra, [a, b]))
}

/// Bray–Curtis dissimilarity Σ |aᵢ − bᵢ| / Σ |aᵢ + bᵢ|.
///
/// Returns `None` on a length mismatch or if the denominator is zero.
pub fn bray_curtis_distance_simd<E: SimdFloat>(a: &[E], b: &[E]) -> Option<E> {
    if !same_len(a, b) {
        return None;
    }
    let (num, den) = fold(BrayCurtis, [a, b]);
    if den == E::zero() {
        None
    } else {
        Some(num / den)
    }
}

/// Chi-squared distance Σ (aᵢ − bᵢ)² / (aᵢ + bᵢ); `0/0` terms count as 0.
/// Intended for non-negative data such as histograms.
pub fn chi_square_distance_simd<E: SimdFloat>(a: &[E], b: &[E]) -> Option<E> {
    same_len(a, b).then(|| fold(ChiSquare, [a, b]))
}

/// Histogram intersection Σ min(aᵢ, bᵢ).
pub fn histogram_intersection_simd<E: SimdFloat>(a: &[E], b: &[E]) -> Option<E> {
    same_len(a, b).then(|| fold(MinMaxSums, [a, b]).0)
}

/// Ruzicka similarity (weighted Jaccard) Σ min(aᵢ, bᵢ) / Σ max(aᵢ, bᵢ).
///
/// Returns `None` on a length mismatch or if Σ max is zero.
///
/// ```
/// use similarity::simd::ruzicka_similarity_simd;
/// assert_eq!(ruzicka_similarity_simd(&[1.0, 2.0, 0.0], &[2.0, 2.0, 1.0]), Some(0.6));
/// ```
pub fn ruzicka_similarity_simd<E: SimdFloat>(a: &[E], b: &[E]) -> Option<E> {
    if !same_len(a, b) {
        return None;
    }
    let (mn, mx) = fold(MinMaxSums, [a, b]);
    if mx == E::zero() {
        None
    } else {
        Some(mn / mx)
    }
}

/// Number of positions where `a[i] != b[i]` (Hamming distance).
///
/// ```
/// use similarity::simd::hamming_distance_simd;
/// assert_eq!(hamming_distance_simd(&[1.0, 2.0, 3.0], &[1.0, 0.0, 3.0]), Some(1));
/// ```
pub fn hamming_distance_simd<E: SimdFloat>(a: &[E], b: &[E]) -> Option<usize> {
    same_len(a, b).then(|| {
        let c: E = fold(NotEqualCount, [a, b]);
        c.to_f64_().round() as usize
    })
}

// ===========================================================================
// Correlation and error metrics
// ===========================================================================

#[inline]
fn co_moments<E: SimdFloat>(a: &[E], b: &[E]) -> (E, E, E) {
    let n = from_usize::<E>(a.len());
    let (sa, sb) = fold(Sum2, [a, b]);
    fold(
        CoMoments {
            mean_a: sa / n,
            mean_b: sb / n,
        },
        [a, b],
    )
}

/// Pearson correlation coefficient *r* ∈ [−1, 1].
///
/// Uses an exact two-pass algorithm: one SIMD pass for both means and one
/// fused pass for the covariance and both variances. Returns `None` on a
/// length mismatch, for empty input, or if either input is constant.
///
/// ```
/// use similarity::simd::pearson_correlation_simd;
/// let r = pearson_correlation_simd(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]).unwrap();
/// assert!((r - 1.0_f64).abs() < 1e-12);
/// ```
pub fn pearson_correlation_simd<E: SimdFloat>(a: &[E], b: &[E]) -> Option<E> {
    if !same_len(a, b) || a.is_empty() {
        return None;
    }
    let (num, va, vb) = co_moments(a, b);
    let den = (va * vb).sqrt();
    if den == E::zero() {
        None
    } else {
        Some((num / den).max(-E::one()).min(E::one()))
    }
}

/// Pearson correlation **distance**, defined as the negative Pearson *r*
/// (same convention as [`crate::pearson_correlation_distance`]).
///
/// Returns `None` on a length mismatch or when exactly one denominator
/// term vanishes with a non-zero numerator; returns `Some(0.0)` when both
/// inputs are constant.
pub fn pearson_correlation_distance_simd<E: SimdFloat>(a: &[E], b: &[E]) -> Option<E> {
    if !same_len(a, b) {
        return None;
    }
    if a.is_empty() {
        return Some(E::zero());
    }
    let (num, va, vb) = co_moments(a, b);
    let den = (va * vb).sqrt();
    if den == E::zero() {
        if num == E::zero() {
            Some(E::zero())
        } else {
            None
        }
    } else {
        Some(-num / den)
    }
}

/// Population covariance. Returns `None` on a length mismatch or for empty
/// input.
pub fn covariance_simd<E: SimdFloat>(a: &[E], b: &[E]) -> Option<E> {
    if !same_len(a, b) || a.is_empty() {
        return None;
    }
    Some(co_moments(a, b).0 / from_usize(a.len()))
}

/// Mean squared error.
pub fn mean_squared_error_simd<E: SimdFloat>(actual: &[E], predicted: &[E]) -> Option<E> {
    if actual.is_empty() {
        return None;
    }
    squared_euclidean_distance_simd(actual, predicted).map(|s| s / from_usize(actual.len()))
}

/// Root mean squared error.
pub fn root_mean_squared_error_simd<E: SimdFloat>(actual: &[E], predicted: &[E]) -> Option<E> {
    mean_squared_error_simd(actual, predicted).map(Float::sqrt)
}

/// Mean absolute error.
pub fn mean_absolute_error_simd<E: SimdFloat>(actual: &[E], predicted: &[E]) -> Option<E> {
    if actual.is_empty() {
        return None;
    }
    manhattan_distance_simd(actual, predicted).map(|s| s / from_usize(actual.len()))
}

/// Coefficient of determination R² = 1 − SS_res / SS_tot.
///
/// Returns `None` on a length mismatch, empty input, or constant `actual`.
pub fn r_squared_simd<E: SimdFloat>(actual: &[E], predicted: &[E]) -> Option<E> {
    if !same_len(actual, predicted) || actual.is_empty() {
        return None;
    }
    let mean = mean_simd(actual)?;
    let ss_tot = fold(SumSqDev { mean }, [actual]);
    if ss_tot == E::zero() {
        return None;
    }
    let ss_res = fold(SqEuclid, [actual, predicted]);
    Some(E::one() - ss_res / ss_tot)
}

// ===========================================================================
// Vectorised elementary functions
// ===========================================================================

/// Natural logarithm of every element, in place (≈1 ulp).
/// `ln 0 = −∞`, negative inputs and NaN give NaN.
///
/// ```
/// use similarity::simd::ln_simd;
/// let mut x = vec![1.0, std::f64::consts::E, 0.5];
/// ln_simd(&mut x);
/// assert!((x[1] - 1.0).abs() < 1e-15 && (x[2] + 2f64.ln()).abs() < 1e-15);
/// ```
pub fn ln_simd<E: SimdFloat>(x: &mut [E]) {
    map(Ln, x, [])
}

/// `e^x` of every element, in place (≈1–2 ulp). Results below about
/// `1e-307` (`f64`) / `2e-38` (`f32`) flush to zero.
pub fn exp_simd<E: SimdFloat>(x: &mut [E]) {
    map(Exp, x, [])
}

/// Square root of every element, in place.
pub fn sqrt_simd<E: SimdFloat>(x: &mut [E]) {
    map(Sqrt, x, [])
}

/// `x^y` of every non-negative element, in place (`exp(y·ln x)`).
/// Negative bases give NaN; `0^y` is `0` for `y > 0`, `1` for `y = 0`.
///
/// ```
/// use similarity::simd::pow_simd;
/// let mut x = vec![4.0_f32, 9.0, 0.0];
/// pow_simd(&mut x, 0.5);
/// assert!((x[0] - 2.0).abs() < 1e-6 && (x[1] - 3.0).abs() < 1e-6 && x[2] == 0.0);
/// ```
pub fn pow_simd<E: SimdFloat>(x: &mut [E], y: E) {
    map(Pow { y }, x, [])
}

// ===========================================================================
// Entropy and divergences (inputs: non-negative weights, normalised internally)
// ===========================================================================

#[inline]
fn valid_total<E: SimdFloat>(s: E) -> bool {
    s > E::zero() && s.is_finite()
}

/// Shannon entropy (nats) of the distribution proportional to `x`.
///
/// One SIMD pass computes Σx and Σx·ln x; the entropy follows as
/// `ln S − (Σ x ln x)/S` without a separate normalisation pass. Returns
/// `None` for empty input, a zero total or a negative weight.
///
/// ```
/// use similarity::simd::shannon_entropy_simd;
/// let h = shannon_entropy_simd(&[1.0, 2.0, 3.0, 4.0]).unwrap();
/// assert!((h - 1.2798542258336676_f64).abs() < 1e-14);
/// ```
pub fn shannon_entropy_simd<E: SimdFloat>(x: &[E]) -> Option<E> {
    let (sum, xlx, neg) = fold(XLogX, [x]);
    if neg > E::zero() || !valid_total(sum) {
        return None;
    }
    Some((sum.ln() - xlx / sum).max(E::zero()))
}

/// Kullback–Leibler divergence D_KL(p ‖ q) of the distributions
/// proportional to `p` and `q`, in one SIMD pass. `+∞` when `q` has zero
/// mass where `p` has positive mass. `None` on a length mismatch, empty
/// input, zero total or negative weight.
pub fn kl_divergence_simd<E: SimdFloat>(p: &[E], q: &[E]) -> Option<E> {
    if !same_len(p, q) {
        return None;
    }
    let [sp, sq, kl, _ce, bad, neg] = fold(KlTerms, [p, q]);
    if neg > E::zero() || !valid_total(sp) || !valid_total(sq) {
        return None;
    }
    if bad > E::zero() {
        return Some(E::infinity());
    }
    Some((kl / sp + (sq / sp).ln()).max(E::zero()))
}

/// Cross-entropy H(p, q) = −Σ pᵢ ln qᵢ of the normalised distributions.
pub fn cross_entropy_simd<E: SimdFloat>(p: &[E], q: &[E]) -> Option<E> {
    if !same_len(p, q) {
        return None;
    }
    let [sp, sq, _kl, ce, bad, neg] = fold(KlTerms, [p, q]);
    if neg > E::zero() || !valid_total(sp) || !valid_total(sq) {
        return None;
    }
    if bad > E::zero() {
        return Some(E::infinity());
    }
    Some(sq.ln() - ce / sp)
}

/// Jensen–Shannon divergence (nats, ≤ ln 2) of the normalised
/// distributions.
pub fn js_divergence_simd<E: SimdFloat>(p: &[E], q: &[E]) -> Option<E> {
    if !same_len(p, q) {
        return None;
    }
    let [sp, sq, neg] = fold(SumsNeg, [p, q]);
    if neg > E::zero() || !valid_total(sp) || !valid_total(sq) {
        return None;
    }
    let t = fold(JsTerms { sa: E::one() / sp, sb: E::one() / sq }, [p, q]);
    Some((t * E::from_f64_(0.5)).max(E::zero()))
}

/// Jensen–Shannon distance √(JSD / ln 2) ∈ [0, 1] (a metric).
pub fn js_distance_simd<E: SimdFloat>(p: &[E], q: &[E]) -> Option<E> {
    js_divergence_simd(p, q).map(|d| (d / E::from_f64_(core::f64::consts::LN_2)).sqrt().min(E::one()))
}

/// Bhattacharyya coefficient Σ √(pᵢ qᵢ) of the normalised distributions,
/// computed in a single SIMD pass as Σ√(pq) / √(Σp·Σq).
pub fn bhattacharyya_coefficient_simd<E: SimdFloat>(p: &[E], q: &[E]) -> Option<E> {
    if !same_len(p, q) {
        return None;
    }
    let [sp, sq, root, neg] = fold(BhattacharyyaTerms, [p, q]);
    if neg > E::zero() || !valid_total(sp) || !valid_total(sq) {
        return None;
    }
    Some((root / (sp * sq).sqrt()).min(E::one()))
}

/// Bhattacharyya distance −ln(BC).
pub fn bhattacharyya_distance_simd<E: SimdFloat>(p: &[E], q: &[E]) -> Option<E> {
    bhattacharyya_coefficient_simd(p, q).map(|bc| -bc.ln())
}

/// Hellinger distance √(1 − BC) ∈ [0, 1].
pub fn hellinger_distance_simd<E: SimdFloat>(p: &[E], q: &[E]) -> Option<E> {
    bhattacharyya_coefficient_simd(p, q).map(|bc| (E::one() - bc).max(E::zero()).sqrt())
}

/// Total variation distance ½ Σ |pᵢ − qᵢ| of the normalised distributions.
pub fn total_variation_distance_simd<E: SimdFloat>(p: &[E], q: &[E]) -> Option<E> {
    if !same_len(p, q) {
        return None;
    }
    let [sp, sq, neg] = fold(SumsNeg, [p, q]);
    if neg > E::zero() || !valid_total(sp) || !valid_total(sq) {
        return None;
    }
    let l1 = fold(ScaledL1 { sa: E::one() / sp, sb: E::one() / sq }, [p, q]);
    Some(l1 * E::from_f64_(0.5))
}

// ===========================================================================
// Spectral entropy (Li et al., Nature Methods 2021)
// ===========================================================================

/// Entropy-based intensity weighting used by spectral entropy similarity.
///
/// Returns `(entropy, weights)` where `weights` sum to one. If the
/// spectrum's entropy is below 3 the intensities are raised to the power
/// `0.25 + 0.25·S` (via SIMD `pow`) and renormalised; otherwise they are
/// only normalised. Returns `None` for empty input, a zero total or a
/// negative intensity.
///
/// Matches [`crate::Spectrum::get_entropy_and_weighted_intensity`] up to
/// floating-point rounding (that method returns un-normalised intensities
/// in the high-entropy branch).
pub fn spectral_entropy_weights_simd<E: SimdFloat>(intensities: &[E]) -> Option<(E, Vec<E>)> {
    let entropy = shannon_entropy_simd(intensities)?;
    let mut w = intensities.to_vec();
    let cutoff = E::from_f64_(3.0);
    if entropy < cutoff {
        let weight = E::from_f64_(0.25) + E::from_f64_(0.25) * entropy;
        pow_simd(&mut w, weight);
        if !normalize_sum_simd(&mut w) {
            return None;
        }
        let e = shannon_entropy_simd(&w)?;
        Some((e, w))
    } else {
        normalize_sum_simd(&mut w);
        Some((entropy, w))
    }
}

/// Spectral entropy similarity of two **aligned** intensity vectors
/// (`a[i]` and `b[i]` belong to the same m/z), in `[0, 1]`.
///
/// For spectra whose peaks are not aligned yet, use
/// [`crate::calculate_entropy_similarity_matched`] or
/// [`crate::flash_entropy::FlashEntropyIndex`]. Returns `None` on a length
/// mismatch, empty input, zero total or negative intensity.
///
/// ```
/// use similarity::simd::entropy_similarity_simd;
/// let s = entropy_similarity_simd(&[1.0, 2.0, 0.0], &[1.0, 2.0, 0.0]).unwrap();
/// assert!((s - 1.0_f64).abs() < 1e-12);
/// ```
pub fn entropy_similarity_simd<E: SimdFloat>(a: &[E], b: &[E]) -> Option<E> {
    if !same_len(a, b) {
        return None;
    }
    let (ea, wa) = spectral_entropy_weights_simd(a)?;
    let (eb, wb) = spectral_entropy_weights_simd(b)?;
    let merged = add_simd(&wa, &wb)?;
    let em = shannon_entropy_simd(&merged)?;
    let s = E::one() - (E::from_f64_(2.0) * em - ea - eb) / E::from_f64_(4.0f64.ln());
    Some(s.max(E::zero()).min(E::one()))
}

/// Writes `(a + b)·ln(a + b) − a·ln a − b·ln b` for every `b` into `out`
/// (`blnb[i]` must hold `b[i]·ln b[i]`). Used by flash entropy search.
pub(crate) fn entropy_gain_simd<E: SimdFloat>(a: E, b: &[E], blnb: &[E], out: &mut [E]) {
    let alna = if a > E::zero() { a * a.ln() } else { E::zero() };
    map(EntropyGain { a, alna }, out, [b, blnb])
}

// ===========================================================================
// Cross-correlation
// ===========================================================================

struct CrossCorrelate<'a, E> {
    x: &'a [E],
    y: &'a [E],
    out: &'a mut [E],
    lag_offset: usize,
}

impl<'a, E: SimdFloat> WithSimd for CrossCorrelate<'a, E> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, s: S) {
        let (x, y) = (self.x, self.y);
        let (x_len, y_len) = (x.len(), y.len());
        for (k, out) in self.out.iter_mut().enumerate() {
            let lag = self.lag_offset + k;
            let x_start = lag.saturating_sub(y_len - 1);
            let y_start = (y_len - 1).saturating_sub(lag);
            let n = (x_len - x_start).min(y_len - y_start);
            let acc = fold_in(s, &Dot, [&x[x_start..x_start + n], &y[y_start..y_start + n]]);
            *out = <Dot as engine::Fold<E, 2>>::finish(&Dot, s, acc);
        }
    }
}

/// Full linear cross-correlation (`numpy.correlate(x, y, "full")`).
///
/// Output has length `x.len() + y.len() − 1`; element `k` is the dot
/// product of the overlap of `x` and `y` shifted by `k − (y.len() − 1)`.
/// Each overlap is a SIMD dot product; instruction-set dispatch happens
/// once for the whole call. Returns an empty `Vec` if either input is
/// empty.
///
/// ```
/// use similarity::simd::cross_correlate_simd;
/// let r = cross_correlate_simd(&[1.0, 2.0, 3.0], &[0.0, 1.0, 0.5]);
/// assert_eq!(r, vec![0.5, 2.0, 3.5, 3.0, 0.0]);
/// ```
pub fn cross_correlate_simd<E: SimdFloat>(x: &[E], y: &[E]) -> Vec<E> {
    if x.is_empty() || y.is_empty() {
        return Vec::new();
    }
    let mut out = vec![E::zero(); x.len() + y.len() - 1];
    isa::dispatch(CrossCorrelate {
        x,
        y,
        out: &mut out,
        lag_offset: 0,
    });
    out
}

/// Parallel + SIMD cross-correlation: lags are split into chunks processed
/// on the rayon thread pool, each chunk using the SIMD kernel.
#[cfg(feature = "parallel")]
pub fn cross_correlate_simd_parallel<E: SimdFloat>(x: &[E], y: &[E]) -> Vec<E> {
    if x.is_empty() || y.is_empty() {
        return Vec::new();
    }
    let total = x.len() + y.len() - 1;
    let mut out = vec![E::zero(); total];
    let chunk = total.div_ceil(rayon::current_num_threads() * 4).max(64);
    out.par_chunks_mut(chunk).enumerate().for_each(|(ci, part)| {
        isa::dispatch(CrossCorrelate {
            x,
            y,
            out: part,
            lag_offset: ci * chunk,
        });
    });
    out
}

/// Index of the maximum of the SIMD cross-correlation (the lag with the
/// best alignment, using the same indexing as
/// [`crate::find_time_shift`]). Returns `None` if either input is empty.
pub fn find_time_shift_simd<E: SimdFloat>(x: &[E], y: &[E]) -> Option<usize> {
    let cc = cross_correlate_simd(x, y);
    argmax(&cc)
}

/// Index of the largest non-NaN value.
pub(crate) fn argmax<E: Float>(v: &[E]) -> Option<usize> {
    let mut best: Option<(usize, E)> = None;
    for (i, &x) in v.iter().enumerate() {
        if x.is_nan() {
            continue;
        }
        match best {
            Some((_, b)) if x <= b => {}
            _ => best = Some((i, x)),
        }
    }
    best.map(|(i, _)| i)
}

#[cfg(test)]
mod legacy_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_math;
