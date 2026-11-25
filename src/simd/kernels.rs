//! SIMD kernels expressed with the [`Fold`] / [`Map`] engine.

use super::element::SimdFloat;
use super::engine::{Fold, Map};
use pulp::Simd;

/// Declares a single-accumulator fold summing `$step` over `N` inputs.
macro_rules! sum_fold {
    ($(#[$m:meta])* $name:ident, $n:literal, |$s:ident, $x:ident| $step:expr) => {
        $(#[$m])*
        #[derive(Clone, Copy)]
        pub(crate) struct $name;

        impl<E: SimdFloat> Fold<E, $n> for $name {
            type Acc<S: Simd> = E::V<S>;
            type Out = E;

            #[inline(always)]
            fn init<S: Simd>(&self, s: S) -> E::V<S> {
                E::v_splat(s, E::zero())
            }
            #[inline(always)]
            fn step<S: Simd>(&self, $s: S, acc: E::V<S>, $x: [E::V<S>; $n]) -> E::V<S> {
                let v: E::V<S> = $step;
                E::v_add($s, acc, v)
            }
            #[inline(always)]
            fn merge<S: Simd>(&self, s: S, a: E::V<S>, b: E::V<S>) -> E::V<S> {
                E::v_add(s, a, b)
            }
            #[inline(always)]
            fn finish<S: Simd>(&self, s: S, acc: E::V<S>) -> E {
                E::v_reduce_sum(s, acc)
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Unary reductions
// ---------------------------------------------------------------------------

sum_fold!(
    /// Σ xᵢ
    Sum, 1, |s, x| { let _ = s; x[0] }
);
sum_fold!(
    /// Σ xᵢ²
    SumSq, 1, |s, x| E::v_mul(s, x[0], x[0])
);
sum_fold!(
    /// Σ |xᵢ|
    SumAbs, 1, |s, x| E::v_abs(s, x[0])
);

/// Σ (xᵢ − μ)²
#[derive(Clone, Copy)]
pub(crate) struct SumSqDev<E> {
    pub mean: E,
}

impl<E: SimdFloat> Fold<E, 1> for SumSqDev<E> {
    type Acc<S: Simd> = E::V<S>;
    type Out = E;

    #[inline(always)]
    fn init<S: Simd>(&self, s: S) -> E::V<S> {
        E::v_splat(s, E::zero())
    }
    #[inline(always)]
    fn step<S: Simd>(&self, s: S, acc: E::V<S>, x: [E::V<S>; 1]) -> E::V<S> {
        let d = E::v_sub(s, x[0], E::v_splat(s, self.mean));
        E::v_mul_add(s, d, d, acc)
    }
    #[inline(always)]
    fn merge<S: Simd>(&self, s: S, a: E::V<S>, b: E::V<S>) -> E::V<S> {
        E::v_add(s, a, b)
    }
    #[inline(always)]
    fn finish<S: Simd>(&self, s: S, acc: E::V<S>) -> E {
        E::v_reduce_sum(s, acc)
    }
    #[inline(always)]
    fn pad(&self, _i: usize) -> E {
        self.mean
    }
}

/// (min xᵢ, max xᵢ). Padding uses the first element so it never wins.
#[derive(Clone, Copy)]
pub(crate) struct MinMax<E> {
    pub first: E,
}

impl<E: SimdFloat> Fold<E, 1> for MinMax<E> {
    type Acc<S: Simd> = (E::V<S>, E::V<S>);
    type Out = (E, E);

    #[inline(always)]
    fn init<S: Simd>(&self, s: S) -> Self::Acc<S> {
        let f = E::v_splat(s, self.first);
        (f, f)
    }
    #[inline(always)]
    fn step<S: Simd>(&self, s: S, acc: Self::Acc<S>, x: [E::V<S>; 1]) -> Self::Acc<S> {
        (E::v_min(s, acc.0, x[0]), E::v_max(s, acc.1, x[0]))
    }
    #[inline(always)]
    fn merge<S: Simd>(&self, s: S, a: Self::Acc<S>, b: Self::Acc<S>) -> Self::Acc<S> {
        (E::v_min(s, a.0, b.0), E::v_max(s, a.1, b.1))
    }
    #[inline(always)]
    fn finish<S: Simd>(&self, s: S, acc: Self::Acc<S>) -> (E, E) {
        (E::v_reduce_min(s, acc.0), E::v_reduce_max(s, acc.1))
    }
    #[inline(always)]
    fn pad(&self, _i: usize) -> E {
        self.first
    }
}

// ---------------------------------------------------------------------------
// Binary reductions
// ---------------------------------------------------------------------------

/// Σ aᵢ·bᵢ (fused multiply-add accumulation).
#[derive(Clone, Copy)]
pub(crate) struct Dot;

impl<E: SimdFloat> Fold<E, 2> for Dot {
    type Acc<S: Simd> = E::V<S>;
    type Out = E;

    #[inline(always)]
    fn init<S: Simd>(&self, s: S) -> E::V<S> {
        E::v_splat(s, E::zero())
    }
    #[inline(always)]
    fn step<S: Simd>(&self, s: S, acc: E::V<S>, x: [E::V<S>; 2]) -> E::V<S> {
        E::v_mul_add(s, x[0], x[1], acc)
    }
    #[inline(always)]
    fn merge<S: Simd>(&self, s: S, a: E::V<S>, b: E::V<S>) -> E::V<S> {
        E::v_add(s, a, b)
    }
    #[inline(always)]
    fn finish<S: Simd>(&self, s: S, acc: E::V<S>) -> E {
        E::v_reduce_sum(s, acc)
    }
}

/// Σ (aᵢ − bᵢ)²
#[derive(Clone, Copy)]
pub(crate) struct SqEuclid;

impl<E: SimdFloat> Fold<E, 2> for SqEuclid {
    type Acc<S: Simd> = E::V<S>;
    type Out = E;

    #[inline(always)]
    fn init<S: Simd>(&self, s: S) -> E::V<S> {
        E::v_splat(s, E::zero())
    }
    #[inline(always)]
    fn step<S: Simd>(&self, s: S, acc: E::V<S>, x: [E::V<S>; 2]) -> E::V<S> {
        let d = E::v_sub(s, x[0], x[1]);
        E::v_mul_add(s, d, d, acc)
    }
    #[inline(always)]
    fn merge<S: Simd>(&self, s: S, a: E::V<S>, b: E::V<S>) -> E::V<S> {
        E::v_add(s, a, b)
    }
    #[inline(always)]
    fn finish<S: Simd>(&self, s: S, acc: E::V<S>) -> E {
        E::v_reduce_sum(s, acc)
    }
}

sum_fold!(
    /// Σ |aᵢ − bᵢ|
    Manhattan, 2, |s, x| E::v_abs(s, E::v_sub(s, x[0], x[1]))
);

sum_fold!(
    /// Σ |aᵢ − bᵢ| / (|aᵢ| + |bᵢ|), terms with a zero denominator count as 0.
    Canberra, 2, |s, x| {
        let num = E::v_abs(s, E::v_sub(s, x[0], x[1]));
        let den = E::v_add(s, E::v_abs(s, x[0]), E::v_abs(s, x[1]));
        let zero = E::v_splat(s, E::zero());
        let is_zero = E::v_eq(s, den, zero);
        let safe_den = E::v_select(s, is_zero, E::v_splat(s, E::one()), den);
        E::v_select(s, is_zero, zero, E::v_div(s, num, safe_den))
    }
);

sum_fold!(
    /// Σ (aᵢ − bᵢ)² / (aᵢ + bᵢ), terms with a zero denominator count as 0.
    ChiSquare, 2, |s, x| {
        let d = E::v_sub(s, x[0], x[1]);
        let den = E::v_add(s, x[0], x[1]);
        let zero = E::v_splat(s, E::zero());
        let is_zero = E::v_eq(s, den, zero);
        let safe_den = E::v_select(s, is_zero, E::v_splat(s, E::one()), den);
        E::v_select(s, is_zero, zero, E::v_div(s, E::v_mul(s, d, d), safe_den))
    }
);

sum_fold!(
    /// Number of positions where aᵢ ≠ bᵢ (returned as a float count).
    NotEqualCount, 2, |s, x| {
        let eq = E::v_eq(s, x[0], x[1]);
        E::v_select(s, eq, E::v_splat(s, E::zero()), E::v_splat(s, E::one()))
    }
);

sum_fold!(
    /// Σ wᵢ (aᵢ − bᵢ)²
    WeightedSqEuclid, 3, |s, x| {
        let d = E::v_sub(s, x[0], x[1]);
        E::v_mul(s, x[2], E::v_mul(s, d, d))
    }
);

sum_fold!(
    /// Σ wᵢ |aᵢ − bᵢ|
    WeightedManhattan, 3, |s, x| E::v_mul(s, x[2], E::v_abs(s, E::v_sub(s, x[0], x[1])))
);

/// max |aᵢ − bᵢ|
#[derive(Clone, Copy)]
pub(crate) struct Chebyshev;

impl<E: SimdFloat> Fold<E, 2> for Chebyshev {
    type Acc<S: Simd> = E::V<S>;
    type Out = E;

    #[inline(always)]
    fn init<S: Simd>(&self, s: S) -> E::V<S> {
        E::v_splat(s, E::zero())
    }
    #[inline(always)]
    fn step<S: Simd>(&self, s: S, acc: E::V<S>, x: [E::V<S>; 2]) -> E::V<S> {
        E::v_max(s, acc, E::v_abs(s, E::v_sub(s, x[0], x[1])))
    }
    #[inline(always)]
    fn merge<S: Simd>(&self, s: S, a: E::V<S>, b: E::V<S>) -> E::V<S> {
        E::v_max(s, a, b)
    }
    #[inline(always)]
    fn finish<S: Simd>(&self, s: S, acc: E::V<S>) -> E {
        E::v_reduce_max(s, acc)
    }
}

/// Three sums in one pass: (Σ aᵢbᵢ, Σ aᵢ², Σ bᵢ²). Used by cosine and Tanimoto.
#[derive(Clone, Copy)]
pub(crate) struct DotNorms;

impl<E: SimdFloat> Fold<E, 2> for DotNorms {
    type Acc<S: Simd> = (E::V<S>, E::V<S>, E::V<S>);
    type Out = (E, E, E);

    #[inline(always)]
    fn init<S: Simd>(&self, s: S) -> Self::Acc<S> {
        let z = E::v_splat(s, E::zero());
        (z, z, z)
    }
    #[inline(always)]
    fn step<S: Simd>(&self, s: S, acc: Self::Acc<S>, x: [E::V<S>; 2]) -> Self::Acc<S> {
        (
            E::v_mul_add(s, x[0], x[1], acc.0),
            E::v_mul_add(s, x[0], x[0], acc.1),
            E::v_mul_add(s, x[1], x[1], acc.2),
        )
    }
    #[inline(always)]
    fn merge<S: Simd>(&self, s: S, a: Self::Acc<S>, b: Self::Acc<S>) -> Self::Acc<S> {
        (E::v_add(s, a.0, b.0), E::v_add(s, a.1, b.1), E::v_add(s, a.2, b.2))
    }
    #[inline(always)]
    fn finish<S: Simd>(&self, s: S, acc: Self::Acc<S>) -> (E, E, E) {
        (E::v_reduce_sum(s, acc.0), E::v_reduce_sum(s, acc.1), E::v_reduce_sum(s, acc.2))
    }
}

/// Two sums in one pass: (Σ |aᵢ − bᵢ|, Σ |aᵢ + bᵢ|). Used by Bray–Curtis.
#[derive(Clone, Copy)]
pub(crate) struct BrayCurtis;

impl<E: SimdFloat> Fold<E, 2> for BrayCurtis {
    type Acc<S: Simd> = (E::V<S>, E::V<S>);
    type Out = (E, E);

    #[inline(always)]
    fn init<S: Simd>(&self, s: S) -> Self::Acc<S> {
        let z = E::v_splat(s, E::zero());
        (z, z)
    }
    #[inline(always)]
    fn step<S: Simd>(&self, s: S, acc: Self::Acc<S>, x: [E::V<S>; 2]) -> Self::Acc<S> {
        (
            E::v_add(s, acc.0, E::v_abs(s, E::v_sub(s, x[0], x[1]))),
            E::v_add(s, acc.1, E::v_abs(s, E::v_add(s, x[0], x[1]))),
        )
    }
    #[inline(always)]
    fn merge<S: Simd>(&self, s: S, a: Self::Acc<S>, b: Self::Acc<S>) -> Self::Acc<S> {
        (E::v_add(s, a.0, b.0), E::v_add(s, a.1, b.1))
    }
    #[inline(always)]
    fn finish<S: Simd>(&self, s: S, acc: Self::Acc<S>) -> (E, E) {
        (E::v_reduce_sum(s, acc.0), E::v_reduce_sum(s, acc.1))
    }
}

/// Two sums in one pass: (Σ min(aᵢ, bᵢ), Σ max(aᵢ, bᵢ)).
/// Used by histogram intersection and the Ruzicka (weighted Jaccard) index.
#[derive(Clone, Copy)]
pub(crate) struct MinMaxSums;

impl<E: SimdFloat> Fold<E, 2> for MinMaxSums {
    type Acc<S: Simd> = (E::V<S>, E::V<S>);
    type Out = (E, E);

    #[inline(always)]
    fn init<S: Simd>(&self, s: S) -> Self::Acc<S> {
        let z = E::v_splat(s, E::zero());
        (z, z)
    }
    #[inline(always)]
    fn step<S: Simd>(&self, s: S, acc: Self::Acc<S>, x: [E::V<S>; 2]) -> Self::Acc<S> {
        (
            E::v_add(s, acc.0, E::v_min(s, x[0], x[1])),
            E::v_add(s, acc.1, E::v_max(s, x[0], x[1])),
        )
    }
    #[inline(always)]
    fn merge<S: Simd>(&self, s: S, a: Self::Acc<S>, b: Self::Acc<S>) -> Self::Acc<S> {
        (E::v_add(s, a.0, b.0), E::v_add(s, a.1, b.1))
    }
    #[inline(always)]
    fn finish<S: Simd>(&self, s: S, acc: Self::Acc<S>) -> (E, E) {
        (E::v_reduce_sum(s, acc.0), E::v_reduce_sum(s, acc.1))
    }
}

/// Two sums in one pass: (Σ aᵢ, Σ bᵢ).
#[derive(Clone, Copy)]
pub(crate) struct Sum2;

impl<E: SimdFloat> Fold<E, 2> for Sum2 {
    type Acc<S: Simd> = (E::V<S>, E::V<S>);
    type Out = (E, E);

    #[inline(always)]
    fn init<S: Simd>(&self, s: S) -> Self::Acc<S> {
        let z = E::v_splat(s, E::zero());
        (z, z)
    }
    #[inline(always)]
    fn step<S: Simd>(&self, s: S, acc: Self::Acc<S>, x: [E::V<S>; 2]) -> Self::Acc<S> {
        (E::v_add(s, acc.0, x[0]), E::v_add(s, acc.1, x[1]))
    }
    #[inline(always)]
    fn merge<S: Simd>(&self, s: S, a: Self::Acc<S>, b: Self::Acc<S>) -> Self::Acc<S> {
        (E::v_add(s, a.0, b.0), E::v_add(s, a.1, b.1))
    }
    #[inline(always)]
    fn finish<S: Simd>(&self, s: S, acc: Self::Acc<S>) -> (E, E) {
        (E::v_reduce_sum(s, acc.0), E::v_reduce_sum(s, acc.1))
    }
}

/// Centred co-moments in one pass:
/// (Σ (aᵢ−μₐ)(bᵢ−μ_b), Σ (aᵢ−μₐ)², Σ (bᵢ−μ_b)²).
#[derive(Clone, Copy)]
pub(crate) struct CoMoments<E> {
    pub mean_a: E,
    pub mean_b: E,
}

impl<E: SimdFloat> Fold<E, 2> for CoMoments<E> {
    type Acc<S: Simd> = (E::V<S>, E::V<S>, E::V<S>);
    type Out = (E, E, E);

    #[inline(always)]
    fn init<S: Simd>(&self, s: S) -> Self::Acc<S> {
        let z = E::v_splat(s, E::zero());
        (z, z, z)
    }
    #[inline(always)]
    fn step<S: Simd>(&self, s: S, acc: Self::Acc<S>, x: [E::V<S>; 2]) -> Self::Acc<S> {
        let da = E::v_sub(s, x[0], E::v_splat(s, self.mean_a));
        let db = E::v_sub(s, x[1], E::v_splat(s, self.mean_b));
        (
            E::v_mul_add(s, da, db, acc.0),
            E::v_mul_add(s, da, da, acc.1),
            E::v_mul_add(s, db, db, acc.2),
        )
    }
    #[inline(always)]
    fn merge<S: Simd>(&self, s: S, a: Self::Acc<S>, b: Self::Acc<S>) -> Self::Acc<S> {
        (E::v_add(s, a.0, b.0), E::v_add(s, a.1, b.1), E::v_add(s, a.2, b.2))
    }
    #[inline(always)]
    fn finish<S: Simd>(&self, s: S, acc: Self::Acc<S>) -> (E, E, E) {
        (E::v_reduce_sum(s, acc.0), E::v_reduce_sum(s, acc.1), E::v_reduce_sum(s, acc.2))
    }
    #[inline(always)]
    fn pad(&self, i: usize) -> E {
        if i == 0 {
            self.mean_a
        } else {
            self.mean_b
        }
    }
}

// ---------------------------------------------------------------------------
// Element-wise maps
// ---------------------------------------------------------------------------

/// out = out · c
#[derive(Clone, Copy)]
pub(crate) struct Scale<E>(pub E);

impl<E: SimdFloat> Map<E, 0> for Scale<E> {
    #[inline(always)]
    fn apply<S: Simd>(&self, s: S, cur: E::V<S>, _x: [E::V<S>; 0]) -> E::V<S> {
        E::v_mul(s, cur, E::v_splat(s, self.0))
    }
}

/// out = (out − shift) · c
#[derive(Clone, Copy)]
pub(crate) struct ShiftScale<E> {
    pub shift: E,
    pub scale: E,
}

impl<E: SimdFloat> Map<E, 0> for ShiftScale<E> {
    #[inline(always)]
    fn apply<S: Simd>(&self, s: S, cur: E::V<S>, _x: [E::V<S>; 0]) -> E::V<S> {
        E::v_mul(s, E::v_sub(s, cur, E::v_splat(s, self.shift)), E::v_splat(s, self.scale))
    }
}

/// out = out + c · x
#[derive(Clone, Copy)]
pub(crate) struct Axpy<E>(pub E);

impl<E: SimdFloat> Map<E, 1> for Axpy<E> {
    #[inline(always)]
    fn apply<S: Simd>(&self, s: S, cur: E::V<S>, x: [E::V<S>; 1]) -> E::V<S> {
        E::v_mul_add(s, E::v_splat(s, self.0), x[0], cur)
    }
}

/// out = a + b
#[derive(Clone, Copy)]
pub(crate) struct Add2;

impl<E: SimdFloat> Map<E, 2> for Add2 {
    #[inline(always)]
    fn apply<S: Simd>(&self, s: S, _cur: E::V<S>, x: [E::V<S>; 2]) -> E::V<S> {
        E::v_add(s, x[0], x[1])
    }
}

/// out = a − b
#[derive(Clone, Copy)]
pub(crate) struct Sub2;

impl<E: SimdFloat> Map<E, 2> for Sub2 {
    #[inline(always)]
    fn apply<S: Simd>(&self, s: S, _cur: E::V<S>, x: [E::V<S>; 2]) -> E::V<S> {
        E::v_sub(s, x[0], x[1])
    }
}

/// out = a ⊙ b
#[derive(Clone, Copy)]
pub(crate) struct Mul2;

impl<E: SimdFloat> Map<E, 2> for Mul2 {
    #[inline(always)]
    fn apply<S: Simd>(&self, s: S, _cur: E::V<S>, x: [E::V<S>; 2]) -> E::V<S> {
        E::v_mul(s, x[0], x[1])
    }
}

/// out = clamp(out, lo, hi)
#[derive(Clone, Copy)]
pub(crate) struct Clamp<E> {
    pub lo: E,
    pub hi: E,
}

impl<E: SimdFloat> Map<E, 0> for Clamp<E> {
    #[inline(always)]
    fn apply<S: Simd>(&self, s: S, cur: E::V<S>, _x: [E::V<S>; 0]) -> E::V<S> {
        E::v_min(s, E::v_max(s, cur, E::v_splat(s, self.lo)), E::v_splat(s, self.hi))
    }
}

// ---------------------------------------------------------------------------
// Transcendental maps
// ---------------------------------------------------------------------------

/// out = ln(out)
#[derive(Clone, Copy)]
pub(crate) struct Ln;

impl<E: SimdFloat> Map<E, 0> for Ln {
    #[inline(always)]
    fn apply<S: Simd>(&self, s: S, cur: E::V<S>, _x: [E::V<S>; 0]) -> E::V<S> {
        E::v_ln(s, cur)
    }
}

/// out = exp(out)
#[derive(Clone, Copy)]
pub(crate) struct Exp;

impl<E: SimdFloat> Map<E, 0> for Exp {
    #[inline(always)]
    fn apply<S: Simd>(&self, s: S, cur: E::V<S>, _x: [E::V<S>; 0]) -> E::V<S> {
        E::v_exp(s, cur)
    }
}

/// out = sqrt(out)
#[derive(Clone, Copy)]
pub(crate) struct Sqrt;

impl<E: SimdFloat> Map<E, 0> for Sqrt {
    #[inline(always)]
    fn apply<S: Simd>(&self, s: S, cur: E::V<S>, _x: [E::V<S>; 0]) -> E::V<S> {
        E::v_sqrt(s, cur)
    }
}

/// out = out^y for out ≥ 0 (negative bases give NaN), computed as exp(y·ln out).
#[derive(Clone, Copy)]
pub(crate) struct Pow<E> {
    pub y: E,
}

impl<E: SimdFloat> Map<E, 0> for Pow<E> {
    #[inline(always)]
    fn apply<S: Simd>(&self, s: S, cur: E::V<S>, _x: [E::V<S>; 0]) -> E::V<S> {
        let zero = E::v_splat(s, E::zero());
        let at_zero = if self.y == E::zero() {
            E::one()
        } else if self.y > E::zero() {
            E::zero()
        } else {
            E::infinity()
        };
        let r = E::v_exp(s, E::v_mul(s, E::v_splat(s, self.y), E::v_ln(s, cur)));
        E::v_select(s, E::v_eq(s, cur, zero), E::v_splat(s, at_zero), r)
    }
}

/// out = (a + b)·ln(a + b) − a·ln a − b·ln b with a = scalar `a`, b = x₀,
/// `blnb` = x₁ (precomputed b·ln b). The flash-entropy per-peak score.
#[derive(Clone, Copy)]
pub(crate) struct EntropyGain<E> {
    pub a: E,
    pub alna: E,
}

impl<E: SimdFloat> Map<E, 2> for EntropyGain<E> {
    #[inline(always)]
    fn apply<S: Simd>(&self, s: S, _cur: E::V<S>, x: [E::V<S>; 2]) -> E::V<S> {
        let ab = E::v_add(s, E::v_splat(s, self.a), x[0]);
        let t = E::v_mul(s, ab, E::v_ln(s, ab));
        E::v_sub(s, E::v_sub(s, t, E::v_splat(s, self.alna)), x[1])
    }
}

// ---------------------------------------------------------------------------
// Information-theoretic folds
// ---------------------------------------------------------------------------

#[inline(always)]
fn neg_count<E: SimdFloat, S: Simd>(s: S, x: E::V<S>) -> E::V<S> {
    let zero = E::v_splat(s, E::zero());
    E::v_select(s, E::v_gt(s, zero, x), E::v_splat(s, E::one()), zero)
}

/// (Σ x, Σ x·ln x over x > 0, #negative)
#[derive(Clone, Copy)]
pub(crate) struct XLogX;

impl<E: SimdFloat> Fold<E, 1> for XLogX {
    type Acc<S: Simd> = (E::V<S>, E::V<S>, E::V<S>);
    type Out = (E, E, E);

    #[inline(always)]
    fn init<S: Simd>(&self, s: S) -> Self::Acc<S> {
        let z = E::v_splat(s, E::zero());
        (z, z, z)
    }
    #[inline(always)]
    fn step<S: Simd>(&self, s: S, acc: Self::Acc<S>, x: [E::V<S>; 1]) -> Self::Acc<S> {
        let zero = E::v_splat(s, E::zero());
        let pos = E::v_gt(s, x[0], zero);
        let t = E::v_select(s, pos, E::v_mul(s, x[0], E::v_ln(s, x[0])), zero);
        (E::v_add(s, acc.0, x[0]), E::v_add(s, acc.1, t), E::v_add(s, acc.2, neg_count::<E, S>(s, x[0])))
    }
    #[inline(always)]
    fn merge<S: Simd>(&self, s: S, a: Self::Acc<S>, b: Self::Acc<S>) -> Self::Acc<S> {
        (E::v_add(s, a.0, b.0), E::v_add(s, a.1, b.1), E::v_add(s, a.2, b.2))
    }
    #[inline(always)]
    fn finish<S: Simd>(&self, s: S, acc: Self::Acc<S>) -> (E, E, E) {
        (E::v_reduce_sum(s, acc.0), E::v_reduce_sum(s, acc.1), E::v_reduce_sum(s, acc.2))
    }
}

/// Raw-weight KL / cross-entropy accumulator:
/// (Σp, Σq, Σ_{p>0,q>0} p·ln(p/q), Σ_{p>0,q>0} p·ln q, #(p>0 ∧ q=0), #negative)
#[derive(Clone, Copy)]
pub(crate) struct KlTerms;

impl<E: SimdFloat> Fold<E, 2> for KlTerms {
    type Acc<S: Simd> = [E::V<S>; 6];
    type Out = [E; 6];

    #[inline(always)]
    fn init<S: Simd>(&self, s: S) -> Self::Acc<S> {
        [E::v_splat(s, E::zero()); 6]
    }
    #[inline(always)]
    fn step<S: Simd>(&self, s: S, acc: Self::Acc<S>, x: [E::V<S>; 2]) -> Self::Acc<S> {
        let (p, q) = (x[0], x[1]);
        let zero = E::v_splat(s, E::zero());
        let one = E::v_splat(s, E::one());
        let p_pos = E::v_gt(s, p, zero);
        let q_pos = E::v_gt(s, q, zero);
        let lnq = E::v_ln(s, q);
        let lnp = E::v_ln(s, p);
        let both = |v: E::V<S>| E::v_select(s, p_pos, E::v_select(s, q_pos, v, zero), zero);
        let kl = both(E::v_mul(s, p, E::v_sub(s, lnp, lnq)));
        let ce = both(E::v_mul(s, p, lnq));
        let bad = E::v_select(s, p_pos, E::v_select(s, q_pos, zero, one), zero);
        let neg = E::v_add(s, neg_count::<E, S>(s, p), neg_count::<E, S>(s, q));
        [
            E::v_add(s, acc[0], p),
            E::v_add(s, acc[1], q),
            E::v_add(s, acc[2], kl),
            E::v_add(s, acc[3], ce),
            E::v_add(s, acc[4], bad),
            E::v_add(s, acc[5], neg),
        ]
    }
    #[inline(always)]
    fn merge<S: Simd>(&self, s: S, a: Self::Acc<S>, b: Self::Acc<S>) -> Self::Acc<S> {
        core::array::from_fn(|i| E::v_add(s, a[i], b[i]))
    }
    #[inline(always)]
    fn finish<S: Simd>(&self, s: S, acc: Self::Acc<S>) -> [E; 6] {
        core::array::from_fn(|i| E::v_reduce_sum(s, acc[i]))
    }
}

/// (Σp, Σq, Σ√(p·q), #negative)
#[derive(Clone, Copy)]
pub(crate) struct BhattacharyyaTerms;

impl<E: SimdFloat> Fold<E, 2> for BhattacharyyaTerms {
    type Acc<S: Simd> = [E::V<S>; 4];
    type Out = [E; 4];

    #[inline(always)]
    fn init<S: Simd>(&self, s: S) -> Self::Acc<S> {
        [E::v_splat(s, E::zero()); 4]
    }
    #[inline(always)]
    fn step<S: Simd>(&self, s: S, acc: Self::Acc<S>, x: [E::V<S>; 2]) -> Self::Acc<S> {
        let neg = E::v_add(s, neg_count::<E, S>(s, x[0]), neg_count::<E, S>(s, x[1]));
        let zero = E::v_splat(s, E::zero());
        let pq = E::v_max(s, E::v_mul(s, x[0], x[1]), zero);
        [
            E::v_add(s, acc[0], x[0]),
            E::v_add(s, acc[1], x[1]),
            E::v_add(s, acc[2], E::v_sqrt(s, pq)),
            E::v_add(s, acc[3], neg),
        ]
    }
    #[inline(always)]
    fn merge<S: Simd>(&self, s: S, a: Self::Acc<S>, b: Self::Acc<S>) -> Self::Acc<S> {
        core::array::from_fn(|i| E::v_add(s, a[i], b[i]))
    }
    #[inline(always)]
    fn finish<S: Simd>(&self, s: S, acc: Self::Acc<S>) -> [E; 4] {
        core::array::from_fn(|i| E::v_reduce_sum(s, acc[i]))
    }
}

/// (Σp, Σq, #negative)
#[derive(Clone, Copy)]
pub(crate) struct SumsNeg;

impl<E: SimdFloat> Fold<E, 2> for SumsNeg {
    type Acc<S: Simd> = [E::V<S>; 3];
    type Out = [E; 3];

    #[inline(always)]
    fn init<S: Simd>(&self, s: S) -> Self::Acc<S> {
        [E::v_splat(s, E::zero()); 3]
    }
    #[inline(always)]
    fn step<S: Simd>(&self, s: S, acc: Self::Acc<S>, x: [E::V<S>; 2]) -> Self::Acc<S> {
        let neg = E::v_add(s, neg_count::<E, S>(s, x[0]), neg_count::<E, S>(s, x[1]));
        [E::v_add(s, acc[0], x[0]), E::v_add(s, acc[1], x[1]), E::v_add(s, acc[2], neg)]
    }
    #[inline(always)]
    fn merge<S: Simd>(&self, s: S, a: Self::Acc<S>, b: Self::Acc<S>) -> Self::Acc<S> {
        core::array::from_fn(|i| E::v_add(s, a[i], b[i]))
    }
    #[inline(always)]
    fn finish<S: Simd>(&self, s: S, acc: Self::Acc<S>) -> [E; 3] {
        core::array::from_fn(|i| E::v_reduce_sum(s, acc[i]))
    }
}

/// Σ [P ln(P/M) + Q ln(Q/M)] with P = p·sa, Q = q·sb, M = (P+Q)/2.
#[derive(Clone, Copy)]
pub(crate) struct JsTerms<E> {
    pub sa: E,
    pub sb: E,
}

impl<E: SimdFloat> Fold<E, 2> for JsTerms<E> {
    type Acc<S: Simd> = E::V<S>;
    type Out = E;

    #[inline(always)]
    fn init<S: Simd>(&self, s: S) -> E::V<S> {
        E::v_splat(s, E::zero())
    }
    #[inline(always)]
    fn step<S: Simd>(&self, s: S, acc: E::V<S>, x: [E::V<S>; 2]) -> E::V<S> {
        let zero = E::v_splat(s, E::zero());
        let p = E::v_mul(s, x[0], E::v_splat(s, self.sa));
        let q = E::v_mul(s, x[1], E::v_splat(s, self.sb));
        let half = E::v_splat(s, E::from_f64_(0.5));
        let lnm = E::v_ln(s, E::v_mul(s, E::v_add(s, p, q), half));
        let tp = E::v_select(s, E::v_gt(s, p, zero), E::v_mul(s, p, E::v_sub(s, E::v_ln(s, p), lnm)), zero);
        let tq = E::v_select(s, E::v_gt(s, q, zero), E::v_mul(s, q, E::v_sub(s, E::v_ln(s, q), lnm)), zero);
        E::v_add(s, acc, E::v_add(s, tp, tq))
    }
    #[inline(always)]
    fn merge<S: Simd>(&self, s: S, a: E::V<S>, b: E::V<S>) -> E::V<S> {
        E::v_add(s, a, b)
    }
    #[inline(always)]
    fn finish<S: Simd>(&self, s: S, acc: E::V<S>) -> E {
        E::v_reduce_sum(s, acc)
    }
}

/// Σ |p·sa − q·sb|
#[derive(Clone, Copy)]
pub(crate) struct ScaledL1<E> {
    pub sa: E,
    pub sb: E,
}

impl<E: SimdFloat> Fold<E, 2> for ScaledL1<E> {
    type Acc<S: Simd> = E::V<S>;
    type Out = E;

    #[inline(always)]
    fn init<S: Simd>(&self, s: S) -> E::V<S> {
        E::v_splat(s, E::zero())
    }
    #[inline(always)]
    fn step<S: Simd>(&self, s: S, acc: E::V<S>, x: [E::V<S>; 2]) -> E::V<S> {
        let d = E::v_sub(s, E::v_mul(s, x[0], E::v_splat(s, self.sa)), E::v_mul(s, x[1], E::v_splat(s, self.sb)));
        E::v_add(s, acc, E::v_abs(s, d))
    }
    #[inline(always)]
    fn merge<S: Simd>(&self, s: S, a: E::V<S>, b: E::V<S>) -> E::V<S> {
        E::v_add(s, a, b)
    }
    #[inline(always)]
    fn finish<S: Simd>(&self, s: S, acc: E::V<S>) -> E {
        E::v_reduce_sum(s, acc)
    }
}
