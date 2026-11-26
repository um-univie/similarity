//! Generic SIMD execution engine.
//!
//! Two kinds of kernels cover almost every metric in this crate:
//!
//! * [`Fold`]: a reduction over `N` equally long input slices
//!   (e.g. dot product, Manhattan distance, cosine similarity).
//! * [`Map`]: an element-wise update of an output slice from `N` inputs
//!   (e.g. scaling, normalisation, AXPY).
//!
//! The drivers in this module take care of runtime instruction-set
//! dispatch ([`pulp::Arch`]), splitting slices into SIMD registers,
//! 4-way accumulator unrolling for instruction-level parallelism, and the
//! final partial register. Kernels only describe what happens per register.

use super::element::SimdFloat;
use pulp::{Simd, WithSimd};

/// Largest lane count of any supported instruction set (AVX-512 `f32`).
const MAX_LANES: usize = 16;

/// A slice split into whole SIMD registers and a scalar tail.
type Split<'a, E, S> = (&'a [<E as SimdFloat>::V<S>], &'a [E]);

/// A SIMD reduction over `N` slices of equal length.
pub(crate) trait Fold<E: SimdFloat, const N: usize> {
    type Acc<S: Simd>: Copy;
    type Out;

    fn init<S: Simd>(&self, s: S) -> Self::Acc<S>;
    fn step<S: Simd>(&self, s: S, acc: Self::Acc<S>, x: [E::V<S>; N]) -> Self::Acc<S>;
    fn merge<S: Simd>(&self, s: S, a: Self::Acc<S>, b: Self::Acc<S>) -> Self::Acc<S>;
    fn finish<S: Simd>(&self, s: S, acc: Self::Acc<S>) -> Self::Out;

    /// Value that fills unused lanes of input `i` in the last register.
    /// It must make those lanes contribute nothing to the result.
    #[inline(always)]
    fn pad(&self, _i: usize) -> E {
        E::zero()
    }
}

/// An element-wise SIMD update `out[i] = f(out[i], x_0[i], .., x_{N-1}[i])`.
pub(crate) trait Map<E: SimdFloat, const N: usize> {
    fn apply<S: Simd>(&self, s: S, cur: E::V<S>, x: [E::V<S>; N]) -> E::V<S>;
}

#[inline(always)]
fn gather<E: SimdFloat, S: Simd, const N: usize>(heads: &[&[E::V<S>]; N], i: usize) -> [E::V<S>; N] {
    core::array::from_fn(|j| heads[j][i])
}

#[inline(always)]
fn load_tail<E: SimdFloat, S: Simd>(s: S, tail: &[E], pad: E) -> E::V<S> {
    let lanes = E::lanes::<S>();
    let mut buf = [pad; MAX_LANES];
    buf[..tail.len()].copy_from_slice(tail);
    E::v_partial_load(s, &buf[..lanes])
}

/// Runs a [`Fold`] kernel inside an already selected instruction set.
///
/// Uses 8 independent accumulators on instruction sets with 32 vector
/// registers (AVX-512, NEON) and 4 otherwise (AVX2), which hides FMA latency
/// without spilling registers.
#[inline(always)]
pub(crate) fn fold_in<E, F, S, const N: usize>(s: S, f: &F, xs: [&[E]; N]) -> F::Acc<S>
where
    E: SimdFloat,
    F: Fold<E, N>,
    S: Simd,
{
    if S::REGISTER_COUNT >= 32 {
        fold_unrolled::<E, F, S, N, 8>(s, f, xs)
    } else {
        fold_unrolled::<E, F, S, N, 4>(s, f, xs)
    }
}

#[inline(always)]
fn fold_unrolled<E, F, S, const N: usize, const U: usize>(s: S, f: &F, xs: [&[E]; N]) -> F::Acc<S>
where
    E: SimdFloat,
    F: Fold<E, N>,
    S: Simd,
{
    let split: [Split<'_, E, S>; N] = core::array::from_fn(|j| E::as_simd::<S>(xs[j]));
    let heads: [&[E::V<S>]; N] = core::array::from_fn(|j| split[j].0);
    let nv = if N == 0 { 0 } else { heads[0].len() };
    let tail_len = if N == 0 { 0 } else { split[0].1.len() };

    let init = f.init(s);
    let mut acc = [init; U];
    let mut i = 0;
    while i + U <= nv {
        for (k, a) in acc.iter_mut().enumerate() {
            *a = f.step(s, *a, gather::<E, S, N>(&heads, i + k));
        }
        i += U;
    }
    // pairwise tree merge
    let mut width = U;
    while width > 1 {
        width /= 2;
        for k in 0..width {
            acc[k] = f.merge(s, acc[k], acc[k + width]);
        }
    }
    let mut a = acc[0];
    while i < nv {
        a = f.step(s, a, gather::<E, S, N>(&heads, i));
        i += 1;
    }
    if tail_len > 0 {
        let v: [E::V<S>; N] = core::array::from_fn(|j| load_tail::<E, S>(s, split[j].1, f.pad(j)));
        a = f.step(s, a, v);
    }
    a
}

struct FoldOp<'a, E, F, const N: usize> {
    f: F,
    xs: [&'a [E]; N],
}

impl<'a, E, F, const N: usize> WithSimd for FoldOp<'a, E, F, N>
where
    E: SimdFloat,
    F: Fold<E, N>,
{
    type Output = F::Out;

    #[inline(always)]
    fn with_simd<S: Simd>(self, s: S) -> Self::Output {
        let acc = fold_in(s, &self.f, self.xs);
        self.f.finish(s, acc)
    }
}

/// Dispatches a [`Fold`] kernel to the best instruction set available at
/// runtime. All slices must have the same length (checked by callers).
#[inline]
pub(crate) fn fold<E, F, const N: usize>(f: F, xs: [&[E]; N]) -> F::Out
where
    E: SimdFloat,
    F: Fold<E, N>,
{
    debug_assert!(xs.iter().all(|x| x.len() == xs[0].len()));
    super::isa::dispatch(FoldOp { f, xs })
}

struct MapOp<'a, 'b, E, F, const N: usize> {
    f: F,
    out: &'b mut [E],
    xs: [&'a [E]; N],
}

impl<'a, 'b, E, F, const N: usize> WithSimd for MapOp<'a, 'b, E, F, N>
where
    E: SimdFloat,
    F: Map<E, N>,
{
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, s: S) {
        let f = self.f;
        let (out_head, out_tail) = E::as_mut_simd::<S>(self.out);
        let split: [Split<'_, E, S>; N] = core::array::from_fn(|j| E::as_simd::<S>(self.xs[j]));
        let heads: [&[E::V<S>]; N] = core::array::from_fn(|j| split[j].0);

        for (i, o) in out_head.iter_mut().enumerate() {
            *o = f.apply(s, *o, gather::<E, S, N>(&heads, i));
        }
        if !out_tail.is_empty() {
            let cur = E::v_partial_load(s, out_tail);
            let v: [E::V<S>; N] = core::array::from_fn(|j| E::v_partial_load(s, split[j].1));
            E::v_partial_store(s, out_tail, f.apply(s, cur, v));
        }
    }
}

/// Dispatches a [`Map`] kernel. All slices must have the same length.
#[inline]
pub(crate) fn map<E, F, const N: usize>(f: F, out: &mut [E], xs: [&[E]; N])
where
    E: SimdFloat,
    F: Map<E, N>,
{
    debug_assert!(xs.iter().all(|x| x.len() == out.len()));
    super::isa::dispatch(MapOp { f, out, xs })
}
