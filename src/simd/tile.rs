//! Register-blocked kernels for all-pairs dot products and squared
//! Euclidean distances (the inner loop of `cdist` / `pdist`).
//!
//! A naive all-pairs loop loads two registers for every fused multiply-add.
//! The micro-kernel here computes an `R × C` tile of results at once: per
//! step it loads `R` registers from rows of `A` and `C` from rows of `B` and
//! performs `R · C` FMAs, so loads per FMA drop from 2 to `(R + C) / (R · C)`.
//! Callers block the work so the rows of one tile stay in cache.

use super::element::SimdFloat;
use pulp::{Simd, WithSimd};

/// A row split into whole SIMD registers and a scalar tail.
type Split<'a, E, S> = (&'a [<E as SimdFloat>::V<S>], &'a [E]);

/// Pairwise operation computed by a tile.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum TileOp {
    /// `Σ a·b`
    Dot,
    /// `Σ (a − b)²`
    SqDiff,
}

/// Fills `out[i * ldo + j]` with `op(a[i], b[j])` for all `i`, `j`. With
/// `upper = Some(diag)`, column groups lying entirely in `j < i + diag` are
/// skipped (the caller fills them by mirroring). All rows must have the
/// same length.
pub(crate) fn tile_block<E: SimdFloat>(
    op: TileOp,
    a: &[&[E]],
    b: &[&[E]],
    out: &mut [E],
    ldo: usize,
    upper: Option<isize>,
) {
    super::isa::dispatch(Block { op, a, b, out, ldo, upper })
}

struct Block<'a, 'o, E> {
    op: TileOp,
    a: &'a [&'a [E]],
    b: &'a [&'a [E]],
    out: &'o mut [E],
    ldo: usize,
    upper: Option<isize>,
}

impl<E: SimdFloat> WithSimd for Block<'_, '_, E> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, s: S) -> Self::Output {
        let Block { op, a, b, out, ldo, upper } = self;
        if S::REGISTER_COUNT >= 32 {
            rows_by::<E, S, 4>(s, op, a, b, out, ldo, upper);
        } else {
            rows_by::<E, S, 2>(s, op, a, b, out, ldo, upper);
        }
    }
}

#[inline(always)]
fn rows_by<E: SimdFloat, S: Simd, const R: usize>(
    s: S,
    op: TileOp,
    a: &[&[E]],
    b: &[&[E]],
    out: &mut [E],
    ldo: usize,
    upper: Option<isize>,
) {
    let full = a.len() / R * R;
    let mut i = 0;
    while i < full {
        row_group::<E, S, R>(s, op, &a[i..i + R], i, b, out, ldo, upper);
        i += R;
    }
    while i < a.len() {
        row_group::<E, S, 1>(s, op, &a[i..i + 1], i, b, out, ldo, upper);
        i += 1;
    }
}

/// One group of `R` rows of `A` against all rows of `B`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn row_group<E: SimdFloat, S: Simd, const R: usize>(
    s: S,
    op: TileOp,
    a: &[&[E]],
    i0: usize,
    b: &[&[E]],
    out: &mut [E],
    ldo: usize,
    upper: Option<isize>,
) {
    const C: usize = 4;
    let ar: [&[E]; R] = core::array::from_fn(|r| a[r]);
    // Skip whole column groups that lie strictly below the diagonal.
    let start = match upper {
        Some(diag) => (i0 as isize + diag).max(0) as usize / C * C,
        None => 0,
    }
    .min(b.len());
    let full = start + (b.len() - start) / C * C;
    let mut j = start;
    while j < full {
        let br: [&[E]; C] = core::array::from_fn(|c| b[j + c]);
        let t = micro::<E, S, R, C>(s, op, ar, br);
        for (r, row) in t.iter().enumerate() {
            out[(i0 + r) * ldo + j..][..C].copy_from_slice(row);
        }
        j += C;
    }
    while j < b.len() {
        let t = micro::<E, S, R, 1>(s, op, ar, [b[j]]);
        for (r, row) in t.iter().enumerate() {
            out[(i0 + r) * ldo + j] = row[0];
        }
        j += 1;
    }
}

/// `R × C` tile of `op(a[r], b[c])`.
#[inline(always)]
fn micro<E: SimdFloat, S: Simd, const R: usize, const C: usize>(
    s: S,
    op: TileOp,
    a: [&[E]; R],
    b: [&[E]; C],
) -> [[E; C]; R] {
    let n = a[0].len();
    let ah: [Split<'_, E, S>; R] = core::array::from_fn(|r| E::as_simd::<S>(&a[r][..n]));
    let bh: [Split<'_, E, S>; C] = core::array::from_fn(|c| E::as_simd::<S>(&b[c][..n]));
    let nv = ah[0].0.len();
    let zero = E::v_splat(s, E::zero());
    let mut acc = [[zero; C]; R];

    #[inline(always)]
    fn step<E: SimdFloat, S: Simd, const R: usize, const C: usize>(
        s: S,
        op: TileOp,
        acc: &mut [[E::V<S>; C]; R],
        av: [E::V<S>; R],
        bv: [E::V<S>; C],
    ) {
        for r in 0..R {
            for c in 0..C {
                acc[r][c] = match op {
                    TileOp::Dot => E::v_mul_add(s, av[r], bv[c], acc[r][c]),
                    TileOp::SqDiff => {
                        let d = E::v_sub(s, av[r], bv[c]);
                        E::v_mul_add(s, d, d, acc[r][c])
                    }
                };
            }
        }
    }

    for k in 0..nv {
        // SAFETY: every row has `nv` whole registers (all rows have length `n`).
        let av: [E::V<S>; R] = core::array::from_fn(|r| unsafe { *ah[r].0.get_unchecked(k) });
        let bv: [E::V<S>; C] = core::array::from_fn(|c| unsafe { *bh[c].0.get_unchecked(k) });
        step::<E, S, R, C>(s, op, &mut acc, av, bv);
    }
    if !ah[0].1.is_empty() {
        // Zero-padded tails contribute nothing to either operation.
        let av: [E::V<S>; R] = core::array::from_fn(|r| E::v_partial_load(s, ah[r].1));
        let bv: [E::V<S>; C] = core::array::from_fn(|c| E::v_partial_load(s, bh[c].1));
        step::<E, S, R, C>(s, op, &mut acc, av, bv);
    }
    core::array::from_fn(|r| core::array::from_fn(|c| E::v_reduce_sum(s, acc[r][c])))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::isa::for_each_isa;
    use rand::Rng;

    fn check<E: SimdFloat>(tol: f64) {
        let mut rng = rand::rng();
        for &(m, n, d) in &[(1, 1, 1), (3, 5, 7), (4, 4, 16), (9, 13, 33), (6, 2, 0), (17, 11, 100)] {
            let a: Vec<Vec<E>> =
                (0..m).map(|_| (0..d).map(|_| E::from_f64_(rng.random_range(-1.0..1.0))).collect()).collect();
            let b: Vec<Vec<E>> =
                (0..n).map(|_| (0..d).map(|_| E::from_f64_(rng.random_range(-1.0..1.0))).collect()).collect();
            let ar: Vec<&[E]> = a.iter().map(|r| r.as_slice()).collect();
            let br: Vec<&[E]> = b.iter().map(|r| r.as_slice()).collect();
            for op in [TileOp::Dot, TileOp::SqDiff] {
                for_each_isa(|isa| {
                    let mut out = vec![E::nan(); m * n];
                    tile_block(op, &ar, &br, &mut out, n, None);
                    for i in 0..m {
                        for j in 0..n {
                            let want: f64 = a[i]
                                .iter()
                                .zip(&b[j])
                                .map(|(x, y)| {
                                    let (x, y) = (x.to_f64_(), y.to_f64_());
                                    if op == TileOp::Dot { x * y } else { (x - y) * (x - y) }
                                })
                                .sum();
                            let got = out[i * n + j].to_f64_();
                            assert!((got - want).abs() <= tol * (1.0 + want.abs()), "{isa:?} {op:?} {m}x{n}x{d} ({i},{j}): {got} vs {want}");
                        }
                    }
                });
            }
        }
    }

    #[test]
    fn tiles_match_naive() {
        check::<f64>(1e-12);
        check::<f32>(1e-4);
    }

    #[test]
    fn upper_skips_only_lower_groups() {
        let rows: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64; 5]).collect();
        let r: Vec<&[f64]> = rows.iter().map(|x| x.as_slice()).collect();
        for_each_isa(|_| {
            let mut out = vec![f64::NAN; 100];
            tile_block(TileOp::SqDiff, &r, &r, &mut out, 10, Some(0));
            for i in 0..10 {
                for j in i..10 {
                    assert_eq!(out[i * 10 + j], 5.0 * ((i as f64 - j as f64).powi(2)));
                }
            }
        });
    }
}
