//! GEMM-style kernel for all-pairs dot products and squared Euclidean
//! distances (the inner loop of `cdist` / `pdist` / `pairwise`).
//!
//! The layout follows BLIS-style `gemm`. `B` is packed into slabs of
//! `NR = V × lanes` rows stored dimension-major, so one SIMD load reads the
//! same dimension of `V × lanes` different rows; `A` is packed into
//! micro-panels of `MR` rows. The micro-kernel keeps an `MR × NR` block of
//! results in registers and, for every dimension, broadcasts one value per
//! row of `A` and updates the block with `MR × V` fused multiply-adds, so no
//! horizontal sums are needed. The depth is split into blocks of `KC` and
//! the columns into blocks that fit in L2. Squared distances are accumulated as `Σ (a − b)²`, so
//! they keep full accuracy (unlike the `|a|² + |b|² − 2a·b` shortcut).

use super::element::SimdFloat;
use pulp::{Simd, WithSimd};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Pairwise operation computed by the kernel.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum TileOp {
    /// `Σ a·b`
    Dot,
    /// `Σ (a − b)²`
    SqDiff,
}

/// Depth (`k`) block: one packed micro-panel of `A` (`MR × KC`) stays in
/// L1 while it is multiplied with every panel of a column block.
const KC: usize = 256;

/// Target size of one column block of packed `B` (kept in L2).
const L2_BLOCK_BYTES: usize = 256 * 1024;

/// Callback applied to every finished output row: `(row index, first
/// column, row)`, where `row` holds the columns `first..first + row.len()`.
pub(crate) type RowFinish<'f, E> = &'f (dyn Fn(usize, usize, &mut [E]) + Sync);

/// How the results are laid out in `out`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Layout {
    /// Row-major `a.len() × b.len()`, every entry computed.
    Full,
    /// Row-major `n × n` with `a` and `b` the same rows; only entries with
    /// `j ≥ i` are guaranteed to be computed (the caller mirrors the rest).
    Upper,
    /// Strict upper triangle in SciPy `pdist` order (`(0,1), (0,2), …,
    /// (1,2), …`, length `n(n−1)/2`) with `a` and `b` the same rows.
    Condensed,
}

/// Offset of row `i` in the condensed layout of an `n × n` matrix.
#[inline]
pub(crate) fn condensed_offset(i: usize, n: usize) -> usize {
    i * n - i * (i + 1) / 2
}

/// Splits a condensed `n × n` result into the slices of row blocks
/// `[k · rows, (k + 1) · rows)`.
pub(crate) fn condensed_blocks<E>(mut out: &mut [E], n: usize, rows: usize) -> Vec<&mut [E]> {
    let mut blocks = Vec::with_capacity(n.div_ceil(rows.max(1)));
    for r0 in (0..n).step_by(rows.max(1)) {
        let r1 = (r0 + rows).min(n);
        let (head, tail) = out.split_at_mut(condensed_offset(r1, n) - condensed_offset(r0, n));
        blocks.push(head);
        out = tail;
    }
    blocks
}

/// Fills `out` with `op(a[i], b[j])` in the given layout and then calls
/// `finish` on each output row. All rows must have the same length.
pub(crate) fn all_pairs<E: SimdFloat>(
    op: TileOp,
    a: &[&[E]],
    b: &[&[E]],
    out: &mut [E],
    layout: Layout,
    finish: RowFinish<'_, E>,
) {
    let (m, n) = (a.len(), b.len());
    if m == 0 || n == 0 {
        return;
    }
    // Work is split over row blocks of `A`. With few rows and many columns
    // that leaves cores idle, so compute the transposed problem instead
    // (both operations are symmetric) and transpose the result.
    if layout == Layout::Full && m < 256 && n >= 4 * m {
        let mut t = vec![E::zero(); n * m];
        all_pairs(op, b, a, &mut t, Layout::Full, &|_, _, _| {});
        let fill = |(i, row): (usize, &mut [E])| {
            for (j, v) in row.iter_mut().enumerate() {
                *v = t[j * m + i];
            }
            finish(i, 0, row);
        };
        #[cfg(feature = "parallel")]
        out.par_chunks_mut(n).enumerate().for_each(fill);
        #[cfg(not(feature = "parallel"))]
        out.chunks_mut(n).enumerate().for_each(fill);
        return;
    }
    super::isa::dispatch(AllPairs { op, a, b, out, layout, finish })
}

struct AllPairs<'a, 'o, 'f, E> {
    op: TileOp,
    a: &'a [&'a [E]],
    b: &'a [&'a [E]],
    out: &'o mut [E],
    layout: Layout,
    finish: RowFinish<'f, E>,
}

impl<E: SimdFloat> WithSimd for AllPairs<'_, '_, '_, E> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, s: S) -> Self::Output {
        // Register tile: MR rows × V registers. 8 × 3 uses 24 accumulators
        // (NEON, AVX-512: 32 registers); 6 × 2 uses 12 (AVX2: 16 registers).
        if S::REGISTER_COUNT >= 32 {
            self.run::<S, 8, 3>(s)
        } else {
            self.run::<S, 6, 2>(s)
        }
    }
}

impl<E: SimdFloat> AllPairs<'_, '_, '_, E> {
    #[inline(always)]
    fn run<S: Simd, const MR: usize, const V: usize>(self, s: S) {
        let AllPairs { op, a, b, out, layout, finish } = self;
        let (n, d) = (b.len(), a[0].len());
        if d == 0 {
            out.iter_mut().for_each(|v| *v = E::zero());
        }
        let nr = V * E::lanes::<S>();
        let packed = pack_b(b, nr, d);
        // Up to 8 micro-panels per task, fewer if that would leave cores idle.
        #[cfg(feature = "parallel")]
        let threads = rayon::current_num_threads().max(1);
        #[cfg(not(feature = "parallel"))]
        let threads = 1;
        let m = a.len();
        let mb = (m / (2 * threads) / MR).clamp(1, 8) * MR;
        let task = |(blk, chunk): (usize, &mut [E])| {
            let i0 = blk * mb;
            let rows = mb.min(m - i0);
            let block = RowBlock::<E, MR, V> { op, a, i0, rows, packed: &packed, n, out: chunk, layout, finish };
            s.vectorize(block);
        };
        if layout == Layout::Condensed {
            let blocks = condensed_blocks(out, n, mb);
            #[cfg(feature = "parallel")]
            blocks.into_par_iter().enumerate().for_each(task);
            #[cfg(not(feature = "parallel"))]
            blocks.into_iter().enumerate().for_each(task);
            return;
        }
        #[cfg(feature = "parallel")]
        out.par_chunks_mut(mb * n).enumerate().for_each(task);
        #[cfg(not(feature = "parallel"))]
        out.chunks_mut(mb * n).enumerate().for_each(task);
    }
}

/// Packs `b` for the micro-kernel. For each depth block `k0..k0 + kc` and
/// each panel `q` of `nr` rows there is one contiguous `kc × nr` slab with
/// `slab[p * nr + c] = b[q * nr + c][k0 + p]` (zero for padding rows). The
/// slab starts at `k0 · panels · nr + q · kc · nr`.
fn pack_b<E: SimdFloat>(b: &[&[E]], nr: usize, d: usize) -> Vec<E> {
    let panels = b.len().div_ceil(nr);
    let mut packed = vec![E::zero(); panels * nr * d];
    let mut k0 = 0;
    while k0 < d {
        let kc = KC.min(d - k0);
        let slabs = &mut packed[k0 * panels * nr..(k0 + kc) * panels * nr];
        let fill = |(q, slab): (usize, &mut [E])| {
            for c in 0..nr {
                if let Some(row) = b.get(q * nr + c) {
                    for (p, &x) in row[k0..k0 + kc].iter().enumerate() {
                        slab[p * nr + c] = x;
                    }
                }
            }
        };
        // Packing small inputs in parallel costs more than it saves.
        #[cfg(feature = "parallel")]
        if slabs.len() >= 1 << 18 {
            slabs.par_chunks_mut(kc * nr).enumerate().for_each(fill);
        } else {
            slabs.chunks_mut(kc * nr).enumerate().for_each(fill);
        }
        #[cfg(not(feature = "parallel"))]
        slabs.chunks_mut(kc * nr).enumerate().for_each(fill);
        k0 += kc;
    }
    packed
}

/// One parallel task: a block of up to `8 · MR` rows of `A` against all of `B`.
struct RowBlock<'a, 'o, 'f, E, const MR: usize, const V: usize> {
    op: TileOp,
    a: &'a [&'a [E]],
    /// Index of the first row of this block.
    i0: usize,
    /// Number of rows in this block.
    rows: usize,
    packed: &'a [E],
    n: usize,
    /// The output rows of this block.
    out: &'o mut [E],
    layout: Layout,
    finish: RowFinish<'f, E>,
}

impl<E: SimdFloat, const MR: usize, const V: usize> WithSimd for RowBlock<'_, '_, '_, E, MR, V> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, s: S) -> Self::Output {
        match self.op {
            TileOp::Dot => self.run::<S, false>(s),
            TileOp::SqDiff => self.run::<S, true>(s),
        }
    }
}

impl<E: SimdFloat, const MR: usize, const V: usize> RowBlock<'_, '_, '_, E, MR, V> {
    /// Where global row `i` lives in `self.out`: `(start, first column,
    /// length)`.
    #[inline(always)]
    fn row_range(&self, i: usize) -> (usize, usize, usize) {
        let n = self.n;
        match self.layout {
            Layout::Full | Layout::Upper => ((i - self.i0) * n, 0, n),
            Layout::Condensed => {
                (condensed_offset(i, n) - condensed_offset(self.i0, n), i + 1, n - i - 1)
            }
        }
    }

    #[inline(always)]
    fn run<S: Simd, const SQ: bool>(self, s: S) {
        let n = self.n;
        let rows = self.rows;
        let lanes = E::lanes::<S>();
        let nr = V * lanes;
        let d = self.a[0].len();
        let panels = n.div_ceil(nr);
        let groups = rows.div_ceil(MR);
        let per_block = (L2_BLOCK_BYTES / (KC * nr * core::mem::size_of::<E>())).max(1);
        let mut apack = vec![E::zero(); groups * MR * KC.min(d)];

        let mut k0 = 0;
        while k0 < d {
            let kc = KC.min(d - k0);
            // Pack MR-row micro-panels of A: ap[g][p * MR + r] = a[i0 + g·MR + r][k0 + p].
            for g in 0..groups {
                let ap = &mut apack[g * MR * kc..(g + 1) * MR * kc];
                for r in 0..MR {
                    let i = g * MR + r;
                    if i < rows {
                        for (p, &x) in self.a[self.i0 + i][k0..k0 + kc].iter().enumerate() {
                            ap[p * MR + r] = x;
                        }
                    } else {
                        (0..kc).for_each(|p| ap[p * MR + r] = E::zero());
                    }
                }
            }
            let slabs = &self.packed[k0 * panels * nr..];
            let mut qb = 0;
            while qb < panels {
                let qe = (qb + per_block).min(panels);
                for g in 0..groups {
                    let first = self.i0 + g * MR;
                    let q_start = if self.layout == Layout::Full { qb } else { qb.max(first / nr) };
                    let ap = &apack[g * MR * kc..(g + 1) * MR * kc];
                    for q in q_start..qe {
                        let bp = E::as_simd::<S>(&slabs[q * kc * nr..(q + 1) * kc * nr]).0;
                        let acc = micro::<E, S, MR, V, SQ>(s, ap, bp, kc);
                        let j0 = q * nr;
                        let w = nr.min(n - j0);
                        for (r, accr) in acc.iter().enumerate().take(rows.saturating_sub(g * MR).min(MR)) {
                            let (start, c0, len) = self.row_range(first + r);
                            if j0 + w <= c0 {
                                continue;
                            }
                            if j0 < c0 {
                                // Tile straddles the first stored column
                                // (condensed diagonal): store lane by lane.
                                let mut tmp = [E::zero(); 64];
                                for (v, &x) in accr.iter().enumerate() {
                                    E::as_mut_simd::<S>(&mut tmp[v * lanes..(v + 1) * lanes]).0[0] = x;
                                }
                                let row = &mut self.out[start..start + len];
                                for j in c0..j0 + w {
                                    let t = tmp[j - j0];
                                    row[j - c0] = if k0 == 0 { t } else { row[j - c0] + t };
                                }
                                continue;
                            }
                            let row = &mut self.out[start + j0 - c0..][..w];
                            for (v, &x) in accr.iter().enumerate() {
                                let start = v * lanes;
                                if start >= w {
                                    break;
                                }
                                if start + lanes <= w {
                                    // Full vector: plain store. pulp's partial
                                    // load/store pick a mask at run time,
                                    // which is much slower (on AVX2 an
                                    // indirect call per access).
                                    let dst = &mut E::as_mut_simd::<S>(&mut row[start..start + lanes]).0[0];
                                    *dst = if k0 == 0 { x } else { E::v_add(s, x, *dst) };
                                } else {
                                    let dst = &mut row[start..w];
                                    let x = if k0 == 0 { x } else { E::v_add(s, x, E::v_partial_load(s, dst)) };
                                    E::v_partial_store(s, dst, x);
                                }
                            }
                        }
                    }
                }
                qb = qe;
            }
            k0 += kc;
        }
        for i in self.i0..self.i0 + rows {
            let (start, c0, len) = self.row_range(i);
            (self.finish)(i, c0, &mut self.out[start..start + len]);
        }
    }
}

/// `MR × (V · lanes)` block of results for one packed micro-panel of `A`
/// (`ap`, `kc × MR`) and one packed slab of `B` (`bp`, `kc × V` registers).
#[inline(always)]
#[allow(clippy::needless_range_loop)] // index loops keep the tile in registers
fn micro<E: SimdFloat, S: Simd, const MR: usize, const V: usize, const SQ: bool>(
    s: S,
    ap: &[E],
    bp: &[E::V<S>],
    kc: usize,
) -> [[E::V<S>; V]; MR] {
    let zero = E::v_splat(s, E::zero());
    let mut acc = [[zero; V]; MR];
    assert!(ap.len() == kc * MR && bp.len() == kc * V);
    for p in 0..kc {
        // SAFETY: bounds checked by the assertion above.
        let bv: [E::V<S>; V] = core::array::from_fn(|v| unsafe { *bp.get_unchecked(p * V + v) });
        for r in 0..MR {
            let av = E::v_splat(s, unsafe { *ap.get_unchecked(p * MR + r) });
            for v in 0..V {
                acc[r][v] = if SQ {
                    let t = E::v_sub(s, av, bv[v]);
                    E::v_mul_add(s, t, t, acc[r][v])
                } else {
                    E::v_mul_add(s, av, bv[v], acc[r][v])
                };
            }
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::isa::for_each_isa;
    use rand::Rng;

    fn check<E: SimdFloat>(tol: f64) {
        let mut rng = rand::rng();
        for &(m, n, d) in &[(1, 1, 1), (3, 5, 7), (8, 16, 16), (9, 13, 33), (6, 2, 0), (70, 37, 100), (130, 3, 5), (2, 50, 300), (100, 30, 513), (1, 9, 0)] {
            let a: Vec<Vec<E>> =
                (0..m).map(|_| (0..d).map(|_| E::from_f64_(rng.random_range(-1.0..1.0))).collect()).collect();
            let b: Vec<Vec<E>> =
                (0..n).map(|_| (0..d).map(|_| E::from_f64_(rng.random_range(-1.0..1.0))).collect()).collect();
            let ar: Vec<&[E]> = a.iter().map(|r| r.as_slice()).collect();
            let br: Vec<&[E]> = b.iter().map(|r| r.as_slice()).collect();
            for op in [TileOp::Dot, TileOp::SqDiff] {
                for_each_isa(|isa| {
                    let mut out = vec![E::nan(); m * n];
                    let calls = std::sync::atomic::AtomicUsize::new(0);
                    all_pairs(op, &ar, &br, &mut out, Layout::Full, &|_, _, _| {
                        calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    });
                    assert_eq!(calls.into_inner(), m);
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
                            assert!(
                                (got - want).abs() <= tol * (1.0 + want.abs()),
                                "{isa:?} {op:?} {m}x{n}x{d} ({i},{j}): {got} vs {want}"
                            );
                        }
                    }
                });
            }
        }
    }

    #[test]
    fn kernel_matches_naive() {
        check::<f64>(1e-12);
        check::<f32>(1e-4);
    }

    #[test]
    fn upper_computes_the_upper_triangle() {
        let rows: Vec<Vec<f64>> = (0..150).map(|i| vec![i as f64; 5]).collect();
        let r: Vec<&[f64]> = rows.iter().map(|x| x.as_slice()).collect();
        for_each_isa(|_| {
            let mut out = vec![f64::NAN; 150 * 150];
            all_pairs(TileOp::SqDiff, &r, &r, &mut out, Layout::Upper, &|_, _, _| {});
            for i in 0..150 {
                for j in i..150 {
                    assert_eq!(out[i * 150 + j], 5.0 * ((i as f64 - j as f64).powi(2)));
                }
            }
        });
    }

    #[test]
    fn condensed_matches_full() {
        let mut rng = rand::rng();
        for &(n, d) in &[(1, 3), (2, 1), (7, 5), (50, 17), (150, 300), (333, 9)] {
            let rows: Vec<Vec<f32>> = (0..n).map(|_| (0..d).map(|_| rng.random_range(-1.0..1.0)).collect()).collect();
            let r: Vec<&[f32]> = rows.iter().map(|x| x.as_slice()).collect();
            for op in [TileOp::Dot, TileOp::SqDiff] {
                for_each_isa(|isa| {
                    let mut full = vec![f32::NAN; n * n];
                    all_pairs(op, &r, &r, &mut full, Layout::Full, &|_, _, _| {});
                    let mut cond = vec![f32::NAN; n * (n - 1) / 2];
                    let seen = std::sync::Mutex::new(vec![false; n]);
                    all_pairs(op, &r, &r, &mut cond, Layout::Condensed, &|i, c0, row| {
                        assert_eq!((c0, row.len()), (i + 1, n - i - 1));
                        seen.lock().unwrap()[i] = true;
                    });
                    assert!(seen.into_inner().unwrap().iter().all(|&x| x));
                    let mut k = 0;
                    for i in 0..n {
                        for j in i + 1..n {
                            assert_eq!(cond[k], full[i * n + j], "{isa:?} {op:?} n={n} d={d} ({i},{j})");
                            k += 1;
                        }
                    }
                });
            }
        }
    }
}
