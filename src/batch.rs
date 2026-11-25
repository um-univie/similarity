//! # Batch similarity: pairwise matrices, cross-distance and k-nearest search
//!
//! These functions evaluate a [`Metric`] over many vectors at once. Each
//! single comparison runs on the portable-SIMD kernels of [`crate::simd`];
//! with the `parallel` feature (default) rows are also spread over the
//! rayon thread pool.
//!
//! Rows can be anything that derefs to a slice (`Vec<f32>`, `&[f64]`,
//! `[f64; 3]`, …). Results are stored row-major in a flat `Vec`. Undefined
//! values (e.g. cosine similarity with a zero vector) are reported as `NaN`.
//!
//! ```
//! use similarity::batch::{pairwise, Metric};
//!
//! let rows = vec![vec![0.0_f32, 0.0], vec![3.0, 4.0], vec![6.0, 8.0]];
//! let d = pairwise(&rows, Metric::Euclidean).unwrap();
//! assert_eq!(d.get(0, 1), 5.0);
//! assert_eq!(d.get(2, 0), 10.0);
//! ```

use crate::simd::{self, SimdFloat};
use num_traits::Float;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Metrics available for batch evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Metric {
    /// Dot product (similarity, larger is closer).
    Dot,
    /// Cosine similarity (similarity).
    Cosine,
    /// `1 − cosine` (distance).
    CosineDistance,
    /// Euclidean / L2 distance.
    Euclidean,
    /// Squared Euclidean distance.
    SquaredEuclidean,
    /// Manhattan / L1 distance.
    Manhattan,
    /// Chebyshev / L∞ distance.
    Chebyshev,
    /// Canberra distance.
    Canberra,
    /// Bray–Curtis dissimilarity.
    BrayCurtis,
    /// Chi-squared distance.
    ChiSquare,
    /// Continuous Tanimoto similarity.
    Tanimoto,
    /// Ruzicka (weighted Jaccard) similarity.
    Ruzicka,
    /// Pearson correlation coefficient (similarity).
    Pearson,
}

impl Metric {
    /// `true` if larger values mean "more alike" (similarities), `false`
    /// for distances.
    pub fn is_similarity(self) -> bool {
        matches!(
            self,
            Metric::Dot | Metric::Cosine | Metric::Tanimoto | Metric::Ruzicka | Metric::Pearson
        )
    }

    /// `true` if `d(a, b) == d(b, a)` (holds for every metric here).
    pub fn is_symmetric(self) -> bool {
        true
    }

    /// Evaluates this metric on one pair with the SIMD kernels.
    /// Returns `None` on a length mismatch or an undefined value.
    pub fn eval<E: SimdFloat>(self, a: &[E], b: &[E]) -> Option<E> {
        match self {
            Metric::Dot => simd::dot_product_simd(a, b),
            Metric::Cosine => simd::cosine_similarity_simd(a, b),
            Metric::CosineDistance => simd::cosine_distance_simd(a, b),
            Metric::Euclidean => simd::euclidean_distance_simd(a, b),
            Metric::SquaredEuclidean => simd::squared_euclidean_distance_simd(a, b),
            Metric::Manhattan => simd::manhattan_distance_simd(a, b),
            Metric::Chebyshev => simd::chebyshev_distance_simd(a, b),
            Metric::Canberra => simd::canberra_distance_simd(a, b),
            Metric::BrayCurtis => simd::bray_curtis_distance_simd(a, b),
            Metric::ChiSquare => simd::chi_square_distance_simd(a, b),
            Metric::Tanimoto => simd::tanimoto_similarity_simd(a, b),
            Metric::Ruzicka => simd::ruzicka_similarity_simd(a, b),
            Metric::Pearson => simd::pearson_correlation_simd(a, b),
        }
    }
}

/// Error returned by batch functions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchError {
    /// A row has a different length than the first row.
    DimensionMismatch { row: usize, expected: usize, found: usize },
    /// `k` was zero or the flat buffer is not a multiple of `dim`.
    InvalidArgument(&'static str),
}

impl core::fmt::Display for BatchError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BatchError::DimensionMismatch { row, expected, found } => {
                write!(f, "row {row} has length {found}, expected {expected}")
            }
            BatchError::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
        }
    }
}

impl std::error::Error for BatchError {}

/// Dense row-major matrix of results.
#[derive(Debug, Clone, PartialEq)]
pub struct DistanceMatrix<E> {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Row-major values, `data[i * cols + j]`.
    pub data: Vec<E>,
}

impl<E: Copy> DistanceMatrix<E> {
    /// Value at row `i`, column `j`.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> E {
        self.data[i * self.cols + j]
    }

    /// Row `i` as a slice.
    #[inline]
    pub fn row(&self, i: usize) -> &[E] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// Iterator over rows.
    pub fn iter_rows(&self) -> impl Iterator<Item = &[E]> {
        self.data.chunks_exact(self.cols.max(1)).take(self.rows)
    }
}

fn check_dims<E, R: AsRef<[E]>>(rows: &[R], dim: Option<usize>) -> Result<usize, BatchError> {
    let expected = match dim {
        Some(d) => d,
        None => rows.first().map(|r| r.as_ref().len()).unwrap_or(0),
    };
    for (i, r) in rows.iter().enumerate() {
        let found = r.as_ref().len();
        if found != expected {
            return Err(BatchError::DimensionMismatch { row: i, expected, found });
        }
    }
    Ok(expected)
}

/// Per-row L2 norms, used to turn cosine into a single dot product per pair.
fn norms<E: SimdFloat, R: AsRef<[E]> + Sync>(rows: &[R]) -> Vec<E> {
    #[cfg(feature = "parallel")]
    {
        rows.par_iter().map(|r| simd::l2_norm_simd(r.as_ref())).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        rows.iter().map(|r| simd::l2_norm_simd(r.as_ref())).collect()
    }
}

/// Builds a closure evaluating `metric` for (row i of A, row j of B). For
/// cosine metrics the norms are precomputed once so each pair costs one
/// SIMD dot product instead of a three-way reduction.
fn pair_fn<'a, E, R1, R2>(
    a: &'a [R1],
    b: &'a [R2],
    metric: Metric,
    same: bool,
) -> impl Fn(usize, usize) -> E + Sync + 'a
where
    E: SimdFloat,
    R1: AsRef<[E]> + Sync,
    R2: AsRef<[E]> + Sync,
{
    let cos = matches!(metric, Metric::Cosine | Metric::CosineDistance);
    let na = if cos { norms(a) } else { Vec::new() };
    let nb = if cos && !same { norms(b) } else { Vec::new() };
    move |i, j| {
        let (x, y) = (a[i].as_ref(), b[j].as_ref());
        if cos {
            let nbj = if same { na[j] } else { nb[j] };
            let den = na[i] * nbj;
            if den == E::zero() {
                return E::nan();
            }
            let dot = simd::dot_product_simd(x, y).unwrap_or_else(E::nan);
            let c = dot / den;
            if metric == Metric::Cosine {
                c
            } else {
                E::one() - c
            }
        } else {
            metric.eval(x, y).unwrap_or_else(E::nan)
        }
    }
}

/// Full `n × n` matrix of `metric` between all rows.
///
/// Only the upper triangle is computed; it is mirrored into the lower
/// triangle. The diagonal is computed too (so cosine of a zero row is
/// `NaN`, as everywhere else).
///
/// ```
/// use similarity::batch::{pairwise, Metric};
/// let rows = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
/// let m = pairwise(&rows, Metric::Cosine).unwrap();
/// assert!((m.get(0, 0) - 1.0_f64).abs() < 1e-12);
/// assert!(m.get(0, 1).abs() < 1e-12);
/// ```
pub fn pairwise<E, R>(rows: &[R], metric: Metric) -> Result<DistanceMatrix<E>, BatchError>
where
    E: SimdFloat,
    R: AsRef<[E]> + Sync,
{
    check_dims(rows, None)?;
    let n = rows.len();
    let f = pair_fn(rows, rows, metric, true);
    let mut data = vec![E::zero(); n * n];

    let fill_row = |i: usize, out: &mut [E]| {
        for (j, o) in out.iter_mut().enumerate().skip(i) {
            *o = f(i, j);
        }
    };
    #[cfg(feature = "parallel")]
    data.par_chunks_mut(n.max(1)).enumerate().for_each(|(i, r)| fill_row(i, r));
    #[cfg(not(feature = "parallel"))]
    data.chunks_mut(n.max(1)).enumerate().for_each(|(i, r)| fill_row(i, r));

    for i in 0..n {
        for j in 0..i {
            data[i * n + j] = data[j * n + i];
        }
    }
    Ok(DistanceMatrix { rows: n, cols: n, data })
}

/// Condensed upper-triangle vector of `metric` (like SciPy `pdist`):
/// entries `(0,1), (0,2), …, (0,n−1), (1,2), …` with length `n(n−1)/2`.
pub fn pdist<E, R>(rows: &[R], metric: Metric) -> Result<Vec<E>, BatchError>
where
    E: SimdFloat,
    R: AsRef<[E]> + Sync,
{
    check_dims(rows, None)?;
    let n = rows.len();
    let f = pair_fn(rows, rows, metric, true);
    let row_vals = |i: usize| (i + 1..n).map(|j| f(i, j)).collect::<Vec<_>>();
    #[cfg(feature = "parallel")]
    let parts: Vec<Vec<E>> = (0..n).into_par_iter().map(row_vals).collect();
    #[cfg(not(feature = "parallel"))]
    let parts: Vec<Vec<E>> = (0..n).map(row_vals).collect();
    Ok(parts.concat())
}

/// `a.len() × b.len()` matrix of `metric` between every row of `a` and
/// every row of `b` (like SciPy `cdist`).
///
/// ```
/// use similarity::batch::{cdist, Metric};
/// let a = [[0.0, 0.0]];
/// let b = [[3.0, 4.0], [1.0, 0.0]];
/// let m = cdist(&a, &b, Metric::Manhattan).unwrap();
/// assert_eq!(m.row(0), &[7.0, 1.0]);
/// ```
pub fn cdist<E, R1, R2>(a: &[R1], b: &[R2], metric: Metric) -> Result<DistanceMatrix<E>, BatchError>
where
    E: SimdFloat,
    R1: AsRef<[E]> + Sync,
    R2: AsRef<[E]> + Sync,
{
    let dim = check_dims(a, None)?;
    if !a.is_empty() {
        check_dims(b, Some(dim))?;
    } else {
        check_dims(b, None)?;
    }
    let (m, n) = (a.len(), b.len());
    let f = pair_fn(a, b, metric, false);
    let mut data = vec![E::zero(); m * n];
    let fill_row = |i: usize, out: &mut [E]| {
        for (j, o) in out.iter_mut().enumerate() {
            *o = f(i, j);
        }
    };
    #[cfg(feature = "parallel")]
    data.par_chunks_mut(n.max(1)).enumerate().for_each(|(i, r)| fill_row(i, r));
    #[cfg(not(feature = "parallel"))]
    data.chunks_mut(n.max(1)).enumerate().for_each(|(i, r)| fill_row(i, r));
    Ok(DistanceMatrix { rows: m, cols: n, data })
}

/// `metric(query, row)` for every row.
pub fn one_to_many<E, R>(query: &[E], rows: &[R], metric: Metric) -> Result<Vec<E>, BatchError>
where
    E: SimdFloat,
    R: AsRef<[E]> + Sync,
{
    check_dims(rows, Some(query.len()))?;
    let eval = |r: &R| metric.eval(query, r.as_ref()).unwrap_or_else(E::nan);
    #[cfg(feature = "parallel")]
    {
        Ok(rows.par_iter().map(eval).collect())
    }
    #[cfg(not(feature = "parallel"))]
    {
        Ok(rows.iter().map(eval).collect())
    }
}

/// A search hit: row index and metric value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbor<E> {
    pub index: usize,
    pub score: E,
}

/// Returns `true` if score `x` ranks before `y` for this metric.
#[inline]
fn better<E: Float>(metric: Metric, x: E, y: E) -> bool {
    if metric.is_similarity() {
        x > y
    } else {
        x < y
    }
}

fn select_top_k<E: SimdFloat>(scores: Vec<E>, k: usize, metric: Metric) -> Vec<Neighbor<E>> {
    let mut hits: Vec<Neighbor<E>> = scores
        .into_iter()
        .enumerate()
        .filter(|(_, s)| !s.is_nan())
        .map(|(index, score)| Neighbor { index, score })
        .collect();
    let cmp = |a: &Neighbor<E>, b: &Neighbor<E>| {
        if better(metric, a.score, b.score) {
            core::cmp::Ordering::Less
        } else if better(metric, b.score, a.score) {
            core::cmp::Ordering::Greater
        } else {
            a.index.cmp(&b.index)
        }
    };
    if k < hits.len() {
        hits.select_nth_unstable_by(k, cmp);
        hits.truncate(k);
    }
    hits.sort_by(cmp);
    hits
}

/// The `k` rows closest to `query`, best first.
///
/// "Closest" means smallest for distances and largest for similarities
/// (see [`Metric::is_similarity`]). Rows with an undefined score are
/// skipped. Uses an `O(n)` selection rather than a full sort.
///
/// ```
/// use similarity::batch::{top_k, Metric};
/// let db = vec![vec![0.0_f32, 1.0], vec![5.0, 5.0], vec![0.1, 0.9]];
/// let hits = top_k(&[0.0, 1.0], &db, 2, Metric::Euclidean).unwrap();
/// assert_eq!(hits[0].index, 0);
/// assert_eq!(hits[1].index, 2);
/// ```
pub fn top_k<E, R>(query: &[E], rows: &[R], k: usize, metric: Metric) -> Result<Vec<Neighbor<E>>, BatchError>
where
    E: SimdFloat,
    R: AsRef<[E]> + Sync,
{
    if k == 0 {
        return Err(BatchError::InvalidArgument("k must be > 0"));
    }
    let scores = one_to_many(query, rows, metric)?;
    Ok(select_top_k(scores, k, metric))
}

/// [`top_k`] for many queries at once (parallel over queries).
pub fn top_k_many<E, R1, R2>(
    queries: &[R1],
    rows: &[R2],
    k: usize,
    metric: Metric,
) -> Result<Vec<Vec<Neighbor<E>>>, BatchError>
where
    E: SimdFloat,
    R1: AsRef<[E]> + Sync,
    R2: AsRef<[E]> + Sync,
{
    if k == 0 {
        return Err(BatchError::InvalidArgument("k must be > 0"));
    }
    let m = cdist(queries, rows, metric)?;
    let per = |row: &[E]| select_top_k(row.to_vec(), k, metric);
    #[cfg(feature = "parallel")]
    {
        Ok(m.data.par_chunks(m.cols.max(1)).take(m.rows).map(per).collect())
    }
    #[cfg(not(feature = "parallel"))]
    {
        Ok(m.iter_rows().map(per).collect())
    }
}

/// Views a flat row-major buffer as rows of length `dim`.
///
/// ```
/// use similarity::batch::{rows_from_flat, pairwise, Metric};
/// let flat = [0.0, 0.0, 3.0, 4.0];
/// let rows = rows_from_flat(&flat, 2).unwrap();
/// assert_eq!(pairwise(&rows, Metric::Euclidean).unwrap().get(0, 1), 5.0);
/// ```
pub fn rows_from_flat<E>(data: &[E], dim: usize) -> Result<Vec<&[E]>, BatchError> {
    #[allow(clippy::manual_is_multiple_of)] // keeps MSRV below 1.87
    if dim == 0 || data.len() % dim != 0 {
        return Err(BatchError::InvalidArgument("data length must be a positive multiple of dim"));
    }
    Ok(data.chunks_exact(dim).collect())
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use rand::Rng;

    fn rows(n: usize, d: usize) -> Vec<Vec<f64>> {
        let mut rng = rand::rng();
        (0..n).map(|_| (0..d).map(|_| rng.random_range(0.0..1.0)).collect()).collect()
    }

    const ALL: [Metric; 13] = [
        Metric::Dot,
        Metric::Cosine,
        Metric::CosineDistance,
        Metric::Euclidean,
        Metric::SquaredEuclidean,
        Metric::Manhattan,
        Metric::Chebyshev,
        Metric::Canberra,
        Metric::BrayCurtis,
        Metric::ChiSquare,
        Metric::Tanimoto,
        Metric::Ruzicka,
        Metric::Pearson,
    ];

    #[test]
    fn pairwise_matches_eval() {
        let r = rows(9, 13);
        for m in ALL {
            let pm = pairwise(&r, m).unwrap();
            let pd = pdist(&r, m).unwrap();
            assert_eq!(pd.len(), 9 * 8 / 2);
            let mut k = 0;
            for i in 0..9 {
                for j in 0..9 {
                    let e = m.eval(&r[i], &r[j]).unwrap();
                    assert!((pm.get(i, j) - e).abs() < 1e-10, "{m:?} {i} {j}");
                    if j > i {
                        assert!((pd[k] - e).abs() < 1e-10);
                        k += 1;
                    }
                }
            }
        }
    }

    #[test]
    fn cdist_and_topk() {
        let a = rows(4, 7);
        let b = rows(25, 7);
        for m in ALL {
            let cm = cdist(&a, &b, m).unwrap();
            assert_eq!((cm.rows, cm.cols), (4, 25));
            for i in 0..4 {
                for j in 0..25 {
                    assert!((cm.get(i, j) - m.eval(&a[i], &b[j]).unwrap()).abs() < 1e-10);
                }
                let hits = top_k(&a[i], &b, 5, m).unwrap();
                assert_eq!(hits.len(), 5);
                let mut all: Vec<(usize, f64)> = cm.row(i).iter().copied().enumerate().collect();
                if m.is_similarity() {
                    all.sort_by(|x, y| y.1.total_cmp(&x.1));
                } else {
                    all.sort_by(|x, y| x.1.total_cmp(&y.1));
                }
                let expected: Vec<usize> = all.iter().take(5).map(|x| x.0).collect();
                let got: Vec<usize> = hits.iter().map(|h| h.index).collect();
                assert_eq!(got, expected, "{m:?}");
            }
            let many = top_k_many(&a, &b, 3, m).unwrap();
            assert_eq!(many.len(), 4);
        }
    }

    #[test]
    fn errors_and_edge_cases() {
        let bad = vec![vec![1.0, 2.0], vec![1.0]];
        assert_eq!(
            pairwise(&bad, Metric::Dot).unwrap_err(),
            BatchError::DimensionMismatch { row: 1, expected: 2, found: 1 }
        );
        assert!(top_k(&[1.0], &[[1.0]], 0, Metric::Dot).is_err());
        assert!(rows_from_flat(&[1.0, 2.0, 3.0], 2).is_err());
        let empty: Vec<Vec<f32>> = vec![];
        assert_eq!(pairwise(&empty, Metric::Euclidean).unwrap().data.len(), 0);
        let zero = [[0.0_f32, 0.0], [1.0, 0.0]];
        let m = pairwise(&zero, Metric::Cosine).unwrap();
        assert!(m.get(0, 1).is_nan());
        let hits = top_k(&[0.0, 0.0], &zero, 2, Metric::Cosine).unwrap();
        assert!(hits.is_empty());
    }
}
