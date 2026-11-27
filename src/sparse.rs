//! # Sparse vectors
//!
//! [`SparseVector`] stores the non-zero entries of a vector as sorted
//! `(index, value)` pairs. Pairwise metrics walk both index lists in a
//! single merge (switching to galloping search when one vector is much
//! shorter than the other), so the cost depends on the number of non-zeros
//! rather than the dimension.
//!
//! For many-against-many comparisons, [`SparseIndex`] builds an inverted
//! index over a collection of vectors. A query then only touches rows that
//! share at least one feature with it, which makes [`SparseIndex::top_k`] and
//! [`sparse_cdist`] fast for fingerprints, bag-of-words counts and binned
//! spectra.
//!
//! ```
//! use similarity::sparse::{SparseVector, SparseIndex, SparseMetric};
//!
//! let a = SparseVector::new(vec![1, 4, 9], vec![1.0, 2.0, 3.0]).unwrap();
//! let b = SparseVector::new(vec![4, 9, 12], vec![2.0, 3.0, 5.0]).unwrap();
//! assert_eq!(a.dot(&b), 13.0);
//! assert_eq!(a.jaccard_index(&b), 0.5);
//!
//! let index = SparseIndex::new(&[a.clone(), b.clone()]);
//! let hits = index.top_k(&a, 1, SparseMetric::Cosine);
//! assert_eq!(hits[0].index, 0);
//! ```

use crate::batch::{DistanceMatrix, Neighbor};
use num_traits::Float;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Error returned when building a [`SparseVector`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseError {
    /// `indices` and `values` have different lengths.
    LengthMismatch { indices: usize, values: usize },
    /// Indices are not strictly increasing at this position.
    Unsorted { position: usize },
}

impl core::fmt::Display for SparseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SparseError::LengthMismatch { indices, values } => {
                write!(f, "{indices} indices but {values} values")
            }
            SparseError::Unsorted { position } => {
                write!(f, "indices must be strictly increasing (position {position})")
            }
        }
    }
}

impl std::error::Error for SparseError {}

/// A sparse vector: strictly increasing indices with one value each.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SparseVector<E> {
    indices: Vec<u32>,
    values: Vec<E>,
}

impl<E: Float> SparseVector<E> {
    /// Builds a vector from sorted, unique indices and their values.
    pub fn new(indices: Vec<u32>, values: Vec<E>) -> Result<Self, SparseError> {
        if indices.len() != values.len() {
            return Err(SparseError::LengthMismatch { indices: indices.len(), values: values.len() });
        }
        if let Some(p) = indices.windows(2).position(|w| w[0] >= w[1]) {
            return Err(SparseError::Unsorted { position: p + 1 });
        }
        Ok(Self { indices, values })
    }

    /// Builds a vector from `(index, value)` pairs in any order. Values of
    /// repeated indices are added up and zeros are dropped.
    pub fn from_pairs(mut pairs: Vec<(u32, E)>) -> Self {
        pairs.sort_unstable_by_key(|p| p.0);
        let mut indices: Vec<u32> = Vec::with_capacity(pairs.len());
        let mut values: Vec<E> = Vec::with_capacity(pairs.len());
        for (i, v) in pairs {
            if indices.last() == Some(&i) {
                let last = values.len() - 1;
                values[last] = values[last] + v;
            } else {
                indices.push(i);
                values.push(v);
            }
        }
        let mut out = Self { indices, values };
        out.drop_zeros();
        out
    }

    /// Keeps the non-zero entries of a dense slice.
    ///
    /// # Panics
    /// If `dense` has more than `u32::MAX` elements.
    pub fn from_dense(dense: &[E]) -> Self {
        assert!(dense.len() <= u32::MAX as usize, "dimension exceeds u32::MAX");
        let (indices, values) =
            dense.iter().enumerate().filter(|(_, v)| !v.is_zero()).map(|(i, &v)| (i as u32, v)).unzip();
        Self { indices, values }
    }

    /// A binary vector (all values `1`) from sorted, unique indices, e.g. the
    /// on-bits of a molecular fingerprint.
    pub fn binary(indices: Vec<u32>) -> Result<Self, SparseError> {
        let n = indices.len();
        Self::new(indices, vec![E::one(); n])
    }

    /// Dense copy of length `dim`. Entries at or beyond `dim` are dropped.
    pub fn to_dense(&self, dim: usize) -> Vec<E> {
        let mut out = vec![E::zero(); dim];
        for (&i, &v) in self.indices.iter().zip(&self.values) {
            if let Some(o) = out.get_mut(i as usize) {
                *o = v;
            }
        }
        out
    }

    fn drop_zeros(&mut self) {
        let mut k = 0;
        for j in 0..self.indices.len() {
            if !self.values[j].is_zero() {
                self.indices[k] = self.indices[j];
                self.values[k] = self.values[j];
                k += 1;
            }
        }
        self.indices.truncate(k);
        self.values.truncate(k);
    }

    /// Indices of the stored entries.
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Values of the stored entries.
    pub fn values(&self) -> &[E] {
        &self.values
    }

    /// Number of stored entries.
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// `true` if nothing is stored.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Sum of squared values.
    pub fn squared_norm(&self) -> E {
        self.values.iter().fold(E::zero(), |s, &v| s + v * v)
    }

    /// Euclidean norm.
    pub fn norm(&self) -> E {
        self.squared_norm().sqrt()
    }

    /// Visits every index shared by both vectors as `f(value_a, value_b)`.
    fn for_each_common(&self, other: &Self, mut f: impl FnMut(E, E)) {
        let (a, b) = (&self.indices, &other.indices);
        // Galloping pays off when one side is much shorter.
        if a.len() * 16 < b.len() || b.len() * 16 < a.len() {
            let (small, large, swap) = if a.len() < b.len() { (self, other, false) } else { (other, self, true) };
            let mut lo = 0;
            for (k, &i) in small.indices.iter().enumerate() {
                match large.indices[lo..].binary_search(&i) {
                    Ok(p) => {
                        let (x, y) = (small.values[k], large.values[lo + p]);
                        if swap { f(y, x) } else { f(x, y) }
                        lo += p + 1;
                    }
                    Err(p) => lo += p,
                }
                if lo >= large.indices.len() {
                    break;
                }
            }
            return;
        }
        let (mut i, mut j) = (0, 0);
        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                core::cmp::Ordering::Less => i += 1,
                core::cmp::Ordering::Greater => j += 1,
                core::cmp::Ordering::Equal => {
                    f(self.values[i], other.values[j]);
                    i += 1;
                    j += 1;
                }
            }
        }
    }

    /// Visits the union of both index sets as `f(value_a, value_b)`, with
    /// zero for a missing side.
    fn for_each_union(&self, other: &Self, mut f: impl FnMut(E, E)) {
        let (a, b) = (&self.indices, &other.indices);
        let (mut i, mut j) = (0, 0);
        while i < a.len() || j < b.len() {
            if j == b.len() || (i < a.len() && a[i] < b[j]) {
                f(self.values[i], E::zero());
                i += 1;
            } else if i == a.len() || b[j] < a[i] {
                f(E::zero(), other.values[j]);
                j += 1;
            } else {
                f(self.values[i], other.values[j]);
                i += 1;
                j += 1;
            }
        }
    }

    /// Dot product.
    pub fn dot(&self, other: &Self) -> E {
        let mut s = E::zero();
        self.for_each_common(other, |x, y| s = s + x * y);
        s
    }

    /// Cosine similarity. `None` if either vector has zero norm.
    pub fn cosine_similarity(&self, other: &Self) -> Option<E> {
        let den = self.norm() * other.norm();
        if den.is_zero() {
            None
        } else {
            Some(self.dot(other) / den)
        }
    }

    /// Squared Euclidean distance (computed exactly over the union of indices).
    pub fn squared_euclidean_distance(&self, other: &Self) -> E {
        let mut s = E::zero();
        self.for_each_union(other, |x, y| s = s + (x - y) * (x - y));
        s
    }

    /// Euclidean distance.
    pub fn euclidean_distance(&self, other: &Self) -> E {
        self.squared_euclidean_distance(other).sqrt()
    }

    /// Manhattan (L1) distance.
    pub fn manhattan_distance(&self, other: &Self) -> E {
        let mut s = E::zero();
        self.for_each_union(other, |x, y| s = s + (x - y).abs());
        s
    }

    /// Jaccard index of the index sets (values are ignored). `0` for two
    /// empty vectors, as in [`crate::jaccard_index`].
    pub fn jaccard_index(&self, other: &Self) -> E {
        let mut common = 0usize;
        self.for_each_common(other, |_, _| common += 1);
        jaccard_from_counts(common, self.nnz(), other.nnz())
    }

    /// Continuous Tanimoto similarity `a·b / (|a|² + |b|² − a·b)`. Equals
    /// the Jaccard index for binary vectors. `None` if both are zero.
    pub fn tanimoto_similarity(&self, other: &Self) -> Option<E> {
        tanimoto_from_dot(self.dot(other), self.squared_norm(), other.squared_norm())
    }

    /// Ruzicka (weighted Jaccard) similarity `Σ min / Σ max` for
    /// non-negative values. `None` if `Σ max` is zero.
    pub fn ruzicka_similarity(&self, other: &Self) -> Option<E> {
        let (mut lo, mut hi) = (E::zero(), E::zero());
        self.for_each_union(other, |x, y| {
            lo = lo + x.min(y);
            hi = hi + x.max(y);
        });
        if hi.is_zero() {
            None
        } else {
            Some(lo / hi)
        }
    }
}

fn jaccard_from_counts<E: Float>(common: usize, na: usize, nb: usize) -> E {
    let union = na + nb - common;
    if union == 0 {
        E::zero()
    } else {
        E::from(common).unwrap() / E::from(union).unwrap()
    }
}

fn tanimoto_from_dot<E: Float>(dot: E, sa: E, sb: E) -> Option<E> {
    let den = sa + sb - dot;
    if den.is_zero() {
        None
    } else {
        Some(dot / den)
    }
}

/// Metrics supported by [`SparseIndex`] and [`sparse_cdist`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SparseMetric {
    /// Dot product (similarity).
    Dot,
    /// Cosine similarity.
    Cosine,
    /// `1 − cosine`.
    CosineDistance,
    /// Euclidean distance, from norms and the dot product.
    Euclidean,
    /// Squared Euclidean distance, from norms and the dot product.
    SquaredEuclidean,
    /// Continuous Tanimoto similarity.
    Tanimoto,
    /// Jaccard index of the index sets.
    Jaccard,
}

impl SparseMetric {
    /// `true` if larger values mean more similar.
    pub fn is_similarity(self) -> bool {
        matches!(self, Self::Dot | Self::Cosine | Self::Tanimoto | Self::Jaccard)
    }
}

/// An inverted index (feature → rows containing it) over sparse vectors.
#[derive(Debug, Clone)]
pub struct SparseIndex<E> {
    /// Distinct features, sorted.
    features: Vec<u32>,
    /// `offsets[f]..offsets[f + 1]` are the postings of `features[f]`.
    offsets: Vec<usize>,
    rows: Vec<u32>,
    values: Vec<E>,
    sq_norms: Vec<E>,
    nnz: Vec<usize>,
}

impl<E: Float + Send + Sync> SparseIndex<E> {
    /// Builds the index.
    ///
    /// # Panics
    /// If there are more than `u32::MAX` vectors.
    pub fn new(vectors: &[SparseVector<E>]) -> Self {
        assert!(vectors.len() <= u32::MAX as usize, "too many vectors");
        let mut postings: Vec<(u32, u32, E)> = vectors
            .iter()
            .enumerate()
            .flat_map(|(r, v)| v.indices.iter().zip(&v.values).map(move |(&f, &x)| (f, r as u32, x)))
            .collect();
        postings.sort_unstable_by_key(|p| (p.0, p.1));
        let mut features = Vec::new();
        let mut offsets = vec![];
        for (k, p) in postings.iter().enumerate() {
            if features.last() != Some(&p.0) {
                features.push(p.0);
                offsets.push(k);
            }
        }
        offsets.push(postings.len());
        Self {
            features,
            offsets,
            rows: postings.iter().map(|p| p.1).collect(),
            values: postings.iter().map(|p| p.2).collect(),
            sq_norms: vectors.iter().map(|v| v.squared_norm()).collect(),
            nnz: vectors.iter().map(|v| v.nnz()).collect(),
        }
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.sq_norms.len()
    }

    /// `true` if no vectors are indexed.
    pub fn is_empty(&self) -> bool {
        self.sq_norms.is_empty()
    }

    /// Dot products and shared-feature counts of `query` with every row.
    fn accumulate(&self, query: &SparseVector<E>) -> (Vec<E>, Vec<u32>) {
        let mut dot = vec![E::zero(); self.len()];
        let mut common = vec![0u32; self.len()];
        let mut lo = 0;
        for (&f, &x) in query.indices.iter().zip(&query.values) {
            match self.features[lo..].binary_search(&f) {
                Ok(p) => {
                    let k = lo + p;
                    for q in self.offsets[k]..self.offsets[k + 1] {
                        let r = self.rows[q] as usize;
                        dot[r] = dot[r] + x * self.values[q];
                        common[r] += 1;
                    }
                    lo = k + 1;
                }
                Err(p) => lo += p,
            }
            if lo >= self.features.len() {
                break;
            }
        }
        (dot, common)
    }

    /// `metric(query, row)` for every indexed row, in row order. Undefined
    /// values (cosine with a zero vector, Tanimoto of two zero vectors) are
    /// `NaN`.
    pub fn scores(&self, query: &SparseVector<E>, metric: SparseMetric) -> Vec<E> {
        let (dot, common) = self.accumulate(query);
        let (qs, qn) = (query.squared_norm(), query.nnz());
        let qnorm = qs.sqrt();
        dot.iter()
            .zip(&common)
            .enumerate()
            .map(|(r, (&d, &c))| {
                let rs = self.sq_norms[r];
                match metric {
                    SparseMetric::Dot => d,
                    SparseMetric::Cosine | SparseMetric::CosineDistance => {
                        let den = qnorm * rs.sqrt();
                        if den.is_zero() {
                            E::nan()
                        } else if metric == SparseMetric::Cosine {
                            d / den
                        } else {
                            E::one() - d / den
                        }
                    }
                    SparseMetric::SquaredEuclidean => (qs + rs - d - d).max(E::zero()),
                    SparseMetric::Euclidean => (qs + rs - d - d).max(E::zero()).sqrt(),
                    SparseMetric::Tanimoto => tanimoto_from_dot(d, qs, rs).unwrap_or_else(E::nan),
                    SparseMetric::Jaccard => jaccard_from_counts(c as usize, qn, self.nnz[r]),
                }
            })
            .collect()
    }

    /// The `k` best rows for `query` (largest similarity or smallest
    /// distance), best first, ties by row index. `NaN` scores are skipped.
    pub fn top_k(&self, query: &SparseVector<E>, k: usize, metric: SparseMetric) -> Vec<Neighbor<E>> {
        let mut hits: Vec<Neighbor<E>> = self
            .scores(query, metric)
            .into_iter()
            .enumerate()
            .filter(|(_, s)| !s.is_nan())
            .map(|(index, score)| Neighbor { index, score })
            .collect();
        let sim = metric.is_similarity();
        let cmp = |a: &Neighbor<E>, b: &Neighbor<E>| {
            let o = if sim { b.score.partial_cmp(&a.score) } else { a.score.partial_cmp(&b.score) };
            o.unwrap_or(core::cmp::Ordering::Equal).then(a.index.cmp(&b.index))
        };
        if k == 0 {
            return Vec::new();
        }
        if k < hits.len() {
            hits.select_nth_unstable_by(k, cmp);
            hits.truncate(k);
        }
        hits.sort_by(cmp);
        hits
    }

    /// [`top_k`](Self::top_k) for many queries (parallel with the
    /// `parallel` feature).
    pub fn top_k_many(&self, queries: &[SparseVector<E>], k: usize, metric: SparseMetric) -> Vec<Vec<Neighbor<E>>> {
        #[cfg(feature = "parallel")]
        {
            queries.par_iter().map(|q| self.top_k(q, k, metric)).collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            queries.iter().map(|q| self.top_k(q, k, metric)).collect()
        }
    }
}

/// `a.len() × b.len()` matrix of `metric` between sparse vectors, computed
/// through an inverted index over `b` (parallel over rows of `a`).
pub fn sparse_cdist<E: Float + Send + Sync>(
    a: &[SparseVector<E>],
    b: &[SparseVector<E>],
    metric: SparseMetric,
) -> DistanceMatrix<E> {
    let index = SparseIndex::new(b);
    #[cfg(feature = "parallel")]
    let rows: Vec<Vec<E>> = a.par_iter().map(|q| index.scores(q, metric)).collect();
    #[cfg(not(feature = "parallel"))]
    let rows: Vec<Vec<E>> = a.iter().map(|q| index.scores(q, metric)).collect();
    DistanceMatrix { rows: a.len(), cols: b.len(), data: rows.concat() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn random(rng: &mut impl Rng, dim: u32, density: f64) -> SparseVector<f64> {
        let mut pairs = Vec::new();
        for i in 0..dim {
            if rng.random_bool(density) {
                pairs.push((i, rng.random_range(0.1..2.0)));
            }
        }
        SparseVector::from_pairs(pairs)
    }

    #[test]
    fn construction() {
        assert!(SparseVector::new(vec![1, 1], vec![1.0, 2.0]).is_err());
        assert!(SparseVector::new(vec![2, 1], vec![1.0, 2.0]).is_err());
        assert!(SparseVector::<f64>::new(vec![1], vec![]).is_err());
        let v = SparseVector::from_pairs(vec![(5, 1.0), (2, 2.0), (5, 3.0), (7, 0.0)]);
        assert_eq!(v.indices(), &[2, 5]);
        assert_eq!(v.values(), &[2.0, 4.0]);
        let d = [0.0, 1.5, 0.0, -2.0];
        let s = SparseVector::from_dense(&d);
        assert_eq!(s.nnz(), 2);
        assert_eq!(s.to_dense(4), d.to_vec());
        assert_eq!(s.to_dense(2), vec![0.0, 1.5]);
        assert_eq!(SparseVector::<f32>::binary(vec![1, 3]).unwrap().values(), &[1.0, 1.0]);
    }

    #[test]
    fn pairwise_matches_dense() {
        let mut rng = rand::rng();
        for &(da, db) in &[(0.1, 0.1), (0.5, 0.02), (0.01, 0.6), (0.0, 0.3)] {
            let (a, b) = (random(&mut rng, 400, da), random(&mut rng, 400, db));
            let (x, y) = (a.to_dense(400), b.to_dense(400));
            let close = |p: f64, q: f64| (p - q).abs() < 1e-9 * (1.0 + q.abs());
            assert!(close(a.dot(&b), x.iter().zip(&y).map(|(p, q)| p * q).sum()));
            assert!(close(a.squared_euclidean_distance(&b), x.iter().zip(&y).map(|(p, q)| (p - q) * (p - q)).sum()));
            assert!(close(a.manhattan_distance(&b), x.iter().zip(&y).map(|(p, q)| (p - q).abs()).sum()));
            match (a.cosine_similarity(&b), crate::simd::cosine_similarity_simd(&x, &y)) {
                (Some(s), Some(d)) => assert!(close(s, d)),
                (s, _) => assert!(s.is_none() && (a.is_empty() || b.is_empty())),
            }
            if let Some(t) = a.tanimoto_similarity(&b) {
                assert!(close(t, crate::simd::tanimoto_similarity_simd(&x, &y).unwrap()));
            }
            if let Some(r) = a.ruzicka_similarity(&b) {
                assert!(close(r, crate::simd::ruzicka_similarity_simd(&x, &y).unwrap()));
            }
            let sa: std::collections::HashSet<u32> = a.indices().iter().copied().collect();
            let sb: std::collections::HashSet<u32> = b.indices().iter().copied().collect();
            assert!(close(a.jaccard_index(&b), crate::jaccard_index(&sa, &sb)));
        }
        // binary Tanimoto == Jaccard
        let p = SparseVector::<f64>::binary(vec![1, 2, 3, 8]).unwrap();
        let q = SparseVector::<f64>::binary(vec![2, 3, 9]).unwrap();
        assert_eq!(p.tanimoto_similarity(&q), Some(p.jaccard_index(&q)));
    }

    #[test]
    fn index_matches_pairwise() {
        let mut rng = rand::rng();
        let db: Vec<SparseVector<f64>> = (0..80).map(|i| random(&mut rng, 300, if i == 3 { 0.0 } else { 0.05 })).collect();
        let qs: Vec<SparseVector<f64>> = (0..10).map(|_| random(&mut rng, 300, 0.08)).collect();
        let metrics = [
            SparseMetric::Dot,
            SparseMetric::Cosine,
            SparseMetric::CosineDistance,
            SparseMetric::Euclidean,
            SparseMetric::SquaredEuclidean,
            SparseMetric::Tanimoto,
            SparseMetric::Jaccard,
        ];
        for metric in metrics {
            let m = sparse_cdist(&qs, &db, metric);
            assert_eq!((m.rows, m.cols), (10, 80));
            for (i, q) in qs.iter().enumerate() {
                for (j, r) in db.iter().enumerate() {
                    let want = match metric {
                        SparseMetric::Dot => q.dot(r),
                        SparseMetric::Cosine => q.cosine_similarity(r).unwrap_or(f64::NAN),
                        SparseMetric::CosineDistance => q.cosine_similarity(r).map_or(f64::NAN, |c| 1.0 - c),
                        SparseMetric::Euclidean => q.euclidean_distance(r),
                        SparseMetric::SquaredEuclidean => q.squared_euclidean_distance(r),
                        SparseMetric::Tanimoto => q.tanimoto_similarity(r).unwrap_or(f64::NAN),
                        SparseMetric::Jaccard => q.jaccard_index(r),
                    };
                    let got = m.get(i, j);
                    assert!((got.is_nan() && want.is_nan()) || (got - want).abs() < 1e-9, "{metric:?} ({i},{j}): {got} vs {want}");
                }
            }
            let index = SparseIndex::new(&db);
            let hits = index.top_k_many(&qs, 5, metric);
            for (i, h) in hits.iter().enumerate() {
                assert_eq!(h.len(), 5);
                let row = m.row(i);
                for w in h.windows(2) {
                    if metric.is_similarity() {
                        assert!(w[0].score >= w[1].score);
                    } else {
                        assert!(w[0].score <= w[1].score);
                    }
                }
                let best = row.iter().copied().filter(|x| !x.is_nan()).fold(
                    if metric.is_similarity() { f64::NEG_INFINITY } else { f64::INFINITY },
                    |a, b| if metric.is_similarity() { a.max(b) } else { a.min(b) },
                );
                assert_eq!(h[0].score, best);
            }
        }
        assert!(SparseIndex::<f32>::new(&[]).is_empty());
    }
}
