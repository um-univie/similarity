//! Property-based tests: metric axioms, SIMD vs scalar agreement and
//! agreement between the different implementations of the same score.

use proptest::prelude::*;
use similarity::batch::{self, Metric};
use similarity::flash_entropy::{FlashEntropyIndex, FlashEntropyOptions};
use similarity::library_search::{LibrarySearchOptions, SpectralLibrary};
use similarity::sparse::{SparseIndex, SparseMetric, SparseVector};
use similarity::{distance_metrics as dm, divergences as dv, simd, MzTolerance, Spectrum};

/// `n` cases unless `PROPTEST_CASES` is set (for longer local runs).
fn cases(n: u32) -> ProptestConfig {
    let mut c = ProptestConfig::default();
    if std::env::var_os("PROPTEST_CASES").is_none() {
        c.cases = n;
    }
    c
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a.is_nan() && b.is_nan()) || (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
}

/// Two vectors of the same length (1..=70, so every SIMD tail length occurs).
fn pair(lo: f64, hi: f64) -> impl Strategy<Value = (Vec<f64>, Vec<f64>)> {
    (1usize..=70).prop_flat_map(move |n| (prop::collection::vec(lo..hi, n), prop::collection::vec(lo..hi, n)))
}

fn triple(lo: f64, hi: f64) -> impl Strategy<Value = (Vec<f64>, Vec<f64>, Vec<f64>)> {
    (1usize..=40).prop_flat_map(move |n| {
        (
            prop::collection::vec(lo..hi, n),
            prop::collection::vec(lo..hi, n),
            prop::collection::vec(lo..hi, n),
        )
    })
}

fn naive_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

proptest! {
    #![proptest_config(cases(200))]

    #[test]
    fn simd_matches_naive((a, b) in pair(-100.0, 100.0)) {
        let sq: f64 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
        prop_assert!(close(simd::dot_product_simd(&a, &b).unwrap(), naive_dot(&a, &b), 1e-10));
        prop_assert!(close(simd::squared_euclidean_distance_simd(&a, &b).unwrap(), sq, 1e-10));
        prop_assert!(close(simd::manhattan_distance_simd(&a, &b).unwrap(), dm::manhattan_distance(&a, &b).unwrap(), 1e-10));
        prop_assert!(close(simd::chebyshev_distance_simd(&a, &b).unwrap(), dm::chebyshev_distance(&a, &b).unwrap(), 0.0));
        prop_assert!(close(simd::canberra_distance_simd(&a, &b).unwrap(), dm::canberra_distance(&a, &b).unwrap(), 1e-10));
        prop_assert!(close(simd::sum_simd(&a), a.iter().sum(), 1e-10));

        // f32 kernels against an f64 reference
        let (a32, b32): (Vec<f32>, Vec<f32>) = (a.iter().map(|&x| x as f32).collect(), b.iter().map(|&x| x as f32).collect());
        let (a64, b64): (Vec<f64>, Vec<f64>) = (a32.iter().map(|&x| x as f64).collect(), b32.iter().map(|&x| x as f64).collect());
        let scale: f64 = a64.iter().zip(&b64).map(|(x, y)| (x * y).abs()).sum::<f64>() + 1.0;
        let got = simd::dot_product_simd(&a32, &b32).unwrap() as f64;
        prop_assert!((got - naive_dot(&a64, &b64)).abs() <= 1e-5 * scale);
    }

    #[test]
    fn distances_are_symmetric_and_zero_on_identity((a, b) in pair(-10.0, 10.0)) {
        for metric in [Metric::Euclidean, Metric::Manhattan, Metric::Chebyshev, Metric::Canberra, Metric::CosineDistance] {
            let (ab, ba) = (metric.eval(&a, &b), metric.eval(&b, &a));
            match (ab, ba) {
                (Some(x), Some(y)) => prop_assert!(close(x, y, 1e-12), "{metric:?}"),
                (x, y) => prop_assert_eq!(x.is_none(), y.is_none()),
            }
            if metric != Metric::CosineDistance {
                prop_assert_eq!(metric.eval(&a, &a).unwrap(), 0.0, "{:?}", metric);
            }
        }
    }

    #[test]
    fn triangle_inequality((a, b, c) in triple(-10.0, 10.0)) {
        for metric in [Metric::Euclidean, Metric::Manhattan, Metric::Chebyshev] {
            let d = |x: &[f64], y: &[f64]| metric.eval(x, y).unwrap();
            prop_assert!(d(&a, &c) <= d(&a, &b) + d(&b, &c) + 1e-9, "{metric:?}");
        }
    }

    #[test]
    fn divergence_properties((p, q, r) in triple(0.0, 1.0)) {
        prop_assume!(p.iter().sum::<f64>() > 1e-3 && q.iter().sum::<f64>() > 1e-3 && r.iter().sum::<f64>() > 1e-3);
        // SIMD and scalar agree
        prop_assert!(close(simd::js_divergence_simd(&p, &q).unwrap(), dv::js_divergence(&p, &q).unwrap(), 1e-9));
        prop_assert!(close(simd::hellinger_distance_simd(&p, &q).unwrap(), dv::hellinger_distance(&p, &q).unwrap(), 1e-9));
        prop_assert!(close(simd::shannon_entropy_simd(&p).unwrap(), dv::shannon_entropy(&p).unwrap(), 1e-9));
        // JS distance and Hellinger are metrics bounded by 1
        for f in [simd::js_distance_simd::<f64>, simd::hellinger_distance_simd::<f64>] {
            let (pq, qr, pr) = (f(&p, &q).unwrap(), f(&q, &r).unwrap(), f(&p, &r).unwrap());
            prop_assert!((0.0..=1.0 + 1e-12).contains(&pq));
            prop_assert!(close(pq, f(&q, &p).unwrap(), 1e-9));
            prop_assert!(pr <= pq + qr + 1e-7);
        }
        // KL is non-negative and zero on identity
        if let Some(kl) = simd::kl_divergence_simd(&p, &q) {
            prop_assert!(kl >= -1e-12);
        }
        prop_assert!(simd::kl_divergence_simd(&p, &p).unwrap().abs() < 1e-12);
    }

    #[test]
    fn vector_ln_exp_match_std(xs in prop::collection::vec(prop_oneof![1e-300f64..1e300, 1e-6f64..10.0, -700.0f64..700.0], 1..64)) {
        let pos: Vec<f64> = xs.iter().map(|x| x.abs()).collect();
        let mut l = pos.clone();
        simd::ln_simd(&mut l);
        for (x, y) in pos.iter().zip(&l) {
            prop_assert!(close(*y, x.ln(), 4.0 * f64::EPSILON) || (x.ln().abs() < 1e-300 && y.abs() < 1e-15), "ln({x})");
        }
        let small: Vec<f64> = xs.iter().map(|x| x.clamp(-700.0, 700.0)).collect();
        let mut e = small.clone();
        simd::exp_simd(&mut e);
        for (x, y) in small.iter().zip(&e) {
            prop_assert!(close(*y, x.exp(), 4.0 * f64::EPSILON), "exp({x})");
        }
    }

    #[test]
    fn tiled_cdist_matches_eval(
        (m, n, d) in (1usize..20, 1usize..20, 0usize..40),
        seed in any::<u64>(),
    ) {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let a: Vec<Vec<f32>> = (0..m).map(|_| (0..d).map(|_| rng.random_range(-1.0..1.0)).collect()).collect();
        let b: Vec<Vec<f32>> = (0..n).map(|_| (0..d).map(|_| rng.random_range(-1.0..1.0)).collect()).collect();
        for metric in [Metric::Dot, Metric::Cosine, Metric::Euclidean, Metric::SquaredEuclidean] {
            let c = batch::cdist(&a, &b, metric).unwrap();
            for (i, ai) in a.iter().enumerate() {
                for (j, bj) in b.iter().enumerate() {
                    let want = metric.eval(ai, bj).map_or(f64::NAN, f64::from);
                    prop_assert!(close(c.get(i, j) as f64, want, 1e-5), "{metric:?} ({i},{j})");
                }
            }
        }
    }

    #[test]
    fn sparse_matches_dense(
        a in prop::collection::vec(prop_oneof![3 => Just(0.0f64), 1 => 0.1f64..5.0], 0..120),
        b in prop::collection::vec(prop_oneof![3 => Just(0.0f64), 1 => 0.1f64..5.0], 0..120),
    ) {
        let dim = a.len().max(b.len());
        let (mut x, mut y) = (a.clone(), b.clone());
        x.resize(dim, 0.0);
        y.resize(dim, 0.0);
        let (sa, sb) = (SparseVector::from_dense(&x), SparseVector::from_dense(&y));
        prop_assert!(close(sa.dot(&sb), naive_dot(&x, &y), 1e-10));
        prop_assert!(close(sa.manhattan_distance(&sb), x.iter().zip(&y).map(|(p, q)| (p - q).abs()).sum(), 1e-10));
        prop_assert_eq!(sa.to_dense(dim), x.clone());
        let idx = SparseIndex::new(std::slice::from_ref(&sb));
        prop_assert!(close(idx.scores(&sa, SparseMetric::Dot)[0], sa.dot(&sb), 1e-10));
        prop_assert!(close(idx.scores(&sa, SparseMetric::Jaccard)[0], sa.jaccard_index(&sb), 1e-12));
        if dim > 0 {
            prop_assert!(close(sa.euclidean_distance(&sb), simd::euclidean_distance_simd(&x, &y).unwrap(), 1e-10));
        }
    }
}

fn spectrum() -> impl Strategy<Value = Spectrum<f64>> {
    prop::collection::vec((50.0f64..500.0, 0.0f64..100.0), 0..25).prop_map(|peaks| {
        let (mz, int): (Vec<f64>, Vec<f64>) = peaks.into_iter().unzip();
        Spectrum::from_arrays(&mz, &int).unwrap()
    })
}

proptest! {
    #![proptest_config(cases(64))]

    #[test]
    fn spectral_scores_agree(lib in prop::collection::vec(spectrum(), 1..12), query in spectrum()) {
        let tol = MzTolerance::Absolute(0.05);
        let flash = FlashEntropyIndex::new(&lib, FlashEntropyOptions::absolute(0.05));
        let sl = SpectralLibrary::new(&lib, LibrarySearchOptions::entropy(tol));
        let f = flash.search(&query);
        for (i, (s, _)) in sl.score_all(&query).into_iter().enumerate() {
            prop_assert!((0.0..=1.0).contains(&s));
            prop_assert!(close(s, f[i], 1e-9), "lib {i}: {s} vs {}", f[i]);
        }
        // a cleaned spectrum scores 1 against itself
        for (i, spec) in lib.iter().enumerate() {
            if spec.peaks.iter().any(|p| p.intensity > 0.0) {
                prop_assert!(close(flash.search(spec)[i], 1.0, 1e-9));
            }
        }
        // matched entropy similarity is symmetric
        let m = similarity::calculate_entropy_similarity_matched(&query, &lib[0], tol);
        let r = similarity::calculate_entropy_similarity_matched(&lib[0], &query, tol);
        prop_assert!(close(m, r, 1e-9));
    }
}
