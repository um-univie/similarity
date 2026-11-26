//! Property-style tests: every SIMD kernel against a straightforward scalar
//! reference, for both precisions and for every length in `0..LEN` so that
//! all head/unroll/tail combinations of every instruction set are covered.

use super::*;
use rand::Rng;

const LEN: usize = 70;

fn rv(n: usize) -> Vec<f64> {
    let mut rng = rand::rng();
    (0..n).map(|_| rng.random_range(-5.0..5.0)).collect()
}

fn rv_pos(n: usize) -> Vec<f64> {
    let mut rng = rand::rng();
    (0..n)
        .map(|_| if rng.random_bool(0.2) { 0.0 } else { rng.random_range(0.0..5.0) })
        .collect()
}

fn f32v(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
}

fn opt_close(a: Option<f64>, b: Option<f64>, tol: f64) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => close(x, y, tol),
        _ => false,
    }
}

/// Checks a two-slice metric in f64 and f32 against a reference.
fn check2(
    name: &str,
    gen: fn(usize) -> Vec<f64>,
    f64f: fn(&[f64], &[f64]) -> Option<f64>,
    f32f: fn(&[f32], &[f32]) -> Option<f32>,
    reference: fn(&[f64], &[f64]) -> Option<f64>,
) {
    for n in 0..LEN {
        let a = gen(n);
        let b = gen(n);
        let r = reference(&a, &b);
        let got = f64f(&a, &b);
        assert!(opt_close(got, r, 1e-10), "{name} f64 n={n}: {got:?} vs {r:?}");
        let r32 = reference(&f32v(&a).iter().map(|&x| x as f64).collect::<Vec<_>>(),
                            &f32v(&b).iter().map(|&x| x as f64).collect::<Vec<_>>());
        let got32 = f32f(&f32v(&a), &f32v(&b)).map(|x| x as f64);
        assert!(opt_close(got32, r32, 1e-3), "{name} f32 n={n}: {got32:?} vs {r32:?}");
    }
    // length mismatch
    assert!(f64f(&[1.0, 2.0], &[1.0]).is_none(), "{name} mismatch");
}

fn zip<'a>(a: &'a [f64], b: &'a [f64]) -> impl Iterator<Item = (f64, f64)> + 'a {
    a.iter().copied().zip(b.iter().copied())
}

#[test]
fn dot() {
    super::isa::for_each_isa(|_isa| {
        check2("dot", rv, dot_product_simd, dot_product_simd, |a, b| Some(zip(a, b).map(|(x, y)| x * y).sum()));
    });
}

#[test]
fn sq_euclid() {
    super::isa::for_each_isa(|_isa| {
        check2("sqeuc", rv, squared_euclidean_distance_simd, squared_euclidean_distance_simd,
            |a, b| Some(zip(a, b).map(|(x, y)| (x - y) * (x - y)).sum()));
        check2("euc", rv, euclidean_distance_simd, euclidean_distance_simd,
            |a, b| Some(zip(a, b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()));
    });
}

#[test]
fn manhattan_chebyshev() {
    super::isa::for_each_isa(|_isa| {
        check2("man", rv, manhattan_distance_simd, manhattan_distance_simd,
            |a, b| Some(zip(a, b).map(|(x, y)| (x - y).abs()).sum()));
        check2("cheb", rv, chebyshev_distance_simd, chebyshev_distance_simd,
            |a, b| Some(zip(a, b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)));
    });
}

fn cos_ref(a: &[f64], b: &[f64]) -> Option<f64> {
    let d: f64 = zip(a, b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum();
    let nb: f64 = b.iter().map(|x| x * x).sum();
    let den = (na * nb).sqrt();
    (den != 0.0).then(|| d / den)
}

#[test]
fn cosine_family() {
    super::isa::for_each_isa(|_isa| {
        check2("cos", rv, cosine_similarity_simd, cosine_similarity_simd, cos_ref);
        check2("cosd", rv, cosine_distance_simd, cosine_distance_simd, |a, b| cos_ref(a, b).map(|c| 1.0 - c));
        check2("tanimoto", rv, tanimoto_similarity_simd, tanimoto_similarity_simd, |a, b| {
            let d: f64 = zip(a, b).map(|(x, y)| x * y).sum();
            let na: f64 = a.iter().map(|x| x * x).sum();
            let nb: f64 = b.iter().map(|x| x * x).sum();
            let den = na + nb - d;
            (den != 0.0).then(|| d / den)
        });
        check2("dice", rv, dice_similarity_simd, dice_similarity_simd, |a, b| {
            let d: f64 = zip(a, b).map(|(x, y)| x * y).sum();
            let den: f64 = a.iter().map(|x| x * x).sum::<f64>() + b.iter().map(|x| x * x).sum::<f64>();
            (den != 0.0).then(|| 2.0 * d / den)
        });
    });
}

#[test]
fn ratio_distances() {
    super::isa::for_each_isa(|_isa| {
        check2("canberra", rv_pos, canberra_distance_simd, canberra_distance_simd, |a, b| {
            Some(zip(a, b).map(|(x, y)| {
                let d = x.abs() + y.abs();
                if d == 0.0 { 0.0 } else { (x - y).abs() / d }
            }).sum())
        });
        check2("bray", rv_pos, bray_curtis_distance_simd, bray_curtis_distance_simd, |a, b| {
            let n: f64 = zip(a, b).map(|(x, y)| (x - y).abs()).sum();
            let d: f64 = zip(a, b).map(|(x, y)| (x + y).abs()).sum();
            (d != 0.0).then(|| n / d)
        });
        check2("chi2", rv_pos, chi_square_distance_simd, chi_square_distance_simd, |a, b| {
            Some(zip(a, b).map(|(x, y)| if x + y == 0.0 { 0.0 } else { (x - y).powi(2) / (x + y) }).sum())
        });
        check2("hist", rv_pos, histogram_intersection_simd, histogram_intersection_simd,
            |a, b| Some(zip(a, b).map(|(x, y)| x.min(y)).sum()));
        check2("ruzicka", rv_pos, ruzicka_similarity_simd, ruzicka_similarity_simd, |a, b| {
            let mn: f64 = zip(a, b).map(|(x, y)| x.min(y)).sum();
            let mx: f64 = zip(a, b).map(|(x, y)| x.max(y)).sum();
            (mx != 0.0).then(|| mn / mx)
        });
    });
}

#[test]
fn hamming() {
    super::isa::for_each_isa(|_isa| {
        for n in 0..LEN {
            let a = rv(n);
            let mut b = a.clone();
            let mut expected = 0;
            for (i, v) in b.iter_mut().enumerate() {
                if i % 3 == 0 {
                    *v += 1.0;
                    expected += 1;
                }
            }
            assert_eq!(hamming_distance_simd(&a, &b), Some(expected));
            assert_eq!(hamming_distance_simd(&f32v(&a), &f32v(&b)), Some(expected));
        }
    });
}

#[test]
fn weighted() {
    super::isa::for_each_isa(|_isa| {
        for n in 0..LEN {
            let (a, b, w) = (rv(n), rv(n), rv_pos(n));
            let r: f64 = (0..n).map(|i| w[i] * (a[i] - b[i]).powi(2)).sum();
            assert!(close(weighted_squared_euclidean_distance_simd(&a, &b, &w).unwrap(), r, 1e-10));
            assert!(close(weighted_euclidean_distance_simd(&a, &b, &w).unwrap(), r.sqrt(), 1e-10));
            let r: f64 = (0..n).map(|i| w[i] * (a[i] - b[i]).abs()).sum();
            assert!(close(weighted_manhattan_distance_simd(&a, &b, &w).unwrap(), r, 1e-10));
        }
        assert!(weighted_euclidean_distance_simd(&[1.0], &[1.0], &[]).is_none());
    });
}

fn pearson_ref(a: &[f64], b: &[f64]) -> (f64, f64, f64) {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    zip(a, b).fold((0.0, 0.0, 0.0), |(c, va, vb), (x, y)| {
        (c + (x - ma) * (y - mb), va + (x - ma).powi(2), vb + (y - mb).powi(2))
    })
}

#[test]
fn correlation() {
    super::isa::for_each_isa(|_isa| {
        for n in 1..LEN {
            let (a, b) = (rv(n), rv(n));
            let (c, va, vb) = pearson_ref(&a, &b);
            let den = (va * vb).sqrt();
            let expected = (den != 0.0).then(|| c / den);
            assert!(opt_close(pearson_correlation_simd(&a, &b), expected, 1e-9), "n={n}");
            if expected.is_some() { assert!(opt_close(pearson_correlation_distance_simd(&a, &b), expected.map(|r| -r), 1e-9)); }
            assert!(close(covariance_simd(&a, &b).unwrap(), c / n as f64, 1e-9));
            let r32 = pearson_correlation_simd(&f32v(&a), &f32v(&b)).map(|x| x as f64);
            assert!(opt_close(r32, expected, 1e-3), "f32 n={n}");
        }
        assert_eq!(pearson_correlation_simd::<f64>(&[], &[]), None);
        assert_eq!(pearson_correlation_simd(&[1.0, 1.0], &[2.0, 3.0]), None);
    });
}

#[test]
fn stats() {
    super::isa::for_each_isa(|_isa| {
        for n in 0..LEN {
            let x = rv(n);
            assert!(close(sum_simd(&x), x.iter().sum(), 1e-10));
            assert!(close(l1_norm_simd(&x), x.iter().map(|v| v.abs()).sum(), 1e-10));
            assert!(close(squared_l2_norm_simd(&x), x.iter().map(|v| v * v).sum(), 1e-10));
            if n == 0 {
                assert_eq!(mean_simd(&x), None);
                assert_eq!(min_max_simd(&x), None);
                continue;
            }
            let mean = x.iter().sum::<f64>() / n as f64;
            assert!(close(mean_simd(&x).unwrap(), mean, 1e-10));
            let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
            assert!(close(variance_simd(&x, 0).unwrap(), var, 1e-10));
            assert!(close(std_dev_simd(&x, 0).unwrap(), var.sqrt(), 1e-10));
            let mn = x.iter().copied().fold(f64::INFINITY, f64::min);
            let mx = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            assert_eq!(min_max_simd(&x), Some((mn, mx)));
            assert_eq!(min_max_simd(&f32v(&x)), Some((mn as f32, mx as f32)));
        }
        assert_eq!(variance_simd(&[1.0], 1), None);
    });
}

#[test]
fn error_metrics() {
    super::isa::for_each_isa(|_isa| {
        for n in 1..LEN {
            let (a, b) = (rv(n), rv(n));
            let mse = zip(&a, &b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / n as f64;
            let mae = zip(&a, &b).map(|(x, y)| (x - y).abs()).sum::<f64>() / n as f64;
            assert!(close(mean_squared_error_simd(&a, &b).unwrap(), mse, 1e-10));
            assert!(close(root_mean_squared_error_simd(&a, &b).unwrap(), mse.sqrt(), 1e-10));
            assert!(close(mean_absolute_error_simd(&a, &b).unwrap(), mae, 1e-10));
            let mean = a.iter().sum::<f64>() / n as f64;
            let ss_tot: f64 = a.iter().map(|v| (v - mean).powi(2)).sum();
            if n > 1 {
                let r2 = 1.0 - mse * n as f64 / ss_tot;
                assert!(close(r_squared_simd(&a, &b).unwrap(), r2, 1e-9));
            }
        }
        assert_eq!(r_squared_simd(&[1.0, 2.0], &[1.0, 2.0]), Some(1.0));
    });
}

#[test]
fn transforms() {
    super::isa::for_each_isa(|_isa| {
        for n in 0..LEN {
            let x = rv(n);
            let mut y = rv(n);
            let y0 = y.clone();
            axpy_simd(&mut y, 0.5, &x).unwrap();
            for i in 0..n {
                assert!(close(y[i], y0[i] + 0.5 * x[i], 1e-12));
            }
            let mut s = x.clone();
            scale_simd(&mut s, -2.0);
            assert!(s.iter().zip(&x).all(|(a, b)| *a == -2.0 * b));
            let add = add_simd(&x, &y0).unwrap();
            let sub = sub_simd(&x, &y0).unwrap();
            let mul = mul_simd(&x, &y0).unwrap();
            for i in 0..n {
                assert_eq!(add[i], x[i] + y0[i]);
                assert_eq!(sub[i], x[i] - y0[i]);
                assert_eq!(mul[i], x[i] * y0[i]);
            }
            let mut c = x.clone();
            clamp_simd(&mut c, -1.0, 1.0);
            assert!(c.iter().zip(&x).all(|(a, b)| *a == b.clamp(-1.0, 1.0)));
            if n >= 2 {
                let mut z = x.clone();
                assert!(zscore_simd(&mut z));
                assert!(mean_simd(&z).unwrap().abs() < 1e-10);
                assert!(close(std_dev_simd(&z, 0).unwrap(), 1.0, 1e-10));
                let mut m = x.clone();
                assert!(normalize_min_max_simd(&mut m));
                let (lo, hi) = min_max_simd(&m).unwrap();
                assert!(lo.abs() < 1e-12 && (hi - 1.0).abs() < 1e-12);
                let mut u = x.clone();
                assert!(normalize_l2_simd(&mut u));
                assert!(close(l2_norm_simd(&u), 1.0, 1e-12));
                let mut p = rv_pos(n);
                p[0] = 1.0;
                assert!(normalize_sum_simd(&mut p));
                assert!(close(sum_simd(&p), 1.0, 1e-12));
            }
        }
        let mut zero = vec![0.0; 5];
        assert!(!normalize_l2_simd(&mut zero));
        assert!(!zscore_simd(&mut [3.0, 3.0]));
        assert!(axpy_simd(&mut [1.0], 1.0, &[]).is_none());
    });
}

#[test]
fn cross_correlation_generic_and_parallel() {
    super::isa::for_each_isa(|_isa| {
        for (nx, ny) in [(1, 1), (5, 3), (3, 5), (17, 9), (64, 64), (100, 37)] {
            let (x, y) = (rv(nx), rv(ny));
            let reference = crate::cross_correlate(&x, &y);
            let got = cross_correlate_simd(&x, &y);
            assert_eq!(got.len(), reference.len());
            for (g, r) in got.iter().zip(&reference) {
                assert!(close(*g, *r, 1e-10));
            }
            #[cfg(feature = "parallel")]
            {
                let par = cross_correlate_simd_parallel(&x, &y);
                assert_eq!(par.len(), got.len());
                    for (p, g) in par.iter().zip(&got) {
                        assert!(close(*p, *g, 1e-10));
                    }
            }
            let got32 = cross_correlate_simd(&f32v(&x), &f32v(&y));
            for (g, r) in got32.iter().zip(&reference) {
                assert!(close(*g as f64, *r, 1e-3));
            }
            assert_eq!(find_time_shift_simd(&x, &y), crate::find_time_shift(&x, &y));
        }
        assert!(cross_correlate_simd::<f32>(&[], &[1.0]).is_empty());
        assert_eq!(find_time_shift_simd::<f64>(&[], &[]), None);
    });
}

#[test]
fn angular() {
    super::isa::for_each_isa(|_isa| {
        assert!(close(angular_distance_simd(&[1.0, 0.0], &[0.0, 1.0]).unwrap(), 0.5, 1e-12));
        assert!(close(angular_distance_simd(&[1.0, 0.0], &[-1.0, 0.0]).unwrap(), 1.0, 1e-12));
        assert!(angular_distance_simd(&[1.0, 2.0], &[1.0, 2.0]).unwrap().abs() < 1e-6);
    });
}
