//! Accuracy tests for the vectorised `ln`/`exp`/`pow` and the SIMD
//! information-theoretic metrics, on every ISA available on the host.

use super::isa::for_each_isa;
use super::*;
use rand::Rng;

fn ulp_diff_f64(a: f64, b: f64) -> u64 {
    if a == b {
        return 0;
    }
    if a.is_nan() || b.is_nan() || a.signum() != b.signum() {
        return u64::MAX;
    }
    (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs()
}

fn ulp_diff_f32(a: f32, b: f32) -> u32 {
    if a == b {
        return 0;
    }
    if a.is_nan() || b.is_nan() || a.signum() != b.signum() {
        return u32::MAX;
    }
    (a.to_bits() as i32 - b.to_bits() as i32).unsigned_abs()
}

/// Log-uniform samples over the whole positive range incl. subnormals.
fn positives(n: usize) -> Vec<f64> {
    let mut rng = rand::rng();
    let mut v: Vec<f64> = (0..n).map(|_| 2f64.powf(rng.random_range(-1070.0..1023.0))).collect();
    v.extend([1.0, 2.0, 0.5, f64::MIN_POSITIVE, 5e-324, f64::MAX, std::f64::consts::SQRT_2, 1.0 - 1e-16, 1.0 + 2e-16]);
    v
}

#[test]
fn ln_f64_accuracy() {
    let xs = positives(100_000);
    for_each_isa(|isa| {
        let mut y = xs.clone();
        ln_simd(&mut y);
        let worst = xs.iter().zip(&y).map(|(x, y)| ulp_diff_f64(*y, x.ln())).max().unwrap();
        assert!(worst <= 2, "{isa:?}: ln f64 worst error {worst} ulp");
    });
}

#[test]
fn ln_f32_accuracy() {
    let mut rng = rand::rng();
    let mut xs: Vec<f32> = (0..100_000).map(|_| 2f32.powf(rng.random_range(-148.0..127.0))).collect();
    xs.extend([1.0, f32::MIN_POSITIVE, 1e-45, f32::MAX]);
    for_each_isa(|isa| {
        let mut y = xs.clone();
        ln_simd(&mut y);
        let worst = xs.iter().zip(&y).map(|(x, y)| ulp_diff_f32(*y, x.ln())).max().unwrap();
        assert!(worst <= 2, "{isa:?}: ln f32 worst error {worst} ulp");
    });
}

#[test]
fn ln_special_values() {
    for_each_isa(|isa| {
        let mut x = vec![0.0, -1.0, f64::INFINITY, f64::NAN, -0.0, f64::NEG_INFINITY];
        ln_simd(&mut x);
        assert_eq!(x[0], f64::NEG_INFINITY, "{isa:?}");
        assert!(x[1].is_nan());
        assert_eq!(x[2], f64::INFINITY);
        assert!(x[3].is_nan());
        assert_eq!(x[4], f64::NEG_INFINITY);
        assert!(x[5].is_nan());
    });
}

#[test]
fn exp_accuracy() {
    let mut rng = rand::rng();
    let xs: Vec<f64> = (0..100_000).map(|_| rng.random_range(-707.0..709.0)).collect();
    let xs32: Vec<f32> = (0..100_000).map(|_| rng.random_range(-86.0f32..88.0)).collect();
    for_each_isa(|isa| {
        let mut y = xs.clone();
        exp_simd(&mut y);
        let worst = xs.iter().zip(&y).map(|(x, y)| ulp_diff_f64(*y, x.exp())).max().unwrap();
        assert!(worst <= 2, "{isa:?}: exp f64 worst error {worst} ulp");
        let mut y = xs32.clone();
        exp_simd(&mut y);
        let worst = xs32.iter().zip(&y).map(|(x, y)| ulp_diff_f32(*y, x.exp())).max().unwrap();
        assert!(worst <= 2, "{isa:?}: exp f32 worst error {worst} ulp");
        let mut s = vec![0.0, 1000.0, -1000.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 709.7];
        exp_simd(&mut s);
        assert_eq!(s[0], 1.0);
        assert_eq!(s[1], f64::INFINITY);
        assert_eq!(s[2], 0.0);
        assert!(s[3].is_nan());
        assert_eq!(s[4], f64::INFINITY);
        assert_eq!(s[5], 0.0);
        assert!(ulp_diff_f64(s[6], 709.7f64.exp()) <= 2);
    });
}

#[test]
fn pow_and_sqrt() {
    let mut rng = rand::rng();
    let xs: Vec<f64> = (0..10_000).map(|_| rng.random_range(0.0..1000.0)).collect();
    for_each_isa(|isa| {
        for y in [0.25, 0.5, 0.75, 1.0, 2.0, -1.5] {
            let mut v = xs.clone();
            pow_simd(&mut v, y);
            for (a, b) in xs.iter().zip(&v) {
                let r = a.powf(y);
                assert!((b - r).abs() <= 1e-13 * r.abs().max(1e-300), "{isa:?} {a}^{y}: {b} vs {r}");
            }
        }
        let mut z = vec![0.0, 0.0, 0.0];
        pow_simd(&mut z, 0.0);
        assert_eq!(z, [1.0, 1.0, 1.0]);
        let mut v = xs.clone();
        sqrt_simd(&mut v);
        assert!(xs.iter().zip(&v).all(|(a, b)| *b == a.sqrt()));
    });
}

fn rand_weights(n: usize, zeros: bool) -> Vec<f64> {
    let mut rng = rand::rng();
    (0..n)
        .map(|_| if zeros && rng.random_bool(0.2) { 0.0 } else { rng.random_range(0.0..10.0) })
        .collect()
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * (1.0 + a.abs().max(b.abs())) || (a.is_infinite() && a == b)
}

#[test]
fn divergences_match_scalar() {
    use crate::divergences as d;
    for_each_isa(|isa| {
        for n in 1..70 {
            let p = rand_weights(n, true);
            let mut q = rand_weights(n, false);
            q[0] += 0.1;
            if p.iter().sum::<f64>() == 0.0 {
                continue;
            }
            let chk = |name: &str, a: Option<f64>, b: Option<f64>| match (a, b) {
                (Some(x), Some(y)) => assert!(close(x, y), "{isa:?} {name} n={n}: {x} vs {y}"),
                (x, y) => assert_eq!(x.is_none(), y.is_none(), "{isa:?} {name} n={n}"),
            };
            chk("H", shannon_entropy_simd(&p), d::shannon_entropy(&p));
            chk("KL", kl_divergence_simd(&p, &q), d::kl_divergence(&p, &q));
            chk("KLinf", kl_divergence_simd(&q, &p), d::kl_divergence(&q, &p));
            chk("CE", cross_entropy_simd(&p, &q), d::cross_entropy(&p, &q));
            chk("JS", js_divergence_simd(&p, &q), d::js_divergence(&p, &q));
            chk("JSd", js_distance_simd(&p, &q), d::js_distance(&p, &q));
            chk("BC", bhattacharyya_coefficient_simd(&p, &q), d::bhattacharyya_coefficient(&p, &q));
            chk("Hel", hellinger_distance_simd(&p, &q), d::hellinger_distance(&p, &q));
            chk("TV", total_variation_distance_simd(&p, &q), d::total_variation_distance(&p, &q));
        }
        assert_eq!(shannon_entropy_simd::<f64>(&[]), None);
        assert_eq!(shannon_entropy_simd(&[1.0, -1.0, 2.0]), None);
        assert_eq!(kl_divergence_simd(&[1.0], &[1.0, 2.0]), None);
        assert_eq!(kl_divergence_simd(&[0.0, 0.0], &[1.0, 2.0]), None);
        let h32 = shannon_entropy_simd(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        assert!((h32 - 1.279_854_2).abs() < 1e-6);
    });
}

#[test]
fn spectral_entropy_matches_reference() {
    use crate::{calculate_entropy_similarity, Spectrum};
    for_each_isa(|isa| {
        for n in 1..40 {
            let a = rand_weights(n, true);
            let b = rand_weights(n, true);
            if a.iter().sum::<f64>() == 0.0 || b.iter().sum::<f64>() == 0.0 {
                continue;
            }
            let mz: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();
            let sa = Spectrum::from_arrays(&mz, &a).unwrap();
            let sb = Spectrum::from_arrays(&mz, &b).unwrap();
            let reference = calculate_entropy_similarity(&sa, &sb);
            let got = entropy_similarity_simd(&a, &b).unwrap();
            // The reference skips normalisation in its high-entropy branch,
            // which only matters when both spectra exceed the cutoff.
            let (ea, _) = sa.get_entropy_and_weighted_intensity();
            let (eb, _) = sb.get_entropy_and_weighted_intensity();
            if ea < 3.0 && eb < 3.0 {
                assert!((got - reference).abs() < 1e-10, "{isa:?} n={n}: {got} vs {reference}");
            }
            assert!((0.0..=1.0).contains(&got));
        }
    });
}

#[test]
fn report_isas() {
    eprintln!("available ISAs: {:?}, detected: {:?}", super::Isa::available(), super::Isa::detected());
    assert!(super::Isa::available().contains(&super::Isa::detected()));
}
