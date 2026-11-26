//! Benchmarks for the generic SIMD kernels (f32 vs f64), the new distance
//! metrics against their scalar counterparts, and batch evaluation.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::Rng;
use similarity::batch::{self, Metric};
use similarity::library_search::{LibrarySearchOptions, SpectralLibrary};
use similarity::distance_metrics;
use similarity::divergences;
use similarity::flash_entropy::{FlashEntropyIndex, FlashEntropyOptions};
use similarity::{calculate_entropy_similarity_matched, MzTolerance, Spectrum};
use similarity::simd::*;
use std::hint::black_box;

fn vec_f64(n: usize) -> Vec<f64> {
    let mut rng = rand::rng();
    (0..n).map(|_| rng.random_range(0.0..1.0)).collect()
}

fn f32s(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

const SIZES: [usize; 4] = [64, 1024, 16384, 262144];

fn bench_precision(c: &mut Criterion) {
    let mut g = c.benchmark_group("simd_precision_dot");
    for n in SIZES {
        let (a, b) = (vec_f64(n), vec_f64(n));
        let (a32, b32) = (f32s(&a), f32s(&b));
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::new("f64", n), &n, |bch, _| {
            bch.iter(|| dot_product_simd(black_box(&a), black_box(&b)))
        });
        g.bench_with_input(BenchmarkId::new("f32", n), &n, |bch, _| {
            bch.iter(|| dot_product_simd(black_box(&a32), black_box(&b32)))
        });
    }
    g.finish();
}

macro_rules! scalar_vs_simd {
    ($fn_name:ident, $group:literal, $scalar:path, $simd:path) => {
        fn $fn_name(c: &mut Criterion) {
            let mut g = c.benchmark_group($group);
            for n in SIZES {
                let (a, b) = (vec_f64(n), vec_f64(n));
                g.throughput(Throughput::Elements(n as u64));
                g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bch, _| {
                    bch.iter(|| $scalar(black_box(&a[..]), black_box(&b[..])))
                });
                g.bench_with_input(BenchmarkId::new("simd", n), &n, |bch, _| {
                    bch.iter(|| $simd(black_box(&a[..]), black_box(&b[..])))
                });
            }
            g.finish();
        }
    };
}

scalar_vs_simd!(bench_chebyshev, "chebyshev", distance_metrics::chebyshev_distance, chebyshev_distance_simd);
scalar_vs_simd!(bench_canberra, "canberra", distance_metrics::canberra_distance, canberra_distance_simd);
scalar_vs_simd!(bench_bray_curtis, "bray_curtis", distance_metrics::bray_curtis_distance, bray_curtis_distance_simd);
scalar_vs_simd!(bench_tanimoto, "tanimoto", distance_metrics::tanimoto_similarity, tanimoto_similarity_simd);
scalar_vs_simd!(bench_ruzicka, "ruzicka", distance_metrics::ruzicka_similarity, ruzicka_similarity_simd);

fn bench_stats(c: &mut Criterion) {
    let mut g = c.benchmark_group("variance");
    for n in SIZES {
        let x = vec_f64(n);
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bch, _| {
            bch.iter(|| {
                let x = black_box(&x);
                let m = x.iter().sum::<f64>() / x.len() as f64;
                x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / x.len() as f64
            })
        });
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bch, _| {
            bch.iter(|| variance_simd(black_box(&x), 0))
        });
    }
    g.finish();
}

fn bench_batch(c: &mut Criterion) {
    let mut g = c.benchmark_group("batch_pairwise_cosine");
    g.sample_size(20);
    for (n, d) in [(256, 128), (1024, 128)] {
        let rows: Vec<Vec<f32>> = (0..n).map(|_| f32s(&vec_f64(d))).collect();
        g.bench_with_input(BenchmarkId::new("naive_scalar", n), &n, |bch, _| {
            bch.iter(|| {
                let mut out = vec![0.0f32; n * n];
                for i in 0..n {
                    for j in 0..n {
                        out[i * n + j] = similarity::cosine_similarity(&rows[i], &rows[j]).unwrap() as f32;
                    }
                }
                out
            })
        });
        g.bench_with_input(BenchmarkId::new("batch", n), &n, |bch, _| {
            bch.iter(|| batch::pairwise(black_box(&rows), Metric::Cosine).unwrap())
        });
    }
    g.finish();

    let mut g = c.benchmark_group("batch_cdist");
    g.sample_size(10);
    for d in [32, 128, 768] {
        let a: Vec<Vec<f32>> = (0..2000).map(|_| f32s(&vec_f64(d))).collect();
        let b: Vec<Vec<f32>> = (0..2000).map(|_| f32s(&vec_f64(d))).collect();
        for metric in [Metric::Cosine, Metric::Euclidean] {
            g.bench_with_input(BenchmarkId::new(format!("{metric:?}_2000x2000"), d), &d, |bch, _| {
                bch.iter(|| batch::cdist(black_box(&a), black_box(&b), metric).unwrap())
            });
        }
    }
    g.finish();

    let mut g = c.benchmark_group("batch_top_k");
    let db: Vec<Vec<f32>> = (0..100_000).map(|_| f32s(&vec_f64(64))).collect();
    let q = f32s(&vec_f64(64));
    g.sample_size(20);
    g.bench_function("euclidean_100k_x_64_k10", |bch| {
        bch.iter(|| batch::top_k(black_box(&q), &db, 10, Metric::Euclidean).unwrap())
    });
    g.finish();
}

fn bench_transcendental(c: &mut Criterion) {
    let mut g = c.benchmark_group("ln_throughput");
    for n in [1024, 65536] {
        let x: Vec<f64> = vec_f64(n).iter().map(|v| v + 1e-3).collect();
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::new("std", n), &n, |bch, _| {
            let mut y = x.clone();
            bch.iter(|| {
                y.copy_from_slice(&x);
                y.iter_mut().for_each(|v| *v = v.ln());
                black_box(&y);
            })
        });
        g.bench_with_input(BenchmarkId::new("simd", n), &n, |bch, _| {
            let mut y = x.clone();
            bch.iter(|| {
                y.copy_from_slice(&x);
                ln_simd(&mut y);
                black_box(&y);
            })
        });
    }
    g.finish();

    let mut g = c.benchmark_group("information");
    for n in [1024, 65536] {
        let (p, q) = (vec_f64(n), vec_f64(n));
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::new("entropy_scalar", n), &n, |bch, _| {
            bch.iter(|| divergences::shannon_entropy(black_box(&p)))
        });
        g.bench_with_input(BenchmarkId::new("entropy_simd", n), &n, |bch, _| {
            bch.iter(|| shannon_entropy_simd(black_box(&p)))
        });
        g.bench_with_input(BenchmarkId::new("kl_scalar", n), &n, |bch, _| {
            bch.iter(|| divergences::kl_divergence(black_box(&p), black_box(&q)))
        });
        g.bench_with_input(BenchmarkId::new("kl_simd", n), &n, |bch, _| {
            bch.iter(|| kl_divergence_simd(black_box(&p), black_box(&q)))
        });
        g.bench_with_input(BenchmarkId::new("js_scalar", n), &n, |bch, _| {
            bch.iter(|| divergences::js_divergence(black_box(&p), black_box(&q)))
        });
        g.bench_with_input(BenchmarkId::new("js_simd", n), &n, |bch, _| {
            bch.iter(|| js_divergence_simd(black_box(&p), black_box(&q)))
        });
        g.bench_with_input(BenchmarkId::new("hellinger_scalar", n), &n, |bch, _| {
            bch.iter(|| divergences::hellinger_distance(black_box(&p), black_box(&q)))
        });
        g.bench_with_input(BenchmarkId::new("hellinger_simd", n), &n, |bch, _| {
            bch.iter(|| hellinger_distance_simd(black_box(&p), black_box(&q)))
        });
    }
    g.finish();
}

fn random_spectrum(rng: &mut impl Rng, n_peaks: usize) -> Spectrum<f64> {
    let mz: Vec<f64> = (0..n_peaks).map(|_| (rng.random_range(50.0..1000.0f64) * 100.0).round() / 100.0).collect();
    let it: Vec<f64> = (0..n_peaks).map(|_| rng.random_range(1.0..1000.0)).collect();
    Spectrum::from_arrays(&mz, &it).unwrap()
}

fn bench_flash(c: &mut Criterion) {
    let mut rng = rand::rng();
    let library: Vec<Spectrum<f64>> = (0..100_000).map(|_| random_spectrum(&mut rng, 30)).collect();
    let query = random_spectrum(&mut rng, 30);
    let opts = FlashEntropyOptions::absolute(0.01);
    let mut g = c.benchmark_group("library_search_100k");
    g.sample_size(10);
    g.bench_function("index_build", |bch| bch.iter(|| FlashEntropyIndex::new(black_box(&library), opts)));
    let index = FlashEntropyIndex::new(&library, opts);
    g.bench_function("flash_search", |bch| bch.iter(|| index.search(black_box(&query))));
    g.bench_function("bruteforce_matched", |bch| {
        bch.iter(|| {
            library
                .iter()
                .map(|l| calculate_entropy_similarity_matched(black_box(&query), l, MzTolerance::Absolute(0.01)))
                .collect::<Vec<_>>()
        })
    });
    let precursors: Vec<f64> = (0..library.len()).map(|_| rng.random_range(100.0..1000.0)).collect();
    let entropy = LibrarySearchOptions::entropy(MzTolerance::Absolute(0.01));
    let lib = SpectralLibrary::new(&library, entropy);
    g.bench_function("library_all_entropy", |bch| bch.iter(|| lib.search(black_box(&query), None, 10)));
    let filtered = SpectralLibrary::with_precursors(
        &library,
        &precursors,
        entropy.with_precursor_tolerance(MzTolerance::Ppm(10.0)),
    )
    .unwrap();
    g.bench_function("library_precursor_10ppm", |bch| bch.iter(|| filtered.search(black_box(&query), Some(500.0), 10)));
    g.finish();
}

criterion_group!(
    benches,
    bench_precision,
    bench_chebyshev,
    bench_canberra,
    bench_bray_curtis,
    bench_tanimoto,
    bench_ruzicka,
    bench_stats,
    bench_batch,
    bench_transcendental,
    bench_flash
);
criterion_main!(benches);
