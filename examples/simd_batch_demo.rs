//! Tour of the SIMD kernels, batch search, new metrics and spectral matching.
//!
//! Run with: `cargo run --release --example simd_batch_demo`

use rand::Rng;
use similarity::batch::{pairwise, top_k, Metric};
use similarity::simd::*;
use similarity::{
    calculate_entropy_similarity_matched, divergences, spectral_cosine_similarity, statistics,
    MzTolerance, Spectrum,
};
use std::time::Instant;

fn main() {
    let mut rng = rand::rng();

    // --- SIMD kernels work for f32 and f64 --------------------------------
    let a: Vec<f32> = (0..1_000).map(|_| rng.random_range(0.0..1.0)).collect();
    let b: Vec<f32> = (0..1_000).map(|_| rng.random_range(0.0..1.0)).collect();
    println!("cosine (f32)     = {:.4}", cosine_similarity_simd(&a, &b).unwrap());
    println!("bray-curtis      = {:.4}", bray_curtis_distance_simd(&a, &b).unwrap());
    println!("tanimoto         = {:.4}", tanimoto_similarity_simd(&a, &b).unwrap());
    println!("pearson r        = {:.4}", pearson_correlation_simd(&a, &b).unwrap());
    println!("spearman rho     = {:.4}", statistics::spearman_correlation(&a, &b).unwrap());
    println!("jensen-shannon   = {:.4}", divergences::js_distance(&a, &b).unwrap());

    // --- k-nearest neighbours over 200k embeddings -------------------------
    let db: Vec<Vec<f32>> = (0..200_000)
        .map(|_| (0..64).map(|_| rng.random_range(-1.0..1.0)).collect())
        .collect();
    let query = db[42].clone();
    let t = Instant::now();
    let hits = top_k(&query, &db, 5, Metric::Cosine).unwrap();
    println!("\ntop-5 cosine of 200k x 64 in {:?}:", t.elapsed());
    for h in &hits {
        println!("  row {:>6}  score {:.4}", h.index, h.score);
    }

    let t = Instant::now();
    let m = pairwise(&db[..2_000], Metric::Euclidean).unwrap();
    println!("2000 x 2000 Euclidean matrix in {:?} (d[0,1] = {:.3})", t.elapsed(), m.get(0, 1));

    // --- spectral matching with ppm tolerance ------------------------------
    let query = Spectrum::from_arrays(
        &[91.0540, 119.0492, 136.0757, 150.0914, 192.1383],
        &[16.6, 35.8, 178.2, 38.5, 999.0],
    )
    .unwrap();
    let reference = Spectrum::from_arrays(
        &[91.0541, 119.0491, 136.0758, 192.1384, 211.0543],
        &[999.0, 605.1, 738.6, 270.7, 37.9],
    )
    .unwrap();
    let tol = MzTolerance::Ppm(10.0);
    let (cos, n) = spectral_cosine_similarity(&query, &reference, tol, 0.0, 0.5);
    println!("\nspectral cosine (sqrt intensities) = {cos:.4} with {n} matched peaks");
    println!(
        "matched entropy similarity          = {:.4}",
        calculate_entropy_similarity_matched(&query, &reference, tol)
    );
}
