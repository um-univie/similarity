//! Times `A · Bᵀ` (f32, all multi-threaded) with similarity's kernels and
//! with faer, the `gemm` crate and `matrixmultiply`, and prints Markdown
//! tables in GFLOP/s. The symmetric table compares similarity's `pairwise`
//! with faer's triangular product (both compute one triangle, like BLAS
//! SYRK); its rate counts the full `2·n²·d` operations, so it can exceed the
//! machine's peak. Run with `cargo run --release` (add `--features avx512`
//! to enable similarity's AVX-512 kernels).

use faer::linalg::matmul::triangular::{self, BlockStructure};
use faer::{linalg::matmul, Accum, Mat, MatRef, Par};
use rand::Rng;
use similarity::batch::{cdist, dense_backend, pairwise, Metric};
use similarity::simd::Isa;
use std::time::Instant;

const SIZES: [usize; 3] = [2000, 4000, 8000];
const DIMS: [usize; 5] = [32, 128, 768, 2048, 4096];

/// Best of a few runs after one warm-up, fewer runs for long ones.
fn best<F: FnMut()>(mut f: F) -> f64 {
    let t = Instant::now();
    f();
    let reps = if t.elapsed().as_secs_f64() > 0.5 { 2 } else { 6 };
    (0..reps)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    let mut rng = rand::rng();
    println!(
        "similarity ISA: `{:?}`, dense backend: `{}`, threads: {}\n",
        Isa::detected(),
        dense_backend(),
        std::thread::available_parallelism().map_or(1, |n| n.get())
    );
    let mut rect = vec![
        "| n | d | similarity dot | faer | gemm | matrixmultiply | similarity euclidean | max abs diff |".to_string(),
        "|---|---|---|---|---|---|---|---|".to_string(),
    ];
    let mut sym = vec![
        "| n | d | similarity pairwise | faer triangular |".to_string(),
        "|---|---|---|---|".to_string(),
    ];
    for n in SIZES {
        for d in DIMS {
            let a: Vec<f32> = (0..n * d).map(|_| rng.random_range(-1.0..1.0)).collect();
            let rows: Vec<&[f32]> = a.chunks(d).collect();
            let fa = MatRef::from_row_major_slice(&a, n, d);
            let gflops = |t: f64| format!("{:.0}", 2.0 * (n * n * d) as f64 / t / 1e9);

            let mut ours = Vec::new();
            let t_ours = best(|| ours = cdist(&rows, &rows, Metric::Dot).unwrap().data);
            let t_euc = best(|| drop(cdist(&rows, &rows, Metric::Euclidean).unwrap()));
            let t_pair = best(|| drop(pairwise(&rows, Metric::Dot).unwrap()));

            let mut f = Mat::<f32>::zeros(n, n);
            let t_faer = best(|| matmul::matmul(f.as_mut(), Accum::Replace, fa, fa.transpose(), 1.0, Par::rayon(0)));
            let t_ftri = best(|| {
                triangular::matmul(
                    f.as_mut(), BlockStructure::TriangularUpper, Accum::Replace,
                    fa, BlockStructure::Rectangular, fa.transpose(), BlockStructure::Rectangular,
                    1.0, Par::rayon(0),
                )
            });

            let mut c = vec![0f32; n * n];
            // SAFETY: all pointers and strides describe the buffers above.
            let t_gemm = best(|| unsafe {
                gemm::gemm(
                    n, n, d, c.as_mut_ptr(), 1, n as isize, false,
                    a.as_ptr(), 1, d as isize, a.as_ptr(), d as isize, 1,
                    0.0f32, 1.0f32, false, false, false, gemm::Parallelism::Rayon(0),
                );
            });
            let mut c2 = vec![0f32; n * n];
            // SAFETY: as above.
            let t_mm = best(|| unsafe {
                matrixmultiply::sgemm(
                    n, d, n, 1.0, a.as_ptr(), d as isize, 1, a.as_ptr(), 1, d as isize, 0.0,
                    c2.as_mut_ptr(), n as isize, 1,
                );
            });
            let diff = ours.iter().zip(&c).chain(ours.iter().zip(&c2)).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
            rect.push(format!(
                "| {n} | {d} | {} | {} | {} | {} | {} | {diff:.1e} |",
                gflops(t_ours), gflops(t_faer), gflops(t_gemm), gflops(t_mm), gflops(t_euc)
            ));
            sym.push(format!("| {n} | {d} | {} | {} |", gflops(t_pair), gflops(t_ftri)));
            eprintln!("done n={n} d={d}");
        }
    }
    println!("**Rectangular `A · Aᵀ`, GFLOP/s**\n\n{}\n", rect.join("\n"));
    println!("**Symmetric (one triangle), GFLOP/s of the full product**\n\n{}", sym.join("\n"));
}
