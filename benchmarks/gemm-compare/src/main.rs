//! Times `A · Bᵀ` (2000 × 2000, f32) with similarity's kernel, the `gemm`
//! crate and `matrixmultiply`, all multi-threaded, and prints a Markdown
//! table. Run with `cargo run --release` (add `--features avx512` to enable
//! similarity's AVX-512 kernels).

use rand::Rng;
use similarity::batch::{cdist, Metric};
use similarity::simd::Isa;
use std::time::Instant;

fn best<F: FnMut()>(mut f: F) -> f64 {
    f();
    (0..7)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    let mut rng = rand::rng();
    let (m, n) = (2000usize, 2000usize);
    println!("similarity ISA: `{:?}`, threads: {}\n", Isa::detected(), std::thread::available_parallelism().map_or(1, |n| n.get()));
    println!("| d | similarity dot | gemm | matrixmultiply | similarity cosine | similarity euclidean | max abs diff |");
    println!("|---|---|---|---|---|---|---|");
    for k in [32usize, 128, 768, 2048] {
        let a: Vec<f32> = (0..m * k).map(|_| rng.random_range(-1.0..1.0)).collect();
        let b: Vec<f32> = (0..n * k).map(|_| rng.random_range(-1.0..1.0)).collect();
        let ar: Vec<&[f32]> = a.chunks(k).collect();
        let br: Vec<&[f32]> = b.chunks(k).collect();

        let mut ours = Vec::new();
        let t_ours = best(|| ours = cdist(&ar, &br, Metric::Dot).unwrap().data);
        let t_cos = best(|| drop(cdist(&ar, &br, Metric::Cosine).unwrap()));
        let t_euc = best(|| drop(cdist(&ar, &br, Metric::Euclidean).unwrap()));

        let mut c = vec![0f32; m * n];
        // SAFETY: all pointers and strides describe the buffers above.
        let t_gemm = best(|| unsafe {
            gemm::gemm(
                m, n, k, c.as_mut_ptr(), 1, n as isize, false,
                a.as_ptr(), 1, k as isize, b.as_ptr(), k as isize, 1,
                0.0f32, 1.0f32, false, false, false, gemm::Parallelism::Rayon(0),
            );
        });
        let mut c2 = vec![0f32; m * n];
        // SAFETY: as above.
        let t_mm = best(|| unsafe {
            matrixmultiply::sgemm(
                m, k, n, 1.0, a.as_ptr(), k as isize, 1, b.as_ptr(), 1, k as isize, 0.0,
                c2.as_mut_ptr(), n as isize, 1,
            );
        });
        let diff = ours.iter().zip(&c).chain(ours.iter().zip(&c2)).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        let ms = |t: f64| format!("{:.2} ms", t * 1e3);
        println!(
            "| {k} | {} | {} | {} | {} | {} | {diff:.1e} |",
            ms(t_ours), ms(t_gemm), ms(t_mm), ms(t_cos), ms(t_euc)
        );
    }
}
