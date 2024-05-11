use criterion::{black_box, criterion_group, criterion_main, Criterion};
use similarity::similarity_metrics::cosine_similarity;

fn bench_cosine_similarity_f32(c: &mut Criterion) {
    let slice1 = (0..100000).map(|x| x as f32).collect::<Vec<f32>>();
    let slice2 = (0..100000).map(|x| x as f32).collect::<Vec<f32>>();
    c.bench_function("cosine_similarity", |b| {
        b.iter(|| black_box(cosine_similarity(&slice1, &slice2)))
    });
}

fn bench_cosine_similarity_f64(c: &mut Criterion) {
    let slice1 = (0..100000).map(|x| x as f64).collect::<Vec<f64>>();
    let slice2 = (0..100000).map(|x| x as f64).collect::<Vec<f64>>();
    c.bench_function("cosine_similarity", |b| {
        b.iter(|| black_box(cosine_similarity(&slice1, &slice2)))
    });
}

// Obviously this wraps around, but it's just to test the performance of the function
fn bench_cosine_similarity_u8(c: &mut Criterion) {
    let slice1 = (0..100000).map(|x| x as u8).collect::<Vec<_>>();
    let slice2 = (0..100000).map(|x| x as u8).collect::<Vec<_>>();
    c.bench_function("cosine_similarity", |b| {
        b.iter(|| black_box(cosine_similarity(&slice1, &slice2)))
    });
}

fn bench_cosine_similarity_u16(c: &mut Criterion) {
    let slice1 = (0..100000).map(|x| x as u16).collect::<Vec<_>>();
    let slice2 = (0..100000).map(|x| x as u16).collect::<Vec<_>>();
    c.bench_function("cosine_similarity", |b| {
        b.iter(|| black_box(cosine_similarity(&slice1, &slice2)))
    });
}

fn bench_cosine_similarity_u32(c: &mut Criterion) {
    let slice1 = (0..100000).map(|x| x as i32).collect::<Vec<i32>>();
    let slice2 = (0..100000).map(|x| x as i32).collect::<Vec<i32>>();
    c.bench_function("cosine_similarity", |b| {
        b.iter(|| black_box(cosine_similarity(&slice1, &slice2)))
    });
}

criterion_group!(
    benches,
    bench_cosine_similarity_f32,
    bench_cosine_similarity_f64,
    bench_cosine_similarity_u8,
    bench_cosine_similarity_u16,
    bench_cosine_similarity_u32
);
criterion_main!(benches);
