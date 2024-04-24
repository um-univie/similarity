use criterion::{black_box,criterion_group, criterion_main, Criterion};
use similarity::similarity_metrics::cosine_similarity;

fn bench_cosine_similarity(c: &mut Criterion) {
    let slice1 = (0..1000).map(|x| x as f64).collect::<Vec<f64>>();
    let slice2 = (0..1000).map(|x| x as f64).collect::<Vec<f64>>();
    c.bench_function("cosine_similarity", |b| b.iter(|| cosine_similarity(&slice1, &slice2)) );
}

criterion_group!(benches, bench_cosine_similarity);
criterion_main!(benches);
