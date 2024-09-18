use criterion::{black_box,criterion_group, criterion_main, Criterion};
use similarity::similarity_metrics::*;

pub fn bench_cross_correlate(c: &mut Criterion) {
    let data = (0..10000).map(|_| {
        rand::random::<f64>()
    }).collect::<Vec<_>>();

    c.bench_function("cross_correlate", |b| {
        b.iter(|| {
            cross_correlate(black_box(&data), black_box(&data));
        });
    });

    c.bench_function("cross_correlate", |b| {
        b.iter(|| {
            cross_correlate_fft(black_box(&data), black_box(&data));
        });
    });
}


criterion_group!(benches, bench_cross_correlate);
criterion_main!(benches);
