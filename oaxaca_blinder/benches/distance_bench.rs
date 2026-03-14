use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::DMatrix;
use oaxaca_blinder::matching::distance::MahalanobisDistance;

fn bench_mahalanobis(c: &mut Criterion) {
    let n = 10;
    let inv_cov = DMatrix::identity(n, n);
    let metric = MahalanobisDistance::from_inv_covariance(inv_cov);
    let a: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let b: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5).collect();

    c.bench_function("mahalanobis_distance", |b_bench| {
        b_bench.iter(|| black_box(metric.distance(black_box(&a), black_box(&b))));
    });
}

criterion_group!(benches, bench_mahalanobis);
criterion_main!(benches);
