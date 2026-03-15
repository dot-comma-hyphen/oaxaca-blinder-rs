use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::{DMatrix, DVector};
use oaxaca_blinder::math::logit::logit;

fn bench_logit(c: &mut Criterion) {
    let n = 10000;
    let k = 20;

    // Generate some random-ish data
    let mut x = DMatrix::zeros(n, k);
    for i in 0..n {
        x[(i, 0)] = 1.0; // Intercept
        for j in 1..k {
            x[(i, j)] = ((i as f64 * 0.1) + (j as f64)).sin();
        }
    }

    let true_beta = DVector::from_element(k, 0.1);
    let xb = &x * &true_beta;
    let y = xb.map(|z| if 1.0 / (1.0 + (-z).exp()) > 0.5 { 1.0 } else { 0.0 });

    c.bench_function("logit_hessian_original", |b| {
        b.iter(|| {
            // We only run a few iterations to focus on the cost per iteration
            let _ = logit(black_box(&y), black_box(&x), black_box(5), black_box(1e-6));
        });
    });
}

criterion_group!(benches, bench_logit);
criterion_main!(benches);
