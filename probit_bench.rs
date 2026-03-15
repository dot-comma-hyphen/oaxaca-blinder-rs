use oaxaca_blinder::math::probit;
use nalgebra::{DMatrix, DVector};
use std::time::Instant;

fn main() {
    let n = 5000;
    let k = 50;
    let mut x_vec = Vec::with_capacity(n * k);
    let mut y_vec = Vec::with_capacity(n);
    for i in 0..n {
        let y_val = if i % 2 == 0 { 0.0 } else { 1.0 };
        y_vec.push(y_val);
        for j in 0..k {
            x_vec.push((i as f64) * 0.01 + (j as f64) * 0.02);
        }
    }

    let x = DMatrix::from_vec(n, k, x_vec);
    let y = DVector::from_vec(y_vec);

    let start = Instant::now();
    for _ in 0..10 {
        let _ = probit(&y, &x, 10, 1e-6);
    }
    let duration = start.elapsed();
    println!("Time: {:?}", duration);
}
