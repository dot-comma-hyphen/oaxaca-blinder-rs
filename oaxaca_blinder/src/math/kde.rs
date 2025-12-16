use std::f64::consts::PI;

/// Gaussian kernel function.
fn gaussian_kernel(u: f64) -> f64 {
    (1.0 / (2.0 * PI).sqrt()) * (-0.5 * u * u).exp()
}

/// Performs Kernel Density Estimation using a Gaussian kernel.
///
/// # Arguments
///
/// * `data` - The data points to estimate the density for.
/// * `weights` - Optional weights for each data point. If None, uniform weights are used.
/// * `grid` - The points at which to evaluate the density.
/// * `bandwidth` - The bandwidth parameter (smoothing).
///
/// # Returns
///
/// A vector of density values corresponding to the grid points.
pub fn kde(data: &[f64], weights: Option<&[f64]>, grid: &[f64], bandwidth: f64) -> Vec<f64> {
    let n = data.len();
    let mut density = Vec::with_capacity(grid.len());

    let w: Vec<f64> = if let Some(ws) = weights {
        let sum_w: f64 = ws.iter().sum();
        ws.iter().map(|&x| x / sum_w).collect()
    } else {
        vec![1.0 / n as f64; n]
    };

    for &x in grid {
        let mut sum = 0.0;
        for i in 0..n {
            let u = (x - data[i]) / bandwidth;
            sum += w[i] * gaussian_kernel(u);
        }
        density.push(sum / bandwidth);
    }

    density
}

/// Calculates the Silverman's Rule of Thumb bandwidth.
pub fn silverman_bandwidth(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std_dev = variance.sqrt();

    // IQR
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q1 = sorted[(n * 0.25) as usize];
    let q3 = sorted[(n * 0.75) as usize];
    let iqr = q3 - q1;

    let a = std_dev.min(iqr / 1.34);
    0.9 * a * n.powf(-0.2)
}
