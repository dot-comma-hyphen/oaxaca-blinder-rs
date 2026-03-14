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
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q1 = sorted[(n * 0.25) as usize];
    let q3 = sorted[(n * 0.75) as usize];
    let iqr = q3 - q1;

    let a = std_dev.min(iqr / 1.34);
    0.9 * a * n.powf(-0.2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_kernel() {
        // Test at center
        let val_0 = gaussian_kernel(0.0);
        let expected_0 = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        assert!((val_0 - expected_0).abs() < 1e-8, "gaussian_kernel(0.0) incorrect");

        // Test symmetry
        let val_1 = gaussian_kernel(1.0);
        let val_neg_1 = gaussian_kernel(-1.0);
        assert!((val_1 - val_neg_1).abs() < 1e-8, "gaussian_kernel should be symmetric");

        // Test known value at x=1
        let expected_1 = expected_0 * (-0.5f64).exp();
        assert!((val_1 - expected_1).abs() < 1e-8, "gaussian_kernel(1.0) incorrect");
    }

    #[test]
    fn test_kde_uniform_weights() {
        let data = vec![0.0, 1.0, 2.0];
        let grid = vec![1.0];
        let bandwidth = 1.0;

        let density = kde(&data, None, &grid, bandwidth);
        assert_eq!(density.len(), 1);

        // Calculate expected value
        // w = [1/3, 1/3, 1/3]
        // at x = 1.0:
        // u = [(1-0)/1, (1-1)/1, (1-2)/1] = [1.0, 0.0, -1.0]
        let expected_sum = (1.0 / 3.0) * gaussian_kernel(1.0)
            + (1.0 / 3.0) * gaussian_kernel(0.0)
            + (1.0 / 3.0) * gaussian_kernel(-1.0);
        let expected_density = expected_sum / bandwidth;

        assert!((density[0] - expected_density).abs() < 1e-8, "kde uniform weights incorrect");
    }

    #[test]
    fn test_kde_custom_weights() {
        let data = vec![0.0, 1.0, 2.0];
        let weights = vec![1.0, 2.0, 1.0]; // Will be normalized to [0.25, 0.5, 0.25]
        let grid = vec![1.0];
        let bandwidth = 2.0;

        let density = kde(&data, Some(&weights), &grid, bandwidth);
        assert_eq!(density.len(), 1);

        // w = [0.25, 0.5, 0.25]
        // at x = 1.0, bw = 2.0:
        // u = [(1-0)/2, (1-1)/2, (1-2)/2] = [0.5, 0.0, -0.5]
        let expected_sum = 0.25 * gaussian_kernel(0.5)
            + 0.5 * gaussian_kernel(0.0)
            + 0.25 * gaussian_kernel(-0.5);
        let expected_density = expected_sum / bandwidth;

        assert!((density[0] - expected_density).abs() < 1e-8, "kde custom weights incorrect");
    }

    #[test]
    fn test_silverman_bandwidth() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let bw = silverman_bandwidth(&data);

        // n = 5
        // mean = 3.0
        // var = 2.5
        // std_dev = sqrt(2.5) ≈ 1.5811388
        // IQR:
        // sorted = [1.0, 2.0, 3.0, 4.0, 5.0]
        // q1 (idx 1) = 2.0
        // q3 (idx 3) = 4.0
        // iqr = 2.0
        // a = min(1.5811388, 2.0 / 1.34) = min(1.5811388, 1.4925373) = 1.4925373
        // result = 0.9 * 1.4925373 * 5^(-0.2) ≈ 0.9735606

        let expected_bw = 0.9735846228506357;
        assert!((bw - expected_bw).abs() < 1e-5, "silverman bandwidth incorrect, got {}, expected roughly {}", bw, expected_bw);
    }
}
