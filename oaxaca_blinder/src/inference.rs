//! This module contains functions for statistical inference, primarily bootstrapping.

/// Calculates the standard error, p-value, and confidence interval from a vector of bootstrap estimates.
pub fn bootstrap_stats(estimates: &[f64], point_estimate: f64) -> (f64, f64, (f64, f64)) {
    // Standard error is the standard deviation of the bootstrap estimates.
    let n = estimates.len() as f64;
    let mean: f64 = estimates.iter().sum::<f64>() / n;
    let std_err = (estimates.iter().map(|&val| (val - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();

    // p-value: Two-tailed test for whether the estimate is different from zero.
    // This is calculated as the proportion of bootstrap estimates that are more extreme than the negative of the absolute point estimate.
    let p_value_proportion = estimates.iter().filter(|&&val| val.abs() > point_estimate.abs()).count();
    let p_value = (p_value_proportion as f64) / n;
    // A more standard way for symmetric distributions:
    // let z_score = point_estimate / std_err;
    // let p_value = 2.0 * (1.0 - distrs::Normal::new(0.0, 1.0).unwrap().cdf(z_score.abs()));
    // For now, we'll use the simpler proportion method.

    // Confidence interval using the percentile method.
    let mut sorted_estimates = estimates.to_vec();
    sorted_estimates.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let lower_idx = (0.025 * n).floor() as usize;
    let upper_idx = (0.975 * n).ceil() as usize;
    let ci_lower = sorted_estimates[lower_idx];
    let ci_upper = sorted_estimates[upper_idx];

    (std_err, p_value, (ci_lower, ci_upper))
}
