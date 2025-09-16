//! This module contains functions for statistical inference, primarily bootstrapping.

/// Calculates the standard error, p-value, and confidence interval from a vector of bootstrap estimates.
pub fn bootstrap_stats(estimates: &[f64], point_estimate: f64) -> (f64, f64, (f64, f64)) {
    if estimates.is_empty() {
        return (f64::NAN, f64::NAN, (f64::NAN, f64::NAN));
    }
    // Standard error is the standard deviation of the bootstrap estimates.
    let n = estimates.len() as f64;
    let mean: f64 = estimates.iter().sum::<f64>() / n;
    let std_err = (estimates.iter().map(|&val| (val - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();

    // p-value: Two-tailed test using the percentile method.
    // This is a more robust way to calculate the p-value for bootstrapped estimates.
    let prop_less_than_point = estimates.iter().filter(|&&val| val < point_estimate).count() as f64 / n;
    let p_value = 2.0 * (0.5 - (prop_less_than_point - 0.5).abs());


    // Confidence interval using the percentile method.
    let mut sorted_estimates = estimates.to_vec();
    sorted_estimates.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let lower_idx = (0.025 * n).floor() as usize;
    let upper_idx = ((0.975 * n).floor() as usize).min(estimates.len().saturating_sub(1));
    let ci_lower = sorted_estimates.get(lower_idx).copied().unwrap_or(f64::NAN);
    let ci_upper = sorted_estimates.get(upper_idx).copied().unwrap_or(f64::NAN);

    (std_err, p_value, (ci_lower, ci_upper))
}
