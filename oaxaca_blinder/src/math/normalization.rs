use nalgebra::{DVector};
use crate::math::ols::{OlsResult};
use std::collections::HashMap;

pub fn normalize_categorical_coefficients(
    ols_results: &mut OlsResult,
    predictor_names: &[String],
    categorical_vars: &[String],
    x_mean: &DVector<f64>,
    category_counts: &HashMap<String, usize>,
) -> HashMap<String, f64> {
    let mut base_coeffs = HashMap::new();
    for var in categorical_vars {
        let dummy_indices: Vec<usize> = predictor_names
            .iter()
            .enumerate()
            .filter(|(_, name)| name.starts_with(&format!("{}_", var)))
            .map(|(i, _)| i)
            .collect();

        if dummy_indices.is_empty() {
            continue;
        }

        let mut sum_of_coeffs = 0.0;
        for &i in &dummy_indices {
            sum_of_coeffs += ols_results.coefficients[i];
        }

        let m = category_counts.get(var).cloned().unwrap_or(dummy_indices.len() + 1);
        if m == 0 { continue; }
        let mean_of_coeffs = sum_of_coeffs / (m as f64);

        base_coeffs.insert(var.clone(), -mean_of_coeffs);

        let mut sum_of_means = 0.0;
        for &i in &dummy_indices {
            sum_of_means += x_mean[i];
        }

        let intercept_adjustment = mean_of_coeffs * sum_of_means;

        for &i in &dummy_indices {
            ols_results.coefficients[i] -= mean_of_coeffs;
        }

        // Assuming intercept is at index 0
        ols_results.coefficients[0] += intercept_adjustment;
    }
    base_coeffs
}
