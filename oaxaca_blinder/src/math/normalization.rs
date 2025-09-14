use nalgebra::{DVector};
use crate::math::ols::{OlsResult};

pub fn normalize_categorical_coefficients(
    ols_results: &mut OlsResult,
    predictor_names: &[String],
    categorical_vars: &[String],
    x_mean: &DVector<f64>,
) {
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

        let mean_of_coeffs = sum_of_coeffs / (dummy_indices.len() as f64);

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
}
