use nalgebra::{DVector};
use crate::math::ols::{OlsResult};
use std::collections::HashMap;

pub fn normalize_categorical_coefficients(
    ols_results: &mut OlsResult,
    predictor_names: &[String],
    categorical_vars: &[String],
    _x_mean: &DVector<f64>,
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

        // The intercept adjustment is simply adding the mean of the coefficients.
        // This effectively centers the coefficients around zero, and the intercept absorbs the average effect.
        ols_results.coefficients[0] += mean_of_coeffs;

        for &i in &dummy_indices {
            ols_results.coefficients[i] -= mean_of_coeffs;
        }
    }
    base_coeffs
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_normalize_categorical_coefficients() {
        // Setup: Intercept = 10.0
        // Categorical var "cat" has 3 levels: A, B, C.
        // Coefficients: A (base) = 0.0, B = 2.0, C = 4.0.
        // Means: A = 0.2, B = 0.3, C = 0.5.
        // Average coefficient beta_bar = (0 + 2 + 4) / 3 = 2.0.
        //
        // Correct Normalization:
        // New Intercept = Old Intercept + beta_bar = 10.0 + 2.0 = 12.0.
        // New Coeffs: A = 0 - 2 = -2, B = 2 - 2 = 0, C = 4 - 2 = 2.
        //
        // Check Invariance:
        // Prediction for A: 12 + (-2) = 10. (Matches original 10 + 0)
        // Prediction for B: 12 + 0 = 12. (Matches original 10 + 2)
        // Prediction for C: 12 + 2 = 14. (Matches original 10 + 4)

        let mut coeffs = DVector::from_vec(vec![10.0, 2.0, 4.0]); // Intercept, cat_B, cat_C
        let mut vcov = DMatrix::zeros(3, 3);
        let residuals = DVector::zeros(0); // Empty residuals for test
        let mut ols_result = OlsResult { coefficients: coeffs, vcov, residuals };

        let predictor_names = vec![
            "intercept".to_string(),
            "cat_B".to_string(),
            "cat_C".to_string(),
        ];
        let categorical_vars = vec!["cat".to_string()];
        
        // Dummy means (not used in the new correct logic, but required by signature)
        let x_mean = DVector::from_vec(vec![1.0, 0.3, 0.5]); 
        
        let mut category_counts = HashMap::new();
        category_counts.insert("cat".to_string(), 3);

        normalize_categorical_coefficients(
            &mut ols_result,
            &predictor_names,
            &categorical_vars,
            &x_mean,
            &category_counts,
        );

        // Verify Intercept
        assert!((ols_result.coefficients[0] - 12.0).abs() < 1e-9);

        // Verify Coefficients (B and C shifted by -2.0)
        assert!((ols_result.coefficients[1] - 0.0).abs() < 1e-9); // B was 2.0, now 0.0
        assert!((ols_result.coefficients[2] - 2.0).abs() < 1e-9); // C was 4.0, now 2.0
    }
}
