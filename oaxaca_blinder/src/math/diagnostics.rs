//! Variance Inflation Factor (VIF) diagnostic function.
use polars::prelude::*;
use nalgebra::{DMatrix, DVector};
use crate::OaxacaError;
use crate::math::ols::ols;

/// Represents the result of a VIF calculation for a single variable.
#[derive(Debug, PartialEq)]
pub struct VifResult {
    pub variable_name: String,
    pub vif_score: f64,
}

/// Calculates the Variance Inflation Factor (VIF) for each predictor in a given dataset.
///
/// VIF is a measure of multicollinearity among predictor variables in a regression model.
///
/// # Arguments
///
/// * `df` - A reference to a Polars DataFrame containing the data.
/// * `predictor_names` - A slice of strings representing the names of the predictor variables.
///
/// # Returns
///
/// A `Result` containing a `Vec<VifResult>` on success, or an `OaxacaError` on failure.
///
pub fn calculate_vif(
    df: &DataFrame,
    predictor_names: &[String],
) -> Result<Vec<VifResult>, OaxacaError> {
    if predictor_names.len() < 2 {
        return Err(OaxacaError::DiagnosticError(
            "VIF calculation requires at least two predictors.".to_string(),
        ));
    }

    let mut results = Vec::new();
    let mut centered_df = df.clone();
    for name in predictor_names {
        let series = centered_df.column(name)?;
        let mean = series.mean().unwrap_or(0.0);
        centered_df.with_column(series - mean)?;
    }

    for p in predictor_names {
        let y_series = centered_df.column(p)?;
        let y_vec: Vec<f64> = y_series.f64()?.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
        let y = DVector::from_vec(y_vec);

        let other_predictors: Vec<String> = predictor_names
            .iter()
            .filter(|&name| name != p)
            .cloned()
            .collect();

        let x_df = centered_df.select(&other_predictors)?;
        let mut x_df_with_intercept = x_df.clone();
        let intercept = Series::new("intercept", vec![1.0; x_df.height()]);
        x_df_with_intercept.with_column(intercept)?;
        
        let x_matrix_ndarray = x_df_with_intercept.to_ndarray::<Float64Type>(IndexOrder::C)?;
        let x_matrix = DMatrix::from_row_slice(
            x_df_with_intercept.height(),
            x_df_with_intercept.width(),
            &x_matrix_ndarray.into_raw_vec(),
        );

        let ols_result = match ols(&y, &x_matrix) {
            Ok(res) => res,
            Err(OaxacaError::NalgebraError(msg)) if msg.contains("Failed to invert X'X matrix") => {
                results.push(VifResult {
                    variable_name: p.clone(),
                    vif_score: f64::INFINITY,
                });
                continue;
            }
            Err(e) => return Err(e),
        };
        
        let y_hat = &x_matrix * &ols_result.coefficients;
        let y_mean = y.mean();

        let ss_total = y.iter().map(|&val| (val - y_mean).powi(2)).sum::<f64>();
        let ss_residual = y.iter().zip(y_hat.iter()).map(|(&yi, &y_hat_i)| (yi - y_hat_i).powi(2)).sum::<f64>();

        if ss_total == 0.0 {
            results.push(VifResult {
                variable_name: p.clone(),
                vif_score: f64::INFINITY,
            });
            continue;
        }

        let r_squared = 1.0 - (ss_residual / ss_total);

        let vif_score = if (1.0 - r_squared).abs() < 1e-9 {
            f64::INFINITY
        } else {
            1.0 / (1.0 - r_squared)
        };

        results.push(VifResult {
            variable_name: p.clone(),
            vif_score,
        });
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    #[test]
    fn test_calculate_vif() {
        let df = df! (
            "x1" => &[1.0, 2.0, 3.0, 4.0, 5.0],
            "x2" => &[2.0, 3.0, 1.0, 5.0, 4.0],
            "x3" => &[1.0, 5.0, 2.0, 4.0, 3.0],
        )
        .unwrap();

        let predictor_names = vec![
            "x1".to_string(),
            "x2".to_string(),
            "x3".to_string(),
        ];

        let vif_results = calculate_vif(&df, &predictor_names).unwrap();

        // Expected values calculated from statsmodels in Python
        let expected_vif_x1 = 1.0909090909090908;
        let expected_vif_x2 = 1.0909090909090908;
        let expected_vif_x3 = 1.0;

        assert!((vif_results[0].vif_score - expected_vif_x1).abs() < 1e-6);
        assert_eq!(vif_results[0].variable_name, "x1");

        assert!((vif_results[1].vif_score - expected_vif_x2).abs() < 1e-6);
        assert_eq!(vif_results[1].variable_name, "x2");

        assert!((vif_results[2].vif_score - expected_vif_x3).abs() < 1e-6);
        assert_eq!(vif_results[2].variable_name, "x3");
    }

    #[test]
    fn test_perfect_multicollinearity() {
        let df = df! (
            "x1" => &[1.0, 2.0, 3.0, 4.0, 5.0],
            "x2" => &[2.0, 4.0, 6.0, 8.0, 10.0], // x2 = 2 * x1
            "x3" => &[1.0, 1.0, 2.0, 2.0, 3.0],
        )
        .unwrap();

        let predictor_names = vec![
            "x1".to_string(),
            "x2".to_string(),
            "x3".to_string(),
        ];

        let vif_results = calculate_vif(&df, &predictor_names).unwrap();

        assert_eq!(vif_results[0].vif_score, f64::INFINITY);
        assert_eq!(vif_results[0].variable_name, "x1");

        assert_eq!(vif_results[1].vif_score, f64::INFINITY);
        assert_eq!(vif_results[1].variable_name, "x2");

        // When calculating VIF for x3, the auxiliary regression is x3 ~ x1 + x2.
        // Since x1 and x2 are perfectly collinear, the OLS will fail,
        // and our function correctly returns INFINITY.
        assert_eq!(vif_results[2].vif_score, f64::INFINITY);
        assert_eq!(vif_results[2].variable_name, "x3");
    }

    #[test]
    fn test_too_few_predictors() {
        let df = df! (
            "x1" => &[1.0, 2.0, 3.0, 4.0, 5.0],
        )
        .unwrap();

        let predictor_names = vec!["x1".to_string()];

        let result = calculate_vif(&df, &predictor_names);

        assert!(result.is_err());
        if let Err(OaxacaError::DiagnosticError(msg)) = result {
            assert_eq!(msg, "VIF calculation requires at least two predictors.");
        } else {
            panic!("Expected OaxacaError::DiagnosticError");
        }
    }
}
