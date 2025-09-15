use crate::OaxacaError;
use nalgebra::{DMatrix, DVector};

/// Represents the results of an OLS regression.
#[derive(Debug)]
pub struct OlsResult {
    pub coefficients: DVector<f64>,
    pub vcov: DMatrix<f64>,
}

/// Performs an Ordinary Least Squares (OLS) regression.
///
/// The function calculates the coefficient vector `β` using the formula:
/// `β = (X'X)⁻¹ * X'y`
///
/// # Arguments
///
/// * `y` - A `DVector` representing the outcome variable.
/// * `x` - A `DMatrix` representing the predictor variables. It is crucial that this
///   matrix includes a column of ones if an intercept is desired in the model.
///
/// # Returns
///
/// A `Result` containing the `OlsResult` on success, or an `OaxacaError` if the
/// `X'X` matrix is singular and cannot be inverted.
pub fn ols(y: &DVector<f64>, x: &DMatrix<f64>) -> Result<OlsResult, OaxacaError> {
    // Calculate X'X
    let xtx = x.transpose() * x;

    // Attempt to invert the X'X matrix.
    // This is the most likely point of failure if the predictors are multicollinear.
    let xtx_inv = xtx.try_inverse().ok_or_else(|| {
        OaxacaError::NalgebraError(
            "Failed to invert X'X matrix. This is likely due to perfect multicollinearity in the predictors.".to_string(),
        )
    })?;

    // Calculate X'y
    let xty = x.transpose() * y;

    // Calculate the coefficients: β = (X'X)⁻¹ * X'y
    let coefficients = &xtx_inv * xty;

    // Calculate residuals
    let y_hat = x * &coefficients;
    let residuals = y - y_hat;

    // Calculate residual variance
    let n = x.nrows() as f64;
    let k = x.ncols() as f64;
    let sse = residuals.transpose() * residuals;
    let sigma_squared = sse[(0, 0)] / (n - k);

    // Calculate variance-covariance matrix
    let vcov = xtx_inv * sigma_squared;

    Ok(OlsResult { coefficients, vcov })
}




#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_ols_simple_regression() {
        // Test a simple model: y = 1 + 2x
        // Note: DMatrix::from_vec is column-major.
        let x = DMatrix::from_vec(
            5,
            2,
            vec![
                // Column 1: Intercept
                1.0, 1.0, 1.0, 1.0, 1.0,
                // Column 2: x-values
                0.0, 1.0, 2.0, 3.0, 4.0,
            ],
        );
        let y = DVector::from_vec(vec![1.0, 3.0, 5.0, 7.0, 9.0]);

        let result = ols(&y, &x).expect("OLS calculation failed on valid data");
        let coeffs = result.coefficients;

        // Check that the calculated coefficients are very close to the true values.
        assert_eq!(coeffs.len(), 2);
        assert!((coeffs[0] - 1.0).abs() < 1e-9, "Intercept is incorrect");
        assert!((coeffs[1] - 2.0).abs() < 1e-9, "Slope is incorrect");
    }

    #[test]
    fn test_ols_handles_singular_matrix() {
        // Create a singular matrix by making two columns perfectly correlated.
        // Column 2 is 2 * Column 1.
        let x = DMatrix::from_vec(
            3,
            2,
            vec![
                // Column 1
                1.0, 1.0, 1.0,
                // Column 2
                2.0, 2.0, 2.0,
            ],
        );
        let y = DVector::from_vec(vec![1.0, 2.0, 3.0]);

        let result = ols(&y, &x);

        // Assert that the result is an error and that it's the specific error we expect.
        assert!(result.is_err());
        match result {
            Err(OaxacaError::NalgebraError(msg)) => {
                assert!(msg.contains("Failed to invert X'X matrix"));
            }
            _ => panic!("Expected a NalgebraError for a singular matrix, but got something else."),
        }
    }

    
}
