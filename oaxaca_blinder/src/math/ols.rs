use crate::OaxacaError;
use nalgebra::{DMatrix, DVector};

/// Represents the results of an OLS regression.
#[derive(Debug)]
#[allow(dead_code)]
pub struct OlsResult {
    pub coefficients: DVector<f64>,
    pub vcov: DMatrix<f64>,
    pub residuals: DVector<f64>,
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
/// Performs an Ordinary Least Squares (OLS) or Weighted Least Squares (WLS) regression.
///
/// The function calculates the coefficient vector `β` using the formula:
/// `β = (X'WX)⁻¹ * X'Wy` (where W is the weight matrix, Identity if unweighted)
///
/// # Arguments
///
/// * `y` - A `DVector` representing the outcome variable.
/// * `x` - A `DMatrix` representing the predictor variables. It is crucial that this
///   matrix includes a column of ones if an intercept is desired in the model.
/// * `weights` - An optional `DVector` of sample weights.
///
/// # Returns
///
/// A `Result` containing the `OlsResult` on success, or an `OaxacaError` if the
/// `X'WX` matrix is singular and cannot be inverted.
pub fn ols(
    y: &DVector<f64>,
    x: &DMatrix<f64>,
    weights: Option<&DVector<f64>>,
) -> Result<OlsResult, OaxacaError> {
    let (xtx, xty, n_obs) = if let Some(w) = weights {
        // Weighted Least Squares
        // We want to minimize (y - Xβ)'W(y - Xβ)
        // Normal equations: (X'WX)β = X'Wy

        // Efficiently compute X'WX and X'Wy without creating the full diagonal matrix W
        // X'WX = \sum w_i * x_i * x_i'
        // X'Wy = \sum w_i * x_i * y_i

        // Alternative: Transform data X* = \sqrt{W}X, y* = \sqrt{W}y
        // Then run OLS on X*, y*
        let w_sqrt = w.map(|v| v.sqrt());

        // Scale X by sqrt(weights) row-wise
        let mut x_w = x.clone();
        for j in 0..x.ncols() {
            let mut col = x_w.column_mut(j);
            col.component_mul_assign(&w_sqrt);
        }

        // Scale y by sqrt(weights)
        let y_w = y.component_mul(&w_sqrt);

        let xtx = x_w.transpose() * &x_w;
        let xty = x_w.transpose() * &y_w;

        // Effective sample size? Usually just sum of weights or N?
        // For variance estimation in survey data, it's complicated.
        // But for standard WLS (heteroskedasticity), we use N.
        // If weights are frequency weights, we use sum(w).
        // Let's assume sampling weights/frequency weights -> sum(w).
        let n = w.sum();

        (xtx, xty, n)
    } else {
        // Ordinary Least Squares
        let xtx = x.transpose() * x;
        let xty = x.transpose() * y;
        let n = x.nrows() as f64;
        (xtx, xty, n)
    };

    // Attempt Cholesky decomposition on X'X (or X'WX).
    // This is more numerically stable than explicit inversion and acts as a check for positive-definiteness.
    // X'X should be positive definite if there is no perfect multicollinearity.
    let cholesky = xtx.cholesky().ok_or_else(|| {
        OaxacaError::NalgebraError(
            "Failed to perform Cholesky decomposition. Matrix may be singular or not positive definite due to multicollinearity.".to_string(),
        )
    })?;

    // Calculate coefficients: β = (X'X)⁻¹ * X'y
    // We solve the linear system (X'X) * β = X'y using the Cholesky factor.
    let coefficients = cholesky.solve(&xty);

    // Calculate residuals (Raw residuals: y - Xβ)
    let y_hat = x * &coefficients;
    let residuals = y - y_hat;

    // Calculate residual variance
    // For WLS: e'We / (n - k)
    let k = x.ncols() as f64;

    let sse = if let Some(w) = weights {
        // Weighted Sum of Squared Errors
        let weighted_residuals = residuals.component_mul(w); // e * w
        residuals.dot(&weighted_residuals) // e' * (w * e) = \sum w_i * e_i^2
    } else {
        residuals.norm_squared()
    };

    let sigma_squared = sse / (n_obs - k);

    // Calculate variance-covariance matrix: (X'X)⁻¹ * σ²
    // We can get the inverse from the Cholesky decomposition efficiently.
    let xtx_inv = cholesky.inverse();
    let vcov = xtx_inv * sigma_squared;

    Ok(OlsResult {
        coefficients,
        vcov,
        residuals,
    })
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
                1.0, 1.0, 1.0, 1.0, 1.0, // Column 2: x-values
                0.0, 1.0, 2.0, 3.0, 4.0,
            ],
        );
        let y = DVector::from_vec(vec![1.0, 3.0, 5.0, 7.0, 9.0]);

        let result = ols(&y, &x, None).expect("OLS calculation failed on valid data");
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
                1.0, 1.0, 1.0, // Column 2
                2.0, 2.0, 2.0,
            ],
        );
        let y = DVector::from_vec(vec![1.0, 2.0, 3.0]);

        let result = ols(&y, &x, None);

        // Assert that the result is an error and that it's the specific error we expect.
        assert!(result.is_err());
        match result {
            Err(OaxacaError::NalgebraError(msg)) => {
                assert!(msg.contains("Failed to perform Cholesky decomposition"));
            }
            _ => panic!("Expected a NalgebraError for a singular matrix, but got something else."),
        }
    }
}
