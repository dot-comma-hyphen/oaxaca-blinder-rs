use crate::OaxacaError;
use nalgebra::{DMatrix, DVector};

/// Represents the results of a Logistic Regression.
#[derive(Debug)]
#[allow(dead_code)]
pub struct LogitResult {
    pub coefficients: DVector<f64>,
    pub predicted_probs: DVector<f64>,
    pub converged: bool,
    pub iterations: usize,
}

/// Sigmoid function.
fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// Performs Logistic Regression using Newton-Raphson optimization.
///
/// # Arguments
///
/// * `y` - A binary `DVector` (0.0 or 1.0) representing the outcome.
/// * `x` - A `DMatrix` representing the predictors (should include intercept).
/// * `max_iter` - Maximum number of iterations.
/// * `tol` - Convergence tolerance.
///
/// # Returns
///
/// A `Result` containing the `LogitResult`.
pub fn logit(
    y: &DVector<f64>,
    x: &DMatrix<f64>,
    max_iter: usize,
    tol: f64,
) -> Result<LogitResult, OaxacaError> {
    let _n = x.nrows();
    let k = x.ncols();

    // Initialize coefficients to zero
    let mut beta = DVector::zeros(k);

    for iter in 0..max_iter {
        let xb = x * &beta;
        let probs: DVector<f64> = xb.map(|z| sigmoid(z).clamp(1e-10, 1.0 - 1e-10));

        // Gradient: X' * (y - p)
        let error = y - &probs;
        let mut gradient = DVector::zeros(k);
        let x_slice = x.as_slice();
        let e_slice = error.as_slice();
        for j in 0..k {
            let col_start = j * _n;
            let col = &x_slice[col_start..col_start + _n];
            gradient[j] = col
                .iter()
                .zip(e_slice.iter())
                .map(|(&xj, &ej)| xj * ej)
                .sum::<f64>();
        }

        // Hessian: X' * W * X, where W is diagonal matrix with p * (1-p)
        // Constructing full diagonal matrix W is expensive (N*N).
        // Instead, compute X' * W * X directly.
        // (X' * W * X)_ij = sum_k (X_ki * W_kk * X_kj)
        // W_kk = p_k * (1 - p_k)

        let w_diag: DVector<f64> = probs.map(|p| p * (1.0 - p));

        // Compute Hessian: H = -(X' * W * X), where W is diagonal matrix of w_diag.
        // To avoid full matrix clone and multiple transpositions, we compute the
        // weighted Gram matrix directly using nalgebra's internal data layout.
        let mut hessian = DMatrix::zeros(k, k);
        let x_slice = x.as_slice();
        let w_slice = w_diag.as_slice();

        for j in 0..k {
            let col_j_start = j * _n;
            let col_j = &x_slice[col_j_start..col_j_start + _n];

            for i in j..k {
                let col_i_start = i * _n;
                let col_i = &x_slice[col_i_start..col_i_start + _n];

                let dot = col_i
                    .iter()
                    .zip(col_j.iter())
                    .zip(w_slice.iter())
                    .map(|((&ci, &cj), &w)| ci * cj * w)
                    .sum::<f64>();

                hessian[(i, j)] = -dot;
                if i != j {
                    hessian[(j, i)] = -dot;
                }
            }
        }

        // Update step: beta_new = beta_old - Hessian^-1 * Gradient
        // beta_new = beta_old + (X'WX)^-1 * Gradient

        // Invert Hessian (actually -Hessian to get positive definite matrix for Cholesky usually, but here just invert)
        // The Hessian of log-likelihood is negative definite.
        // We want to maximize likelihood.
        // Update: beta = beta - H^-1 * g
        // Since H is negative, let's work with Information Matrix I = -H.
        // beta = beta + I^-1 * g

        let information_matrix = -hessian;

        // Add ridge regularization for stability if needed? Standard logit usually doesn't unless specified.
        // But for DFL, perfect separation can be an issue.
        // Let's try standard inversion first.

        let inv_info = information_matrix.try_inverse().ok_or_else(|| {
            OaxacaError::NalgebraError(
                "Failed to invert Information Matrix in Logit. Perfect separation?".to_string(),
            )
        })?;

        let step = &inv_info * &gradient;
        beta += &step;

        if step.norm() < tol {
            return Ok(LogitResult {
                coefficients: beta,
                predicted_probs: probs, // Note: this is from start of iter, maybe update one last time?
                converged: true,
                iterations: iter + 1,
            });
        }
    }

    // Calculate final probs
    let xb = x * &beta;
    let probs = xb.map(|z| sigmoid(z).clamp(1e-10, 1.0 - 1e-10));

    Ok(LogitResult {
        coefficients: beta,
        predicted_probs: probs,
        converged: false,
        iterations: max_iter,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_logit_simple_regression() {
        let x = DMatrix::from_vec(
            11,
            2,
            vec![
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // Intercept
                -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, // X
            ],
        );
        let y = DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        let result = logit(&y, &x, 100, 1e-6).expect("Logit calculation failed on valid data");
        let coeffs = result.coefficients;

        assert_eq!(coeffs.len(), 2);
        // values from statsmodels
        assert!((coeffs[0] - 0.6533055).abs() < 1e-4);
        assert!((coeffs[1] - 1.3046124).abs() < 1e-4);
        assert!(result.converged);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_logit_perfect_separation() {
        let x = DMatrix::from_vec(
            6,
            2,
            vec![
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // Intercept
                -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, // X
            ],
        );
        let y = DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let _result = logit(&y, &x, 100, 1e-6);
        // Perfect separation might cause an inversion error or hit max iterations depending on regularization
        // The main thing is that it doesn't crash or panic.
    }

    #[test]
    fn test_logit_singular_matrix() {
        let x = DMatrix::from_vec(
            4,
            2,
            vec![
                1.0, 1.0, 1.0, 1.0, // Col 1
                1.0, 1.0, 1.0, 1.0, // Col 2 (same as Col 1)
            ],
        );
        let y = DVector::from_vec(vec![0.0, 1.0, 0.0, 1.0]);

        let result = logit(&y, &x, 100, 1e-6);
        assert!(result.is_err());
        match result {
            Err(crate::OaxacaError::NalgebraError(msg)) => {
                assert!(msg.contains("Failed to invert Information Matrix"));
            }
            Err(_) => panic!("Expected NalgebraError"),
            Ok(_) => panic!("Expected an error for singular matrix"),
        }
    }

    #[test]
    fn test_logit_max_iterations() {
        let x = DMatrix::from_vec(
            11,
            2,
            vec![
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // Intercept
                -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, // X
            ],
        );
        let y = DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        let result = logit(&y, &x, 1, 1e-6).expect("Logit calculation failed");
        assert!(!result.converged);
        assert_eq!(result.iterations, 1);
    }
}
