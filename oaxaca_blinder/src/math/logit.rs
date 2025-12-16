use crate::OaxacaError;
use nalgebra::{DMatrix, DVector};

/// Represents the results of a Logistic Regression.
#[derive(Debug)]
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
    let n = x.nrows();
    let k = x.ncols();

    // Initialize coefficients to zero
    let mut beta = DVector::zeros(k);

    for iter in 0..max_iter {
        let xb = x * &beta;
        let probs: DVector<f64> = xb.map(sigmoid);

        // Gradient: X' * (y - p)
        let error = y - &probs;
        let gradient = x.transpose() * &error;

        // Hessian: X' * W * X, where W is diagonal matrix with p * (1-p)
        // Constructing full diagonal matrix W is expensive (N*N).
        // Instead, compute X' * W * X directly.
        // (X' * W * X)_ij = sum_k (X_ki * W_kk * X_kj)
        // W_kk = p_k * (1 - p_k)

        let w_diag: DVector<f64> = probs.map(|p| p * (1.0 - p));

        // We can compute X'WX as (X.transpose() * (W_diag.component_mul(X_col)))?
        // Or simpler: Scale rows of X by sqrt(w), then compute Gram matrix.
        // Let X_tilde = sqrt(W) * X
        // Hessian = - X_tilde' * X_tilde

        let sqrt_w: DVector<f64> = w_diag.map(|w| w.sqrt());
        let mut x_tilde = x.clone();
        for i in 0..n {
            let w_i = sqrt_w[i];
            for j in 0..k {
                x_tilde[(i, j)] *= w_i;
            }
        }

        let hessian = -(x_tilde.transpose() * x_tilde);

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
    let probs = xb.map(sigmoid);

    Ok(LogitResult {
        coefficients: beta,
        predicted_probs: probs,
        converged: false,
        iterations: max_iter,
    })
}
