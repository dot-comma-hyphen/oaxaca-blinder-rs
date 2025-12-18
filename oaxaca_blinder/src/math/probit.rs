use crate::OaxacaError;
use nalgebra::{DMatrix, DVector};
use statrs::distribution::{Continuous, ContinuousCDF, Normal};

/// Represents the results of a Probit regression.
#[derive(Debug)]
#[allow(dead_code)]
pub struct ProbitResult {
    pub coefficients: DVector<f64>,
    pub vcov: DMatrix<f64>,
    pub converged: bool,
    pub iterations: usize,
}

/// Performs Probit regression using Newton-Raphson optimization.
///
/// Probit regression models the probability that Y = 1 as Φ(Xβ), where Φ is the standard normal CDF.
///
/// # Arguments
///
/// * `y` - Binary outcome vector (0s and 1s).
/// * `x` - Predictor matrix.
/// * `max_iter` - Maximum number of iterations.
/// * `tol` - Convergence tolerance.
pub fn probit(
    y: &DVector<f64>,
    x: &DMatrix<f64>,
    max_iter: usize,
    tol: f64,
) -> Result<ProbitResult, OaxacaError> {
    let n = x.nrows();
    let k = x.ncols();

    // Initial guess: OLS coefficients (Linear Probability Model)
    // Add small regularization to avoid perfect separation issues initially?
    // Or just start with zeros. OLS is usually a good start.
    // We can use our existing ols function but we need to handle the Result.
    // For simplicity, let's start with zeros.
    let mut beta = DVector::zeros(k);

    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut converged = false;
    let mut iterations = 0;

    // Hessian matrix (Information Matrix)
    let mut h = DMatrix::zeros(k, k);

    for iter in 0..max_iter {
        iterations = iter + 1;

        // Calculate linear predictor: z = Xβ
        let z = x * &beta;

        // Calculate gradient and Hessian
        let mut gradient = DVector::zeros(k);
        h = DMatrix::zeros(k, k);

        // We accumulate gradient and Hessian contributions row by row
        // Gradient = \sum \lambda_i x_i
        // Hessian = \sum -\lambda_i (\lambda_i + z_i) x_i x_i' (Observed)
        // Or Expected Hessian: \sum - \frac{\phi^2}{\Phi(1-\Phi)} x_i x_i'
        // Expected Hessian is generally more stable (Fisher Scoring).

        for i in 0..n {
            let xi = x.row(i).transpose();
            let yi = y[i];
            let zi = z[i];

            let phi = normal.pdf(zi);
            let big_phi = normal.cdf(zi);

            // Avoid division by zero
            let big_phi = big_phi.clamp(1e-10, 1.0 - 1e-10);

            let lambda = if yi > 0.5 {
                phi / big_phi
            } else {
                -phi / (1.0 - big_phi)
            };

            // Gradient contribution
            gradient += &xi * lambda;

            // Expected Hessian contribution (Fisher Scoring)
            // Weight w_i = \frac{\phi^2}{\Phi(1-\Phi)}
            let weight = (phi * phi) / (big_phi * (1.0 - big_phi));
            h -= &xi * xi.transpose() * weight;
        }

        // Newton step: \Delta \beta = -H^{-1} g
        // Since we use Fisher Scoring, H is negative definite.
        // We solve H \Delta \beta = -g

        // Add small regularization to diagonal to ensure invertibility
        for i in 0..k {
            h[(i, i)] -= 1e-9;
        }

        let h_inv = match h.clone().try_inverse() {
            Some(inv) => inv,
            None => {
                return Err(OaxacaError::NalgebraError(
                    "Failed to invert Hessian in Probit".to_string(),
                ))
            }
        };

        let _delta = &h_inv * (-&gradient); // Wait, H is negative. So -H is positive definite.
                                            // delta = (-H)^{-1} * g
                                            // delta = -(H^{-1} * g)
                                            // Actually: beta_new = beta_old - H^{-1} g.
                                            // If H is negative definite, -H^{-1} is positive definite.

        let step = -&h_inv * &gradient;
        beta += &step;

        if step.norm() < tol {
            converged = true;
            break;
        }
    }

    // Variance-Covariance Matrix is -H^{-1} (Inverse of Observed Information)
    // Since we used Expected Information (Fisher Scoring), it's the inverse of the Fisher Information Matrix.
    // Fisher Information I = -E[H]. So V = I^{-1}.
    // In our loop, `h` is the negative of the Fisher Information Matrix (accumulated as -weight * x * x').
    // So `h` is -I.
    // So V = (-h)^{-1} = -(h^{-1}).

    let vcov = match h.try_inverse() {
        Some(inv) => -inv,
        None => {
            return Err(OaxacaError::NalgebraError(
                "Failed to invert Hessian for VCOV".to_string(),
            ))
        }
    };

    Ok(ProbitResult {
        coefficients: beta,
        vcov,
        converged,
        iterations,
    })
}
