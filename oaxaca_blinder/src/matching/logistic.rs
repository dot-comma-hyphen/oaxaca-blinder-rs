use nalgebra::{DMatrix, DVector};
use std::f64::consts::E;

/// Simple Logistic Regression implementation using Newton-Raphson optimization.
#[derive(Debug, Clone)]
pub struct LogisticRegression {
    pub coefficients: DVector<f64>,
}

impl LogisticRegression {
    pub fn new() -> Self {
        Self {
            coefficients: DVector::zeros(0),
        }
    }

    /// Fits the logistic regression model to the data.
    ///
    /// # Arguments
    ///
    /// * `x` - Design matrix (n_samples x n_features). Should include intercept column if desired.
    /// * `y` - Target vector (n_samples). Values should be 0.0 or 1.0.
    /// * `max_iter` - Maximum number of iterations.
    /// * `tol` - Convergence tolerance.
    pub fn fit(
        &mut self,
        x: &DMatrix<f64>,
        y: &DVector<f64>,
        max_iter: usize,
        tol: f64,
    ) -> Result<(), String> {
        let n_features = x.ncols();
        let n_samples = x.nrows();

        if n_samples != y.len() {
            return Err("Number of samples in X and y must match".to_string());
        }

        // Initialize coefficients to zeros
        let mut beta = DVector::zeros(n_features);

        for _iter in 0..max_iter {
            // Calculate probabilities: p = 1 / (1 + exp(-X * beta))
            let xb = x * &beta;
            let p: DVector<f64> = xb.map(|val| 1.0 / (1.0 + E.powf(-val)));

            // Calculate gradient: X^T * (y - p)
            // Note: y - p is the error
            let error = y - &p;
            let gradient = x.transpose() * &error;

            // Calculate Hessian: X^T * W * X
            // W is diagonal matrix with elements p_i * (1 - p_i)
            // To avoid creating a huge diagonal matrix, we can compute X^T * W * X directly
            // or scale rows of X.
            // Let's scale rows of X by sqrt(w) then compute Gram matrix?
            // Or just compute (X^T * W) * X

            // Constructing W explicitly is O(N^2) memory, bad.
            // We need X^T * diag(w) * X
            // Let W_vec = p * (1-p)
            let w_vec: DVector<f64> = p.map(|val| val * (1.0 - val));

            // Compute Hessian efficiently
            // H_jk = sum_i (x_ij * x_ik * w_i)
            // let mut hessian = DMatrix::zeros(n_features, n_features);

            // This loop is O(N * K^2), acceptable if K is small.
            // For larger K, matrix multiplication is better.
            // X_weighted = diag(w) * X
            // But diag(w) is huge.
            // Instead: X_weighted_rows = X.rows * w_i
            // H = X.T * X_weighted_rows

            // Vectorized approach for Hessian
            let mut x_weighted = x.clone();
            for i in 0..n_samples {
                let w = w_vec[i];
                let mut row = x_weighted.row_mut(i);
                row *= w;
            }
            let mut hessian = x.transpose() * x_weighted;

            // Regularization (Ridge) to avoid singular matrix?
            // Add small value to diagonal
            for i in 0..n_features {
                hessian[(i, i)] += 1e-6;
            }

            // Newton step: beta_new = beta_old + H^-1 * gradient
            // Solve H * delta = gradient
            let delta = hessian.lu().solve(&gradient).ok_or("Hessian is singular")?;

            beta += &delta;

            if delta.norm() < tol {
                break;
            }
        }

        self.coefficients = beta;
        Ok(())
    }

    /// Predicts probabilities for new data.
    pub fn predict_proba(&self, x: &DMatrix<f64>) -> DVector<f64> {
        let xb = x * &self.coefficients;
        xb.map(|val| 1.0 / (1.0 + E.powf(-val)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logistic_regression() {
        // Simple OR gate logic
        // X: [[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]] (Intercept, F1, F2)
        // Y: [0, 1, 1, 1]

        let x = DMatrix::from_row_slice(
            4,
            3,
            &[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        );
        let y = DVector::from_vec(vec![0.0, 1.0, 1.0, 1.0]);

        let mut model = LogisticRegression::new();
        model.fit(&x, &y, 100, 1e-6).unwrap();

        let preds = model.predict_proba(&x);

        assert!(preds[0] < 0.5);
        assert!(preds[1] > 0.5);
        assert!(preds[2] > 0.5);
        assert!(preds[3] > 0.5);
    }
}
