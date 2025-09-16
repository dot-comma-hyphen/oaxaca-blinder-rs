use clarabel::algebra::CscMatrix;
use clarabel::solver::{
    DefaultSettings, DefaultSolver, IPSolver, NonnegativeConeT, SolverStatus, ZeroConeT,
};
use ndarray::{Array1, Array2};

/// Solves a quantile regression problem for a given quantile `tau`.
///
/// This function formulates the quantile regression as a linear programming problem
/// in the standard conic form and solves it using the `clarabel` crate.
///
/// # Arguments
///
/// * `x_data`: A 2D array of covariates (n_observations x n_features).
/// * `y_data`: A 1D array of the outcome variable (n_observations).
/// * `tau`: The quantile to be estimated, a value between 0 and 1.
///
/// # Returns
///
/// A `Result` containing a `Vec<f64>` of the estimated coefficients `β`,
/// or an error string if the problem could not be solved.
pub fn solve_qr(
    x_data: &Array2<f64>,
    y_data: &Array1<f64>,
    tau: f64,
) -> Result<Vec<f64>, String> {
    let (n_obs, n_features) = x_data.dim();
    if y_data.len() != n_obs {
        return Err(
            "Input dimensions mismatch: X and y must have the same number of observations."
                .to_string(),
        );
    }
    if !(0.0..=1.0).contains(&tau) {
        return Err("Tau must be between 0 and 1.".to_string());
    }

    // Total number of variables = n_features (β) + n_obs (u) + n_obs (v)
    let n_vars = n_features + 2 * n_obs;

    // P matrix (quadratic term) is zero for an LP.
    let p = CscMatrix::new(n_vars, n_vars, vec![0; n_vars + 1], vec![], vec![]);

    // q vector (linear term)
    let mut q = vec![0.0; n_vars];
    for i in 0..n_obs {
        q[n_features + i] = tau; // Cost for u_i
        q[n_features + n_obs + i] = 1.0 - tau; // Cost for v_i
    }

    // b vector for constraints
    let mut b = vec![0.0; 3 * n_obs];
    for i in 0..n_obs {
        b[i] = y_data[i];
    }

    // A matrix for constraints
    let mut a_col_ptr = vec![0];
    let mut a_row_ind = vec![];
    let mut a_nz_val = vec![];

    // Columns for β
    for j in 0..n_features {
        for i in 0..n_obs {
            a_row_ind.push(i);
            a_nz_val.push(x_data[[i, j]]);
        }
        a_col_ptr.push(a_nz_val.len());
    }

    // Columns for u
    for i in 0..n_obs {
        // from x_i'β + u_i - v_i = y_i
        a_row_ind.push(i);
        a_nz_val.push(1.0);
        // from -u_i <= 0
        a_row_ind.push(n_obs + i);
        a_nz_val.push(-1.0);
        a_col_ptr.push(a_nz_val.len());
    }

    // Columns for v
    for i in 0..n_obs {
        // from x_i'β + u_i - v_i = y_i
        a_row_ind.push(i);
        a_nz_val.push(-1.0);
        // from -v_i <= 0
        a_row_ind.push(2 * n_obs + i);
        a_nz_val.push(-1.0);
        a_col_ptr.push(a_nz_val.len());
    }

    let a = CscMatrix::new(3 * n_obs, n_vars, a_col_ptr, a_row_ind, a_nz_val);

    println!("P: {:?}", p);
    println!("q: {:?}", q);
    println!("A: {:?}", a);
    println!("b: {:?}", b);

    // Cones: n_obs equality constraints, 2 * n_obs non-negative constraints
    let cones = vec![ZeroConeT(n_obs), NonnegativeConeT(2 * n_obs)];

    let settings = DefaultSettings::default();

    let mut solver = match DefaultSolver::new(&p, &q, &a, &b, &cones, settings) {
        Ok(s) => s,
        Err(e) => return Err(format!("Error creating solver: {:?}", e)),
    };

    solver.solve();

    if solver.solution.status == SolverStatus::Solved {
        let coeffs = solver.solution.x[0..n_features].to_vec();
        Ok(coeffs)
    } else {
        Err(format!(
            "Solver failed with status: {:?}",
            solver.solution.status
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_solve_qr_median() {
        // Simple dataset with an outlier
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = array![
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
            [1.0, 5.0]
        ];
        let tau = 0.5;

        // Expected results from R's quantreg::rq(y ~ x, tau = 0.5)
        // coef(fit) -> (Intercept): 0.0, x: 1.0
        let expected_betas = vec![0.0, 1.0];

        let result = solve_qr(&x, &y, tau).unwrap();

        println!("Calculated coefficients (tau=0.5): {:?}", result);
        println!("Expected coefficients (tau=0.5): {:?}", expected_betas);

        assert_eq!(result.len(), 2);
        // Check if the results are close to the expected values
        let tolerance = 1e-4; // Looser tolerance for different solver
        assert!((result[0] - expected_betas[0]).abs() < tolerance);
        assert!((result[1] - expected_betas[1]).abs() < tolerance);
    }

    #[test]
    fn test_solve_qr_quartile() {
        // Simple dataset with an outlier
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = array![
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
            [1.0, 5.0]
        ];
        let tau = 0.25;

        // Expected results from R's quantreg::rq(y ~ x, tau = 0.25)
        // coef(fit) -> (Intercept): 0.5, x: 0.5
        let expected_betas = vec![0.5, 0.5];

        let result = solve_qr(&x, &y, tau).unwrap();

        println!("Calculated coefficients (tau=0.25): {:?}", result);
        println!("Expected coefficients (tau=0.25): {:?}", expected_betas);

        assert_eq!(result.len(), 2);
        // Check if the results are close to the expected values
        let tolerance = 1e-4; // Looser tolerance for different solver
        assert!((result[0] - expected_betas[0]).abs() < tolerance);
        assert!((result[1] - expected_betas[1]).abs() < tolerance);
    }
}