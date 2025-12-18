#![cfg(feature = "pay_equity")]
use crate::engine::{ProblemDefinition, VariableMap};
use anyhow::{Context, Result};
use good_lp::{variable, Expression, SolverModel};
use nalgebra::DVector;
use oaxaca_blinder::OaxacaBuilder;

pub struct PayEquityProblem {
    pub builder: OaxacaBuilder,
    pub target_gap: f64,
}

impl PayEquityProblem {
    pub fn new(builder: OaxacaBuilder, target_gap: f64) -> Self {
        Self {
            builder,
            target_gap,
        }
    }

    fn calculate_coefficients(&self) -> Result<(DVector<f64>, f64)> {
        // 1. Get matrices
        let (x_a, y_a, x_b, y_b, _) = self
            .builder
            .get_data_matrices()
            .map_err(|e| anyhow::anyhow!("Oaxaca Error: {}", e))?;

        // 2. Calculate initial OLS for Group A and B
        let xtx_a = x_a.transpose() * &x_a;
        let xtx_b = x_b.transpose() * &x_b;

        let chol_a = xtx_a
            .cholesky()
            .context("Matrix X_A'X_A is not positive definite")?;
        let beta_a = chol_a.solve(&(&x_a.transpose() * &y_a));

        let chol_b = xtx_b
            .cholesky()
            .context("Matrix X_B'X_B is not positive definite")?;
        let beta_b = chol_b.solve(&(&x_b.transpose() * &y_b));

        // 3. Calculate Means
        // row_mean() calculates the centroid (mean of rows) as a RowVector (1 x k)? Or Column Vector (k x 1)?
        // Previous error: `row_mean().transpose()` was (2, 1). So `row_mean()` is (1, 2).
        // (1, 2) is RowVector of means of columns (k=2).
        let xb_mean = x_b.row_mean();

        // 4. Unexplained Gap
        let diff = &beta_a - &beta_b;

        // Debugging shape mismatch
        if xb_mean.ncols() != diff.nrows() {
            anyhow::bail!(
                "Shape mismatch: xb_mean ({}, {}), diff ({}, {})",
                xb_mean.nrows(),
                xb_mean.ncols(),
                diff.nrows(),
                diff.ncols()
            );
        }

        // Use clone or reference to avoid move
        let unexplained_old = (xb_mean.clone() * &diff)[(0, 0)];

        // 5. Calculate M
        let inv_xtx_b = chol_b.inverse();
        let term1 = xb_mean * inv_xtx_b; // (1 x k)
        let m_row = term1 * x_b.transpose(); // (1 x N)

        // Convert to DVector (column vector).
        let m_mat = m_row.transpose();
        // Construct DVector from column slice.
        let m = DVector::from_column_slice(m_mat.as_slice());

        Ok((m, unexplained_old))
    }
}

impl ProblemDefinition for PayEquityProblem {
    fn define_variables(&self, problem: &mut good_lp::ProblemVariables) -> Result<VariableMap> {
        let (_, _, _, y_b, _) = self
            .builder
            .get_data_matrices()
            .map_err(|e| anyhow::anyhow!("Oaxaca Error: {}", e))?;

        let n = y_b.len();
        let mut map = VariableMap::new();

        for i in 0..n {
            let var = problem.add(variable().min(0));
            map.insert(format!("adj_{}", i), var);
        }

        Ok(map)
    }

    fn define_objective(&self, variables: &VariableMap) -> Expression {
        variables.values().sum()
    }

    fn define_constraints<T: SolverModel>(
        &self,
        variables: &VariableMap,
        problem: &mut T,
    ) -> Result<()> {
        let (m, unexplained_old) = self.calculate_coefficients()?;

        let mut lhs = Expression::from(0);
        let n = m.len();

        for i in 0..n {
            if let Some(var) = variables.get(&format!("adj_{}", i)) {
                lhs += *var * m[i];
            }
        }

        let rhs = unexplained_old - self.target_gap;
        problem.add_constraint(lhs.geq(rhs));

        Ok(())
    }
}
