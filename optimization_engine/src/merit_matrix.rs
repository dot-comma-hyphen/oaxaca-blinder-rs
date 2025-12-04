use crate::engine::{ProblemDefinition, VariableMap};
use good_lp::{Variable, Expression, SolverModel, Constraint, IntoAffineExpression, variable};
use polars::prelude::*;
use anyhow::Result;

pub struct MeritMatrixProblem {
    pub census: DataFrame,
    pub merit_budget: f64,
}

impl MeritMatrixProblem {
    pub fn new(census: DataFrame, merit_budget: f64) -> Self {
        Self { census, merit_budget }
    }
}

impl ProblemDefinition for MeritMatrixProblem {
    fn define_variables(&self, problem: &mut good_lp::ProblemVariables) -> Result<VariableMap> {
        let n = self.census.height();
        let mut map = VariableMap::new();

        for i in 0..n {
            let var = problem.add(variable().min(0).max(0.10));
            map.insert(format!("increase_pct_{}", i), var);
        }

        Ok(map)
    }

    fn define_objective(&self, variables: &VariableMap) -> Expression {
        let ratings = self.census.column("performance_rating").expect("performance_rating column missing").f64().expect("must be float");

        let mut expr = Expression::from(0);

        for i in 0..self.census.height() {
            if let Some(r) = ratings.get(i) {
                if let Some(var) = variables.get(&format!("increase_pct_{}", i)) {
                    expr += *var * (-r);
                }
            }
        }

        expr
    }

    fn define_constraints<T: SolverModel>(&self, variables: &VariableMap, problem: &mut T) -> Result<()> {
        let salaries = self.census.column("salary").expect("salary column missing").f64().expect("must be float");

        let mut cost_expr = Expression::from(0);

        for i in 0..self.census.height() {
            if let Some(s) = salaries.get(i) {
                 if let Some(var) = variables.get(&format!("increase_pct_{}", i)) {
                     cost_expr += *var * s;
                 }
            }
        }

        problem.add_constraint(cost_expr.leq(self.merit_budget));

        Ok(())
    }
}
