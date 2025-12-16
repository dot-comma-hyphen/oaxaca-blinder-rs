use crate::engine::{ProblemDefinition, VariableMap};
use anyhow::Result;
use good_lp::{variable, Expression, SolverModel};
use polars::prelude::*;
use std::collections::HashMap;

pub struct WageScaleProblem {
    pub census: DataFrame,
    pub budget: f64,
    pub min_step_diff: f64,
    pub min_grade_diff: f64,
    pub grades: u32,
    pub steps: u32,
    pub min_wage: f64,
}

impl WageScaleProblem {
    pub fn new(census: DataFrame, budget: f64, grades: u32, steps: u32, min_wage: f64) -> Self {
        Self {
            census,
            budget,
            min_step_diff: 0.03,
            min_grade_diff: 0.10,
            grades,
            steps,
            min_wage,
        }
    }
}

impl ProblemDefinition for WageScaleProblem {
    fn define_variables(&self, problem: &mut good_lp::ProblemVariables) -> Result<VariableMap> {
        let mut map = VariableMap::new();

        for g in 1..=self.grades {
            for s in 1..=self.steps {
                // Enforce min_wage at variable definition level for efficiency
                let var = problem.add(variable().min(self.min_wage));
                map.insert(format!("step_{}_{}", g, s), var);
            }
        }

        Ok(map)
    }

    fn define_objective(&self, variables: &VariableMap) -> Expression {
        let mut total_cost = Expression::from(0);

        let g_col = self.census.column("grade");
        let s_col = self.census.column("step");

        if let (Ok(_g_s), Ok(_s_s)) = (g_col, s_col) {
            let df = self
                .census
                .clone()
                .lazy()
                .group_by([col("grade"), col("step")])
                .agg([len().alias("count")])
                .collect();

            if let Ok(counts) = df {
                let g_ca = counts.column("grade").unwrap().u32().unwrap();
                let s_ca = counts.column("step").unwrap().u32().unwrap();
                let c_ca = counts.column("count").unwrap().u32().unwrap();

                for i in 0..counts.height() {
                    let g = g_ca.get(i).unwrap();
                    let s = s_ca.get(i).unwrap();
                    let c = c_ca.get(i).unwrap() as f64;

                    if let Some(var) = variables.get(&format!("step_{}_{}", g, s)) {
                        total_cost += *var * c;
                    }
                }
            }
        }

        total_cost
    }

    fn define_constraints<T: SolverModel>(
        &self,
        variables: &VariableMap,
        problem: &mut T,
    ) -> Result<()> {
        // 1. Structure Constraints
        for g in 1..=self.grades {
            for s in 1..self.steps {
                let curr = variables[&format!("step_{}_{}", g, s)];
                let next = variables[&format!("step_{}_{}", g, s + 1)];
                problem.add_constraint((next - curr * (1.0 + self.min_step_diff)).geq(0));
            }
        }

        for g in 1..self.grades {
            let curr = variables[&format!("step_{}_{}", g, 1)];
            let next = variables[&format!("step_{}_{}", g + 1, 1)];
            problem.add_constraint((next - curr * (1.0 + self.min_grade_diff)).geq(0));
        }

        // 2. Budget Constraint
        let obj = self.define_objective(variables);
        problem.add_constraint(obj.leq(self.budget));

        // 3. No Pay Cuts (Lower Bound)
        let df = self
            .census
            .clone()
            .lazy()
            .group_by([col("grade"), col("step")])
            .agg([col("salary").max().alias("max_salary")])
            .collect()?;

        let g_ca = df.column("grade")?.u32()?;
        let s_ca = df.column("step")?.u32()?;
        let m_ca = df.column("max_salary")?.f64()?;

        for i in 0..df.height() {
            let g = g_ca.get(i).unwrap();
            let s = s_ca.get(i).unwrap();
            if let Some(max_sal) = m_ca.get(i) {
                if let Some(var) = variables.get(&format!("step_{}_{}", g, s)) {
                    // Ensure grid point is at least the max current salary in that cell
                    // Note: variable() already has min(self.min_wage).
                    // This adds an additional lower bound if max_sal > min_wage.
                    problem.add_constraint(Expression::from(*var).geq(max_sal));
                }
            }
        }

        Ok(())
    }
}
