use anyhow::Result;
use good_lp::{Expression, ResolutionError, Solution, SolverModel, Variable};
use polars::prelude::*;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum OptimizationError {
    #[error("Solver error: {0}")]
    SolverError(#[from] ResolutionError),
    #[error("Data error: {0}")]
    DataError(#[from] PolarsError),
    #[error("Invalid definition: {0}")]
    InvalidDefinition(String),
}

pub type VariableMap = HashMap<String, Variable>;

pub trait ProblemDefinition {
    /// Define variables and return a map for later reference.
    fn define_variables(&self, problem: &mut good_lp::ProblemVariables) -> Result<VariableMap>;

    /// Define the objective function.
    fn define_objective(&self, variables: &VariableMap) -> Expression;

    /// Define constraints and add them to the problem.
    fn define_constraints<T: SolverModel>(
        &self,
        variables: &VariableMap,
        problem: &mut T,
    ) -> Result<()>;
}

pub struct OptimizationEngine;

#[derive(Debug)]
pub struct OptimizationResult {
    pub solution: HashMap<String, f64>,
    pub objective_value: f64,
}

impl Default for OptimizationEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationEngine {
    pub fn new() -> Self {
        Self
    }

    pub fn solve<P: ProblemDefinition>(&self, problem_def: P) -> Result<OptimizationResult> {
        let mut vars = good_lp::variables!();
        let variable_map = problem_def.define_variables(&mut vars)?;

        let objective_expr = problem_def.define_objective(&variable_map);

        // Use Highs solver by default as requested
        // good_lp::solvers::highs refers to the function `highs` in `good_lp::solvers` module.
        // But importing `good_lp::solvers::highs` directly might fail if it's not re-exported properly or is a module.
        // Let's use `good_lp::default_solver` if highs fails, but user wants highs.
        // `good_lp::solvers::highs` is a function `pub fn highs<P: Problem>(problem: P) -> HighsSolver<P>`.
        // I need to make sure I am using it correctly.
        // Try `good_lp::solvers::highs` again but make sure I import it?
        // The error said "expected value, found module `good_lp::solvers::highs`".
        // This means `highs` is a module?
        // In `good_lp` 1.8, `solvers::highs` is a module, the function is `good_lp::solvers::highs::highs`.

        let mut model = vars
            .minimise(objective_expr)
            .using(good_lp::solvers::highs::highs);

        problem_def.define_constraints(&variable_map, &mut model)?;

        let solution = model.solve()?;

        let mut result_map = HashMap::new();
        for (name, var) in &variable_map {
            result_map.insert(name.clone(), solution.value(*var));
        }

        // Re-evaluate objective.
        let obj_expr = problem_def.define_objective(&variable_map);
        let obj_val = solution.eval(obj_expr);

        Ok(OptimizationResult {
            solution: result_map,
            objective_value: obj_val,
        })
    }
}
