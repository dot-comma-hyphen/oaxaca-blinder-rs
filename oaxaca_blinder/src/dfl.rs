use crate::OaxacaError;
use crate::math::logit::logit;
use crate::math::kde::{kde, silverman_bandwidth};
use polars::prelude::*;
use nalgebra::{DMatrix, DVector};
use serde::Serialize;

/// Holds the results of a DiNardo-Fortin-Lemieux (DFL) reweighting analysis.
#[derive(Debug, Serialize)]
pub struct DflResult {
    /// The grid points where the density was evaluated.
    pub grid: Vec<f64>,
    /// The density of the outcome for Group A (Advantaged).
    pub density_a: Vec<f64>,
    /// The density of the outcome for Group B (Disadvantaged/Reference).
    pub density_b: Vec<f64>,
    /// The counterfactual density of Group B if they had the characteristics of Group A.
    pub density_b_counterfactual: Vec<f64>,
}

/// Performs DFL Reweighting to estimate the counterfactual density of Group B.
///
/// # Arguments
///
/// * `df` - The DataFrame containing the data.
/// * `outcome` - The name of the outcome variable.
/// * `group` - The name of the group variable.
/// * `reference_group` - The value of the reference group (Group B).
/// * `predictors` - The list of predictor variables to use for the propensity score.
///
/// # Returns
///
/// A `Result` containing the `DflResult`.
pub fn run_dfl(
    df: &DataFrame,
    outcome: &str,
    group: &str,
    reference_group: &str,
    predictors: &[String],
) -> Result<DflResult, OaxacaError> {
    // 1. Prepare Data for Logit
    // Target: 1 if Group A, 0 if Group B
    let unique_groups = df.column(group)?.unique()?.sort(false, false);
    let group_b_name = reference_group;
    let group_a_name_temp = unique_groups.str()?.get(0).unwrap_or(reference_group);
    let group_a_name = if group_a_name_temp == group_b_name { unique_groups.str()?.get(1).unwrap_or("") } else { group_a_name_temp };
    
    let group_series = df.column(group)?;
    let target_vec: Vec<f64> = group_series.str()?
        .into_iter()
        .map(|opt_s| if opt_s.unwrap_or("") == group_a_name { 1.0 } else { 0.0 })
        .collect();
    let y = DVector::from_vec(target_vec);
    
    // Prepare X matrix (predictors + intercept)
    // Note: We need to handle categorical variables here too, but for simplicity let's assume numeric or pre-processed for now.
    // Ideally we reuse the `prepare_data` logic from OaxacaBuilder, but it's private.
    // For this implementation, let's assume numeric predictors for the MVP.
    // TODO: Support categorical predictors in DFL.
    
    let mut x_cols = Vec::new();
    let intercept = Series::new("intercept", vec![1.0; df.height()]);
    x_cols.push(intercept);
    
    for pred in predictors {
        let col = df.column(pred)?.cast(&DataType::Float64)?;
        x_cols.push(col);
    }
    
    let x_df = DataFrame::new(x_cols).map_err(OaxacaError::from)?;
    let x_matrix = x_df.to_ndarray::<Float64Type>(IndexOrder::Fortran)?;
    let x_vec: Vec<f64> = x_matrix.iter().copied().collect();
    let x = DMatrix::from_row_slice(x_df.height(), x_df.width(), &x_vec);
    
    // 2. Estimate Propensity Score P(A|X) using Logit
    let logit_res = logit(&y, &x, 100, 1e-6)?;
    let probs = logit_res.predicted_probs; // P(A|X)
    
    // 3. Calculate Weights
    // Weight for Group B to look like Group A:
    // psi(x) = (P(A|X) / (1 - P(A|X))) * (P(B) / P(A))
    
    let n = df.height() as f64;
    let n_a = df.filter(&df.column(group)?.equal(group_a_name)?)?.height() as f64;
    let n_b = df.filter(&df.column(group)?.equal(group_b_name)?)?.height() as f64;
    
    let p_a_marginal = n_a / n;
    let p_b_marginal = n_b / n;
    let ratio_marginal = p_b_marginal / p_a_marginal;
    
    let mut weights_counterfactual = Vec::new();
    let outcome_series = df.column(outcome)?.f64()?;
    let mut outcome_b = Vec::new();
    let mut outcome_a = Vec::new();
    
    for i in 0..df.height() {
        let is_group_b = y[i] == 0.0;
        let val = outcome_series.get(i).unwrap_or(0.0);
        
        if is_group_b {
            let p_x = probs[i];
            // Avoid division by zero
            let p_x = p_x.min(0.9999).max(0.0001);
            
            let weight = (p_x / (1.0 - p_x)) * ratio_marginal;
            weights_counterfactual.push(weight);
            outcome_b.push(val);
        } else {
            outcome_a.push(val);
        }
    }
    
    // 4. Compute KDEs
    // Define Grid
    let all_outcomes: Vec<f64> = outcome_series.into_no_null_iter().collect();
    let min_val = all_outcomes.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = all_outcomes.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let range = max_val - min_val;
    let grid_size = 100;
    let step = range / (grid_size as f64);
    let grid: Vec<f64> = (0..grid_size).map(|i| min_val + i as f64 * step).collect();
    
    let bandwidth_a = silverman_bandwidth(&outcome_a);
    let bandwidth_b = silverman_bandwidth(&outcome_b);
    // Use average bandwidth for counterfactual to be comparable? Or B's bandwidth?
    // Usually keep bandwidth consistent or re-estimate. Let's re-estimate or use B's.
    // Using B's bandwidth is safer for the counterfactual of B.
    
    let density_a = kde(&outcome_a, None, &grid, bandwidth_a);
    let density_b = kde(&outcome_b, None, &grid, bandwidth_b);
    let density_b_counterfactual = kde(&outcome_b, Some(&weights_counterfactual), &grid, bandwidth_b);
    
    Ok(DflResult {
        grid,
        density_a,
        density_b,
        density_b_counterfactual,
    })
}
