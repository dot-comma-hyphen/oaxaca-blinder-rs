//! A Rust implementation of the Oaxaca-Blinder decomposition method.
//!
//! This library provides tools to decompose the mean difference in an outcome
//! variable between two groups into an "explained" part (due to differences
//! in observable characteristics) and an "unexplained" part (due to differences
//! in the returns to those characteristics).
//!
//! Currently, the library supports numerical predictors and calculates standard
//! errors using bootstrapping.
//!
//! # Example
//!
//! ```ignore
//! use polars::prelude::*;
//! use oaxaca_blinder::OaxacaBuilder;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let df = df!(
//!         "wage" => &[10.0, 12.0, 11.0, 13.0, 15.0, 20.0, 22.0, 21.0, 23.0, 25.0],
//!         "education" => &[12.0, 16.0, 14.0, 16.0, 18.0, 12.0, 16.0, 14.0, 16.0, 18.0],
//!         "gender" => &["F", "F", "F", "F", "F", "M", "M", "M", "M", "M"]
//!     )?;
//!
//!     let results = OaxacaBuilder::new(df, "wage", "gender", "F")
//!         .predictors(&["education"])
//!         .run()?;
//!
//!     results.summary();
//!     Ok(())
//! }
//! ```
//!
//! ### Quantile Regression Decomposition
//!
//! ```ignore
//! use polars::prelude::*;
//! use oaxaca_blinder::QuantileDecompositionBuilder;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let df = df!(
//!         "wage" => &[10.0, 12.0, 11.0, 13.0, 15.0, 20.0, 22.0, 21.0, 23.0, 25.0, 9.0, 18.0],
//!         "education" => &[12.0, 16.0, 14.0, 16.0, 18.0, 12.0, 16.0, 14.0, 16.0, 18.0, 10.0, 20.0],
//!         "gender" => &["F", "F", "F", "F", "F", "F", "M", "M", "M", "M", "M", "M"]
//!     )?;
//!
//!     let results = QuantileDecompositionBuilder::new(df, "wage", "gender", "F")
//!         .predictors(&["education"])
//!         .quantiles(&[0.25, 0.5, 0.75])
//!         .run()?;
//!
//!     results.summary();
//!     Ok(())
//! }
//! ```

use comfy_table::{Cell, Table};
use getset::Getters;
use nalgebra::{DMatrix, DVector};
use polars::prelude::*;
use serde::Serialize;
use std::collections::HashMap;
use std::fmt;

mod decomposition;
mod inference;
mod math;

pub use crate::decomposition::ReferenceCoefficients;
use crate::decomposition::{
    detailed_decomposition, three_fold_decomposition, two_fold_decomposition, BudgetAdjustment,
    DetailedComponent, ThreeFoldDecomposition, TwoFoldDecomposition,
};
use crate::inference::bootstrap_stats;
use crate::math::normalization::normalize_categorical_coefficients;
use crate::math::ols::ols;
pub mod quantile_decomposition;
use crate::math::rif::calculate_rif;
pub use crate::quantile_decomposition::QuantileDecompositionBuilder;
pub mod jmp;
pub use crate::jmp::decompose_changes;
pub mod dfl;
pub use crate::dfl::run_dfl;
pub mod formula;
pub mod heckman;
use crate::formula::Formula;
pub use crate::heckman::heckman_two_step;
pub mod akm;
pub use crate::akm::{AkmBuilder, AkmResult};
pub mod matching;
pub use crate::matching::engine::MatchingEngine;

#[cfg(feature = "python")]
pub mod python;

/// Error type for the `oaxaca_blinder` library.
#[derive(Debug)]
pub enum OaxacaError {
    /// Wraps a `PolarsError`.
    PolarsError(PolarsError),
    /// Occurs when a specified column name does not exist in the DataFrame.
    ColumnNotFound(String),
    /// Occurs when the grouping variable does not contain exactly two unique, non-null groups.
    InvalidGroupVariable(String),
    /// Occurs when there is an issue with linear algebra operations, such as a singular matrix.
    NalgebraError(String),
    /// Occurs when there is an issue with a diagnostic calculation.
    DiagnosticError(String),
}

impl From<PolarsError> for OaxacaError {
    fn from(err: PolarsError) -> Self {
        OaxacaError::PolarsError(err)
    }
}

impl fmt::Display for OaxacaError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OaxacaError::PolarsError(e) => write!(f, "Polars error: {}", e),
            OaxacaError::ColumnNotFound(s) => write!(f, "Column not found: {}", s),
            OaxacaError::InvalidGroupVariable(s) => write!(f, "Invalid group variable: {}", s),
            OaxacaError::NalgebraError(s) => write!(f, "Nalgebra error: {}", s),
            OaxacaError::DiagnosticError(s) => write!(f, "Diagnostic error: {}", s),
        }
    }
}

impl std::error::Error for OaxacaError {}

/// The main entry point for configuring and running an Oaxaca-Blinder decomposition.
///
/// This struct is created using a builder pattern.
#[derive(Debug, Clone)]
pub struct OaxacaBuilder {
    dataframe: DataFrame,
    outcome: String,
    predictors: Vec<String>,
    categorical_predictors: Vec<String>,
    group: String,
    reference_group: String,
    bootstrap_reps: usize,
    reference_coeffs: ReferenceCoefficients,
    normalization_vars: Vec<String>,
    weights_col: Option<String>,
    selection_outcome: Option<String>,
    selection_predictors: Vec<String>,
}

#[derive(Clone)]
struct SinglePassResult {
    three_fold: ThreeFoldDecomposition,
    two_fold: TwoFoldDecomposition,
    detailed_explained: Vec<DetailedComponent>,
    detailed_unexplained: Vec<DetailedComponent>,
    total_gap: f64,
    residuals_a: DVector<f64>,
    residuals_b: DVector<f64>,
    xa_mean: DVector<f64>,
    xb_mean: DVector<f64>,
    beta_star: DVector<f64>,
}

struct EstimationResult {
    beta_a: DVector<f64>,
    beta_b: DVector<f64>,
    xa_mean: DVector<f64>,
    xb_mean: DVector<f64>,
    predictor_names: Vec<String>,
    residuals_a: DVector<f64>,
    residuals_b: DVector<f64>,
    base_coeffs_a: HashMap<String, f64>,
    base_coeffs_b: HashMap<String, f64>,
}

struct EstimationContext<'a> {
    df_a: &'a DataFrame,
    df_b: &'a DataFrame,
    x_a: &'a DMatrix<f64>,
    y_a: &'a DVector<f64>,
    w_a: &'a Option<DVector<f64>>,
    x_b: &'a DMatrix<f64>,
    y_b: &'a DVector<f64>,
    w_b: &'a Option<DVector<f64>>,
    predictor_names: &'a [String],
    category_counts: &'a HashMap<String, usize>,
}

trait Estimator {
    fn estimate(&self, ctx: &EstimationContext) -> Result<EstimationResult, OaxacaError>;
}

struct OlsEstimator {
    normalization_vars: Vec<String>,
}

impl Estimator for OlsEstimator {
    fn estimate(&self, ctx: &EstimationContext) -> Result<EstimationResult, OaxacaError> {
        let mut ols_a = ols(ctx.y_a, ctx.x_a, ctx.w_a.as_ref())?;
        let mut ols_b = ols(ctx.y_b, ctx.x_b, ctx.w_b.as_ref())?;

        let calculate_mean = |x: &DMatrix<f64>, w: &Option<DVector<f64>>| -> DVector<f64> {
            if let Some(weights) = w {
                let total_weight = weights.sum();
                let mut means = DVector::zeros(x.ncols());
                for j in 0..x.ncols() {
                    let col = x.column(j);
                    means[j] = col.dot(weights) / total_weight;
                }
                means
            } else {
                x.row_mean().transpose()
            }
        };

        let xa_mean = calculate_mean(ctx.x_a, ctx.w_a);
        let xb_mean = calculate_mean(ctx.x_b, ctx.w_b);

        let mut base_coeffs_a = HashMap::new();
        let mut base_coeffs_b = HashMap::new();

        if !self.normalization_vars.is_empty() {
            base_coeffs_a = normalize_categorical_coefficients(
                &mut ols_a,
                ctx.predictor_names,
                &self.normalization_vars,
                &xa_mean,
                ctx.category_counts,
            );
            base_coeffs_b = normalize_categorical_coefficients(
                &mut ols_b,
                ctx.predictor_names,
                &self.normalization_vars,
                &xb_mean,
                ctx.category_counts,
            );
        }

        Ok(EstimationResult {
            beta_a: ols_a.coefficients,
            beta_b: ols_b.coefficients,
            xa_mean,
            xb_mean,
            predictor_names: ctx.predictor_names.to_vec(),
            residuals_a: ols_a.residuals,
            residuals_b: ols_b.residuals,
            base_coeffs_a,
            base_coeffs_b,
        })
    }
}

struct HeckmanEstimator {
    selection_outcome: String,
    selection_predictors: Vec<String>,
}

impl Estimator for HeckmanEstimator {
    fn estimate(&self, ctx: &EstimationContext) -> Result<EstimationResult, OaxacaError> {
        let prepare_selection = |df_group: &DataFrame| -> Result<
            (DMatrix<f64>, DVector<f64>, DMatrix<f64>),
            OaxacaError,
        > {
            let y_sel_series = df_group.column(&self.selection_outcome)?.f64()?;
            let y_sel_vec: Vec<f64> = y_sel_series
                .into_iter()
                .map(|opt| opt.expect("Selection outcome should be clean"))
                .collect();
            let y_sel = DVector::from_vec(y_sel_vec);

            let mut x_sel_df = df_group.select(&self.selection_predictors)?;
            let intercept = Series::new("intercept".into(), vec![1.0; df_group.height()]);
            x_sel_df.with_column(intercept)?;
            let mut cols = vec!["intercept".to_string()];
            cols.extend(self.selection_predictors.clone());
            let x_sel_df = x_sel_df.select(&cols)?;

            let x_sel_mat = x_sel_df.to_ndarray::<Float64Type>(IndexOrder::Fortran)?;
            let x_sel_vec: Vec<f64> = x_sel_mat.iter().copied().collect();
            let x_sel = DMatrix::from_row_slice(x_sel_df.height(), x_sel_df.width(), &x_sel_vec);

            let mask = df_group
                .column(&self.selection_outcome)?
                .as_materialized_series()
                .equal(1)?;
            let df_subset = df_group.filter(&mask)?;

            let mut x_sel_sub_df = df_subset.select(&self.selection_predictors)?;
            x_sel_sub_df.with_column(Series::new(
                "intercept".into(),
                vec![1.0; df_subset.height()],
            ))?;
            let x_sel_sub_df = x_sel_sub_df.select(&cols)?;

            let x_sel_sub_mat = x_sel_sub_df.to_ndarray::<Float64Type>(IndexOrder::Fortran)?;
            let x_sel_sub_vec: Vec<f64> = x_sel_sub_mat.iter().copied().collect();
            let x_sel_sub = DMatrix::from_row_slice(
                x_sel_sub_df.height(),
                x_sel_sub_df.width(),
                &x_sel_sub_vec,
            );

            Ok((x_sel, y_sel, x_sel_sub))
        };

        let (x_sel_a, y_sel_a, x_sel_sub_a) = prepare_selection(ctx.df_a)?;
        let (x_sel_b, y_sel_b, x_sel_sub_b) = prepare_selection(ctx.df_b)?;

        let filter_outcome_data = |x: &DMatrix<f64>,
                                   y: &DVector<f64>,
                                   df: &DataFrame|
         -> Result<(DMatrix<f64>, DVector<f64>), OaxacaError> {
            let mask = df
                .column(&self.selection_outcome)?
                .as_materialized_series()
                .equal(1)?;
            let mut rows = Vec::new();
            let mut y_vals = Vec::new();
            for i in 0..df.height() {
                if mask.get(i) == Some(true) {
                    rows.push(x.row(i).into_owned());
                    y_vals.push(y[i]);
                }
            }
            if rows.is_empty() {
                return Err(OaxacaError::InvalidGroupVariable(
                    "No observed outcomes in group".to_string(),
                ));
            }
            let x_filtered = DMatrix::from_rows(&rows);
            let y_filtered = DVector::from_vec(y_vals);
            Ok((x_filtered, y_filtered))
        };

        let (x_a_filt, y_a_filt) = filter_outcome_data(ctx.x_a, ctx.y_a, ctx.df_a)?;
        let (x_b_filt, y_b_filt) = filter_outcome_data(ctx.x_b, ctx.y_b, ctx.df_b)?;

        let res_a = heckman_two_step(&y_sel_a, &x_sel_a, &y_a_filt, &x_a_filt, &x_sel_sub_a)?;
        let res_b = heckman_two_step(&y_sel_b, &x_sel_b, &y_b_filt, &x_b_filt, &x_sel_sub_b)?;

        let mut beta_a = res_a.outcome_coeffs;
        let mut beta_b = res_b.outcome_coeffs;

        let k = beta_a.len();
        beta_a = beta_a.insert_row(k, res_a.imr_coeff);
        beta_b = beta_b.insert_row(k, res_b.imr_coeff);

        let mut xa_mean = x_a_filt.row_mean().transpose();
        let mut xb_mean = x_b_filt.row_mean().transpose();

        let imr_mean_a = res_a.imr.mean();
        let imr_mean_b = res_b.imr.mean();

        let k_x = xa_mean.len();
        xa_mean = xa_mean.insert_row(k_x, imr_mean_a);
        xb_mean = xb_mean.insert_row(k_x, imr_mean_b);

        let mut final_names = ctx.predictor_names.to_vec();
        final_names.push("IMR".to_string());

        let residuals_a = DVector::zeros(y_a_filt.len());
        let residuals_b = DVector::zeros(y_b_filt.len());

        Ok(EstimationResult {
            beta_a,
            beta_b,
            xa_mean,
            xb_mean,
            predictor_names: final_names,
            residuals_a,
            residuals_b,
            base_coeffs_a: HashMap::new(),
            base_coeffs_b: HashMap::new(),
        })
    }
}
impl OaxacaBuilder {
    /// Creates a new `OaxacaBuilder`.
    ///
    /// # Arguments
    ///
    /// * `dataframe` - A `polars::DataFrame` containing the data for the analysis.
    /// * `outcome` - The name of the column representing the outcome variable (e.g., "wage").
    /// * `group` - The name of the column that divides the data into two groups (e.g., "gender").
    /// * `reference_group` - The value within the `group` column that identifies the reference group (the lower-outcome group, or Group B).
    pub fn new(dataframe: DataFrame, outcome: &str, group: &str, reference_group: &str) -> Self {
        Self {
            dataframe,
            outcome: outcome.to_string(),
            predictors: Vec::new(),
            categorical_predictors: Vec::new(),
            group: group.to_string(),
            reference_group: reference_group.to_string(),
            bootstrap_reps: 100,
            reference_coeffs: ReferenceCoefficients::GroupA,
            normalization_vars: Vec::new(),
            weights_col: None,
            selection_outcome: None,
            selection_predictors: Vec::new(),
        }
    }

    /// Creates a new `OaxacaBuilder` using an R-style formula.
    ///
    /// # Arguments
    ///
    /// * `dataframe` - The Polars DataFrame containing the data.
    /// * `formula` - A string representing the model formula (e.g., "wage ~ education + experience + C(sector)").
    /// * `group` - The name of the column defining the two groups.
    /// * `reference_group` - The value in the `group` column representing the reference group (Group B).
    pub fn from_formula(
        dataframe: DataFrame,
        formula: &str,
        group: &str,
        reference_group: &str,
    ) -> Result<Self, OaxacaError> {
        let parsed_formula = Formula::parse(formula)?;
        Ok(Self {
            dataframe,
            outcome: parsed_formula.outcome,
            predictors: parsed_formula.predictors,
            categorical_predictors: parsed_formula.categorical_predictors,
            group: group.to_string(),
            reference_group: reference_group.to_string(),
            bootstrap_reps: 100,
            reference_coeffs: ReferenceCoefficients::GroupA,
            normalization_vars: Vec::new(),
            weights_col: None,
            selection_outcome: None,
            selection_predictors: Vec::new(),
        })
    }

    /// Sets the reference coefficients for the decomposition.
    ///
    /// The default is `ReferenceCoefficients::GroupB`.
    pub fn reference_coefficients(&mut self, reference: ReferenceCoefficients) -> &mut Self {
        self.reference_coeffs = reference;
        self
    }

    /// Sets the predictor variables for the model.
    ///
    /// # Arguments
    ///
    /// * `predictors` - A slice of strings representing the column names of the predictor variables.
    pub fn predictors(&mut self, predictors: &[&str]) -> &mut Self {
        self.predictors = predictors.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Sets the categorical predictor variables for the model.
    ///
    /// # Arguments
    ///
    /// * `predictors` - A slice of strings representing the column names of the categorical predictor variables.
    pub fn categorical_predictors(&mut self, predictors: &[&str]) -> &mut Self {
        self.categorical_predictors = predictors.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Sets the number of bootstrap replications for standard error calculation.
    ///
    /// # Arguments
    ///
    /// * `reps` - The number of bootstrap samples to generate. Defaults to 100.
    pub fn bootstrap_reps(&mut self, reps: usize) -> &mut Self {
        self.bootstrap_reps = reps;
        self
    }

    /// Sets the categorical variables for which to apply coefficient normalization.
    ///
    /// # Arguments
    ///
    /// * `vars` - A slice of strings representing the column names of the categorical variables to normalize.
    pub fn normalize(&mut self, vars: &[&str]) -> &mut Self {
        self.normalization_vars = vars.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Sets the column name for sample weights.
    ///
    /// # Arguments
    ///
    /// * `weights` - The name of the column containing sample weights.
    pub fn weights(&mut self, weights: &str) -> &mut Self {
        self.weights_col = Some(weights.to_string());
        self
    }

    /// Configures the Heckman selection model.
    ///
    /// # Arguments
    ///
    /// * `outcome` - The binary selection variable (e.g., "employed").
    /// * `predictors` - The predictors for the selection equation (should include exclusion restriction).
    pub fn heckman_selection(&mut self, outcome: &str, predictors: &[&str]) -> &mut Self {
        self.selection_outcome = Some(outcome.to_string());
        self.selection_predictors = predictors.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Exposes the internal data matrices for advanced usage (e.g., optimization).
    /// This method prepares the data (creating dummies, etc.) and returns the matrices for Group A and Group B.
    /// Returns: (X_A, y_A, X_B, y_B, predictor_names)
    pub fn get_data_matrices(
        &self,
    ) -> Result<
        (
            DMatrix<f64>,
            DVector<f64>,
            DMatrix<f64>,
            DVector<f64>,
            Vec<String>,
        ),
        OaxacaError,
    > {
        let df_dirty = self.dataframe.clone();
        let mut df = self.clean_dataframe(&df_dirty)?;

        let mut all_dummy_names = Vec::new();
        // let mut category_counts = std::collections::HashMap::new();

        if !self.categorical_predictors.is_empty() {
            for cat_pred in &self.categorical_predictors {
                let series = df.column(cat_pred)?;
                let (dummies, _, _) =
                    self.create_dummies_manual(series.as_materialized_series())?;
                // category_counts.insert(cat_pred.clone(), m);
                for s in dummies.get_columns() {
                    all_dummy_names.push(s.name().to_string());
                }
                df = df.hstack(dummies.get_columns())?;
            }
        }

        let unique_groups = self
            .dataframe
            .column(&self.group)?
            .unique()?
            .sort(SortOptions {
                descending: false,
                nulls_last: false,
                ..Default::default()
            })?;
        if unique_groups.len() < 2 {
            return Err(OaxacaError::InvalidGroupVariable(
                "Not enough groups for comparison".to_string(),
            ));
        }

        let group_b_name = self.reference_group.as_str();
        let group_a_name_temp = unique_groups
            .str()?
            .get(0)
            .unwrap_or(self.reference_group.as_str());
        let group_a_name = if group_a_name_temp == group_b_name {
            unique_groups.str()?.get(1).unwrap_or("")
        } else {
            group_a_name_temp
        };

        let df_a = df.filter(
            &df.column(&self.group)?
                .as_materialized_series()
                .equal(group_a_name)?,
        )?;
        let df_b = df.filter(
            &df.column(&self.group)?
                .as_materialized_series()
                .equal(group_b_name)?,
        )?;

        let (x_a, y_a, _, predictor_names) = self.prepare_data(&df_a, &all_dummy_names, &[])?;
        let (x_b, y_b, _, _) = self.prepare_data(&df_b, &all_dummy_names, &[])?;

        Ok((x_a, y_a, x_b, y_b, predictor_names))
    }

    fn prepare_data(
        &self,
        df: &DataFrame,
        all_dummy_names: &[String],
        extra_predictors: &[String],
    ) -> Result<
        (
            DMatrix<f64>,
            DVector<f64>,
            Option<DVector<f64>>,
            Vec<String>,
        ),
        OaxacaError,
    > {
        let y_series = df.column(&self.outcome)?.f64()?;
        // Safe to unwrap because we ran clean_dataframe
        let y_vec: Vec<f64> = y_series
            .into_iter()
            .map(|opt| opt.expect("Null values should have been dropped"))
            .collect();
        let y = DVector::from_vec(y_vec);

        let mut current_predictors = self.predictors.clone();
        current_predictors.extend_from_slice(extra_predictors);

        let mut final_predictors: Vec<String> = vec!["intercept".to_string()];
        final_predictors.extend_from_slice(&current_predictors);
        final_predictors.extend_from_slice(all_dummy_names);

        let mut x_df = df.select(&current_predictors)?;
        let intercept_series = Series::new("intercept".into(), vec![1.0; df.height()]);
        x_df.with_column(intercept_series)?;

        for name in all_dummy_names {
            if df
                .get_column_names()
                .iter()
                .any(|s| s.as_str() == name.as_str())
            {
                x_df.with_column(df.column(name)?.clone())?;
            } else {
                let zero_series = Series::new(name.into(), vec![0.0; df.height()]);
                x_df.with_column(zero_series)?;
            }
        }

        let x_df_selected = x_df.select(&final_predictors)?;
        let x_matrix = x_df_selected.to_ndarray::<Float64Type>(IndexOrder::Fortran)?;
        let x_vec: Vec<f64> = x_matrix.iter().copied().collect();
        let final_names = x_df_selected
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();

        let weights = if let Some(w_col) = &self.weights_col {
            let w_series = df.column(w_col)?.f64()?;
            let w_vec: Vec<f64> = w_series
                .into_iter()
                .map(|opt| opt.expect("Null weights should have been dropped"))
                .collect();
            Some(DVector::from_vec(w_vec))
        } else {
            None
        };

        Ok((
            DMatrix::from_row_slice(x_df_selected.height(), x_df_selected.width(), &x_vec),
            y,
            weights,
            final_names,
        ))
    }

    fn create_dummies_manual(
        &self,
        series: &Series,
    ) -> Result<(DataFrame, usize, String), OaxacaError> {
        let unique_vals = series.unique()?.sort(SortOptions {
            descending: false,
            nulls_last: false,
            ..Default::default()
        })?;
        let m = unique_vals.len();
        let mut dummy_vars: Vec<Series> = Vec::new();

        let reference_val = if let Some(s) = unique_vals.str()?.get(0) {
            s
        } else {
            return Err(OaxacaError::InvalidGroupVariable(format!(
                "Could not get reference category for {}",
                series.name()
            )));
        };
        let reference_name = format!("{}_{}", series.name(), reference_val);

        for val in unique_vals.str()?.into_iter().flatten().skip(1) {
            // Skip the first category as the reference
            let dummy_name = format!("{}_{}", series.name(), val);
            let ca = series.equal(val)?;
            let mut dummy_series = ca.into_series();
            dummy_series = dummy_series.cast(&DataType::Float64)?;
            dummy_series.rename(dummy_name.as_str().into());
            dummy_vars.push(dummy_series);
        }

        Ok((
            DataFrame::new(dummy_vars.into_iter().map(Column::Series).collect())
                .map_err(OaxacaError::from)?,
            m,
            reference_name,
        ))
    }

    fn run_single_pass(
        &self,
        df: &DataFrame,
        all_dummy_names: &[String],
        category_counts: &std::collections::HashMap<String, usize>,
        base_categories: &std::collections::HashMap<String, String>,
    ) -> Result<SinglePassResult, OaxacaError> {
        let unique_groups = df.column(&self.group)?.unique()?.sort(SortOptions {
            descending: false,
            nulls_last: false,
            ..Default::default()
        })?;
        if unique_groups.len() < 2 {
            return Err(OaxacaError::InvalidGroupVariable(
                "Not enough groups for comparison".to_string(),
            ));
        }

        let group_b_name = self.reference_group.as_str();
        let group_a_name = unique_groups
            .str()?
            .get(0)
            .unwrap_or(self.reference_group.as_str());
        let group_a_name = if group_a_name == group_b_name {
            unique_groups.str()?.get(1).unwrap_or("")
        } else {
            group_a_name
        };

        let df_a = df.filter(
            &df.column(&self.group)?
                .as_materialized_series()
                .equal(group_a_name)?,
        )?;
        let df_b = df.filter(
            &df.column(&self.group)?
                .as_materialized_series()
                .equal(group_b_name)?,
        )?;
        if df_a.height() == 0 || df_b.height() == 0 {
            return Err(OaxacaError::InvalidGroupVariable(
                "One group has no data".to_string(),
            ));
        }

        let (x_a, y_a, w_a, predictor_names) = self.prepare_data(&df_a, all_dummy_names, &[])?;
        let (x_b, y_b, w_b, _) = self.prepare_data(&df_b, all_dummy_names, &[])?;

        let ctx = EstimationContext {
            df_a: &df_a,
            df_b: &df_b,
            x_a: &x_a,
            y_a: &y_a,
            w_a: &w_a,
            x_b: &x_b,
            y_b: &y_b,
            w_b: &w_b,
            predictor_names: &predictor_names,
            category_counts,
        };

        let estimator: Box<dyn Estimator> = if let Some(sel_outcome) = &self.selection_outcome {
            Box::new(HeckmanEstimator {
                selection_outcome: sel_outcome.clone(),
                selection_predictors: self.selection_predictors.clone(),
            })
        } else {
            Box::new(OlsEstimator {
                normalization_vars: self.normalization_vars.clone(),
            })
        };

        let result = estimator.estimate(&ctx)?;

        // ... Calculate beta_star and decompositions ...
        let beta_a = &result.beta_a;
        let beta_b = &result.beta_b;
        let xa_mean = result.xa_mean;
        let xb_mean = result.xb_mean;
        let final_predictor_names = result.predictor_names;
        let residuals_a = result.residuals_a;
        let residuals_b = result.residuals_b;
        let base_coeffs_a = result.base_coeffs_a;
        let base_coeffs_b = result.base_coeffs_b;

        let mut base_coeffs_star = std::collections::HashMap::new();
        let beta_star_owned: DVector<f64>;
        let beta_star: &DVector<f64> = match self.reference_coeffs {
            ReferenceCoefficients::GroupA => {
                base_coeffs_star = base_coeffs_a.clone();
                beta_a
            }
            ReferenceCoefficients::GroupB => {
                base_coeffs_star = base_coeffs_b.clone();
                beta_b
            }
            ReferenceCoefficients::Pooled | ReferenceCoefficients::Neumark => {
                let mut df_pooled = df_a.vstack(&df_b)?;
                let group_indicator = Series::new(
                    "group_indicator".into(),
                    df_pooled
                        .column(&self.group)?
                        .as_materialized_series()
                        .equal(group_a_name)?
                        .into_series()
                        .cast(&DataType::Float64)?,
                );
                df_pooled.with_column(group_indicator)?;

                let (x_pooled, y_pooled, w_pooled, pooled_predictor_names) = self.prepare_data(
                    &df_pooled,
                    all_dummy_names,
                    &["group_indicator".to_string()],
                )?;

                let mut ols_pooled = ols(&y_pooled, &x_pooled, w_pooled.as_ref())?;

                if !self.normalization_vars.is_empty() {
                    let n_a = df_a.height() as f64;
                    let n_b = df_b.height() as f64;
                    let x_pool_mean = (xa_mean.clone() * n_a + xb_mean.clone() * n_b) / (n_a + n_b);
                    base_coeffs_star = normalize_categorical_coefficients(
                        &mut ols_pooled,
                        &pooled_predictor_names,
                        &self.normalization_vars,
                        &x_pool_mean,
                        category_counts,
                    );
                }
                let group_indicator_idx = pooled_predictor_names
                    .iter()
                    .position(|r| r == "group_indicator")
                    .ok_or_else(|| {
                        OaxacaError::NalgebraError(
                            "group_indicator not found in pooled model predictors".to_string(),
                        )
                    })?;
                beta_star_owned = ols_pooled.coefficients.remove_row(group_indicator_idx);
                &beta_star_owned
            }
            ReferenceCoefficients::Weighted | ReferenceCoefficients::Cotton => {
                let n_a = if let Some(w) = &w_a {
                    w.sum()
                } else {
                    df_a.height() as f64
                };
                let n_b = if let Some(w) = &w_b {
                    w.sum()
                } else {
                    df_b.height() as f64
                };
                let total_n = n_a + n_b;
                if total_n == 0.0 {
                    return Err(OaxacaError::InvalidGroupVariable(
                        "No data in groups for weighted coefficients.".to_string(),
                    ));
                }
                let weight_a = n_a / total_n;
                let weight_b = 1.0 - weight_a;
                if !self.normalization_vars.is_empty() {
                    for var in &self.normalization_vars {
                        let coeff_a = base_coeffs_a.get(var).unwrap_or(&0.0);
                        let coeff_b = base_coeffs_b.get(var).unwrap_or(&0.0);
                        base_coeffs_star
                            .insert(var.clone(), coeff_a * weight_a + coeff_b * weight_b);
                    }
                }
                beta_star_owned = beta_a * weight_a + beta_b * weight_b;
                &beta_star_owned
            }
        };

        let three_fold = three_fold_decomposition(&xa_mean, &xb_mean, beta_a, beta_b);
        let mut two_fold = two_fold_decomposition(&xa_mean, &xb_mean, beta_a, beta_b, beta_star);
        let (mut detailed_explained, mut detailed_unexplained) = detailed_decomposition(
            &xa_mean,
            &xb_mean,
            beta_a,
            beta_b,
            beta_star,
            &final_predictor_names,
        );

        if !self.normalization_vars.is_empty() && self.selection_outcome.is_none() {
            for var in &self.normalization_vars {
                let base_dummy_name = if let Some(name) = base_categories.get(var) {
                    name
                } else {
                    continue;
                };

                let dummy_indices: Vec<usize> = final_predictor_names
                    .iter()
                    .enumerate()
                    .filter(|(_, name)| name.starts_with(&format!("{}_", var)))
                    .map(|(i, _)| i)
                    .collect();

                let xa_mean_base = 1.0 - dummy_indices.iter().map(|&i| xa_mean[i]).sum::<f64>();
                let xb_mean_base = 1.0 - dummy_indices.iter().map(|&i| xb_mean[i]).sum::<f64>();

                let beta_a_base = base_coeffs_a.get(var).cloned().unwrap_or(0.0);
                let beta_b_base = base_coeffs_b.get(var).cloned().unwrap_or(0.0);
                let beta_star_base = base_coeffs_star.get(var).cloned().unwrap_or(0.0);

                let contribution_unexplained = xa_mean_base * (beta_a_base - beta_star_base)
                    + xb_mean_base * (beta_star_base - beta_b_base);

                let contribution_explained = (xa_mean_base - xb_mean_base) * beta_star_base;

                detailed_unexplained.push(DetailedComponent {
                    variable_name: base_dummy_name.clone(),
                    contribution: contribution_unexplained,
                });

                detailed_explained.push(DetailedComponent {
                    variable_name: base_dummy_name.clone(),
                    contribution: contribution_explained,
                });

                two_fold.explained += contribution_explained;
                two_fold.unexplained += contribution_unexplained;
            }
        }

        let total_gap = if let Some(w) = &w_a {
            y_a.dot(w) / w.sum()
        } else {
            y_a.mean()
        } - if let Some(w) = &w_b {
            y_b.dot(w) / w.sum()
        } else {
            y_b.mean()
        };

        Ok(SinglePassResult {
            three_fold,
            two_fold,
            detailed_explained,
            detailed_unexplained,
            total_gap,
            residuals_a,
            residuals_b,
            xa_mean: xa_mean.clone(),
            xb_mean: xb_mean.clone(),
            beta_star: beta_star.clone(),
        })
    }

    /// Performs a RIF-Regression decomposition for a specific quantile.
    ///
    /// This method transforms the outcome variable into its Recentered Influence Function (RIF)
    /// for each group separately and then performs the standard Oaxaca-Blinder decomposition
    /// on the transformed variable. This allows for decomposing the difference in quantiles
    /// (e.g., the 90th percentile gap) into explained and unexplained components.
    ///
    /// # Arguments
    ///
    /// * `quantile` - The target quantile (e.g., 0.9 for the 90th percentile).
    pub fn decompose_quantile(&self, quantile: f64) -> Result<OaxacaResults, OaxacaError> {
        // 1. Clean data first to ensure quantiles are calculated on non-null data
        let df_dirty = self.dataframe.clone();
        let df = self.clean_dataframe(&df_dirty)?;

        // 2. Split data into groups
        let unique_groups = df.column(&self.group)?.unique()?.sort(SortOptions {
            descending: false,
            nulls_last: false,
            ..Default::default()
        })?;
        if unique_groups.len() < 2 {
            return Err(OaxacaError::InvalidGroupVariable(
                "Not enough groups".to_string(),
            ));
        }

        let group_b_name = self.reference_group.as_str();
        let group_a_name_temp = unique_groups
            .str()?
            .get(0)
            .unwrap_or(self.reference_group.as_str());
        let group_a_name = if group_a_name_temp == group_b_name {
            unique_groups.str()?.get(1).unwrap_or("")
        } else {
            group_a_name_temp
        };

        let df_a = df.filter(
            &df.column(&self.group)?
                .as_materialized_series()
                .equal(group_a_name)?,
        )?;
        let df_b = df.filter(
            &df.column(&self.group)?
                .as_materialized_series()
                .equal(group_b_name)?,
        )?;

        // 3. Calculate RIF for each group
        let rif_a = calculate_rif(
            df_a.column(&self.outcome)?.as_materialized_series(),
            quantile,
        )
        .map_err(OaxacaError::PolarsError)?;
        let rif_b = calculate_rif(
            df_b.column(&self.outcome)?.as_materialized_series(),
            quantile,
        )
        .map_err(OaxacaError::PolarsError)?;

        // 4. Replace outcome with RIF
        let mut df_a_mod = df_a.clone();
        df_a_mod.with_column(rif_a)?;

        let mut df_b_mod = df_b.clone();
        df_b_mod.with_column(rif_b)?;

        // 5. Combine back
        let df_mod = df_a_mod.vstack(&df_b_mod)?;

        // 6. Create new builder and run
        let mut builder =
            OaxacaBuilder::new(df_mod, &self.outcome, &self.group, &self.reference_group);
        builder
            .predictors(
                &self
                    .predictors
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>(),
            )
            .categorical_predictors(
                &self
                    .categorical_predictors
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>(),
            )
            .bootstrap_reps(self.bootstrap_reps)
            .reference_coefficients(self.reference_coeffs.clone())
            .normalize(
                &self
                    .normalization_vars
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>(),
            );

        if let Some(w) = &self.weights_col {
            builder.weights(w);
        }

        builder.run()
    }

    /// Helper to drop rows with missing values in relevant columns.
    fn clean_dataframe(&self, df: &DataFrame) -> Result<DataFrame, OaxacaError> {
        let mut cols = vec![self.outcome.clone(), self.group.clone()];
        cols.extend(self.predictors.clone());
        cols.extend(self.categorical_predictors.clone());

        if let Some(w) = &self.weights_col {
            cols.push(w.to_string());
        }
        if let Some(sel_out) = &self.selection_outcome {
            cols.push(sel_out.to_string());
        }
        cols.extend(self.selection_predictors.clone());

        // Ensure all columns exist before trying to drop nulls on them
        for c in &cols {
            if df.column(c).is_err() {
                return Err(OaxacaError::ColumnNotFound(c.clone()));
            }
        }

        let clean_df = df
            .drop_nulls(Some(&cols))
            .map_err(OaxacaError::PolarsError)?;
        Ok(clean_df)
    }

    /// Executes the Oaxaca-Blinder decomposition.
    pub fn run(&self) -> Result<OaxacaResults, OaxacaError> {
        let df_dirty = self.dataframe.clone();
        let mut df = self.clean_dataframe(&df_dirty)?;

        let mut all_dummy_names = Vec::new();
        let mut category_counts = std::collections::HashMap::new();
        let mut base_categories = std::collections::HashMap::new();
        if !self.categorical_predictors.is_empty() {
            for cat_pred in &self.categorical_predictors {
                let series = df.column(cat_pred)?;
                let (dummies, m, base_name) =
                    self.create_dummies_manual(series.as_materialized_series())?;
                category_counts.insert(cat_pred.clone(), m);
                base_categories.insert(cat_pred.clone(), base_name);
                for s in dummies.get_columns() {
                    all_dummy_names.push(s.name().to_string());
                }
                df = df.hstack(dummies.get_columns())?;
            }
        }

        let unique_groups = self
            .dataframe
            .column(&self.group)?
            .unique()?
            .sort(SortOptions {
                descending: false,
                nulls_last: false,
                ..Default::default()
            })?;
        let group_b_name = self.reference_group.as_str();
        let group_a_name_temp = unique_groups
            .str()?
            .get(0)
            .unwrap_or(self.reference_group.as_str());
        let group_a_name = if group_a_name_temp == group_b_name {
            unique_groups.str()?.get(1).unwrap_or("")
        } else {
            group_a_name_temp
        };

        use rayon::prelude::*;

        let point_estimates =
            self.run_single_pass(&df, &all_dummy_names, &category_counts, &base_categories)?;

        let group_a_name_owned = group_a_name.to_string();
        let group_b_name_owned = group_b_name.to_string();

        let bootstrap_results: Vec<SinglePassResult> = (0..self.bootstrap_reps)
            .into_par_iter()
            .filter_map(|_| {
                // Stratified sampling: Sample from Group A and Group B separately
                let df_a = df
                    .filter(
                        &df.column(&self.group)
                            .ok()?
                            .as_materialized_series()
                            .equal(group_a_name_owned.as_str())
                            .ok()?,
                    )
                    .ok()?;
                let df_b = df
                    .filter(
                        &df.column(&self.group)
                            .ok()?
                            .as_materialized_series()
                            .equal(group_b_name_owned.as_str())
                            .ok()?,
                    )
                    .ok()?;

                let sample_a = df_a
                    .sample_n_literal(df_a.height(), true, false, None)
                    .ok()?;
                let sample_b = df_b
                    .sample_n_literal(df_b.height(), true, false, None)
                    .ok()?;

                let sample_df = sample_a.vstack(&sample_b).ok()?;

                self.run_single_pass(
                    &sample_df,
                    &all_dummy_names,
                    &category_counts,
                    &base_categories,
                )
                .ok()
            })
            .collect();

        let successful_bootstraps = bootstrap_results.len();
        if successful_bootstraps < self.bootstrap_reps {
            eprintln!(
                "Warning: {} out of {} bootstrap replications failed and were discarded. The analysis is based on {} successful replications.",
                self.bootstrap_reps - successful_bootstraps, self.bootstrap_reps, successful_bootstraps
            );
        }

        let process_component = |name: &str, point: f64, estimates: Vec<f64>| {
            let (std_err, p_value, (ci_lower, ci_upper)) = bootstrap_stats(&estimates, point);
            ComponentResult {
                name: name.to_string(),
                estimate: point,
                std_err,
                p_value,
                ci_lower,
                ci_upper,
            }
        };

        let two_fold_agg = vec![
            process_component(
                "explained",
                point_estimates.two_fold.explained,
                bootstrap_results
                    .iter()
                    .map(|r| r.two_fold.explained)
                    .collect(),
            ),
            process_component(
                "unexplained",
                point_estimates.two_fold.unexplained,
                bootstrap_results
                    .iter()
                    .map(|r| r.two_fold.unexplained)
                    .collect(),
            ),
        ];
        let three_fold_agg = vec![
            process_component(
                "endowments",
                point_estimates.three_fold.endowments,
                bootstrap_results
                    .iter()
                    .map(|r| r.three_fold.endowments)
                    .collect(),
            ),
            process_component(
                "coefficients",
                point_estimates.three_fold.coefficients,
                bootstrap_results
                    .iter()
                    .map(|r| r.three_fold.coefficients)
                    .collect(),
            ),
            process_component(
                "interaction",
                point_estimates.three_fold.interaction,
                bootstrap_results
                    .iter()
                    .map(|r| r.three_fold.interaction)
                    .collect(),
            ),
        ];

        let detailed_explained = self.process_detailed_components(
            &point_estimates.detailed_explained,
            &bootstrap_results,
            |r| &r.detailed_explained,
            &process_component,
        );
        let detailed_unexplained = self.process_detailed_components(
            &point_estimates.detailed_unexplained,
            &bootstrap_results,
            |r| &r.detailed_unexplained,
            &process_component,
        );

        Ok(OaxacaResults {
            total_gap: point_estimates.total_gap,
            two_fold: TwoFoldResults {
                aggregate: two_fold_agg,
                detailed_explained,
                detailed_unexplained,
            },
            three_fold: DecompositionDetail {
                aggregate: three_fold_agg,
                detailed: Vec::new(),
            },
            n_a: df
                .filter(
                    &df.column(&self.group)?
                        .as_materialized_series()
                        .equal(group_a_name)?,
                )?
                .height(),
            n_b: df
                .filter(
                    &df.column(&self.group)?
                        .as_materialized_series()
                        .equal(group_b_name)?,
                )?
                .height(),
            residuals: point_estimates.residuals_b.iter().copied().collect(),
            xa_mean: point_estimates.xa_mean,
            xb_mean: point_estimates.xb_mean,
            beta_star: point_estimates.beta_star,
        })
    }

    fn process_detailed_components<'a, F>(
        &self,
        point_components: &[DetailedComponent],
        bootstrap_results: &'a [SinglePassResult],
        extract_fn: F,
        process_component: &dyn Fn(&str, f64, Vec<f64>) -> ComponentResult,
    ) -> Vec<ComponentResult>
    where
        F: Fn(&'a SinglePassResult) -> &'a Vec<DetailedComponent> + Sync,
    {
        let mut bootstrap_map: HashMap<String, Vec<f64>> = HashMap::new();
        for r in bootstrap_results.iter() {
            for comp in extract_fn(r) {
                bootstrap_map
                    .entry(comp.variable_name.clone())
                    .or_default()
                    .push(comp.contribution);
            }
        }

        point_components
            .iter()
            .map(|comp| {
                let estimates = bootstrap_map
                    .get(&comp.variable_name)
                    .cloned()
                    .unwrap_or_else(Vec::new);
                process_component(&comp.variable_name, comp.contribution, estimates)
            })
            .collect()
    }
}

/// Holds results for the two-fold decomposition, including detailed components.
#[derive(Debug, Getters, Serialize)]
#[getset(get = "pub")]
pub struct TwoFoldResults {
    /// Aggregate results for the explained and unexplained components.
    aggregate: Vec<ComponentResult>,
    /// Detailed results for the explained component, broken down by variable.
    detailed_explained: Vec<ComponentResult>,
    /// Detailed results for the unexplained component, broken down by variable.
    detailed_unexplained: Vec<ComponentResult>,
}

/// Holds all the results from the Oaxaca-Blinder decomposition.
#[derive(Debug, Getters, Serialize)]
#[getset(get = "pub")]
pub struct OaxacaResults {
    /// The total difference in the mean outcome between the two groups.
    total_gap: f64,
    /// The results of the two-fold decomposition.
    two_fold: TwoFoldResults,
    /// The results of the three-fold decomposition.
    three_fold: DecompositionDetail,
    /// The number of observations in the advantaged group (Group A).
    n_a: usize,
    /// The number of observations in the reference group (Group B).
    n_b: usize,
    /// The residuals of the reference group (Group B) from the decomposition model.
    /// These represent the "unexplained" part of the outcome for each individual.
    residuals: Vec<f64>,
    /// The mean of the predictors for Group A.
    #[serde(skip)]
    xa_mean: DVector<f64>,
    /// The mean of the predictors for Group B.
    #[serde(skip)]
    xb_mean: DVector<f64>,
    /// The reference coefficients used in the decomposition.
    #[serde(skip)]
    beta_star: DVector<f64>,
}

impl OaxacaResults {
    /// Prints a formatted summary of the decomposition results to the console.
    pub fn summary(&self) {
        println!("Oaxaca-Blinder Decomposition Results");
        println!("========================================");
        println!("Group A (Advantaged): {} observations", self.n_a);
        println!("Group B (Reference):  {} observations", self.n_b);
        println!("Total Gap: {:.4}", self.total_gap);
        println!();

        let mut two_fold_table = Table::new();
        two_fold_table.set_header(vec![
            "Component",
            "Estimate",
            "Std. Err.",
            "p-value",
            "95% CI",
        ]);
        for component in self.two_fold.aggregate() {
            let ci = format!("[{:.3}, {:.3}]", component.ci_lower(), component.ci_upper());
            two_fold_table.add_row(vec![
                Cell::new(component.name()),
                Cell::new(format!("{:.4}", component.estimate())),
                Cell::new(format!("{:.4}", component.std_err())),
                Cell::new(format!("{:.4}", component.p_value())),
                Cell::new(ci),
            ]);
        }
        println!("Two-Fold Decomposition");
        println!("{}", two_fold_table);

        let mut explained_table = Table::new();
        explained_table.set_header(vec![
            "Variable",
            "Contribution",
            "Std. Err.",
            "p-value",
            "95% CI",
        ]);
        for component in self.two_fold.detailed_explained() {
            let ci = format!("[{:.3}, {:.3}]", component.ci_lower(), component.ci_upper());
            explained_table.add_row(vec![
                Cell::new(component.name()),
                Cell::new(format!("{:.4}", component.estimate())),
                Cell::new(format!("{:.4}", component.std_err())),
                Cell::new(format!("{:.4}", component.p_value())),
                Cell::new(ci),
            ]);
        }
        println!("\nDetailed Decomposition (Explained)");
        println!("{}", explained_table);

        let mut unexplained_table = Table::new();
        unexplained_table.set_header(vec![
            "Variable",
            "Contribution",
            "Std. Err.",
            "p-value",
            "95% CI",
        ]);
        for component in self.two_fold.detailed_unexplained() {
            let ci = format!("[{:.3}, {:.3}]", component.ci_lower(), component.ci_upper());
            unexplained_table.add_row(vec![
                Cell::new(component.name()),
                Cell::new(format!("{:.4}", component.estimate())),
                Cell::new(format!("{:.4}", component.std_err())),
                Cell::new(format!("{:.4}", component.p_value())),
                Cell::new(ci),
            ]);
        }
        println!("\nDetailed Decomposition (Unexplained)");
        println!("{}", unexplained_table);
    }

    /// Exports the results to a LaTeX table fragment.
    pub fn to_latex(&self) -> String {
        let mut latex = String::new();
        latex.push_str("\\begin{table}[ht]\n");
        latex.push_str("\\centering\n");
        latex.push_str("\\begin{tabular}{lcccc}\n");
        latex.push_str("\\hline\n");
        latex.push_str("Component & Estimate & Std. Err. & p-value & 95\\% CI \\\\\n");
        latex.push_str("\\hline\n");
        latex.push_str("\\multicolumn{5}{l}{\\textit{Two-Fold Decomposition}} \\\\\n");

        for component in self.two_fold.aggregate() {
            latex.push_str(&format!(
                "{} & {:.4} & {:.4} & {:.4} & [{:.3}, {:.3}] \\\\\n",
                component.name(),
                component.estimate(),
                component.std_err(),
                component.p_value(),
                component.ci_lower(),
                component.ci_upper()
            ));
        }
        latex.push_str("\\hline\n");
        latex.push_str("\\end{tabular}\n");
        latex.push_str("\\caption{Oaxaca-Blinder Decomposition Results}\n");
        latex.push_str("\\label{tab:oaxaca_results}\n");
        latex.push_str("\\end{table}\n");
        latex
    }

    /// Exports the results to a Markdown table.
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str("### Oaxaca-Blinder Decomposition Results\n\n");
        md.push_str("| Component | Estimate | Std. Err. | p-value | 95% CI |\n");
        md.push_str("|---|---|---|---|---|\n");

        for component in self.two_fold.aggregate() {
            md.push_str(&format!(
                "| {} | {:.4} | {:.4} | {:.4} | [{:.3}, {:.3}] |\n",
                component.name(),
                component.estimate(),
                component.std_err(),
                component.p_value(),
                component.ci_lower(),
                component.ci_upper()
            ));
        }
        md
    }

    /// Exports the results to a JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Optimizes the allocation of a remediation budget to reduce the pay gap.
    ///
    /// This method identifies the individuals in the reference group (Group B) with the largest
    /// negative unexplained residuals (i.e., those who are most underpaid relative to their
    /// observable characteristics) and calculates the necessary adjustments to bring them
    /// closer to their predicted pay, subject to the budget and target gap constraints.
    ///
    /// # Arguments
    ///
    /// * `budget` - The maximum total amount to spend on adjustments.
    /// * `target_gap` - The desired final pay gap (difference in means).
    ///
    /// # Returns
    ///
    /// A vector of `BudgetAdjustment` structs detailing who should get a raise and how much.
    pub fn optimize_budget(&self, budget: f64, target_gap: f64) -> Vec<BudgetAdjustment> {
        let current_gap = self.total_gap;
        // If the gap is already smaller than or equal to the target, no adjustments needed.
        if current_gap <= target_gap {
            return Vec::new();
        }

        let required_reduction = current_gap - target_gap;
        // Total amount needed to reduce the gap by required_reduction is required_reduction * n_b
        let total_needed = required_reduction * self.n_b as f64;

        // We can't spend more than the budget, and we don't need to spend more than total_needed.
        let effective_budget = budget.min(total_needed);

        // Identify underpaid individuals (negative residuals) in Group B
        let mut candidates: Vec<(usize, f64)> = self
            .residuals
            .iter()
            .enumerate()
            .filter(|(_, &r)| r < 0.0)
            .map(|(i, &r)| (i, r))
            .collect();

        // Sort by residual ascending (most negative first).
        // We want to fix the largest underpayments first.
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut adjustments = Vec::new();
        let mut spent = 0.0;

        for (index, residual) in candidates {
            if spent >= effective_budget {
                break;
            }

            // The maximum raise for this individual is the amount to bring their residual to 0.
            let max_raise = -residual;
            let remaining_budget = effective_budget - spent;

            // Give them the full correction or whatever is left in the budget/needed.
            let raise = if max_raise <= remaining_budget {
                max_raise
            } else {
                remaining_budget
            };

            // Avoid tiny adjustments due to floating point precision
            if raise > 1e-9 {
                adjustments.push(BudgetAdjustment {
                    index,
                    original_residual: residual,
                    adjustment: raise,
                });
                spent += raise;
            }
        }

        adjustments
    }
}

/// Represents a component of the decomposition (e.g., two-fold or three-fold).
#[derive(Debug, Getters, Serialize)]
#[getset(get = "pub")]
pub struct DecompositionDetail {
    /// Aggregate results for this decomposition component (e.g., "Explained", "Unexplained").
    aggregate: Vec<ComponentResult>,
    /// Detailed results broken down by each predictor variable.
    detailed: Vec<ComponentResult>,
}

/// Represents the calculated result for a single component or variable.
#[derive(Debug, Getters, Clone, Serialize)]
#[getset(get = "pub")]
pub struct ComponentResult {
    name: String,
    estimate: f64,
    std_err: f64,
    p_value: f64,
    ci_lower: f64,
    ci_upper: f64,
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
