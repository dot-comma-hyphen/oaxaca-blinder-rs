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

use std::fmt;
use std::collections::HashMap;
use getset::Getters;
use polars::prelude::*;
use nalgebra::{DMatrix, DVector};
use comfy_table::{Table, Cell};

mod math;
mod decomposition;
mod inference;

use crate::math::ols::{ols};
use crate::decomposition::{
    three_fold_decomposition, two_fold_decomposition, detailed_decomposition,
    ThreeFoldDecomposition, TwoFoldDecomposition, DetailedComponent,
};
use crate::math::normalization::normalize_categorical_coefficients;
use crate::inference::bootstrap_stats;
pub use crate::decomposition::ReferenceCoefficients;
pub mod quantile_decomposition;
pub use crate::quantile_decomposition::QuantileDecompositionBuilder;

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
}

impl From<PolarsError> for OaxacaError {
    fn from(err: PolarsError) -> Self { OaxacaError::PolarsError(err) }
}

impl fmt::Display for OaxacaError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OaxacaError::PolarsError(e) => write!(f, "Polars error: {}", e),
            OaxacaError::ColumnNotFound(s) => write!(f, "Column not found: {}", s),
            OaxacaError::InvalidGroupVariable(s) => write!(f, "Invalid group variable: {}", s),
            OaxacaError::NalgebraError(s) => write!(f, "Nalgebra error: {}", s),
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
}

#[derive(Clone)]
struct SinglePassResult {
    three_fold: ThreeFoldDecomposition,
    two_fold: TwoFoldDecomposition,
    detailed_explained: Vec<DetailedComponent>,
    detailed_unexplained: Vec<DetailedComponent>,
    total_gap: f64,
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
        OaxacaBuilder {
            dataframe,
            outcome: outcome.to_string(),
            predictors: Vec::new(),
            categorical_predictors: Vec::new(),
            group: group.to_string(),
            reference_group: reference_group.to_string(),
            bootstrap_reps: 100,
            reference_coeffs: ReferenceCoefficients::default(),
            normalization_vars: Vec::new(),
        }
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


    fn prepare_data(&self, df: &DataFrame, all_dummy_names: &[String], extra_predictors: &[String]) -> Result<(DMatrix<f64>, DVector<f64>, Vec<String>), OaxacaError> {
        let y_series = df.column(&self.outcome)?.f64()?;
        let y_vec: Vec<f64> = y_series.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
        let y = DVector::from_vec(y_vec);

        let mut current_predictors = self.predictors.clone();
        current_predictors.extend_from_slice(extra_predictors);

        let mut final_predictors: Vec<String> = vec!["intercept".to_string()];
        final_predictors.extend_from_slice(&current_predictors);
        final_predictors.extend_from_slice(all_dummy_names);

        let mut x_df = df.select(&current_predictors)?;
        let intercept_series = Series::new("intercept", vec![1.0; df.height()]);
        x_df.with_column(intercept_series)?;

        for name in all_dummy_names {
            if df.get_column_names().contains(&name.as_str()) {
                x_df.with_column(df.column(name)?.clone())?;
            } else {
                let zero_series = Series::new(name, vec![0.0; df.height()]);
                x_df.with_column(zero_series)?;
            }
        }

        let x_df_selected = x_df.select(&final_predictors)?;
        let x_matrix = x_df_selected.to_ndarray::<Float64Type>(IndexOrder::Fortran)?;
        let x_vec: Vec<f64> = x_matrix.iter().copied().collect();
        let final_names = x_df_selected.get_column_names().iter().map(|s| s.to_string()).collect();
        Ok((DMatrix::from_row_slice(x_df_selected.height(), x_df_selected.width(), &x_vec), y, final_names))
    }

    fn create_dummies_manual(&self, series: &Series) -> Result<(DataFrame, usize, String), OaxacaError> {
        let unique_vals = series.unique()?.sort(false, false);
        let m = unique_vals.len();
        let mut dummy_vars: Vec<Series> = Vec::new();

        let reference_val = if let Some(s) = unique_vals.str()?.get(0) {
            s
        } else {
            return Err(OaxacaError::InvalidGroupVariable(format!("Could not get reference category for {}", series.name())));
        };
        let reference_name = format!("{}_{}", series.name(), reference_val);

        for val in unique_vals.str()?.into_iter().flatten().skip(1) { // Skip the first category as the reference
            let dummy_name = format!("{}_{}", series.name(), val);
            let ca = series.equal(val)?;
            let mut dummy_series = ca.into_series();
            dummy_series = dummy_series.cast(&DataType::Float64)?;
            dummy_series.rename(&dummy_name);
            dummy_vars.push(dummy_series);
        }

        Ok((DataFrame::new(dummy_vars).map_err(OaxacaError::from)?, m, reference_name))
    }

    fn run_single_pass(&self, df: &DataFrame, all_dummy_names: &[String], category_counts: &std::collections::HashMap<String, usize>, base_categories: &std::collections::HashMap<String, String>) -> Result<SinglePassResult, OaxacaError> {
        let unique_groups = df.column(&self.group)?.unique()?.sort(false, false);
        if unique_groups.len() < 2 { return Err(OaxacaError::InvalidGroupVariable("Not enough groups for comparison".to_string())); }

        let group_b_name = self.reference_group.as_str();
        let group_a_name = unique_groups.str()?.get(0).unwrap_or(self.reference_group.as_str());
        let group_a_name = if group_a_name == group_b_name { unique_groups.str()?.get(1).unwrap_or("") } else { group_a_name };

        let df_a = df.filter(&df.column(&self.group)?.equal(group_a_name)?)?;
        let df_b = df.filter(&df.column(&self.group)?.equal(group_b_name)?)?;
        if df_a.height() == 0 || df_b.height() == 0 { return Err(OaxacaError::InvalidGroupVariable("One group has no data".to_string())); }

        let (x_a, y_a, predictor_names) = self.prepare_data(&df_a, all_dummy_names, &[])?;
        let (x_b, y_b, _) = self.prepare_data(&df_b, all_dummy_names, &[])?;

        let mut ols_a = ols(&y_a, &x_a)?;
        let mut ols_b = ols(&y_b, &x_b)?;

        let xa_mean = x_a.row_mean().transpose();
        let xb_mean = x_b.row_mean().transpose();

        let mut base_coeffs_a = std::collections::HashMap::new();
        let mut base_coeffs_b = std::collections::HashMap::new();
        if !self.normalization_vars.is_empty() {
            base_coeffs_a = normalize_categorical_coefficients(&mut ols_a, &predictor_names, &self.normalization_vars, &xa_mean, category_counts);
            base_coeffs_b = normalize_categorical_coefficients(&mut ols_b, &predictor_names, &self.normalization_vars, &xb_mean, category_counts);
        }

        let beta_a = &ols_a.coefficients;
        let beta_b = &ols_b.coefficients;

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
            ReferenceCoefficients::Pooled => {
                let mut df_pooled = df_a.vstack(&df_b)?;
                let group_indicator = Series::new("group_indicator", df_pooled.column(&self.group)?.equal(group_a_name)?.into_series().cast(&DataType::Float64)?);
                df_pooled.with_column(group_indicator)?;

                let (x_pooled, y_pooled, pooled_predictor_names) = self.prepare_data(&df_pooled, all_dummy_names, &["group_indicator".to_string()])?;
                
                let mut ols_pooled = ols(&y_pooled, &x_pooled)?;

                if !self.normalization_vars.is_empty() {
                    let n_a = df_a.height() as f64;
                    let n_b = df_b.height() as f64;
                    let x_pool_mean = (xa_mean.clone() * n_a + xb_mean.clone() * n_b) / (n_a + n_b);
                    base_coeffs_star = normalize_categorical_coefficients(&mut ols_pooled, &pooled_predictor_names, &self.normalization_vars, &x_pool_mean, category_counts);
                }
                let group_indicator_idx = pooled_predictor_names.iter().position(|r| r == "group_indicator")
                    .ok_or_else(|| OaxacaError::NalgebraError("group_indicator not found in pooled model predictors".to_string()))?;
                beta_star_owned = ols_pooled.coefficients.remove_row(group_indicator_idx);
                &beta_star_owned
            }
            ReferenceCoefficients::Weighted => {
                let n_a = df_a.height() as f64;
                let n_b = df_b.height() as f64;
                let total_n = n_a + n_b;
                if total_n == 0.0 {
                    return Err(OaxacaError::InvalidGroupVariable("No data in groups for weighted coefficients.".to_string()));
                }
                let weight_a = n_a / total_n;
                let weight_b = 1.0 - weight_a;
                if !self.normalization_vars.is_empty() {
                    for var in &self.normalization_vars {
                        let coeff_a = base_coeffs_a.get(var).unwrap_or(&0.0);
                        let coeff_b = base_coeffs_b.get(var).unwrap_or(&0.0);
                        base_coeffs_star.insert(var.clone(), coeff_a * weight_a + coeff_b * weight_b);
                    }
                }
                beta_star_owned = beta_a * weight_a + beta_b * weight_b;
                &beta_star_owned
            }
        };

        let three_fold = three_fold_decomposition(&xa_mean, &xb_mean, beta_a, beta_b);
        let two_fold = two_fold_decomposition(&xa_mean, &xb_mean, beta_a, beta_b, beta_star);
        let (detailed_explained, mut detailed_unexplained) = detailed_decomposition(&xa_mean, &xb_mean, beta_a, beta_b, beta_star, &predictor_names);

        if !self.normalization_vars.is_empty() {
            for var in &self.normalization_vars {
                let base_dummy_name = if let Some(name) = base_categories.get(var) {
                    name
                } else {
                    continue;
                };

                let dummy_indices: Vec<usize> = predictor_names
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

                let contribution =
                    xa_mean_base * (beta_a_base - beta_star_base) + xb_mean_base * (beta_star_base - beta_b_base);

                detailed_unexplained.push(DetailedComponent {
                    variable_name: base_dummy_name.clone(),
                    contribution,
                });
            }
        }

        let total_gap = y_a.mean() - y_b.mean();

        Ok(SinglePassResult { three_fold, two_fold, detailed_explained, detailed_unexplained, total_gap })
    }

    /// Executes the Oaxaca-Blinder decomposition.
    pub fn run(&self) -> Result<OaxacaResults, OaxacaError> {
        let mut df = self.dataframe.clone();
        let mut all_dummy_names = Vec::new();
        let mut category_counts = std::collections::HashMap::new();
        let mut base_categories = std::collections::HashMap::new();
        if !self.categorical_predictors.is_empty() {
            for cat_pred in &self.categorical_predictors {
                let series = df.column(cat_pred)?;
                let (dummies, m, base_name) = self.create_dummies_manual(series)?;
                category_counts.insert(cat_pred.clone(), m);
                base_categories.insert(cat_pred.clone(), base_name);
                for s in dummies.get_columns() {
                    all_dummy_names.push(s.name().to_string());
                }
                df = df.hstack(dummies.get_columns())?;
            }
        }

        use rayon::prelude::*;

        let point_estimates = self.run_single_pass(&df, &all_dummy_names, &category_counts, &base_categories)?;

        let bootstrap_results: Vec<SinglePassResult> = (0..self.bootstrap_reps)
            .into_par_iter()
            .filter_map(|_| {
                df.sample_n_literal(df.height(), true, false, None)
                    .ok()
                    .and_then(|sample_df| self.run_single_pass(&sample_df, &all_dummy_names, &category_counts, &base_categories).ok())
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
            ComponentResult { name: name.to_string(), estimate: point, std_err, p_value, ci_lower, ci_upper }
        };

        let two_fold_agg = vec![
            process_component("explained", point_estimates.two_fold.explained, bootstrap_results.iter().map(|r| r.two_fold.explained).collect()),
            process_component("unexplained", point_estimates.two_fold.unexplained, bootstrap_results.iter().map(|r| r.two_fold.unexplained).collect()),
        ];
        let three_fold_agg = vec![
            process_component("endowments", point_estimates.three_fold.endowments, bootstrap_results.iter().map(|r| r.three_fold.endowments).collect()),
            process_component("coefficients", point_estimates.three_fold.coefficients, bootstrap_results.iter().map(|r| r.three_fold.coefficients).collect()),
            process_component("interaction", point_estimates.three_fold.interaction, bootstrap_results.iter().map(|r| r.three_fold.interaction).collect()),
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

        let group_b_name = self.reference_group.as_str();
        let unique_groups = self.dataframe.column(&self.group)?.unique()?.sort(false, false);
        let group_a_name = unique_groups.str()?.get(0).unwrap_or(self.reference_group.as_str());
        let group_a_name = if group_a_name == group_b_name { unique_groups.str()?.get(1).unwrap_or("") } else { group_a_name };

        Ok(OaxacaResults {
            total_gap: point_estimates.total_gap,
            two_fold: TwoFoldResults {
                aggregate: two_fold_agg,
                detailed_explained,
                detailed_unexplained,
            },
            three_fold: DecompositionDetail { aggregate: three_fold_agg, detailed: Vec::new() },
            n_a: self.dataframe.filter(&self.dataframe.column(&self.group)?.equal(group_a_name)?)?.height(),
            n_b: self.dataframe.filter(&self.dataframe.column(&self.group)?.equal(group_b_name)?)?.height(),
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
                bootstrap_map.entry(comp.variable_name.clone()).or_default().push(comp.contribution);
            }
        }

        point_components.iter().map(|comp| {
            let estimates = bootstrap_map.get(&comp.variable_name).cloned().unwrap_or_else(Vec::new);
            process_component(&comp.variable_name, comp.contribution, estimates)
        }).collect()
    }
}

/// Holds results for the two-fold decomposition, including detailed components.
#[derive(Debug, Getters)]
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
#[derive(Debug, Getters)]
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
        two_fold_table.set_header(vec!["Component", "Estimate", "Std. Err.", "p-value", "95% CI"]);
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
        explained_table.set_header(vec!["Variable", "Contribution", "Std. Err.", "p-value", "95% CI"]);
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
        unexplained_table.set_header(vec!["Variable", "Contribution", "Std. Err.", "p-value", "95% CI"]);
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
}


/// Represents a component of the decomposition (e.g., two-fold or three-fold).
#[derive(Debug, Getters)]
#[getset(get = "pub")]
pub struct DecompositionDetail {
    /// Aggregate results for this decomposition component (e.g., "Explained", "Unexplained").
    aggregate: Vec<ComponentResult>,
    /// Detailed results broken down by each predictor variable.
    detailed: Vec<ComponentResult>,
}

/// Represents the calculated result for a single component or variable.
#[derive(Debug, Getters, Clone)]
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