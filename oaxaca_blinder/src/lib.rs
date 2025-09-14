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

use std::fmt;
use getset::Getters;
use polars::prelude::*;
use nalgebra::{DMatrix, DVector};
use comfy_table::{Table, Cell};

mod math;
mod decomposition;
mod inference;

use crate::math::ols::{ols, pooled_ols};
use crate::decomposition::{
    three_fold_decomposition, two_fold_decomposition, detailed_decomposition,
    ThreeFoldDecomposition, TwoFoldDecomposition, DetailedComponent,
};
use crate::inference::bootstrap_stats;
pub use crate::decomposition::ReferenceCoefficients;

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
    group: String,
    reference_group: String,
    bootstrap_reps: usize,
    reference_coeffs: ReferenceCoefficients,
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
            group: group.to_string(),
            reference_group: reference_group.to_string(),
            bootstrap_reps: 100,
            reference_coeffs: ReferenceCoefficients::default(),
        }
    }

    /// Sets the reference coefficients for the decomposition.
    ///
    /// The default is `ReferenceCoefficients::GroupB`.
    pub fn reference_coefficients(mut self, reference: ReferenceCoefficients) -> Self {
        self.reference_coeffs = reference;
        self
    }

    /// Sets the predictor variables for the model.
    ///
    /// # Arguments
    ///
    /// * `predictors` - A slice of strings representing the column names of the predictor variables.
    pub fn predictors(mut self, predictors: &[&str]) -> Self {
        self.predictors = predictors.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Sets the number of bootstrap replications for standard error calculation.
    ///
    /// # Arguments
    ///
    /// * `reps` - The number of bootstrap samples to generate. Defaults to 100.
    pub fn bootstrap_reps(mut self, reps: usize) -> Self {
        self.bootstrap_reps = reps;
        self
    }

    fn prepare_data(&self, df: &DataFrame) -> Result<(DMatrix<f64>, DVector<f64>, Vec<String>), OaxacaError> {
        let y_series = df.column(&self.outcome)?.f64()?;
        let y_vec: Vec<f64> = y_series.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
        let y = DVector::from_vec(y_vec);

        let mut predictors_with_intercept = self.predictors.clone();
        predictors_with_intercept.insert(0, "intercept".to_string());

        let mut x_df = df.select(&self.predictors)?;
        let intercept_series = Series::new("intercept", vec![1.0; df.height()]);
        x_df.with_column(intercept_series)?;

        let x_df = x_df.select(&predictors_with_intercept)?;

        let x_matrix = x_df.to_ndarray::<Float64Type>(IndexOrder::Fortran)?;
        let x_vec: Vec<f64> = x_matrix.iter().copied().collect();
        let final_names = x_df.get_column_names().iter().map(|s| s.to_string()).collect();
        Ok((DMatrix::from_row_slice(x_df.height(), x_df.width(), &x_vec), y, final_names))
    }

    fn run_single_pass(&self, df: &DataFrame) -> Result<SinglePassResult, OaxacaError> {
        let unique_groups = df.column(&self.group)?.unique()?.sort(false, false);
        if unique_groups.len() < 2 { return Err(OaxacaError::InvalidGroupVariable("Not enough groups for comparison".to_string())); }

        let group_b_name = self.reference_group.as_str();
        let group_a_name = unique_groups.str()?.get(0).unwrap_or(self.reference_group.as_str());
        let group_a_name = if group_a_name == group_b_name { unique_groups.str()?.get(1).unwrap_or("") } else { group_a_name };

        let df_a = df.filter(&df.column(&self.group)?.equal(group_a_name)?)?;
        let df_b = df.filter(&df.column(&self.group)?.equal(group_b_name)?)?;
        if df_a.height() == 0 || df_b.height() == 0 { return Err(OaxacaError::InvalidGroupVariable("One group has no data".to_string())); }

        let (x_a, y_a, predictor_names) = self.prepare_data(&df_a)?;
        let (x_b, y_b, _) = self.prepare_data(&df_b)?;

        let ols_a = ols(&y_a, &x_a)?;
        let ols_b = ols(&y_b, &x_b)?;
        let beta_a = &ols_a.coefficients;
        let beta_b = &ols_b.coefficients;

        let beta_star_owned: DVector<f64>;
        let beta_star: &DVector<f64> = match self.reference_coeffs {
            ReferenceCoefficients::GroupA => beta_a,
            ReferenceCoefficients::GroupB => beta_b,
            ReferenceCoefficients::Pooled => {
                let ols_pooled = pooled_ols(&y_a, &x_a, &y_b, &x_b)?;
                beta_star_owned = ols_pooled.coefficients;
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
                let weight_b = 1.0 - weight_a; // Avoids a second division
                beta_star_owned = beta_a * weight_a + beta_b * weight_b;
                &beta_star_owned
            }
        };

        let xa_mean = x_a.row_mean().transpose();
        let xb_mean = x_b.row_mean().transpose();

        let three_fold = three_fold_decomposition(&xa_mean, &xb_mean, beta_a, beta_b);
        let two_fold = two_fold_decomposition(&xa_mean, &xb_mean, beta_a, beta_b, beta_star);
        let (detailed_explained, detailed_unexplained) = detailed_decomposition(&xa_mean, &xb_mean, beta_a, beta_b, beta_star, &predictor_names);
        let total_gap = y_a.mean() - y_b.mean();

        Ok(SinglePassResult { three_fold, two_fold, detailed_explained, detailed_unexplained, total_gap })
    }

    /// Executes the Oaxaca-Blinder decomposition.
    pub fn run(&self) -> Result<OaxacaResults, OaxacaError> {
        let point_estimates = self.run_single_pass(&self.dataframe)?;
        let mut bootstrap_results: Vec<SinglePassResult> = Vec::with_capacity(self.bootstrap_reps);
        for _ in 0..self.bootstrap_reps {
             if let Ok(sample_df) = self.dataframe.sample_n_literal(self.dataframe.height(), true, false, None) {
                if let Ok(result) = self.run_single_pass(&sample_df) { bootstrap_results.push(result); }
            }
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

        let detailed_explained: Vec<ComponentResult> = point_estimates.detailed_explained.iter().enumerate().map(|(i, comp)| {
            let estimates = bootstrap_results.iter().map(|r| r.detailed_explained[i].contribution).collect();
            process_component(&comp.variable_name, comp.contribution, estimates)
        }).collect();
        let detailed_unexplained: Vec<ComponentResult> = point_estimates.detailed_unexplained.iter().enumerate().map(|(i, comp)| {
            let estimates = bootstrap_results.iter().map(|r| r.detailed_unexplained[i].contribution).collect();
            process_component(&comp.variable_name, comp.contribution, estimates)
        }).collect();

        let group_b_name = self.reference_group.as_str();
        let unique_groups = self.dataframe.column(&self.group)?.unique()?.sort(false, false);
        let group_a_name = unique_groups.str()?.get(0).unwrap_or(self.reference_group.as_str());
        let group_a_name = if group_a_name == group_b_name { unique_groups.str()?.get(1).unwrap_or("") } else { group_a_name };

        Ok(OaxacaResults {
            total_gap: point_estimates.total_gap,
            two_fold: DecompositionDetail { aggregate: two_fold_agg, detailed: detailed_explained },
            three_fold: DecompositionDetail { aggregate: three_fold_agg, detailed: detailed_unexplained },
            n_a: self.dataframe.filter(&self.dataframe.column(&self.group)?.equal(group_a_name)?)?.height(),
            n_b: self.dataframe.filter(&self.dataframe.column(&self.group)?.equal(group_b_name)?)?.height(),
        })
    }
}

/// Holds all the results from the Oaxaca-Blinder decomposition.
#[derive(Debug, Getters)]
#[getset(get = "pub")]
pub struct OaxacaResults {
    /// The total difference in the mean outcome between the two groups.
    total_gap: f64,
    /// The results of the two-fold decomposition.
    two_fold: DecompositionDetail,
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
        for component in self.two_fold.detailed() {
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
        for component in self.three_fold.detailed() {
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
