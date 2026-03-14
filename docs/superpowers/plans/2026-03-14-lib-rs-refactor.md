# lib.rs Refactor Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the monolithic 1,755-line `oaxaca_blinder/src/lib.rs` into 5 focused modules while preserving the public API surface and consolidating duplicated group-splitting logic.

**Architecture:** Extract error types, result types, estimation internals, builder logic, and display formatting into separate modules. `lib.rs` becomes a thin hub of module declarations and re-exports. A new `split_groups()` helper replaces 5 duplicated group-splitting blocks.

**Tech Stack:** Rust, Polars, Nalgebra, comfy-table (display feature), serde (serialization)

**Spec:** `docs/superpowers/specs/2026-03-14-lib-rs-refactor-design.md`

---

## Chunk 1: Extract foundational modules (error.rs, types.rs, display.rs)

These are the "leaves" of the dependency graph. `error.rs` and `types.rs` are independent; `display.rs` depends on `types.rs` and must come after it. Tasks are sequenced accordingly.

### Task 1: Create `error.rs`

**Files:**
- Create: `oaxaca_blinder/src/error.rs`
- Modify: `oaxaca_blinder/src/lib.rs`

- [ ] **Step 1: Create `error.rs` with `OaxacaError` and impls**

```rust
// oaxaca_blinder/src/error.rs
use polars::prelude::PolarsError;
use std::fmt;

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
    /// Occurs when there is not enough data for an operation.
    InsufficientData(String),
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
            OaxacaError::InsufficientData(s) => write!(f, "Insufficient data: {}", s),
        }
    }
}

impl std::error::Error for OaxacaError {}
```

- [ ] **Step 2: Add `mod error` and `pub use` to `lib.rs`**

At the top of `lib.rs` (after the doc comment), add:

```rust
mod error;
pub use error::OaxacaError;
```

Remove the `OaxacaError` enum definition, `From`, `Display`, and `Error` impls from `lib.rs` (lines 97-133).

- [ ] **Step 3: Verify compilation**

Run: `cargo build -p oaxaca_blinder 2>&1 | head -20`
Expected: Successful build (all other files import `crate::OaxacaError` which is still re-exported)

- [ ] **Step 4: Run tests**

Run: `cargo test -p oaxaca_blinder 2>&1 | tail -5`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add oaxaca_blinder/src/error.rs oaxaca_blinder/src/lib.rs
git commit -m "refactor: extract OaxacaError into error.rs"
```

---

### Task 2: Create `types.rs`

**Files:**
- Create: `oaxaca_blinder/src/types.rs`
- Modify: `oaxaca_blinder/src/lib.rs`

- [ ] **Step 1: Create `types.rs` with result structs, accessors, and `optimize_budget`**

Move these items from `lib.rs`:
- `TwoFoldResults` struct (lines 1441-1452)
- `OaxacaResults` struct (lines 1454-1480)
- `OaxacaResults` impl block with `explained()`, `unexplained()`, `get_summary_table()`, `get_detailed_table()`, `optimize_budget()` (lines 1482-1721)
- `DecompositionDetail` struct (lines 1724-1732)
- `ComponentResult` struct (lines 1734-1745)

```rust
// oaxaca_blinder/src/types.rs
use getset::Getters;
use nalgebra::DVector;
use serde::Serialize;

use crate::decomposition::BudgetAdjustment;

/// Holds results for the two-fold decomposition, including detailed components.
#[derive(Debug, Getters, Serialize)]
#[getset(get = "pub")]
pub struct TwoFoldResults {
    /// Aggregate results for the explained and unexplained components.
    pub aggregate: Vec<ComponentResult>,
    /// Detailed results for the explained component, broken down by variable.
    pub detailed_explained: Vec<ComponentResult>,
    /// Detailed results for the unexplained component, broken down by variable.
    pub detailed_unexplained: Vec<ComponentResult>,
    /// Detailed results for the selection component (Heckman only).
    pub detailed_selection: Vec<ComponentResult>,
}

/// Holds all the results from the Oaxaca-Blinder decomposition.
#[derive(Debug, Getters, Serialize)]
#[getset(get = "pub")]
pub struct OaxacaResults {
    /// The total difference in the mean outcome between the two groups.
    pub total_gap: f64,
    /// The results of the two-fold decomposition.
    pub two_fold: TwoFoldResults,
    /// The results of the three-fold decomposition.
    pub three_fold: DecompositionDetail,
    /// The number of observations in the advantaged group (Group A).
    pub n_a: usize,
    /// The number of observations in the reference group (Group B).
    pub n_b: usize,
    /// The residuals of the reference group (Group B) from the decomposition model.
    pub residuals: Vec<f64>,
    /// The mean of the predictors for Group A.
    #[serde(skip)]
    pub xa_mean: DVector<f64>,
    /// The mean of the predictors for Group B.
    #[serde(skip)]
    pub xb_mean: DVector<f64>,
    /// The reference coefficients used in the decomposition.
    #[serde(skip)]
    pub beta_star: DVector<f64>,
}

impl OaxacaResults {
    pub fn explained(&self) -> &ComponentResult {
        self.two_fold
            .aggregate()
            .iter()
            .find(|c| c.name == "explained")
            .expect("Explained component not found")
    }

    pub fn unexplained(&self) -> &ComponentResult {
        self.two_fold
            .aggregate()
            .iter()
            .find(|c| c.name == "unexplained")
            .expect("Unexplained component not found")
    }

    pub fn get_summary_table(&self) -> Vec<(&String, &ComponentResult)> {
        self.two_fold
            .aggregate()
            .iter()
            .map(|c| (&c.name, c))
            .collect()
    }

    pub fn get_detailed_table(&self) -> Vec<(String, f64, f64)> {
        let mut map = std::collections::HashMap::new();
        for comp in self.two_fold.detailed_explained() {
            map.entry(comp.name().clone()).or_insert((0.0, 0.0)).0 = *comp.estimate();
        }
        for comp in self.two_fold.detailed_unexplained() {
            map.entry(comp.name().clone()).or_insert((0.0, 0.0)).1 = *comp.estimate();
        }
        map.into_iter().map(|(k, (v1, v2))| (k, v1, v2)).collect()
    }

    /// Optimizes the allocation of a remediation budget to reduce the pay gap.
    pub fn optimize_budget(&self, budget: f64, target_gap: f64) -> Vec<BudgetAdjustment> {
        let current_gap = self.total_gap;
        if current_gap <= target_gap {
            return Vec::new();
        }

        let required_reduction = current_gap - target_gap;
        let total_needed = required_reduction * self.n_b as f64;
        let effective_budget = budget.min(total_needed);

        let mut candidates: Vec<(usize, f64)> = self
            .residuals
            .iter()
            .enumerate()
            .filter(|(_, &r)| r < 0.0)
            .map(|(i, &r)| (i, r))
            .collect();

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut adjustments = Vec::new();
        let mut spent = 0.0;

        for (index, residual) in candidates {
            if spent >= effective_budget {
                break;
            }

            let max_raise = -residual;
            let remaining_budget = effective_budget - spent;

            let raise = if max_raise <= remaining_budget {
                max_raise
            } else {
                remaining_budget
            };

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
    /// Aggregate results for this decomposition component.
    pub aggregate: Vec<ComponentResult>,
    /// Detailed results broken down by each predictor variable.
    pub detailed: Vec<ComponentResult>,
}

/// Represents the calculated result for a single component or variable.
#[derive(Debug, Getters, Clone, Serialize)]
#[getset(get = "pub")]
pub struct ComponentResult {
    pub name: String,
    pub estimate: f64,
    pub std_err: f64,
    pub t_stat: f64,
    pub p_value: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
}
```

- [ ] **Step 2: Update `lib.rs`**

Add to `lib.rs`:

```rust
mod types;
pub use types::{ComponentResult, DecompositionDetail, OaxacaResults, TwoFoldResults};
```

Remove from `lib.rs`: the `TwoFoldResults`, `OaxacaResults`, `DecompositionDetail`, `ComponentResult` struct definitions and the `impl OaxacaResults` block containing `explained()`, `unexplained()`, `get_summary_table()`, `get_detailed_table()`, and `optimize_budget()` (lines 1440-1745).

- [ ] **Step 3: Verify compilation**

Run: `cargo build -p oaxaca_blinder 2>&1 | head -20`
Expected: Successful build

- [ ] **Step 4: Run tests**

Run: `cargo test -p oaxaca_blinder 2>&1 | tail -5`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add oaxaca_blinder/src/types.rs oaxaca_blinder/src/lib.rs
git commit -m "refactor: extract result types into types.rs"
```

---

### Task 3: Create `display.rs`

**Files:**
- Create: `oaxaca_blinder/src/display.rs`
- Modify: `oaxaca_blinder/src/lib.rs`

- [ ] **Step 1: Create `display.rs` with formatting methods**

Move the `summary()`, `to_latex()`, `to_markdown()`, `to_json()` methods from the `impl OaxacaResults` block. These are now a separate `impl` block in their own file:

```rust
// oaxaca_blinder/src/display.rs
#[cfg(feature = "display")]
use comfy_table::{Cell, Table};

use crate::types::OaxacaResults;

#[cfg(feature = "display")]
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
}

impl OaxacaResults {
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
}
```

- [ ] **Step 2: Update `lib.rs`**

Add to `lib.rs`:

```rust
mod display;
```

Remove from `lib.rs`:
- The `#[cfg(feature = "display")] use comfy_table::{Cell, Table};` import (line 56-57)
- The `summary()`, `to_latex()`, `to_markdown()`, `to_json()` methods from `impl OaxacaResults` (lines 1518-1646)

- [ ] **Step 3: Verify compilation**

Run: `cargo build -p oaxaca_blinder 2>&1 | head -20`
Expected: Successful build

- [ ] **Step 4: Run tests**

Run: `cargo test -p oaxaca_blinder 2>&1 | tail -5`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add oaxaca_blinder/src/display.rs oaxaca_blinder/src/lib.rs
git commit -m "refactor: extract display/export formatters into display.rs"
```

---

## Chunk 2: Extract estimation.rs and builder.rs

These are the core of the refactor. `estimation.rs` must exist before `builder.rs` can import from it.

### Task 4: Create `estimation.rs`

**Files:**
- Create: `oaxaca_blinder/src/estimation.rs`
- Modify: `oaxaca_blinder/src/lib.rs`

- [ ] **Step 1: Create `estimation.rs` with estimator trait and implementations**

Move from `lib.rs`:
- `EstimationContext<'a>` struct (lines 190-201)
- `EstimationResult` struct (lines 170-188)
- `Estimator` trait (lines 203-205)
- `OlsEstimator` struct + `Estimator` impl (lines 207-272)
- `HeckmanEstimator` struct + `Estimator` impl + helper methods (lines 274-414)

```rust
// oaxaca_blinder/src/estimation.rs
use std::collections::HashMap;

use nalgebra::{DMatrix, DVector};
use polars::prelude::*;

use crate::error::OaxacaError;
use crate::heckman::heckman_two_step;
use crate::math::normalization::normalize_categorical_coefficients;
use crate::math::ols::ols;

pub(crate) struct EstimationResult {
    pub beta_a: DVector<f64>,
    pub beta_b: DVector<f64>,
    pub xa_mean: DVector<f64>,
    pub xb_mean: DVector<f64>,
    pub predictor_names: Vec<String>,
    pub residuals_a: DVector<f64>,
    pub residuals_b: DVector<f64>,
    pub base_coeffs_a: HashMap<String, f64>,
    pub base_coeffs_b: HashMap<String, f64>,
    pub selection_coeffs_a: Option<DVector<f64>>,
    pub selection_coeffs_b: Option<DVector<f64>>,
    pub selection_means_a: Option<DVector<f64>>,
    pub selection_means_b: Option<DVector<f64>>,
    pub selection_names: Option<Vec<String>>,
    pub imr_delta_a: Option<f64>,
    pub imr_delta_b: Option<f64>,
}

pub(crate) struct EstimationContext<'a> {
    pub df_a: &'a DataFrame,
    pub df_b: &'a DataFrame,
    pub x_a: &'a DMatrix<f64>,
    pub y_a: &'a DVector<f64>,
    pub w_a: &'a Option<DVector<f64>>,
    pub x_b: &'a DMatrix<f64>,
    pub y_b: &'a DVector<f64>,
    pub w_b: &'a Option<DVector<f64>>,
    pub predictor_names: &'a [String],
    pub category_counts: &'a HashMap<String, usize>,
}

pub(crate) trait Estimator {
    fn estimate(&self, ctx: &EstimationContext) -> Result<EstimationResult, OaxacaError>;
}

pub(crate) struct OlsEstimator {
    pub normalization_vars: Vec<String>,
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
            selection_coeffs_a: None,
            selection_coeffs_b: None,
            selection_means_a: None,
            selection_means_b: None,
            selection_names: None,
            imr_delta_a: None,
            imr_delta_b: None,
        })
    }
}

pub(crate) struct HeckmanEstimator {
    pub selection_outcome: String,
    pub selection_predictors: Vec<String>,
}

impl Estimator for HeckmanEstimator {
    fn estimate(&self, ctx: &EstimationContext) -> Result<EstimationResult, OaxacaError> {
        let (x_sel_a, y_sel_a, x_sel_sub_a) = self.prepare_selection_data(ctx.df_a)?;
        let (x_sel_b, y_sel_b, x_sel_sub_b) = self.prepare_selection_data(ctx.df_b)?;

        let (x_a_filt, y_a_filt) = self.filter_outcome_rows(ctx.x_a, ctx.y_a, ctx.df_a)?;
        let (x_b_filt, y_b_filt) = self.filter_outcome_rows(ctx.x_b, ctx.y_b, ctx.df_b)?;

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
            selection_coeffs_a: Some(res_a.selection_coeffs),
            selection_coeffs_b: Some(res_b.selection_coeffs),
            selection_means_a: Some(x_sel_a.row_mean().transpose().clone()),
            selection_means_b: Some(x_sel_b.row_mean().transpose().clone()),
            selection_names: Some(ctx.predictor_names.to_vec()),
            imr_delta_a: Some(res_a.imr_delta),
            imr_delta_b: Some(res_b.imr_delta),
        })
    }
}

impl HeckmanEstimator {
    #[allow(clippy::type_complexity)]
    fn prepare_selection_data(
        &self,
        df_group: &DataFrame,
    ) -> Result<(DMatrix<f64>, DVector<f64>, DMatrix<f64>), OaxacaError> {
        let y_sel_series = df_group.column(&self.selection_outcome)?.f64()?;
        let y_sel_vec: Result<Vec<f64>, OaxacaError> = y_sel_series
            .into_iter()
            .map(|opt| {
                opt.ok_or_else(|| {
                    OaxacaError::InvalidGroupVariable(
                        "Selection outcome contains nulls".to_string(),
                    )
                })
            })
            .collect();
        let y_sel = DVector::from_vec(y_sel_vec?);

        let mut x_sel_df = df_group.select(&self.selection_predictors)?;
        let intercept = Series::new("__ob_intercept__".into(), vec![1.0; df_group.height()]);
        x_sel_df.with_column(intercept)?;
        let mut cols = vec!["__ob_intercept__".to_string()];
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
            "__ob_intercept__".into(),
            vec![1.0; df_subset.height()],
        ))?;
        let x_sel_sub_df = x_sel_sub_df.select(&cols)?;

        let x_sel_sub_mat = x_sel_sub_df.to_ndarray::<Float64Type>(IndexOrder::Fortran)?;
        let x_sel_sub_vec: Vec<f64> = x_sel_sub_mat.iter().copied().collect();
        let x_sel_sub =
            DMatrix::from_row_slice(x_sel_sub_df.height(), x_sel_sub_df.width(), &x_sel_sub_vec);

        Ok((x_sel, y_sel, x_sel_sub))
    }

    fn filter_outcome_rows(
        &self,
        x: &DMatrix<f64>,
        y: &DVector<f64>,
        df: &DataFrame,
    ) -> Result<(DMatrix<f64>, DVector<f64>), OaxacaError> {
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
    }
}
```

- [ ] **Step 2: Update `lib.rs`**

Add to `lib.rs`:

```rust
mod estimation;
```

Remove from `lib.rs`: `EstimationResult` (lines 170-188), `EstimationContext` (lines 190-201), `Estimator` trait (lines 203-205), `OlsEstimator` (lines 207-272), `HeckmanEstimator` (lines 274-414).

- [ ] **Step 3: Verify compilation**

Run: `cargo build -p oaxaca_blinder 2>&1 | head -20`
Expected: Successful build

- [ ] **Step 4: Run tests**

Run: `cargo test -p oaxaca_blinder 2>&1 | tail -5`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add oaxaca_blinder/src/estimation.rs oaxaca_blinder/src/lib.rs
git commit -m "refactor: extract estimator trait and implementations into estimation.rs"
```

---

### Task 5: Create `builder.rs` and finalize `lib.rs`

This is the largest task. Move `OaxacaBuilder` and `SinglePassResult` into `builder.rs`, add `GroupSplit` and `split_groups()`, refactor all 5 group-splitting call sites, and reduce `lib.rs` to a thin hub.

**Files:**
- Create: `oaxaca_blinder/src/builder.rs`
- Modify: `oaxaca_blinder/src/lib.rs` (rewrite to hub)

- [ ] **Step 1: Create `builder.rs`**

Move from `lib.rs`:
- `SinglePassResult` struct (lines 154-168, with `#[allow(dead_code)]`)
- `OaxacaBuilder` struct (lines 135-152)
- All `impl OaxacaBuilder` blocks (constructors, config methods, `run`, `run_single_pass`, `decompose_quantile`, `get_data_matrices`, `prepare_data`, `create_dummies_manual`, `clean_dataframe`, `process_detailed_components`)

Add new:
- `GroupSplit` struct
- `split_groups()` method on `OaxacaBuilder`

Replace all 5 inline group-splitting blocks with calls to `self.split_groups()`.

The file header and imports:

```rust
// oaxaca_blinder/src/builder.rs
use std::collections::HashMap;

use nalgebra::{DMatrix, DVector};
use polars::prelude::*;
use rayon::prelude::*;

use crate::decomposition::{
    detailed_decomposition, three_fold_decomposition, two_fold_decomposition, DetailedComponent,
    ReferenceCoefficients, ThreeFoldDecomposition, TwoFoldDecomposition,
};
use crate::error::OaxacaError;
use crate::estimation::{EstimationContext, Estimator, HeckmanEstimator, OlsEstimator};
use crate::formula::Formula;
use crate::inference::bootstrap_stats;
use crate::math::normalization::normalize_categorical_coefficients;
use crate::math::ols::ols;
use crate::math::rif::calculate_rif;
use crate::types::{ComponentResult, DecompositionDetail, OaxacaResults, TwoFoldResults};
```

The `GroupSplit` struct and `split_groups()` method:

```rust
pub(crate) struct GroupSplit {
    pub df_a: DataFrame,
    pub df_b: DataFrame,
    pub group_a_name: String,
    pub group_b_name: String,
}

impl OaxacaBuilder {
    fn split_groups(&self, df: &DataFrame) -> Result<GroupSplit, OaxacaError> {
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

        let group_b_name = self.reference_group.clone();
        let group_a_name_temp = unique_groups
            .str()?
            .get(0)
            .unwrap_or(self.reference_group.as_str())
            .to_string();
        let group_a_name = if group_a_name_temp == group_b_name {
            unique_groups.str()?.get(1).unwrap_or("").to_string()
        } else {
            group_a_name_temp
        };

        let df_a = df.filter(
            &df.column(&self.group)?
                .as_materialized_series()
                .equal(group_a_name.as_str())?,
        )?;
        let df_b = df.filter(
            &df.column(&self.group)?
                .as_materialized_series()
                .equal(group_b_name.as_str())?,
        )?;

        Ok(GroupSplit {
            df_a,
            df_b,
            group_a_name,
            group_b_name,
        })
    }
}
```

The `SinglePassResult` struct (kept as-is with `#[allow(dead_code)]`):

```rust
#[derive(Clone)]
#[allow(dead_code)]
pub(crate) struct SinglePassResult {
    pub three_fold: ThreeFoldDecomposition,
    pub two_fold: TwoFoldDecomposition,
    pub detailed_explained: Vec<DetailedComponent>,
    pub detailed_unexplained: Vec<DetailedComponent>,
    pub total_gap: f64,
    pub residuals_a: DVector<f64>,
    pub residuals_b: DVector<f64>,
    pub xa_mean: DVector<f64>,
    pub xb_mean: DVector<f64>,
    pub beta_star: DVector<f64>,
    pub detailed_selection: Vec<DetailedComponent>,
}
```

The `OaxacaBuilder` struct and all methods follow (moved verbatim from `lib.rs`), with the following changes:

**In `run_single_pass()`:** Replace the inline group-split logic (lines 754-790) with:

```rust
let groups = self.split_groups(df)?;
let df_a = groups.df_a;
let df_b = groups.df_b;
let group_a_name = groups.group_a_name;
// group_b_name not needed here (already in self.reference_group)

if df_a.height() == 0 || df_b.height() == 0 {
    return Err(OaxacaError::InvalidGroupVariable(
        "One group has no data".to_string(),
    ));
}
```

**IMPORTANT:** `group_a_name` is now a `String` (owned by `GroupSplit`), not a `&str`. In the Pooled/Neumark branch (~line 909 in original), the code does `.equal(group_a_name)?`. This must become `.equal(group_a_name.as_str())?` since Polars `equal()` expects `&str`. Search for all uses of `group_a_name` in `run_single_pass` and add `.as_str()` where it is passed to Polars filter/equal methods.

**In `run()`:** The full restructured flow of `run()` should be:

1. Clean dataframe, create dummies (unchanged, lines 1191-1209)
2. **Replace** the inline group-name resolution (lines 1211-1229) AND the bootstrap split (lines 1239-1256) with a single `split_groups()` call:

```rust
// Replace lines 1211-1256 with:
let groups = self.split_groups(&df)?;

let point_estimates =
    self.run_single_pass(&df, &all_dummy_names, &category_counts, &base_categories)?;

// Use owned DataFrames from split_groups for bootstrap sampling
// (replaces the unwrap()-laden filter block at lines 1239-1256)
let df_a_global = groups.df_a;
let df_b_global = groups.df_b;
```

3. Bootstrap loop (unchanged, lines 1258-1281)
4. Bootstrap aggregation (unchanged, lines 1283-1372)
5. **Replace** the n_a/n_b filter block (lines 1386-1399) with:

```rust
n_a: df_a_global.height(),
n_b: df_b_global.height(),
```

This fixes the `.unwrap()` calls in the original bootstrap split by using `?` propagation via `split_groups`.

**In `get_data_matrices()`:** Replace the inline group-split logic (lines 577-616) with:

```rust
let groups = self.split_groups(&df)?;

let (x_a, y_a, _, predictor_names) = self.prepare_data(&groups.df_a, &all_dummy_names, &[])?;
let (x_b, y_b, _, _) = self.prepare_data(&groups.df_b, &all_dummy_names, &[])?;

Ok((x_a, y_a, x_b, y_b, predictor_names))
```

**In `decompose_quantile()`:** Replace the inline group-split logic (lines 1072-1103) with:

```rust
let groups = self.split_groups(&df)?;

let rif_a = calculate_rif(
    groups.df_a.column(&self.outcome)?.as_materialized_series(),
    quantile,
)
.map_err(OaxacaError::PolarsError)?;
let rif_b = calculate_rif(
    groups.df_b.column(&self.outcome)?.as_materialized_series(),
    quantile,
)
.map_err(OaxacaError::PolarsError)?;

let mut df_a_mod = groups.df_a;
df_a_mod.with_column(rif_a)?;
let mut df_b_mod = groups.df_b;
df_b_mod.with_column(rif_b)?;

let df_mod = df_a_mod.vstack(&df_b_mod)?;
```

**In `run()` final n_a/n_b calculation (lines 1386-1399):** Replace the inline filter with:

```rust
n_a: df_a_global.height(),
n_b: df_b_global.height(),
```

Since we already have `df_a_global` and `df_b_global` from the earlier `split_groups` call.

- [ ] **Step 2: Rewrite `lib.rs` as thin hub**

Replace all remaining content of `lib.rs` (everything after the crate doc comment) with:

```rust
mod builder;
mod decomposition;
mod display;
mod error;
mod estimation;
mod inference;
mod math;
mod types;

pub mod akm;
pub mod dfl;
pub mod formula;
pub mod heckman;
pub mod jmp;
pub mod matching;
pub mod quantile_decomposition;

// #[cfg(feature = "python")]
// pub mod python;

pub use akm::{AkmBuilder, AkmResult};
pub use builder::OaxacaBuilder;
pub use decomposition::{BudgetAdjustment, ReferenceCoefficients};
pub use dfl::run_dfl;
pub use error::OaxacaError;
pub use heckman::heckman_two_step;
pub use jmp::decompose_changes;
pub use matching::engine::MatchingEngine;
pub use quantile_decomposition::QuantileDecompositionBuilder;
pub use types::{ComponentResult, DecompositionDetail, OaxacaResults, TwoFoldResults};

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
```

- [ ] **Step 3: Verify compilation of `oaxaca_blinder`**

Run: `cargo build -p oaxaca_blinder 2>&1 | head -30`
Expected: Successful build

- [ ] **Step 4: Run `oaxaca_blinder` tests**

Run: `cargo test -p oaxaca_blinder 2>&1 | tail -10`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add oaxaca_blinder/src/builder.rs oaxaca_blinder/src/lib.rs
git commit -m "refactor: extract OaxacaBuilder into builder.rs with consolidated split_groups()"
```

---

## Chunk 3: Full workspace verification

### Task 6: Verify import paths and full workspace

**Files:** None (verification only)

- [ ] **Step 1: Verify sibling module imports still resolve**

All sibling modules (`jmp.rs`, `dfl.rs`, `heckman.rs`, `quantile_decomposition.rs`, `matching/engine.rs`, `math/diagnostics.rs`, `math/ols.rs`, `math/probit.rs`, `math/logit.rs`, `formula.rs`) import `crate::OaxacaError` and other items via the crate root. Since `lib.rs` re-exports all public types, no import paths need changing. Verify by inspecting any compilation errors in Step 2.

- [ ] **Step 2: Build entire workspace**

Run: `cargo build 2>&1 | tail -10`
Expected: All three crates build successfully (oaxaca_blinder, pay-equity-engine, meridian-mcp)

- [ ] **Step 2: Run all workspace tests**

Run: `cargo test 2>&1 | tail -20`
Expected: All tests pass across all crates

- [ ] **Step 3: Verify no public API changes**

Check that the engine crate and meridian-mcp still compile without changes. If there are errors, they will show up in Step 2. No code changes should be needed in those crates.

- [ ] **Step 4: Verify line counts of new modules**

Run: `wc -l oaxaca_blinder/src/{lib,error,types,estimation,builder,display}.rs`
Expected approximate counts:
- `lib.rs`: ~60 lines
- `error.rs`: ~35 lines
- `types.rs`: ~170 lines
- `estimation.rs`: ~210 lines
- `builder.rs`: ~750 lines
- `display.rs`: ~150 lines

- [ ] **Step 5: Final commit (if any fixups were needed)**

```bash
git add -A
git commit -m "refactor: complete lib.rs decomposition into focused modules"
```
