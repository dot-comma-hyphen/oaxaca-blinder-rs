# Design: Refactor `oaxaca_blinder/src/lib.rs` into focused modules

**Date:** 2026-03-14
**Status:** Approved
**Approach:** Module-per-responsibility (Approach A)

## Problem

`lib.rs` is 1,755 lines containing error types, result types, estimation logic, the builder struct with all methods, display/export formatters, and optimization logic. This makes the file difficult to navigate and reason about.

## Goals

1. Split `lib.rs` into focused modules with clear single responsibilities
2. Consolidate duplicated group-splitting logic (appears 5 times) into a single helper
3. Preserve the public API surface exactly -- no breaking changes for downstream consumers
4. All existing tests must pass without modification

## New Module Structure

### `lib.rs` (~60 lines)

Thin hub: module declarations, re-exports, and the existing crate-level `//!` doc comment (lines 1-54 of current lib.rs, preserved as-is).

```rust
//! A Rust implementation of the Oaxaca-Blinder decomposition method.
//! ... (existing doc comment preserved)

mod error;
mod types;
mod estimation;
mod builder;
mod display;
mod decomposition;
mod inference;
mod math;
pub mod quantile_decomposition;
pub mod jmp;
pub mod dfl;
pub mod formula;
pub mod heckman;
pub mod akm;
pub mod matching;

// #[cfg(feature = "python")]
// pub mod python;

pub use error::OaxacaError;
pub use types::{ComponentResult, DecompositionDetail, TwoFoldResults, OaxacaResults};
pub use builder::OaxacaBuilder;
pub use decomposition::{BudgetAdjustment, ReferenceCoefficients};
pub use quantile_decomposition::QuantileDecompositionBuilder;
pub use jmp::decompose_changes;
pub use dfl::run_dfl;
pub use heckman::heckman_two_step;
pub use akm::{AkmBuilder, AkmResult};
pub use matching::engine::MatchingEngine;
```

**Note:** `python.rs` exists but is commented out in `lib.rs`. The commented-out `pub mod python` line is preserved. When re-enabled, `python.rs` imports `crate::{ComponentResult, OaxacaResults, TwoFoldResults, OaxacaBuilder, run_dfl, MatchingEngine}` -- all of these are covered by the re-exports above, so no additional work is needed for Python bindings compatibility.

**Note:** `akm.rs` defines its own `AkmError` type rather than using `OaxacaError`. This pattern remains unchanged.

### `error.rs` (~35 lines)

- `OaxacaError` enum
- `From<PolarsError>` impl
- `Display` impl
- `Error` impl

### `types.rs` (~170 lines)

Result/output structs and domain accessors:

- `ComponentResult` -- single decomposition component with name, estimate, std_err, t_stat, p_value, ci_lower, ci_upper
- `DecompositionDetail` -- aggregate + detailed vec
- `TwoFoldResults` -- two-fold aggregate with explained/unexplained/selection detail vecs
- `OaxacaResults` -- top-level results struct (total_gap, two_fold, three_fold, n_a, n_b, residuals, xa_mean, xb_mean, beta_star)
- Accessor methods on `OaxacaResults`: `explained()`, `unexplained()`, `get_summary_table()`, `get_detailed_table()`
- `optimize_budget()` stays on `OaxacaResults` as a pure method on results data

**Dependency:** `types.rs` imports `crate::decomposition::BudgetAdjustment` (used by `optimize_budget()` return type). This is a one-way dependency: `types.rs` -> `decomposition.rs`.

### `estimation.rs` (~210 lines)

Internal estimation machinery (all `pub(crate)`):

- `EstimationContext<'a>` -- borrowed references to split group data
- `EstimationResult` -- beta vectors, means, residuals, optional Heckman selection fields
- `Estimator` trait with `fn estimate(&self, ctx: &EstimationContext) -> Result<EstimationResult, OaxacaError>`
- `OlsEstimator` struct + `Estimator` impl
- `HeckmanEstimator` struct + `Estimator` impl + helper methods (`prepare_selection_data`, `filter_outcome_rows`)

**`SinglePassResult` is NOT in this module** -- it belongs in `builder.rs` (see below).

### `builder.rs` (~750 lines)

`OaxacaBuilder` struct and all methods. Also owns `SinglePassResult` and `GroupSplit`.

**Internal types (pub(crate)):**
- `SinglePassResult` -- three-fold, two-fold, detailed components, residuals, means, beta_star. Carries `#[allow(dead_code)]` from the original. Has fields of type `ThreeFoldDecomposition`, `TwoFoldDecomposition`, `DetailedComponent` (from `decomposition.rs`) and `DVector<f64>` (from nalgebra).
- `GroupSplit` -- owned DataFrames and group name strings (no lifetime parameter)

**Constructors:**
- `new()` -- standard constructor
- `from_formula()` -- R-style formula constructor

**Configuration (builder pattern):**
- `predictors()`, `categorical_predictors()`, `bootstrap_reps()`, `reference_coefficients()`, `normalize()`, `weights()`, `heckman_selection()`

**Public entry points:**
- `run()` -- main decomposition with bootstrap
- `decompose_quantile()` -- RIF-regression quantile decomposition
- `get_data_matrices()` -- expose internal matrices for engine crate

**Internal helpers:**
- `split_groups()` -- **new consolidated helper** (see below)
- `prepare_data()` -- build design matrix + outcome vector from DataFrame
- `create_dummies_manual()` -- one-hot encode categorical variables
- `clean_dataframe()` -- drop nulls from relevant columns
- `run_single_pass()` -- single estimation pass (used by both point estimate and bootstrap)
- `process_detailed_components()` -- aggregate bootstrap results for detailed components

**Key imports for `builder.rs`:**
```rust
use crate::error::OaxacaError;
use crate::types::{ComponentResult, DecompositionDetail, TwoFoldResults, OaxacaResults};
use crate::estimation::{EstimationContext, EstimationResult, Estimator, OlsEstimator, HeckmanEstimator};
use crate::decomposition::{
    detailed_decomposition, three_fold_decomposition, two_fold_decomposition,
    DetailedComponent, ThreeFoldDecomposition, TwoFoldDecomposition, ReferenceCoefficients,
};
use crate::inference::bootstrap_stats;
use crate::math::normalization::normalize_categorical_coefficients;
use crate::math::ols::ols;
use crate::math::rif::calculate_rif;
use crate::formula::Formula;
use crate::heckman::heckman_two_step;
```

### `display.rs` (~150 lines)

Separate `impl OaxacaResults` block for output formatting:

- `summary()` -- console table output (behind `#[cfg(feature = "display")]`)
- `to_latex()` -- LaTeX table fragment
- `to_markdown()` -- Markdown table
- `to_json()` -- JSON via serde

**Key imports for `display.rs`:**
```rust
use crate::types::OaxacaResults;
#[cfg(feature = "display")]
use comfy_table::{Cell, Table};
```

## Key Consolidation: `split_groups()`

The group-splitting logic currently appears in 5 locations: `run_single_pass()`, the bootstrap loop inside `run()`, `run()` group-name resolution, `get_data_matrices()`, and `decompose_quantile()`. All are consolidated into a single helper:

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
            .unwrap_or(&self.reference_group)
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

### `split_groups()` usage per call site

| Call site | How it uses `split_groups()` |
|---|---|
| `run_single_pass()` | Direct: `let groups = self.split_groups(df)?;` replaces inline logic |
| `run()` point estimate | Calls `run_single_pass()` which calls `split_groups()` internally |
| `run()` bootstrap loop | Direct: `let groups = self.split_groups(&df)?;` to get `df_a_global`/`df_b_global` before the parallel loop. Also replaces the `.unwrap()` calls with proper `?` propagation. |
| `get_data_matrices()` | Direct: `let groups = self.split_groups(&df)?;` replaces inline logic |
| `decompose_quantile()` | **Special case:** RIF substitution happens before splitting. The method first cleans data, then uses `split_groups()` on the cleaned df to get group DataFrames, then calculates RIF per group and replaces the outcome column, then vstacks and delegates to a child `OaxacaBuilder::new(...).run()`. The split is used for the RIF calculation step, not for final decomposition. |

## Dependency Graph

```
lib.rs (hub)
  ├── error.rs          (no internal deps)
  ├── types.rs          -> error.rs, decomposition.rs (for BudgetAdjustment)
  ├── estimation.rs     -> error.rs, math/ (ols, normalization), heckman.rs, decomposition.rs
  ├── builder.rs        -> error.rs, types.rs, estimation.rs, decomposition.rs,
  │                        inference.rs, math/ (ols, rif, normalization), formula.rs, heckman.rs
  ├── display.rs        -> types.rs
  ├── decomposition.rs  -> error.rs (no internal deps otherwise)
  ├── inference.rs      (no internal deps)
  └── math/             -> error.rs
```

No circular dependencies. All arrows point "downward" from builder -> estimation -> primitives.

## Visibility Rules

- **Public API** (`pub`): `OaxacaError`, `OaxacaBuilder`, `OaxacaResults`, `ComponentResult`, `DecompositionDetail`, `TwoFoldResults`, `BudgetAdjustment`, `ReferenceCoefficients` -- unchanged
- **Crate-internal** (`pub(crate)`): `EstimationResult`, `EstimationContext`, `Estimator`, `OlsEstimator`, `HeckmanEstimator` (in `estimation.rs`); `SinglePassResult`, `GroupSplit` (in `builder.rs`)

## Migration Steps

1. Create `error.rs`, move `OaxacaError` + impls
2. Create `types.rs`, move result structs + accessors + `optimize_budget` (import `BudgetAdjustment` from `decomposition`)
3. Create `estimation.rs`, move `Estimator` trait, `OlsEstimator`, `HeckmanEstimator`, `EstimationContext`, `EstimationResult`
4. Create `builder.rs`, move `OaxacaBuilder` + all methods + `SinglePassResult` (with `#[allow(dead_code)]`), add `GroupSplit` and `split_groups()`, refactor all 5 call sites
5. Create `display.rs`, move formatting methods
6. Rewrite `lib.rs` as thin re-export hub (preserve crate-level `//!` doc comment and commented-out `python` module)
7. Fix all `use` paths across the crate -- key files to check:
   - `quantile_decomposition.rs` (imports `crate::inference::bootstrap_stats`, `crate::OaxacaBuilder`, etc.)
   - `jmp.rs`, `dfl.rs`, `akm.rs`, `heckman.rs` (may import `crate::OaxacaError`)
   - `matching/` modules
   - `python.rs` (commented out, but verify re-exports cover its imports for future re-enablement)
8. Run `cargo test` -- all existing tests must pass
9. Run `cargo build` for the full workspace (engine, meridian-mcp depend on this crate)

## What Does NOT Change

- Public API surface (all re-exports preserved)
- Behavior of any decomposition method
- Test files
- Other modules (`decomposition.rs`, `inference.rs`, `math/`, `jmp.rs`, `dfl.rs`, `heckman.rs`, `akm.rs`, `matching/`, `quantile_decomposition.rs`, `formula.rs`)
- The `engine` and `meridian-mcp` crates (they import via `oaxaca_blinder::` which stays stable)
- `akm.rs` using its own `AkmError` type (not `OaxacaError`)
- `python.rs` remaining commented out
