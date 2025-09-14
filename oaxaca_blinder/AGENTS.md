# Agent Instructions for the `oaxaca_blinder` Rust Project

This document provides guidance for AI agents working on this Rust library.

## 1. Project Overview and Structure

This project is a Rust library that implements the Oaxaca-Blinder decomposition method. The goal is to create a robust, user-friendly, and statistically sound tool for researchers.

The crate is structured into several key modules:
-   `src/lib.rs`: Contains the main public API, primarily the `OaxacaBuilder` and the `OaxacaResults` structs. The main `run()` method, which orchestrates the entire analysis, is implemented here.
-   `src/math/ols.rs`: Implements the core Ordinary Least Squares (OLS) regression logic using the `nalgebra` crate. This is the mathematical foundation of the decomposition.
-   `src/decomposition.rs`: Contains the functions that calculate the aggregate (two-fold, three-fold) and detailed components of the decomposition.
-   `src/inference.rs`: Contains the logic for statistical inference, specifically the bootstrapping engine and functions to calculate standard errors, p-values, and confidence intervals.
-   `tests/`: Contains integration tests. The `integration_test.rs` file provides a good example of how to use the library from end to end.

## 2. Development and Testing

**The single most important command to run is:**

```bash
cargo test --manifest-path oaxaca_blinder/Cargo.toml
```

**IMPORTANT NOTES:**
-   You **must** use `--manifest-path oaxaca_blinder/Cargo.toml` when running `cargo` commands from the root directory (`/app`). Do not use `--workspace` or try to `cd` into the directory, as these have proven unreliable in this environment.
-   Make small, incremental changes. The `polars` API can be complex and has caused compilation issues in the past. It is highly recommended to run `cargo check` or `cargo test` after every significant change to `lib.rs`.
-   All new functionality should be accompanied by unit or integration tests.

## 3. Current Limitations and Future Work

This library is functional but has several key limitations that are priorities for future development.

### High Priority:
1.  **Categorical Variable Support**: This is the most critical missing feature. The library currently only supports **numerical predictors**. To add this, you will need to:
    *   Modify the `OaxacaBuilder` to accept a list of categorical variables.
    *   In the `prepare_data` function in `lib.rs`, use the `polars` API to create dummy variables for these columns. **Be careful**, as this was the source of many compilation errors. Research the `DataFrame::to_dummies` or `DataFrame::columns_to_dummies` methods thoroughly.
    *   Re-implement the coefficient normalization logic (previously in `decomposition.rs`) to correctly handle the detailed decomposition of the unexplained component for these dummy variables, as described in the research paper.

2.  **Configurable Reference Model**: The two-fold decomposition currently defaults to using the disadvantaged group's coefficients as the reference. A key feature would be to allow the user to select other models (e.g., using the advantaged group's coefficients, or a pooled model). This would likely involve:
    *   Adding a method to `OaxacaBuilder` like `.reference_model("pooled")`.
    *   Implementing the logic to calculate the `beta*` vector based on the user's choice (see Section 3 of the research paper).

### Lower Priority:
*   **Weighted Least Squares (WLS)**: Add support for survey weights.
*   **Delta Method for Standard Errors**: Implement the Delta Method as an alternative, faster option for calculating standard errors.
