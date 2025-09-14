# Oaxaca-Blinder Decomposition in Rust

[![crates.io](https://img.shields.io/crates/v/oaxaca_blinder.svg)](https://crates.io/crates/oaxaca_blinder)
[![docs.rs](https://docs.rs/oaxaca_blinder/badge.svg)](https://docs.rs/oaxaca_blinder)

A Rust implementation of the Oaxaca-Blinder decomposition method, designed for performance and ease of use. This library provides tools to decompose the mean difference in an outcome variable between two groups into an "explained" part (due to differences in observable characteristics) and an "unexplained" part (due to differences in the returns to those characteristics).

The library is built on top of the `polars` DataFrame library for data manipulation and `nalgebra` for the underlying linear algebra.

## Features

-   **Two-Fold & Three-Fold Decomposition:** Perform both standard decomposition types.
-   **Detailed Components:** Get a breakdown of contributions for each predictor variable to both the explained and unexplained parts.
-   **Categorical Variable Support:** Automatically handles one-hot encoding and applies coefficient normalization to solve the "identification problem" for detailed decompositions.
-   **Bootstrapped Standard Errors:** Calculate standard errors and confidence intervals for all components using a robust, non-parametric bootstrapping procedure.
-   **Flexible Reference Coefficients:** Choose the reference coefficients (`beta*`) for the decomposition, with support for:
    -   Group A or Group B coefficients.
    -   Coefficients from a pooled model (Neumark's method).
    -   Weighted average of group coefficients (Cotton's method).
-   **Easy-to-Use Builder Pattern:** Configure and run your decomposition with a fluent, chainable API.
-   **Formatted Summary Table:** A built-in `summary()` method prints a clear, publication-style table of the results.

## Decomposition Methodology

The Oaxaca-Blinder method is an "accounting" technique that decomposes the gap in a mean outcome variable between two groups (A and B) into components attributable to group differences in measurable characteristics and differences in the returns to those characteristics.

### The Fundamental Model

The method begins by estimating separate linear regression models for each group:

-   Group A: `Y_A = X_A'β_A + ε_A`
-   Group B: `Y_B = X_B'β_B + ε_B`

A key property of OLS is that the regression line passes through the means, so the mean outcome for each group can be expressed as `E[Y] = E[X]'β`. The total gap is therefore:

`ΔY = E[Y_A] - E[Y_B] = E[X_A]'β_A - E[X_B]'β_B`

### Three-Fold Decomposition

The library implements the three-fold decomposition, which provides a complete algebraic partitioning of the gap:

`ΔY = (E[X_A] - E[X_B])'β_B  +  E[X_B]'(β_A - β_B)  +  (E[X_A] - E[X_B])'(β_A - β_B)`
`     \_______________________/   \___________________/   \________________________________/`
`           Endowments (E)         Coefficients (C)              Interaction (I)`

-   **Endowments (E):** The portion of the gap due to differences in average characteristics (e.g., education, experience), valued at the reference group's returns.
-   **Coefficients (C):** The portion due to differences in the returns to characteristics, valued at the reference group's endowment levels.
-   **Interaction (I):** Accounts for the fact that differences in endowments and coefficients exist simultaneously.

### Two-Fold Decomposition and the Indexing Problem

The more common two-fold decomposition simplifies the three-fold structure by introducing a hypothetical, non-discriminatory reference coefficient vector, `β*`:

`ΔY = (E[X_A] - E[X_B])'β*   +   (E[X_A]'(β_A - β*) + E[X_B]'(β* - β_B))`
`      \__________________/       \___________________________________/`
`            Explained                         Unexplained`

The choice of `β*` is a critical methodological decision known as the **"index number problem"**, as it determines how the interaction term (I) from the three-fold decomposition is allocated between the explained and unexplained components. This library provides several standard choices for `β*` via the `ReferenceCoefficients` enum, allowing you to select the appropriate counterfactual for your analysis.

### Detailed Decomposition and Categorical Variables

A significant challenge arises when decomposing the contribution of individual categorical variables (e.g., "sector", "region"). The estimated coefficients for dummy variables are dependent on which category is chosen as the omitted reference, making a naive detailed decomposition arbitrary.

This library addresses this **identification problem** by implementing a normalization procedure inspired by the work of Yun (2005). Before the detailed decomposition of the unexplained component is calculated, the coefficients for the specified categorical variables are transformed to be invariant to the choice of the base category. This ensures that the detailed results are robust and scientifically valid.

### Statistical Inference

The library uses a non-parametric **bootstrapping** procedure to estimate standard errors and confidence intervals. This resampling method is computationally intensive but robust, as it makes fewer distributional assumptions and correctly accounts for the sampling variation in both the regression coefficients (`β`) and the predictor means (`E[X]`).

## Installation

Add the following to your `Cargo.toml` file:

```toml
[dependencies]
oaxaca_blinder = "0.1.0" # Replace with the latest version
polars = { version = "0.38", features = ["lazy", "csv"] }
```

## How to Use the Library

### 1. Basic Usage

The library is designed around a builder pattern, allowing you to chain methods to configure the decomposition before running it.

```rust
use polars::prelude::*;
use oaxaca_blinder::{OaxacaBuilder, ReferenceCoefficients};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Create a sample DataFrame using Polars
    let df = df!(
        "wage" => &[25.0, 30.0, 35.0, 40.0, 45.0, 20.0, 22.0, 28.0, 32.0, 38.0],
        "education" => &[16.0, 18.0, 14.0, 20.0, 16.0, 12.0, 14.0, 16.0, 12.0, 18.0],
        "experience" => &[10.0, 12.0, 15.0, 20.0, 8.0, 5.0, 8.0, 10.0, 4.0, 14.0],
        "sector" => &["Public", "Private", "Public", "Private", "Public", "Private", "Public", "Private", "Public", "Private"],
        "gender" => &["M", "M", "M", "M", "M", "F", "F", "F", "F", "F"]
    )?;

    // Configure and run the decomposition
    let results = OaxacaBuilder::new(df, "wage", "gender", "F")
        .predictors(&["education", "experience"])
        .categorical_predictors(&["sector"])
        .bootstrap_reps(500)
        .reference_coefficients(ReferenceCoefficients::Pooled)
        .run()?;

    // Print the summary table
    results.summary();

    Ok(())
}
```

### 2. Configuring the Decomposition

The `OaxacaBuilder` provides several methods to customize the analysis:

-   `.new(dataframe, "outcome", "group", "reference_group")`: The entry point to start a new decomposition.
-   `.predictors(&["var1", "var2"])`: Sets the numerical predictor variables.
-   `.categorical_predictors(&["cat1", "cat2"])`: Sets the categorical predictor variables. The library will automatically one-hot encode these and apply normalization for the detailed decomposition.
-   `.bootstrap_reps(n)`: Sets the number of bootstrap replications for calculating standard errors. A higher number (e.g., 500-1000) leads to more stable results but takes longer to compute. Defaults to 100.
-   `.reference_coefficients(ReferenceCoefficients::Variant)`: Sets the `β*` to use. The options are:
    -   `ReferenceCoefficients::GroupA`
    -   `ReferenceCoefficients::GroupB` (Default)
    -   `ReferenceCoefficients::Pooled`
    -   `ReferenceCoefficients::Weighted`
-   `.run()`: Executes the full decomposition and returns a `Result<OaxacaResults, OaxacaError>`.

### 3. Interpreting the Summary Output

The `.summary()` method provides a comprehensive overview of the results.

```text
Oaxaca-Blinder Decomposition Results
========================================
Group A (Advantaged): 5 observations
Group B (Reference):  5 observations
Total Gap: 8.4000

Two-Fold Decomposition
+-------------+----------+-----------+---------+-------------------+
| Component   | Estimate | Std. Err. | p-value | 95% CI            |
+==================================================================+
| explained   | 4.5123   | 1.8921    | 0.4820  | [0.581, 8.132]    |
|-------------+----------+-----------+---------+-------------------|
| unexplained | 3.8877   | 2.4511    | 0.5160  | [-1.211, 8.543]   |
+-------------+----------+-----------+---------+-------------------+

Detailed Decomposition (Explained)
+------------------+--------------+-----------+---------+------------------+
| Variable         | Contribution | Std. Err. | p-value | 95% CI           |
+=========================================================================+
| intercept        | 0.0000       | 0.0000    | NaN     | [0.000, 0.000]   |
|------------------+--------------+-----------+---------+------------------|
| education        | 1.5432       | ...       | ...     | ...              |
|------------------+--------------+-----------+---------+------------------|
| experience       | 2.9691       | ...       | ...     | ...              |
|------------------+--------------+-----------+---------+------------------|
| sector_Public    | ...          | ...       | ...     | ...              |
+------------------+--------------+-----------+---------+------------------+

Detailed Decomposition (Unexplained)
+------------------+--------------+-----------+---------+------------------+
| Variable         | Contribution | Std. Err. | p-value | 95% CI           |
+=========================================================================+
| intercept        | 2.1112       | ...       | ...     | ...              |
|------------------+--------------+-----------+---------+------------------|
| education        | 0.8991       | ...       | ...     | ...              |
|------------------+--------------+-----------+---------+------------------|
| experience       | 0.7774       | ...       | ...     | ...              |
|------------------+--------------+-----------+---------+------------------|
| sector_Public    | ...          | ...       | ...     | ...              |
+------------------+--------------+-----------+---------+------------------+
```

-   **Estimate**: The point estimate for the component's contribution to the gap.
-   **Std. Err.**: The bootstrapped standard error. A smaller value indicates greater precision.
-   **95% CI**: The 95% confidence interval. If this interval does not contain zero, the result is typically considered statistically significant at the 5% level. This is often a more reliable indicator of significance than the p-value in bootstrapped results.

### 4. Programmatic Access to Results

For use in other parts of an application, all results can be accessed directly from the `OaxacaResults` struct returned by `.run()`.

```rust
// ... after running the decomposition in the main function
let results = results; // Assuming `results` is the OaxacaResults object

// --- Accessing Top-Level Information ---
println!("\n--- Programmatic Access ---");
println!("Total Gap: {:.4}", results.total_gap());
println!("Observations in Group A: {}", results.n_a());
println!("Observations in Group B: {}", results.n_b());

// --- Accessing Aggregate Components ---
// The `two_fold()` and `three_fold()` methods return a `DecompositionDetail` struct.
// From there, `.aggregate()` returns a Vec<ComponentResult>.
println!("\nAggregate Two-Fold Components:");
for component in results.two_fold().aggregate() {
    println!(
        "- {}: Estimate={:.4}, CI=[{:.4}, {:.4}]",
        component.name(),
        component.estimate(),
        component.ci_lower(),
        component.ci_upper()
    );
}

// --- Accessing Detailed Components ---
// Use `.detailed()` to get the breakdown by variable.
println!("\nDetailed Unexplained Contributions:");
for component in results.three_fold().detailed() {
    // The three-fold detailed view corresponds to the unexplained part
    println!(
        "- {}: Contribution={:.4}",
        component.name(),
        component.estimate()
    );
}

// Example: Find the specific contribution of 'education' to the explained gap
let explained_details = results.two_fold().detailed();
if let Some(education_explained) = explained_details.iter().find(|c| c.name() == "education") {
    println!("\nExplained contribution of education: {:.4}", education_explained.estimate());
}
```

## License

This project is licensed under the MIT License.
