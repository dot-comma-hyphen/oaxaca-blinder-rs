# Oaxaca-Blinder Decomposition in Rust

A Rust implementation of the Oaxaca-Blinder decomposition method, designed for accuracy and ease of use.

This library provides tools to decompose the mean difference in an outcome variable between two groups into an "explained" part (due to differences in observable characteristics) and an "unexplained" part (due to differences in the returns to those characteristics).

## Current Features

*   Calculates the two-fold (explained/unexplained) and three-fold (endowments/coefficients/interaction) aggregate decompositions.
*   Provides a detailed decomposition, breaking down the gap by individual predictor variables.
*   Uses bootstrapping to calculate standard errors, p-values, and confidence intervals for all components, enabling robust statistical inference.

## Limitations

*   **Numerical Predictors Only**: The current version of the library only supports numerical predictor variables. Support for automatic dummy variable creation for categorical predictors is a priority for a future release.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
oaxaca_blinder = { git = "https://github.com/your-repo/oaxaca_blinder" } # Replace with actual repo URL
```

## Usage

Here is a basic example of how to use the library:

```rust
use polars::prelude::*;
use oaxaca_blinder::{OaxacaBuilder, OaxacaError};

fn run_decomposition() -> Result<(), OaxacaError> {
    // 1. Create a sample DataFrame
    let df = df!(
        "wage" => &[10.0, 12.0, 11.0, 13.0, 15.0, 20.0, 22.0, 21.0, 23.0, 25.0],
        "education" => &[12.0, 16.0, 14.0, 16.0, 18.0, 12.0, 16.0, 14.0, 16.0, 18.0],
        "gender" => &["F", "F", "F", "F", "F", "M", "M", "M", "M", "M"]
    ).unwrap();

    // 2. Set up and run the decomposition
    // Here, we are analyzing the 'wage' gap between 'M' and 'F' gender groups,
    // using 'F' as the reference (disadvantaged) group.
    let results = OaxacaBuilder::new(df, "wage", "gender", "F")
        .predictors(&["education"])
        .bootstrap_reps(500) // Use a higher number for real analysis
        .run()?;

    // 3. Print the summary of results
    results.summary();

    Ok(())
}
```
