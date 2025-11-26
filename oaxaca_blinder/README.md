# Oaxaca-Blinder Decomposition in Rust

[![crates.io](https://img.shields.io/crates/v/oaxaca_blinder.svg)](https://crates.io/crates/oaxaca_blinder)
[![docs.rs](https://docs.rs/oaxaca_blinder/badge.svg)](https://docs.rs/oaxaca_blinder)

A high-performance Rust library for performing Oaxaca-Blinder decomposition, designed for economists, data scientists, and HR analysts. It decomposes the gap in an outcome variable (like wage) between two groups into "explained" (characteristics) and "unexplained" (discrimination/coefficients) components.

## 🚀 Feature Support

| Feature | Support |
| :--- | :---: |
| **OLS Mean Decomposition** | ✅ |
| **Quantile Decomposition (Machado-Mata)** | ✅ |
| **Quantile Decomposition (RIF Regression)** | ✅ |
| **Categorical Normalization (Yun)** | ✅ |
| **Bootstrapped Standard Errors** | ✅ |
| **Budget Optimization Solver** | ✅ |
| **JMP Decomposition (Time Series)** | ✅ |
| **DFL Reweighting (Counterfactuals)** | ✅ |
| **Sample Weights** | ❌ |

---

## 🖥️ Command Line Interface (CLI)

Don't want to write Rust code? You can use the `oaxaca-cli` tool directly from your terminal to analyze CSV files.

### Installation

```bash
cargo install oaxaca_blinder --features cli
```

### Usage

```bash
oaxaca-cli --data wage.csv --outcome wage --group gender --reference F \
    --predictors education experience --categorical sector
```

Supports both `--analysis-type mean` (default) and `--analysis-type quantile`.

---

## ⚡ Quick Start (Rust)

Add to `Cargo.toml`:

```toml
[dependencies]
oaxaca_blinder = "0.1.0"
polars = { version = "0.38", features = ["lazy", "csv"] }
```

### Basic OLS Decomposition

```rust
use polars::prelude::*;
use oaxaca_blinder::{OaxacaBuilder, ReferenceCoefficients};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let df = df!(
        "wage" => &[25.0, 30.0, 35.0, 40.0, 45.0, 20.0, 22.0, 28.0, 32.0, 38.0],
        "education" => &[16.0, 18.0, 14.0, 20.0, 16.0, 12.0, 14.0, 16.0, 12.0, 18.0],
        "gender" => &["M", "M", "M", "M", "M", "F", "F", "F", "F", "F"]
    )?;

    let results = OaxacaBuilder::new(df, "wage", "gender", "F")
        .predictors(&["education"])
        .reference_coefficients(ReferenceCoefficients::Pooled)
        .run()?;

    results.summary();
    Ok(())
}
```

---

## 💰 Policy Simulation: Budget Optimization

**"The Cheapest Fix"**

This unique feature is designed for HR analytics. It answers: *"Given a limited budget, how can we reduce the pay gap as much as possible?"*

It identifies individuals in the disadvantaged group with the largest negative unexplained residuals (i.e., the most "underpaid" relative to their qualifications) and calculates the optimal raises.

```rust
// Scenario: You have $200,000 to reduce the gap to 5%
let adjustments = results.optimize_budget(200_000.0, 0.05);

for adj in adjustments {
    println!("Give ${:.2} raise to employee #{}", adj.adjustment, adj.index);
}
```

---

## 📊 Quantile Decomposition Strategies

The library supports two robust methods for decomposing the wage gap across the distribution:

| Method | Best For... | Builder |
| :--- | :--- | :--- |
| **Machado-Mata (Simulation)** | Constructing full counterfactual distributions and "glass ceiling" analysis. | `QuantileDecompositionBuilder` |
| **RIF Regression (Analytical)** | Fast, detailed decomposition of specific quantiles (e.g., "Why is the 90th percentile gap so large?"). | `OaxacaBuilder::decompose_quantile(0.9)` |

### Example: RIF Decomposition

```rust
// Fast decomposition of the 90th percentile gap
let results = OaxacaBuilder::new(df, "wage", "gender", "F")
    .predictors(&["education", "experience"])
    .decompose_quantile(0.9)?;
```

---

## 📈 Visualizing DFL Reweighting

**DiNardo-Fortin-Lemieux (DFL)** reweighting is a non-parametric alternative that allows you to visualize what the wage distribution of Group B would look like if they had the characteristics of Group A.

The `run_dfl` function returns density vectors perfect for plotting in Python (matplotlib) or Rust (plotters).

```rust
use oaxaca_blinder::run_dfl;

let dfl = run_dfl(&df, "wage", "gender", "F", &["education", "experience"])?;

// dfl.grid                   <- X-axis (Wage levels)
// dfl.density_a              <- Actual Group A Density
// dfl.density_b              <- Actual Group B Density
// dfl.density_b_counterfactual <- "What B would earn with A's characteristics"
```

*Tip: Plot `density_b` vs `density_b_counterfactual` to visualize the "explained" gap.*

---

## ⏱️ Benchmarks

Designed for performance, utilizing Rust's speed and parallelization (Rayon) for bootstrapping.

**Performance vs Python (`statsmodels`)**
*Dataset: 100k rows, 10 predictors*

| Method | Rust (`oaxaca_blinder`) | Python (`statsmodels`) |
| :--- | :--- | :--- |
| **Raw Decomposition** | **0.16s** 🚀 | 0.29s |
| **With 500 Bootstrap Reps** | **3.16s** 🚀 | ~150s (est.) |

*Rust's parallelized bootstrapping makes standard error estimation orders of magnitude faster.*

### Performance Comparison (Real-World Data)

**Rust Implementation**: 4.32 seconds
**R Implementation**: ~1.99 minutes (119.4 seconds)
**Speedup**: ~27.6x faster

### Results Validation

The results are nearly identical, confirming the correctness of the Rust implementation:

| Metric | Rust Result | R Result | Difference |
| :--- | :--- | :--- | :--- |
| **Total Gap** | 3.2101 | 3.210084 | ~0.000016 |
| **Explained** | -0.8097 | -0.8097 | Exact match (4 decimals) |
| **Unexplained** | 4.0198 | 4.0198 | Exact match (4 decimals) |

**Detailed Components (Selected):**

| Variable | Component | Rust Contribution | R Contribution |
| :--- | :--- | :--- | :--- |
| **Education** | Explained | -0.7852 | -0.7852 |
| **Experience** | Unexplained | 2.0670 | 2.0670 |
| **Intercept** | Unexplained | 2.1405 | 2.1405 |

*The minor differences in standard errors (e.g., Rust SE for unexplained is 0.0314 vs R SE 0.0311) are expected due to the random nature of bootstrapping.*

---

## 📚 Theory & Methodology

<details>
<summary><strong>Deep Dive: The Indexing Problem & Reference Groups</strong></summary>

The choice of `β*` determines how the interaction term is allocated between the explained and unexplained components. This library supports:

-   **Group A / Group B:** Uses one group's coefficients as the reference.
-   **Pooled (Neumark):** Uses coefficients from a pooled regression of both groups.
-   **Weighted (Cotton):** Uses a weighted average of the coefficients.

</details>

<details>
<summary><strong>Deep Dive: Categorical Variables (Yun Normalization)</strong></summary>

Standard detailed decomposition is sensitive to the choice of the omitted base category for dummy variables. This library implements **Yun's normalization**, which transforms coefficients to be invariant to the base category choice, ensuring robust detailed results.

</details>

<details>
<summary><strong>Deep Dive: JMP Decomposition</strong></summary>

The **Juhn-Murphy-Pierce (JMP)** method decomposes the *change* in the gap over time into:
1.  **Quantity Effect:** Changes in observable characteristics.
2.  **Price Effect:** Changes in returns to characteristics.
3.  **Gap Effect:** Changes in unobserved residual inequality.

</details>

---

## License

This project is licensed under the MIT License.
