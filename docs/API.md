# API Reference

This document summarizes the primary interfaces and schemas for the `oaxaca-blinder-rs` components.

## 📦 Core Library: `oaxaca_blinder`

### `OaxacaBuilder`
The main entry point for mean-based decompositions.

| Method | Description |
| :--- | :--- |
| `new(df, outcome, group)` | Initialize with Polars DataFrame and variables. |
| `predictors(&[&str])` | Add continuous predictors. |
| `categorical_predictors(&[&str])` | Add categorical variables (auto-dummy encoded). |
| `reference_coefficients(ReferenceCoefficients)` | Set target coefficients (Pooled, GroupA, GroupB). |
| `bootstrap_reps(usize)` | Number of iterations for standard error estimation. |
| `run()` | Execute the decomposition. |

### `QuantileDecompositionBuilder`
Used for Machado-Mata/RIF quantile decompositions.

| Method | Description |
| :--- | :--- |
| `quantile(f64)` | The target quantile (e.g., 0.5 for median). |
| `simulations(usize)` | Number of simulations for Machado-Mata method. |

---

## 🤖 Meridian MCP Server Tools

The MCP server exposes the following tools for use by AI agents.

### `forensic_decomposition`
Performs a full Oaxaca-Blinder pay equity audit.
- **Parameters**: 
    - `csv_content`: String data.
    - `outcome_variable`, `group_variable`, `reference_group`: Field names.
    - `predictors`, `categorical_predictors`: List of strings.
    - `three_fold`: Boolean (standard 2-fold if false).

### `simulate_remediation`
Calculates required wage adjustments to close identified gaps.
- **Parameters**:
    - `budget`: Maximum available budget.
    - `strategy`: "Greedy" (max gap first) or "Equitable" (shared distribution).
    - `target`: "Reference" or "Pooled" coefficients.

### `generate_efficient_frontier`
Simulates the trade-off between Remediation Budget and remaining Statistical Significance of the gap.

---

## 🛠 Extension Engines

### AKM (Fixed Effects)
Available via `AkmBuilder`. Requires high-density longitudinal data with `worker_id` and `firm_id`.

### Matching Engine
Supports Nearest Neighbor and Propensity Score Matching (PSM) for causal audit workflows.
