# Project Architecture

## Key Components

The project is organized as a Rust workspace with two primary members:

### 1. `oaxaca_blinder`
This is the core library implementing various statistical decomposition methods.
- **Purpose**: To perform econometric decompositions on wage gaps or other outcome differentials.
- **Key Modules**:
    - `decomposition.rs`: Standard Oaxaca-Blinder decomposition.
    - `quantile_decomposition.rs`: RIF-Regression based quantile decomposition (Firpo, Fortin, Lemieux).
    - `jpm.rs`: Juhn-Murphy-Pierce decomposition.
    - `dfl.rs`: DiNardo-Fortin-Lemieux (reweighting) decomposition.
    - `matching/`: Propensity score matching logic.
    - `akm.rs`: Abowd-Kramarz-Margolis (AKM) high-dimensional fixed effects.
- **Interface**:
    - **CLI**: `src/main.rs` exposes a command-line tool `oaxaca-cli`.
    - **Python**: `python.rs` and `pyo3` provide Python bindings.

### 2. `optimization_engine`
A separate crate focused on optimization problems related to pay equity and wage scaling.
- **Purpose**: To solve linear programming constraints for wage adjustments.
- **Key Modules**:
    - `pay_equity.rs`: Logic for correcting pay inequities.
    - `wage_scale.rs`: Designing compliant wage scales.
    - `engine.rs`: Core optimization engine.

## Data Flow

1.  **Input**: Data is ingested via **Polars DataFrames** (CSV, Parquet, etc.).
2.  **Processing (Decomposition)**:
    -   The `oaxaca_blinder` crate processes these frames.
    -   Linear algebra operations are handled by **Nalgebra**.
    -   High-performance computing uses **Rayon** for parallelism.
3.  **Processing (Optimization)**:
    -   The `optimization_engine` takes constraints and objectives.
    -   It utilizes **GoodLP** (with HiGHS solver) to find optimal wage allocations.
4.  **Output**:
    -   Results are returned as structs (e.g., `OaxacaBlinderResult`) or printed to stdout (CLI).
    -   Python users receive standard Python objects/dataframes.

## Tech Stack

-   **Language**: Rust (Edition 2021)
-   **Data Processing**: `polars` (Lazy evaluation, high performance)
-   **Math/Stats**:
    -   `nalgebra`: Linear algebra.
    -   `statrs`: Statistical distributions.
    -   `clarabel`: Convex optimization solver.
-   **Optimization**: `good_lp` (Linear Programming interface).
-   **Interop**: `pyo3` and `pyo3-polars` for Python bindings.
-   **CLI**: `clap`.
