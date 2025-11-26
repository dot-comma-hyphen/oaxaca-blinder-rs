## Project Overview

This project is a Rust library and command-line tool for performing Oaxaca-Blinder decomposition. The library is named `oaxaca_blinder` and is built using the `polars` DataFrame library for data manipulation and `nalgebra` for linear algebra.

The core functionality includes:
-   **Two-Fold & Three-Fold Decomposition:** Supports both standard decomposition types.
-   **Quantile Decomposition:** Provides functionality for quantile regression decomposition.
-   **Categorical Variable Support:** Automatically handles one-hot encoding and applies coefficient normalization.
-   **Bootstrapped Standard Errors:** Calculates standard errors and confidence intervals using bootstrapping.
-   **Flexible Reference Coefficients:** Allows choosing the reference coefficients for the decomposition.
-   **Command-Line Interface:** Includes a CLI for running decompositions from the terminal.

## Building and Running

The project is a standard Rust project and can be built using Cargo.

### Building the Library

To build the library, run the following command from the root directory:

```bash
cargo build
```

### Running the CLI

The project includes a command-line interface. To run the CLI, you can use `cargo run` with the `--bin` flag, followed by the desired arguments.

**Example:**

```bash
cargo run --bin oaxaca-cli -- \
    --data oaxaca_blinder/tests/data/wage.csv \
    --outcome wage \
    --group gender \
    --reference F \
    --predictors education,experience \
    --categorical sector
```

### Running Tests

To run the tests, use the following command:

```bash
cargo test
```

## Development Conventions

-   **Code Style:** The code follows standard Rust conventions and is formatted with `rustfmt`.
-   **Error Handling:** The library uses a custom `OaxacaError` enum for error handling.
-   **Dependencies:** The project uses `polars` for DataFrames, `nalgebra` for linear algebra, and `clap` for the CLI.
-   **Modularity:** The code is organized into modules for `math`, `decomposition`, and `inference`.
