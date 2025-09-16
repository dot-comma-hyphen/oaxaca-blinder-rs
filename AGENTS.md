# Agent Instructions

This document provides instructions for AI agents working with the `oaxaca_blinder` codebase.

## Project Overview

This repository contains a Rust library called `oaxaca_blinder` for performing Oaxaca-Blinder decomposition, a statistical method used in econometrics. The library is built on top of `polars` for data manipulation and `nalgebra` for linear algebra.

The main source code for the library is located in `oaxaca_blinder/src/lib.rs`. Integration tests are in `oaxaca_blinder/tests/integration_test.rs`.

## Development Environment & Testing

The development environment for this project appears to be resource-constrained. When running the full test suite with `cargo test`, you may encounter linker errors (`ld terminated with signal 9 [Killed]`), which indicates that the process is running out of memory.

### Recommended Testing Procedure

To work around the memory constraints, it is recommended to run the library's unit tests and integration tests separately.

1.  **Run Unit Tests:** To run only the unit tests, use the following command from the root of the repository:
    ```bash
    cargo test -p oaxaca_blinder --lib
    ```
    This command is much less memory-intensive and should complete successfully.

2.  **Run Integration Tests:** If you need to run the integration tests, it is recommended to run them one at a time to avoid memory issues. You can do this with the following command:
    ```bash
    cargo test -p oaxaca_blinder --test <test_name>
    ```
    Replace `<test_name>` with the name of the test you want to run (e.g., `integration_test`).

By following this procedure, you should be able to verify your changes without running into memory-related issues.
