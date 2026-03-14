# Development Guide

This guide provides instructions for setting up the environment, running tests, and following development guidelines for the `oaxaca-blinder-rs` project.

## 🛠 Prerequisites

- **Rust**: Latest stable version. Install via [rustup](https://rustup.rs/).
- **Cargo-Make**: (Optional but recommended) for workspace automation.

## 🚀 Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/dot-comma-hyphen/oaxaca-blinder-rs.git
   cd oaxaca-blinder-rs
   ```

2. **Build the workspace**:
   ```bash
   cargo build
   ```

## 🧪 Testing

The project uses standard Rust testing patterns.

### Unit Tests
Run unit tests for the core math library:
```bash
cargo test -p oaxaca_blinder
```

### Integration & CLI Tests
The CLI tests use `assert_cmd` to verify the output of the binary:
```bash
cargo test --test cli_tests
```

### Numerical Validation
Check the `verification/` directory for Python scripts that compare Rust results against reference implementations like `statsmodels`.

## 📜 Coding Guidelines

### 1. Error Handling
Always use the custom `OaxacaError` enum in `oaxaca_blinder/src/lib.rs`. Do not use `panic!` in library code. Use `anyhow` only in the MCP server or CLI layers.

### 2. Performance
- Use `Rayon` for any loop that performs statistical simulations (bootstrapping).
- Prefer `nalgebra` for matrix operations rather than manual loops for speed and stability.

### 3. Documentation
All public structures and methods must have doc-comments (`///`). Explain not only *what* the function does, but the *statistical context* if it's a math utility.

### 4. Code Hygiene
- Run `cargo fmt` before committing.
- Run `cargo clippy` to check for common Rust lints.

## 📦 Deployment

### MCP Server
To run the MCP server locally for testing with an LLM:
```bash
cargo run -p meridian-mcp -- --transport stdio
```

### Python Bindings
To build the Python wheels (requires `maturin`):
```bash
maturin build -p oaxaca_blinder --features python
```
