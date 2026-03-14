# Task Decomposition: `oaxaca-blinder-rs` Audit Remediation

This document decomposes the [AUDIT_REPORT.md](file:///home/deji/Documents/Code/comp-audit-suite/oaxaca-blinder-rs/docs/AUDIT_REPORT.md) into independent, parallel workstreams sized for a Jules asynchronous agent.

## Execution Overview
- **Lanes**: 4
- **Critical Path**: Safety (Lane 4) + Auth (Lane 2) -> Math Core (Lane 1) -> Features (Lane 3)
- **Parallelism**: Lane 1, 2, and 4 can start immediately. Lane 3 depends on core stability.

---

## Lane 1: Mathematical Engine (Correctness & Stability)
**Domain**: `oaxaca_blinder` library core algorithms.

| ID | Task | Size | Findings | Files |
|----|------|------|----------|-------|
| T1.1 | **Panic-Proof Algorithms** | Med | C2, H11 | `akm.rs`, `heckman.rs`, `lib.rs` |
| T1.2 | **Robust Input Handling** | Med | H1, H10, H12 | `akm.rs`, `quantile_decomposition.rs`, `inference.rs`, `math/kde.rs`, `engine/analysis.rs` |
| T1.3 | **Statistical Interpolation & Clamping**| Med | H3, H4, H5 | `math/rif.rs`, `math/logit.rs`, `heckman.rs`, `engine/analysis.rs` |
| T1.4 | **Weighted Stats & OLS Fixes** | Small | M1, M2, M10 | `ols.rs` |
| T1.5 | **Numerical Optimization Cleanup** | Med | M3, M5, M7, L6 | `logit.rs`, `akm.rs`, `probit.rs`, `quantile_decomposition.rs` |

---

## Lane 2: MCP Infrastructure (Server & Security)
**Domain**: `meridian-mcp` service and deployment.

| ID | Task | Size | Findings | Files |
|----|------|------|----------|-------|
| T2.1 | **Auth & Access Control Fixes** | Med | C3, C4, H9 | `meridian-mcp/src/main.rs`, `engine/src/access.rs`, `Dockerfile`, `docker-compose.yml` |
| T2.2 | **Async Runtime & Locking Safety** | Med | H7, M8 | `meridian-mcp/src/main.rs` |
| T2.3 | **Resource Hardening & SSE** | Large | H6, H8, H13, M9, L5 | `meridian-mcp/src/main.rs` |
| T2.4 | **MCP Protocol Compliance** | Med | Protocol Violations | `meridian-mcp/src/main.rs` |

---

## Lane 3: Pay Equity Engine (Product & Metrics)
**Domain**: `engine` crate higher-level analysis.

| ID | Task | Size | Findings | Files |
|----|------|------|----------|-------|
| T3.1 | **Singularity & Metric Calculation** | Med | H2, H14 | `engine/src/defensibility.rs`, `engine/src/analysis.rs` |
| T3.2 | **Index Safety & Bound Checks** | Small | M11 | `engine/src/analysis.rs` |
| T3.3 | **API Consistency & Cleanup** | Small | M6, M12, L1-L4 | `lib.rs`, `engine/src/`, `matching/distance.rs`, `meridian-mcp/main.rs` |

---

## Lane 4: FFI Safety (High Priority)
**Domain**: Python/Rust boundary.

| ID | Task | Size | Findings | Files |
|----|------|------|----------|-------|
| T4.1 | **Remediate Memory Safety Violation** | Med | C1 | `oaxaca_blinder/src/python.rs` |

---

## Task Templates for Jules

### Task T1.1: Panic-Proof Algorithms
**Lane**: Math Engine
**Findings**: C2, H11
**Impacted Files**: `oaxaca_blinder/src/akm.rs`, `oaxaca_blinder/src/heckman.rs`
**Acceptance Criteria**:
- Replace all `.unwrap()` and `.expect()` calls in `akm.rs` and `heckman.rs` with safe error propagation using `Result<_, AkmError>`.
- Add unit tests verifying that null or missing firm/worker IDs return a graceful error instead of crashing.
- Ensure `oaxaca_blinder/src/lib.rs` also propagates these errors correctly.

### Task T4.1: Remediate Memory Safety Violation (CRITICAL)
**Lane**: FFI Safety
**Findings**: C1
**Impacted Files**: `oaxaca_blinder/src/python.rs`
**Acceptance Criteria**:
- Remove the unsafe pointer cast in `python.rs:230-236`.
- Implement a safe data transfer mechanism between Python Polars and Rust Polars (e.g., via Arrow IPC stream or pinning versions).
- Verify with `cargo test` and `pytest`.
