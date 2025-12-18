pub mod engine;
pub mod merit_matrix;
#[cfg(feature = "pay_equity")]
pub mod pay_equity;
pub mod wage_scale;

pub use engine::{OptimizationEngine, OptimizationResult, ProblemDefinition};
