use crate::OaxacaError;
use nalgebra::{DMatrix, DVector};
use crate::math::probit::probit;
use crate::math::ols::ols;
use statrs::distribution::{Normal, Continuous, ContinuousCDF};

/// Represents the results of a Heckman Two-Step estimation.
#[derive(Debug)]
pub struct HeckmanResult {
    /// Coefficients from the selection (Probit) equation.
    pub selection_coeffs: DVector<f64>,
    /// Coefficients from the outcome (OLS) equation, excluding the IMR coefficient.
    pub outcome_coeffs: DVector<f64>,
    /// Coefficient on the Inverse Mills Ratio (lambda).
    pub imr_coeff: f64,
    /// The Inverse Mills Ratio values for the outcome sample.
    pub imr: DVector<f64>,
}

/// Performs Heckman Two-Step estimation.
///
/// # Arguments
///
/// * `y_select` - Binary selection variable (N x 1).
/// * `x_select` - Predictors for the selection equation (N x K1).
/// * `y_outcome` - Outcome variable for the selected sample (M x 1).
/// * `x_outcome` - Predictors for the outcome equation for the selected sample (M x K2).
/// * `x_select_subset` - Predictors for the selection equation corresponding to the selected sample (M x K1).
///
/// # Returns
///
/// `HeckmanResult` containing coefficients and IMR.
pub fn heckman_two_step(
    y_select: &DVector<f64>,
    x_select: &DMatrix<f64>,
    y_outcome: &DVector<f64>,
    x_outcome: &DMatrix<f64>,
    x_select_subset: &DMatrix<f64>,
) -> Result<HeckmanResult, OaxacaError> {
    // 1. Probit Regression on Selection Equation
    let probit_res = probit(y_select, x_select, 100, 1e-6)?;
    let gamma = probit_res.coefficients;
    
    // 2. Calculate Inverse Mills Ratio (IMR) for the selected sample
    // IMR = phi(z'gamma) / Phi(z'gamma)
    let normal = Normal::new(0.0, 1.0).unwrap();
    let z_gamma = x_select_subset * &gamma;
    
    let imr_vec: Vec<f64> = z_gamma.iter().map(|&zg| {
        let phi = normal.pdf(zg);
        let big_phi = normal.cdf(zg);
        if big_phi < 1e-10 { 0.0 } else { phi / big_phi }
    }).collect();
    let imr = DVector::from_vec(imr_vec);
    
    // 3. OLS on Outcome Equation with IMR
    // Augment x_outcome with IMR
    let mut x_augmented = x_outcome.clone();
    // Insert IMR as the last column
    x_augmented = x_augmented.insert_column(x_outcome.ncols(), 0.0);
    x_augmented.set_column(x_outcome.ncols(), &imr);
    
    // Run OLS
    let ols_res = ols(y_outcome, &x_augmented, None)?;
    
    // Extract coefficients
    let full_coeffs = ols_res.coefficients;
    let k_outcome = x_outcome.ncols();
    let outcome_coeffs = full_coeffs.rows(0, k_outcome).into_owned();
    let imr_coeff = full_coeffs[k_outcome];
    
    Ok(HeckmanResult {
        selection_coeffs: gamma,
        outcome_coeffs,
        imr_coeff,
        imr,
    })
}
