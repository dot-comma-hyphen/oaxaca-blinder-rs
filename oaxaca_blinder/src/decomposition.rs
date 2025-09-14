use nalgebra::DVector;

/// Represents the choice of reference coefficients for the two-fold decomposition.
#[derive(Debug, Clone)]
pub enum ReferenceCoefficients {
    GroupA,
    GroupB,
}

/// Holds the results of the three-fold decomposition.
#[derive(Debug, Clone)]
pub struct ThreeFoldDecomposition {
    pub endowments: f64,
    pub coefficients: f64,
    pub interaction: f64,
}

/// Holds the results of the two-fold decomposition.
#[derive(Debug, Clone)]
pub struct TwoFoldDecomposition {
    pub explained: f64,
    pub unexplained: f64,
}

/// Represents the contribution of a single variable to a decomposition component.
#[derive(Debug, PartialEq, Clone)]
pub struct DetailedComponent {
    pub variable_name: String,
    pub contribution: f64,
}

/// Computes the three-fold Oaxaca-Blinder decomposition.
pub fn three_fold_decomposition(
    xa_mean: &DVector<f64>,
    xb_mean: &DVector<f64>,
    beta_a: &DVector<f64>,
    beta_b: &DVector<f64>,
) -> ThreeFoldDecomposition {
    let diff_x = xa_mean - xb_mean;
    let diff_beta = beta_a - beta_b;
    let endowments = diff_x.dot(beta_b);
    let coefficients = xb_mean.dot(&diff_beta);
    let interaction = diff_x.dot(&diff_beta);
    ThreeFoldDecomposition { endowments, coefficients, interaction }
}

/// Computes the two-fold decomposition from the three-fold components.
pub fn two_fold_from_three_fold(
    three_fold: &ThreeFoldDecomposition,
    reference: &ReferenceCoefficients,
) -> TwoFoldDecomposition {
    match reference {
        ReferenceCoefficients::GroupB => TwoFoldDecomposition {
            explained: three_fold.endowments,
            unexplained: three_fold.coefficients + three_fold.interaction,
        },
        ReferenceCoefficients::GroupA => TwoFoldDecomposition {
            explained: three_fold.endowments + three_fold.interaction,
            unexplained: three_fold.coefficients,
        },
    }
}

/// Computes the detailed decomposition for both explained and unexplained parts.
pub fn detailed_decomposition(
    xa_mean: &DVector<f64>,
    xb_mean: &DVector<f64>,
    beta_a: &DVector<f64>,
    beta_b: &DVector<f64>,
    predictor_names: &[String],
) -> (Vec<DetailedComponent>, Vec<DetailedComponent>) {
    let diff_x = xa_mean - xb_mean;
    let diff_beta = beta_a - beta_b;

    let explained = diff_x.iter().zip(beta_b.iter()).enumerate().map(|(i, (dx, b))| DetailedComponent {
        variable_name: predictor_names[i].clone(),
        contribution: dx * b,
    }).collect();

    let unexplained = xb_mean.iter().zip(diff_beta.iter()).enumerate().map(|(i, (x, db))| DetailedComponent {
        variable_name: predictor_names[i].clone(),
        contribution: x * db,
    } ).collect();

    (explained, unexplained)
}


#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    #[test]
    fn test_three_fold_decomposition() {
        let xa_mean = DVector::from_vec(vec![1.0, 5.0]);
        let xb_mean = DVector::from_vec(vec![1.0, 3.0]);
        let beta_a = DVector::from_vec(vec![2.0, 4.0]);
        let beta_b = DVector::from_vec(vec![1.0, 3.0]);
        let result = three_fold_decomposition(&xa_mean, &xb_mean, &beta_a, &beta_b);
        assert!((result.endowments - 6.0).abs() < 1e-9);
        assert!((result.coefficients - 4.0).abs() < 1e-9);
        assert!((result.interaction - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_two_fold_decomposition() {
        let three_fold = ThreeFoldDecomposition { endowments: 6.0, coefficients: 4.0, interaction: 2.0 };
        let two_fold_b = two_fold_from_three_fold(&three_fold, &ReferenceCoefficients::GroupB);
        assert!((two_fold_b.explained - 6.0).abs() < 1e-9);
        assert!((two_fold_b.unexplained - 6.0).abs() < 1e-9);
        let two_fold_a = two_fold_from_three_fold(&three_fold, &ReferenceCoefficients::GroupA);
        assert!((two_fold_a.explained - 8.0).abs() < 1e-9);
        assert!((two_fold_a.unexplained - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_detailed_decomposition_sums() {
        let predictor_names = vec!["intercept".to_string(), "age".to_string()];
        let beta_a = DVector::from_vec(vec![2.0, 4.0]);
        let beta_b = DVector::from_vec(vec![1.0, 3.0]);
        let xa_mean = DVector::from_vec(vec![1.0, 5.0]);
        let xb_mean = DVector::from_vec(vec![1.0, 3.0]);

        let (explained_detailed, unexplained_detailed) = detailed_decomposition(&xa_mean, &xb_mean, &beta_a, &beta_b, &predictor_names);

        let total_explained: f64 = explained_detailed.iter().map(|c| c.contribution).sum();
        let total_unexplained: f64 = unexplained_detailed.iter().map(|c| c.contribution).sum();

        assert!((total_explained - 6.0).abs() < 1e-9);
        assert!((total_unexplained - 4.0).abs() < 1e-9);
    }
}
