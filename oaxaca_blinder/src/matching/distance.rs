use nalgebra::{DMatrix, DVector};
use std::fmt::Debug;

/// Trait for distance metrics used in matching.
pub trait DistanceMetric: Send + Sync + Debug {
    /// Calculates the distance between two vectors.
    fn distance(&self, a: &[f64], b: &[f64]) -> f64;
}

/// Euclidean distance metric.
#[derive(Debug, Clone, Default)]
pub struct EuclideanDistance;

impl DistanceMetric for EuclideanDistance {
    fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

/// Mahalanobis distance metric.
///
/// This metric accounts for the correlations between variables by using the inverse covariance matrix.
#[derive(Debug, Clone)]
pub struct MahalanobisDistance {
    pub inv_covariance: DMatrix<f64>,
}

impl MahalanobisDistance {
    /// Creates a new `MahalanobisDistance` metric from a data matrix.
    ///
    /// The covariance matrix is calculated from the provided data and then inverted.
    pub fn new(data: &DMatrix<f64>) -> Result<Self, String> {
        let n = data.nrows();
        if n < 2 {
            return Err("Not enough data points to calculate covariance".to_string());
        }

        // Calculate covariance matrix
        let centered = data.row_mean().transpose();
        let mut centered_data = data.clone();
        for i in 0..n {
            let mut row = centered_data.row_mut(i);
            row -= &centered;
        }
        
        let covariance = (centered_data.transpose() * centered_data) / ((n - 1) as f64);

        // Calculate inverse covariance
        // Use pseudo-inverse or regular inverse with check
        let inv_covariance = covariance.try_inverse()
            .ok_or_else(|| "Covariance matrix is singular and cannot be inverted".to_string())?;

        Ok(Self { inv_covariance })
    }
    
    /// Creates a new `MahalanobisDistance` from a pre-calculated inverse covariance matrix.
    pub fn from_inv_covariance(inv_covariance: DMatrix<f64>) -> Self {
        Self { inv_covariance }
    }
}

impl DistanceMetric for MahalanobisDistance {
    fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
        let diff_vec: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
        let diff = DVector::from_vec(diff_vec);
        
        // d = sqrt( (x-y)^T * S^-1 * (x-y) )
        let dist_sq = diff.dot(&(&self.inv_covariance * &diff));
        
        if dist_sq < 0.0 {
            0.0 // Should not happen theoretically for PSD matrices, but floating point errors
        } else {
            dist_sq.sqrt()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::RowDVector;

    #[test]
    fn test_euclidean_distance() {
        let metric = EuclideanDistance;
        let a = &[1.0, 2.0];
        let b = &[4.0, 6.0];
        // sqrt((1-4)^2 + (2-6)^2) = sqrt(9 + 16) = 5
        assert_eq!(metric.distance(a, b), 5.0);
    }

    #[test]
    fn test_mahalanobis_distance() {
        // Simple case: uncorrelated variables with unit variance (should be same as Euclidean)
        let data = DMatrix::from_row_slice(3, 2, &[
            1.0, 0.0,
            0.0, 1.0,
            -1.0, -1.0
        ]);
        
        // Covariance of this specific matrix might not be exactly identity, let's construct one manually
        // Identity inverse covariance
        let inv_cov = DMatrix::identity(2, 2);
        let metric = MahalanobisDistance::from_inv_covariance(inv_cov);
        
        let a = &[1.0, 2.0];
        let b = &[4.0, 6.0];
        assert_eq!(metric.distance(a, b), 5.0);
        
        // Case with scaling
        let inv_cov_scaled = DMatrix::from_row_slice(2, 2, &[
            0.25, 0.0, // Variance 4 for x
            0.0, 1.0   // Variance 1 for y
        ]);
        let metric_scaled = MahalanobisDistance::from_inv_covariance(inv_cov_scaled);
        // dist^2 = (3^2 * 0.25) + (4^2 * 1) = 2.25 + 16 = 18.25
        // dist = sqrt(18.25) approx 4.272
        let d = metric_scaled.distance(a, b);
        assert!((d - 18.25f64.sqrt()).abs() < 1e-6);
    }
}
