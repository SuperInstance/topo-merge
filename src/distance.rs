//! Distance matrix computation with multiple metric support.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Supported distance metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Metric {
    /// Euclidean (L2) distance.
    Euclidean,
    /// Cosine distance: 1 - cos(a, b).
    Cosine,
}

/// A symmetric distance matrix stored in upper-triangle form.
///
/// For n points, stores n*(n-1)/2 distances.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DistanceMatrix {
    /// Number of points.
    pub n: usize,
    /// Upper-triangle distances, row-major: dist(i,j) for i < j.
    /// Index: i * n - i * (i + 1) / 2 + j - i - 1
    pub data: Vec<f64>,
    /// The metric used.
    pub metric: Metric,
}

impl DistanceMatrix {
    /// Compute distance matrix from a slice of coordinate vectors.
    pub fn compute(coords: &[Vec<f64>], metric: Metric) -> Result<Self, DistanceError> {
        if coords.is_empty() {
            return Ok(Self {
                n: 0,
                data: vec![],
                metric,
            });
        }
        let dim = coords[0].len();
        for (i, c) in coords.iter().enumerate() {
            if c.len() != dim {
                return Err(DistanceError::DimensionMismatch {
                    index: i,
                    expected: dim,
                    actual: c.len(),
                });
            }
        }
        let n = coords.len();
        let mut data = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let d = match metric {
                    Metric::Euclidean => euclidean(&coords[i], &coords[j]),
                    Metric::Cosine => cosine_distance(&coords[i], &coords[j]),
                };
                data.push(d);
            }
        }
        Ok(Self { n, data, metric })
    }

    /// Build from an upper-triangle flat array (must have exactly n*(n-1)/2 entries).
    pub fn from_upper_triangle(n: usize, data: Vec<f64>, metric: Metric) -> Result<Self, DistanceError> {
        let expected = n * (n - 1) / 2;
        if data.len() != expected {
            return Err(DistanceError::InvalidSize {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { n, data, metric })
    }

    /// Get distance between points i and j. Returns 0.0 if i == j.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        if i == j {
            return 0.0;
        }
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        let idx = a * self.n - a * (a + 1) / 2 + b - a - 1;
        self.data[idx]
    }

    /// Set distance between points i and j (i != j).
    pub fn set(&mut self, i: usize, j: usize, val: f64) {
        assert_ne!(i, j, "cannot set self-distance");
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        let idx = a * self.n - a * (a + 1) / 2 + b - a - 1;
        self.data[idx] = val;
    }

    /// All unique distances sorted.
    pub fn sorted_distances(&self) -> Vec<f64> {
        let mut d = self.data.clone();
        d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        d
    }

    /// Maximum distance.
    pub fn max_distance(&self) -> f64 {
        self.data.iter().copied().fold(0.0_f64, f64::max)
    }

    /// Minimum distance (excluding self-distances).
    pub fn min_distance(&self) -> f64 {
        self.data
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    }

    /// Number of point pairs.
    pub fn pair_count(&self) -> usize {
        self.data.len()
    }
}

/// Errors for distance matrix operations.
#[derive(Debug, thiserror::Error)]
pub enum DistanceError {
    #[error("dimension mismatch at index {index}: expected {expected}, got {actual}")]
    DimensionMismatch {
        index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("invalid size: expected {expected} distances, got {actual}")]
    InvalidSize { expected: usize, actual: usize },
}

/// Euclidean distance.
fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Cosine distance: 1 - cos(a, b).
fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a < 1e-15 || norm_b < 1e-15 {
        return 1.0; // treat zero vectors as maximally distant
    }
    1.0 - dot / (norm_a * norm_b)
}

impl fmt::Display for DistanceMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DistanceMatrix(n={}, pairs={}, metric={:?})",
            self.n,
            self.data.len(),
            self.metric
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_basic() {
        let _coords = vec![vec![0.0], vec![3.0], vec![3.0, 4.0].clone()];
        // just 1D for first two
        let dm = DistanceMatrix::compute(
            &[vec![0.0, 0.0], vec![3.0, 4.0]],
            Metric::Euclidean,
        )
        .unwrap();
        assert!((dm.get(0, 1) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_self_distance_zero() {
        let dm = DistanceMatrix::compute(
            &[vec![1.0, 2.0], vec![3.0, 4.0]],
            Metric::Euclidean,
        )
        .unwrap();
        assert!((dm.get(0, 0)).abs() < 1e-10);
        assert!((dm.get(1, 1)).abs() < 1e-10);
    }

    #[test]
    fn test_symmetry() {
        let dm = DistanceMatrix::compute(
            &[vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
            Metric::Euclidean,
        )
        .unwrap();
        assert!((dm.get(0, 1) - dm.get(1, 0)).abs() < 1e-10);
        assert!((dm.get(0, 2) - dm.get(2, 0)).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_triangle() {
        let dm = DistanceMatrix::compute(
            &[vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
            Metric::Euclidean,
        )
        .unwrap();
        // 0-1: 1.0, 0-2: 1.0, 1-2: sqrt(2)
        assert!((dm.get(0, 1) - 1.0).abs() < 1e-10);
        assert!((dm.get(0, 2) - 1.0).abs() < 1e-10);
        assert!((dm.get(1, 2) - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_identical() {
        let dm = DistanceMatrix::compute(
            &[vec![1.0, 0.0], vec![2.0, 0.0]],
            Metric::Cosine,
        )
        .unwrap();
        assert!(dm.get(0, 1).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let dm = DistanceMatrix::compute(
            &[vec![1.0, 0.0], vec![0.0, 1.0]],
            Metric::Cosine,
        )
        .unwrap();
        assert!((dm.get(0, 1) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_opposite() {
        let dm = DistanceMatrix::compute(
            &[vec![1.0, 0.0], vec![-1.0, 0.0]],
            Metric::Cosine,
        )
        .unwrap();
        assert!((dm.get(0, 1) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty() {
        let dm = DistanceMatrix::compute(&[], Metric::Euclidean).unwrap();
        assert_eq!(dm.n, 0);
        assert!(dm.data.is_empty());
    }

    #[test]
    fn test_single_point() {
        let dm = DistanceMatrix::compute(&[vec![1.0, 2.0]], Metric::Euclidean).unwrap();
        assert_eq!(dm.n, 1);
        assert!(dm.data.is_empty());
    }

    #[test]
    fn test_from_upper_triangle() {
        let dm = DistanceMatrix::from_upper_triangle(3, vec![1.0, 2.0, 3.0], Metric::Euclidean).unwrap();
        assert!((dm.get(0, 1) - 1.0).abs() < 1e-10);
        assert!((dm.get(0, 2) - 2.0).abs() < 1e-10);
        assert!((dm.get(1, 2) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_from_upper_triangle_wrong_size() {
        let result = DistanceMatrix::from_upper_triangle(3, vec![1.0, 2.0], Metric::Euclidean);
        assert!(result.is_err());
    }

    #[test]
    fn test_sorted_distances() {
        let dm = DistanceMatrix::compute(
            &[vec![0.0], vec![3.0], vec![1.0]],
            Metric::Euclidean,
        )
        .unwrap();
        let sorted = dm.sorted_distances();
        assert_eq!(sorted.len(), 3);
        assert!(sorted[0] <= sorted[1]);
        assert!(sorted[1] <= sorted[2]);
    }

    #[test]
    fn test_max_min() {
        let dm = DistanceMatrix::compute(
            &[vec![0.0], vec![1.0], vec![10.0]],
            Metric::Euclidean,
        )
        .unwrap();
        assert!((dm.min_distance() - 1.0).abs() < 1e-10);
        assert!((dm.max_distance() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_set_distance() {
        let mut dm = DistanceMatrix::from_upper_triangle(2, vec![5.0], Metric::Euclidean).unwrap();
        dm.set(0, 1, 10.0);
        assert!((dm.get(0, 1) - 10.0).abs() < 1e-10);
        dm.set(1, 0, 20.0);
        assert!((dm.get(0, 1) - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_serialize_deserialize() {
        let dm = DistanceMatrix::compute(
            &[vec![0.0, 0.0], vec![1.0, 1.0]],
            Metric::Euclidean,
        )
        .unwrap();
        let json = serde_json::to_string(&dm).unwrap();
        let dm2: DistanceMatrix = serde_json::from_str(&json).unwrap();
        assert_eq!(dm, dm2);
    }

    #[test]
    fn test_display() {
        let dm = DistanceMatrix::compute(&[vec![0.0], vec![1.0]], Metric::Euclidean).unwrap();
        let s = format!("{}", dm);
        assert!(s.contains("n=2"));
    }
}
