//! Belief state representation: a weighted point cloud in R^n.

use serde::{Deserialize, Serialize};
use std::fmt;

/// A single point in a belief state with an associated weight.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WeightedPoint {
    /// The point coordinates in R^n.
    pub coords: Vec<f64>,
    /// Weight/confidence of this point. Must be > 0.
    pub weight: f64,
}

impl WeightedPoint {
    /// Create a new weighted point.
    pub fn new(coords: Vec<f64>, weight: f64) -> Self {
        Self { coords, weight: if weight <= 0.0 { 1.0 } else { weight } }
    }

    /// Create a point with unit weight.
    pub fn unit(coords: Vec<f64>) -> Self {
        Self { coords, weight: 1.0 }
    }

    /// Dimensionality of the point.
    pub fn dim(&self) -> usize {
        self.coords.len()
    }
}

/// A belief state: a weighted point cloud in R^n.
///
/// Represents an agent's belief about a system as a collection of weighted points
/// in some metric space.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BeliefState {
    /// The weighted points forming the belief.
    pub points: Vec<WeightedPoint>,
    /// Dimensionality of the space (all points must agree).
    pub dimension: usize,
    /// Optional label identifying the source agent.
    pub agent_id: Option<String>,
}

/// Errors for belief state operations.
#[derive(Debug, thiserror::Error)]
pub enum BeliefError {
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("empty belief state has no centroid")]
    EmptyState,
    #[error("weight must be positive, got {0}")]
    NonPositiveWeight(f64),
}

impl BeliefState {
    /// Create a new belief state from weighted points.
    pub fn new(points: Vec<WeightedPoint>) -> Result<Self, BeliefError> {
        if points.is_empty() {
            return Ok(Self {
                points,
                dimension: 0,
                agent_id: None,
            });
        }
        let dim = points[0].dim();
        for p in points.iter() {
            if p.dim() != dim {
                return Err(BeliefError::DimensionMismatch {
                    expected: dim,
                    actual: p.dim(),
                });
            }
            if p.weight <= 0.0 {
                return Err(BeliefError::NonPositiveWeight(p.weight));
            }
        }
        Ok(Self {
            points,
            dimension: dim,
            agent_id: None,
        })
    }

    /// Create a belief state with an agent ID.
    pub fn with_agent_id(mut self, id: impl Into<String>) -> Self {
        self.agent_id = Some(id.into());
        self
    }

    /// Create an empty belief state in the given dimension.
    pub fn empty(dimension: usize) -> Self {
        Self {
            points: vec![],
            dimension,
            agent_id: None,
        }
    }

    /// Create a belief state from unweighted coordinates (all unit weight).
    pub fn from_coords(coords: &[Vec<f64>]) -> Result<Self, BeliefError> {
        let points: Vec<WeightedPoint> = coords
            .iter()
            .map(|c| WeightedPoint::unit(c.clone()))
            .collect();
        Self::new(points)
    }

    /// Number of points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the belief state is empty.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Compute the weighted centroid of the point cloud.
    pub fn centroid(&self) -> Result<Vec<f64>, BeliefError> {
        if self.points.is_empty() {
            return Err(BeliefError::EmptyState);
        }
        let dim = self.dimension;
        let total_weight: f64 = self.points.iter().map(|p| p.weight).sum();
        let mut centroid = vec![0.0; dim];
        for p in &self.points {
            for (i, &c) in p.coords.iter().enumerate() {
                centroid[i] += c * p.weight;
            }
        }
        for c in centroid.iter_mut() {
            *c /= total_weight;
        }
        Ok(centroid)
    }

    /// Compute the weighted spread (root-mean-square distance from centroid).
    pub fn spread(&self) -> Result<f64, BeliefError> {
        let centroid = self.centroid()?;
        let total_weight: f64 = self.points.iter().map(|p| p.weight).sum();
        let mut sum_sq = 0.0;
        for p in &self.points {
            let dist_sq: f64 = p
                .coords
                .iter()
                .zip(centroid.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            sum_sq += p.weight * dist_sq;
        }
        Ok((sum_sq / total_weight).sqrt())
    }

    /// Compute the union of two belief states. Points from both are combined.
    pub fn union(&self, other: &BeliefState) -> Result<BeliefState, BeliefError> {
        if self.dimension != 0 && other.dimension != 0 && self.dimension != other.dimension {
            return Err(BeliefError::DimensionMismatch {
                expected: self.dimension,
                actual: other.dimension,
            });
        }
        let dim = if self.dimension == 0 {
            other.dimension
        } else {
            self.dimension
        };
        let mut combined = self.points.clone();
        combined.extend(other.points.iter().cloned());
        Ok(BeliefState {
            points: combined,
            dimension: dim,
            agent_id: None,
        })
    }

    /// Get just the coordinate vectors (without weights).
    pub fn coords(&self) -> Vec<&[f64]> {
        self.points.iter().map(|p| p.coords.as_slice()).collect()
    }

    /// Total weight of all points.
    pub fn total_weight(&self) -> f64 {
        self.points.iter().map(|p| p.weight).sum()
    }

    /// Add a point to the belief state.
    pub fn add_point(&mut self, point: WeightedPoint) -> Result<(), BeliefError> {
        if self.dimension != 0 && point.dim() != self.dimension {
            return Err(BeliefError::DimensionMismatch {
                expected: self.dimension,
                actual: point.dim(),
            });
        }
        if point.weight <= 0.0 {
            return Err(BeliefError::NonPositiveWeight(point.weight));
        }
        if self.dimension == 0 {
            self.dimension = point.dim();
        }
        self.points.push(point);
        Ok(())
    }
}

impl fmt::Display for BeliefState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let agent = self.agent_id.as_deref().unwrap_or("unknown");
        write!(
            f,
            "BeliefState(agent={}, dim={}, n={})",
            agent,
            self.dimension,
            self.points.len()
        )
    }
}

impl fmt::Display for WeightedPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Point(w={:.3}, [", self.weight)?;
        for (i, c) in self.coords.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:.3}", c)?;
        }
        write!(f, "])")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_belief_state() {
        let bs = BeliefState::empty(3);
        assert!(bs.is_empty());
        assert_eq!(bs.dimension, 3);
        assert!(bs.centroid().is_err());
    }

    #[test]
    fn test_from_coords() {
        let bs = BeliefState::from_coords(&[
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ])
        .unwrap();
        assert_eq!(bs.len(), 3);
        assert_eq!(bs.dimension, 2);
    }

    #[test]
    fn test_dimension_mismatch() {
        let result = BeliefState::new(vec![
            WeightedPoint::unit(vec![0.0, 0.0]),
            WeightedPoint::unit(vec![1.0, 0.0, 0.0]),
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn test_non_positive_weight() {
        let result = BeliefState::new(vec![WeightedPoint { coords: vec![0.0], weight: 0.0 }]);
        assert!(result.is_err());
        let result = BeliefState::new(vec![WeightedPoint { coords: vec![0.0], weight: -1.0 }]);
        assert!(result.is_err());
    }

    #[test]
    fn test_centroid_unit_weights() {
        let bs = BeliefState::from_coords(&[
            vec![0.0, 0.0],
            vec![2.0, 0.0],
            vec![0.0, 2.0],
        ])
        .unwrap();
        let c = bs.centroid().unwrap();
        assert!((c[0] - 2.0 / 3.0).abs() < 1e-10);
        assert!((c[1] - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_centroid_weighted() {
        let bs = BeliefState::new(vec![
            WeightedPoint::new(vec![0.0], 1.0),
            WeightedPoint::new(vec![10.0], 9.0),
        ])
        .unwrap();
        let c = bs.centroid().unwrap();
        // weighted: (0*1 + 10*9) / 10 = 9.0
        assert!((c[0] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_spread() {
        let bs = BeliefState::from_coords(&[vec![0.0], vec![2.0]]).unwrap();
        let s = bs.spread().unwrap();
        // centroid = 1.0, distances = 1.0 and 1.0, rms = 1.0
        assert!((s - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_union() {
        let bs1 = BeliefState::from_coords(&[vec![0.0, 0.0]]).unwrap();
        let bs2 = BeliefState::from_coords(&[vec![1.0, 1.0]]).unwrap();
        let union = bs1.union(&bs2).unwrap();
        assert_eq!(union.len(), 2);
        assert_eq!(union.dimension, 2);
    }

    #[test]
    fn test_union_dimension_mismatch() {
        let bs1 = BeliefState::from_coords(&[vec![0.0]]).unwrap();
        let bs2 = BeliefState::from_coords(&[vec![0.0, 0.0]]).unwrap();
        assert!(bs1.union(&bs2).is_err());
    }

    #[test]
    fn test_union_empty() {
        let bs1 = BeliefState::empty(2);
        let bs2 = BeliefState::from_coords(&[vec![1.0, 1.0]]).unwrap();
        let union = bs1.union(&bs2).unwrap();
        assert_eq!(union.len(), 1);
    }

    #[test]
    fn test_add_point() {
        let mut bs = BeliefState::empty(2);
        bs.add_point(WeightedPoint::unit(vec![1.0, 2.0])).unwrap();
        assert_eq!(bs.len(), 1);
        assert_eq!(bs.dimension, 2);
    }

    #[test]
    fn test_add_point_dim_mismatch() {
        let mut bs = BeliefState::from_coords(&[vec![0.0, 0.0]]).unwrap();
        let result = bs.add_point(WeightedPoint::unit(vec![1.0]));
        assert!(result.is_err());
    }

    #[test]
    fn test_serialize_deserialize() {
        let bs = BeliefState::from_coords(&[
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ])
        .unwrap()
        .with_agent_id("agent-1");
        let json = serde_json::to_string(&bs).unwrap();
        let bs2: BeliefState = serde_json::from_str(&json).unwrap();
        assert_eq!(bs, bs2);
        assert_eq!(bs2.agent_id.as_deref(), Some("agent-1"));
    }

    #[test]
    fn test_total_weight() {
        let bs = BeliefState::new(vec![
            WeightedPoint::new(vec![0.0], 2.0),
            WeightedPoint::new(vec![1.0], 3.0),
        ])
        .unwrap();
        assert!((bs.total_weight() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_display() {
        let bs = BeliefState::from_coords(&[vec![0.0]]).unwrap().with_agent_id("a1");
        let s = format!("{}", bs);
        assert!(s.contains("a1"));
    }

    #[test]
    fn test_weighted_point_display() {
        let p = WeightedPoint::new(vec![1.0, 2.0], 0.5);
        let s = format!("{}", p);
        assert!(s.contains("0.500"));
    }
}
