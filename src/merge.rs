//! Topological merger: combine N belief states using persistent homology.

use crate::belief::BeliefState;
use crate::confidence::ConfidenceMap;
use crate::distance::{DistanceMatrix, Metric};
use crate::persistence::PersistenceDiagram;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Result of a topological merge.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MergeResult {
    /// The merged belief state.
    pub merged_belief: BeliefState,
    /// Persistence diagram of the merged state.
    pub merged_diagram: PersistenceDiagram,
    /// Confidence scores for merged features.
    pub confidence: ConfidenceMap,
    /// Distance metric used.
    pub metric: Metric,
    /// Number of input agents.
    pub agent_count: usize,
}

/// Parameters for the merge algorithm.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MergeParams {
    /// Distance metric to use.
    pub metric: Metric,
    /// Minimum number of agents that must observe a feature for it to be "confirmed".
    pub consensus_threshold: usize,
}

impl Default for MergeParams {
    fn default() -> Self {
        Self {
            metric: Metric::Euclidean,
            consensus_threshold: 2,
        }
    }
}

/// Topological merger for combining agent belief states.
#[derive(Debug, Clone)]
pub struct TopologicalMerger {
    params: MergeParams,
}

impl TopologicalMerger {
    /// Create a new merger with default parameters.
    pub fn new() -> Self {
        Self {
            params: MergeParams::default(),
        }
    }

    /// Create a merger with custom parameters.
    pub fn with_params(params: MergeParams) -> Self {
        Self { params }
    }

    /// Merge multiple belief states into one.
    ///
    /// Algorithm:
    /// 1. Compute the union of all belief states
    /// 2. Build distance matrix for the union
    /// 3. Compute persistence diagram of the union
    /// 4. Compute individual persistence diagrams for each agent
    /// 5. Score features by cross-agent agreement
    /// 6. Return merged belief + diagram + confidence
    pub fn merge(&self, beliefs: &[BeliefState]) -> Result<MergeResult, MergeError> {
        if beliefs.is_empty() {
            return Err(MergeError::NoAgents);
        }

        // Validate dimension consistency
        let dim = beliefs[0].dimension;
        for (i, b) in beliefs.iter().enumerate() {
            if b.dimension != dim {
                return Err(MergeError::DimensionMismatch {
                    agent: i,
                    expected: dim,
                    actual: b.dimension,
                });
            }
        }

        // Step 1: Compute union
        let mut union_belief = beliefs[0].clone();
        for b in &beliefs[1..] {
            union_belief = union_belief.union(b).map_err(|e| MergeError::BeliefError(e.to_string()))?;
        }

        if union_belief.is_empty() {
            return Err(MergeError::EmptyUnion);
        }

        // Step 2: Build distance matrix on union
        let coords: Vec<Vec<f64>> = union_belief.points.iter().map(|p| p.coords.clone()).collect();
        let dm = DistanceMatrix::compute(&coords, self.params.metric)
            .map_err(|e| MergeError::DistanceError(e.to_string()))?;

        // Step 3: Compute merged persistence diagram
        let merged_diagram = PersistenceDiagram::compute(&dm);

        // Step 4: Compute individual diagrams
        let individual_diagrams: Vec<PersistenceDiagram> = beliefs
            .iter()
            .filter(|b| !b.is_empty())
            .map(|b| {
                let agent_coords: Vec<Vec<f64>> = b.points.iter().map(|p| p.coords.clone()).collect();
                let agent_dm = DistanceMatrix::compute(&agent_coords, self.params.metric).unwrap();
                PersistenceDiagram::compute(&agent_dm)
            })
            .collect();

        // Step 5: Build confidence map
        let confidence = ConfidenceMap::build(&merged_diagram, &individual_diagrams);

        Ok(MergeResult {
            merged_belief: union_belief,
            merged_diagram,
            confidence,
            metric: self.params.metric,
            agent_count: beliefs.len(),
        })
    }

    /// Quick merge without full persistence computation.
    /// Just unions the beliefs and computes basic stats.
    pub fn quick_merge(&self, beliefs: &[BeliefState]) -> Result<BeliefState, MergeError> {
        if beliefs.is_empty() {
            return Err(MergeError::NoAgents);
        }
        let mut union_belief = beliefs[0].clone();
        for b in &beliefs[1..] {
            union_belief = union_belief.union(b).map_err(|e| MergeError::BeliefError(e.to_string()))?;
        }
        Ok(union_belief)
    }
}

impl Default for TopologicalMerger {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors during merging.
#[derive(Debug, thiserror::Error)]
pub enum MergeError {
    #[error("no agents provided")]
    NoAgents,
    #[error("dimension mismatch in agent {agent}: expected {expected}, got {actual}")]
    DimensionMismatch {
        agent: usize,
        expected: usize,
        actual: usize,
    },
    #[error("union of belief states is empty")]
    EmptyUnion,
    #[error("belief error: {0}")]
    BeliefError(String),
    #[error("distance error: {0}")]
    DistanceError(String),
}

impl fmt::Display for MergeResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "MergeResult:")?;
        writeln!(f, "  agents: {}", self.agent_count)?;
        writeln!(f, "  points: {}", self.merged_belief.len())?;
        writeln!(f, "  diagram: {}", self.merged_diagram)?;
        writeln!(f, "  confidence avg: {:.3}", self.confidence.average_score())?;
        let confirmed = self.confidence.confirmed().len();
        let provisional = self.confidence.provisional().len();
        let rejected = self.confidence.rejected().len();
        writeln!(f, "  features: {} confirmed, {} provisional, {} rejected", confirmed, provisional, rejected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief::WeightedPoint;

    fn make_belief(coords: &[Vec<f64>], agent: &str) -> BeliefState {
        BeliefState::from_coords(coords)
            .unwrap()
            .with_agent_id(agent)
    }

    #[test]
    fn test_merge_two_agents() {
        let a = make_belief(&[vec![0.0, 0.0], vec![1.0, 0.0]], "a");
        let b = make_belief(&[vec![0.5, 0.5], vec![1.5, 0.5]], "b");
        let merger = TopologicalMerger::new();
        let result = merger.merge(&[a, b]).unwrap();
        assert_eq!(result.agent_count, 2);
        assert_eq!(result.merged_belief.len(), 4);
    }

    #[test]
    fn test_merge_three_agents() {
        let a = make_belief(&[vec![0.0], vec![1.0]], "a");
        let b = make_belief(&[vec![0.5], vec![1.5]], "b");
        let c = make_belief(&[vec![2.0]], "c");
        let merger = TopologicalMerger::new();
        let result = merger.merge(&[a, b, c]).unwrap();
        assert_eq!(result.merged_belief.len(), 5);
        assert!(result.merged_diagram.features.len() > 0);
    }

    #[test]
    fn test_merge_no_agents() {
        let merger = TopologicalMerger::new();
        let result = merger.merge(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_dimension_mismatch() {
        let a = make_belief(&[vec![0.0, 0.0]], "a");
        let b = make_belief(&[vec![0.0]], "b");
        let merger = TopologicalMerger::new();
        let result = merger.merge(&[a, b]);
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_single_agent() {
        let a = make_belief(&[vec![0.0, 0.0], vec![1.0, 0.0]], "a");
        let merger = TopologicalMerger::new();
        let result = merger.merge(&[a]).unwrap();
        assert_eq!(result.agent_count, 1);
        assert_eq!(result.merged_belief.len(), 2);
    }

    #[test]
    fn test_merge_with_cosine_metric() {
        let params = MergeParams {
            metric: Metric::Cosine,
            consensus_threshold: 1,
        };
        let a = make_belief(&[vec![1.0, 0.0], vec![0.0, 1.0]], "a");
        let b = make_belief(&[vec![1.0, 0.0]], "b");
        let merger = TopologicalMerger::with_params(params);
        let result = merger.merge(&[a, b]).unwrap();
        assert_eq!(result.metric, Metric::Cosine);
    }

    #[test]
    fn test_quick_merge() {
        let a = make_belief(&[vec![0.0]], "a");
        let b = make_belief(&[vec![1.0]], "b");
        let merger = TopologicalMerger::new();
        let result = merger.quick_merge(&[a, b]).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_quick_merge_no_agents() {
        let merger = TopologicalMerger::new();
        let result = merger.quick_merge(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_result_display() {
        let a = make_belief(&[vec![0.0], vec![1.0]], "a");
        let b = make_belief(&[vec![0.5]], "b");
        let merger = TopologicalMerger::new();
        let result = merger.merge(&[a, b]).unwrap();
        let s = format!("{}", result);
        assert!(s.contains("agents: 2"));
    }

    #[test]
    fn test_merge_identical_agents() {
        let a = make_belief(&[vec![0.0, 0.0], vec![1.0, 0.0]], "a");
        let b = make_belief(&[vec![0.0, 0.0], vec![1.0, 0.0]], "b");
        let merger = TopologicalMerger::new();
        let result = merger.merge(&[a, b]).unwrap();
        // High confidence since agents agree
        let avg = result.confidence.average_score();
        assert!(avg > 0.1, "identical agents should have reasonable confidence, got {}", avg);
    }

    #[test]
    fn test_merge_default() {
        let merger = TopologicalMerger::default();
        let a = make_belief(&[vec![0.0]], "a");
        let result = merger.merge(&[a]).unwrap();
        assert_eq!(result.agent_count, 1);
    }

    #[test]
    fn test_merge_params_default() {
        let params = MergeParams::default();
        assert_eq!(params.metric, Metric::Euclidean);
        assert_eq!(params.consensus_threshold, 2);
    }

    #[test]
    fn test_merge_serialize_deserialize() {
        let a = make_belief(&[vec![0.0], vec![1.0]], "a");
        let b = make_belief(&[vec![0.5]], "b");
        let merger = TopologicalMerger::new();
        let result = merger.merge(&[a, b]).unwrap();
        let json = serde_json::to_string(&result).unwrap();
        let result2: MergeResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result, result2);
    }

    #[test]
    fn test_merge_weighted_beliefs() {
        let a = BeliefState::new(vec![
            WeightedPoint::new(vec![0.0], 2.0),
            WeightedPoint::new(vec![1.0], 1.0),
        ])
        .unwrap()
        .with_agent_id("heavy");
        let b = BeliefState::new(vec![
            WeightedPoint::new(vec![0.0], 0.5),
        ])
        .unwrap()
        .with_agent_id("light");
        let merger = TopologicalMerger::new();
        let result = merger.merge(&[a, b]).unwrap();
        assert_eq!(result.merged_belief.len(), 3);
        // Total weight should be preserved
        let total: f64 = result.merged_belief.points.iter().map(|p| p.weight).sum();
        assert!((total - 3.5).abs() < 1e-10);
    }
}
