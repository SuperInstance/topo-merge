//! Confidence scoring for merged topological features.

use crate::persistence::{PersistenceDiagram, PersistenceFeature};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Confidence level for a merged feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum ConfidenceLevel {
    /// Feature observed by many agents with high persistence.
    Confirmed,
    /// Feature observed by some agents, moderate persistence.
    Provisional,
    /// Feature not well-supported, low persistence or few observers.
    Rejected,
}

impl fmt::Display for ConfidenceLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfidenceLevel::Confirmed => write!(f, "confirmed"),
            ConfidenceLevel::Provisional => write!(f, "provisional"),
            ConfidenceLevel::Rejected => write!(f, "rejected"),
        }
    }
}

/// A confidence score for a single merged feature.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureConfidence {
    /// Index of the feature in the merged diagram.
    pub feature_index: usize,
    /// The persistence feature.
    pub feature: PersistenceFeature,
    /// How many agents observed this feature.
    pub agent_count: usize,
    /// Total number of agents.
    pub total_agents: usize,
    /// Persistence value (death - birth).
    pub persistence: f64,
    /// Proximity score: how close agents' individual features are (0-1).
    pub proximity: f64,
    /// Raw confidence score (0-1).
    pub score: f64,
    /// Categorized confidence level.
    pub level: ConfidenceLevel,
}

impl Serialize for FeatureConfidence {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Helper {
            feature_index: usize,
            feature: PersistenceFeature,
            agent_count: usize,
            total_agents: usize,
            persistence: Option<f64>,
            proximity: f64,
            score: f64,
            level: ConfidenceLevel,
        }
        Helper {
            feature_index: self.feature_index,
            feature: self.feature.clone(),
            agent_count: self.agent_count,
            total_agents: self.total_agents,
            persistence: if self.persistence == f64::INFINITY { None } else { Some(self.persistence) },
            proximity: self.proximity,
            score: self.score,
            level: self.level,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for FeatureConfidence {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Helper {
            feature_index: usize,
            feature: PersistenceFeature,
            agent_count: usize,
            total_agents: usize,
            persistence: Option<f64>,
            proximity: f64,
            score: f64,
            level: ConfidenceLevel,
        }
        let h = Helper::deserialize(deserializer)?;
        Ok(FeatureConfidence {
            feature_index: h.feature_index,
            feature: h.feature,
            agent_count: h.agent_count,
            total_agents: h.total_agents,
            persistence: h.persistence.unwrap_or(f64::INFINITY),
            proximity: h.proximity,
            score: h.score,
            level: h.level,
        })
    }
}

impl FeatureConfidence {
    /// Compute a raw confidence score from components.
    pub fn compute(
        feature: PersistenceFeature,
        feature_index: usize,
        agent_count: usize,
        total_agents: usize,
        proximity: f64,
        max_persistence: f64,
    ) -> Self {
        let persistence = feature.persistence();
        let persistence_norm = if max_persistence > 0.0 && persistence.is_finite() {
            persistence / max_persistence
        } else if persistence.is_infinite() {
            1.0
        } else {
            0.0
        };

        let agent_fraction = if total_agents > 0 {
            agent_count as f64 / total_agents as f64
        } else {
            0.0
        };

        // Weighted combination: 40% agent agreement, 40% persistence, 20% proximity
        let score = 0.4 * agent_fraction + 0.4 * persistence_norm + 0.2 * proximity;

        let level = if score >= 0.6 {
            ConfidenceLevel::Confirmed
        } else if score >= 0.3 {
            ConfidenceLevel::Provisional
        } else {
            ConfidenceLevel::Rejected
        };

        Self {
            feature_index,
            feature,
            agent_count,
            total_agents,
            persistence,
            proximity,
            score,
            level,
        }
    }
}

/// A map from features to confidence scores.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConfidenceMap {
    /// Confidence scores for each feature.
    pub scores: Vec<FeatureConfidence>,
    /// Total number of agents.
    pub total_agents: usize,
}

impl ConfidenceMap {
    /// Build a confidence map from merged diagram, individual diagrams, and agent count.
    pub fn build(
        merged: &PersistenceDiagram,
        individual_diagrams: &[PersistenceDiagram],
    ) -> Self {
        let total_agents = individual_diagrams.len();
        let max_persistence = merged
            .features
            .iter()
            .filter(|f| f.death.is_finite())
            .map(|f| f.persistence())
            .fold(0.0_f64, f64::max);

        let scores: Vec<FeatureConfidence> = merged
            .features
            .iter()
            .enumerate()
            .map(|(idx, feature)| {
                // Count how many agents have a similar feature
                let (agent_count, proximity) = count_observing_agents(
                    feature,
                    individual_diagrams,
                );

                FeatureConfidence::compute(
                    feature.clone(),
                    idx,
                    agent_count,
                    total_agents,
                    proximity,
                    max_persistence,
                )
            })
            .collect();

        Self {
            scores,
            total_agents,
        }
    }

    /// Get features at a specific confidence level.
    pub fn at_level(&self, level: ConfidenceLevel) -> Vec<&FeatureConfidence> {
        self.scores.iter().filter(|s| s.level == level).collect()
    }

    /// Get confirmed features.
    pub fn confirmed(&self) -> Vec<&FeatureConfidence> {
        self.at_level(ConfidenceLevel::Confirmed)
    }

    /// Get provisional features.
    pub fn provisional(&self) -> Vec<&FeatureConfidence> {
        self.at_level(ConfidenceLevel::Provisional)
    }

    /// Get rejected features.
    pub fn rejected(&self) -> Vec<&FeatureConfidence> {
        self.at_level(ConfidenceLevel::Rejected)
    }

    /// Average confidence score.
    pub fn average_score(&self) -> f64 {
        if self.scores.is_empty() {
            return 0.0;
        }
        self.scores.iter().map(|s| s.score).sum::<f64>() / self.scores.len() as f64
    }
}

/// Count how many agents have a feature similar to the given one.
/// Returns (count, average_proximity).
fn count_observing_agents(
    feature: &PersistenceFeature,
    individual_diagrams: &[PersistenceDiagram],
) -> (usize, f64) {
    let mut count = 0;
    let mut total_prox = 0.0;
    let threshold = 0.5; // matching threshold

    for diag in individual_diagrams {
        let best_match = diag
            .features
            .iter()
            .filter(|f| f.dim == feature.dim)
            .map(|f| {
                // Proximity based on birth/death closeness
                let b_diff = (f.birth - feature.birth).abs();
                let d_diff = if f.death.is_infinite() && feature.death.is_infinite() {
                    0.0
                } else if f.death.is_infinite() || feature.death.is_infinite() {
                    f64::INFINITY
                } else {
                    (f.death - feature.death).abs()
                };
                if d_diff.is_infinite() {
                    0.0
                } else {
                    let dist = (b_diff.powi(2) + d_diff.powi(2)).sqrt();
                    let max_range = feature.persistence().max(1.0);
                    (1.0 - (dist / max_range).min(1.0)).max(0.0)
                }
            })
            .fold(0.0_f64, f64::max);

        if best_match > threshold {
            count += 1;
            total_prox += best_match;
        }
    }

    let avg_prox = if count > 0 {
        total_prox / count as f64
    } else {
        0.0
    };

    (count, avg_prox)
}

impl fmt::Display for ConfidenceMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "ConfidenceMap({} features, {} agents):", self.scores.len(), self.total_agents)?;
        for s in &self.scores {
            writeln!(
                f,
                "  H{} [{:.2},{:.2}] → {} (score={:.3}, agents={}/{})",
                s.feature.dim,
                s.feature.birth,
                if s.feature.death.is_infinite() {
                    f64::INFINITY
                } else {
                    s.feature.death
                },
                s.level,
                s.score,
                s.agent_count,
                s.total_agents,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::{DistanceMatrix, Metric};

    fn simple_diagram() -> PersistenceDiagram {
        PersistenceDiagram {
            features: vec![
                PersistenceFeature {
                    dim: 0,
                    birth: 0.0,
                    death: 1.0,
                },
                PersistenceFeature {
                    dim: 0,
                    birth: 0.0,
                    death: f64::INFINITY,
                },
            ],
        }
    }

    #[test]
    fn test_confidence_levels() {
        assert_ne!(ConfidenceLevel::Confirmed, ConfidenceLevel::Rejected);
        assert_ne!(ConfidenceLevel::Provisional, ConfidenceLevel::Confirmed);
    }

    #[test]
    fn test_feature_confidence_compute() {
        let fc = FeatureConfidence::compute(
            PersistenceFeature {
                dim: 0,
                birth: 0.0,
                death: 2.0,
            },
            0,
            3,
            3,
            0.9,
            2.0,
        );
        // agent_fraction = 3/3 = 1.0, persistence_norm = 2.0/2.0 = 1.0, proximity = 0.9
        // score = 0.4*1.0 + 0.4*1.0 + 0.2*0.9 = 0.4 + 0.4 + 0.18 = 0.98
        assert!((fc.score - 0.98).abs() < 0.01);
        assert_eq!(fc.level, ConfidenceLevel::Confirmed);
    }

    #[test]
    fn test_feature_confidence_low() {
        let fc = FeatureConfidence::compute(
            PersistenceFeature {
                dim: 0,
                birth: 0.0,
                death: 0.1,
            },
            0,
            1,
            5,
            0.2,
            10.0,
        );
        assert!(fc.score < 0.3);
        assert_eq!(fc.level, ConfidenceLevel::Rejected);
    }

    #[test]
    fn test_confidence_map_build() {
        let merged = simple_diagram();
        let ind1 = simple_diagram();
        let ind2 = simple_diagram();
        let map = ConfidenceMap::build(&merged, &[ind1, ind2]);
        assert_eq!(map.scores.len(), 2);
        assert_eq!(map.total_agents, 2);
    }

    #[test]
    fn test_confidence_map_levels() {
        let dm = DistanceMatrix::compute(&[vec![0.0], vec![1.0]], Metric::Euclidean).unwrap();
        let pd = PersistenceDiagram::compute(&dm);
        let map = ConfidenceMap::build(&pd, &[pd.clone(), pd.clone()]);
        let confirmed = map.confirmed();
        let provisional = map.provisional();
        let rejected = map.rejected();
        assert_eq!(confirmed.len() + provisional.len() + rejected.len(), map.scores.len());
    }

    #[test]
    fn test_average_score() {
        let dm = DistanceMatrix::compute(&[vec![0.0], vec![1.0]], Metric::Euclidean).unwrap();
        let pd = PersistenceDiagram::compute(&dm);
        let map = ConfidenceMap::build(&pd, &[pd.clone()]);
        let avg = map.average_score();
        assert!(avg >= 0.0 && avg <= 1.0);
    }

    #[test]
    fn test_empty_confidence_map() {
        let pd = PersistenceDiagram::empty();
        let map = ConfidenceMap::build(&pd, &[]);
        assert_eq!(map.scores.len(), 0);
        assert!((map.average_score()).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_level_display() {
        assert_eq!(format!("{}", ConfidenceLevel::Confirmed), "confirmed");
        assert_eq!(format!("{}", ConfidenceLevel::Provisional), "provisional");
        assert_eq!(format!("{}", ConfidenceLevel::Rejected), "rejected");
    }

    #[test]
    fn test_confidence_map_display() {
        let map = ConfidenceMap {
            scores: vec![],
            total_agents: 3,
        };
        let s = format!("{}", map);
        assert!(s.contains("3 agents"));
    }
}
