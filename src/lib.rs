//! # topo-merge
//!
//! **Topological merging of agent belief states using persistent homology.**
//!
//! When multiple agents observe the same system — sensors tracking a target, robots
//! mapping a room, nodes reaching consensus — each forms its own "belief state": a
//! weighted point cloud encoding what it thinks reality looks like. This crate merges
//! those beliefs by asking a structural question: *do these agents see the same shape?*
//!
//! # How It Works
//!
//! 1. Compute the **union** of all agents' belief states into one combined point cloud
//! 2. Build a **distance matrix** (Euclidean or Cosine) over the union
//! 3. Compute a **Vietoris-Rips filtration** — a sequence of simplicial complexes growing
//!    as the distance threshold ε increases
//! 4. Extract a **persistence diagram** — features (connected components, loops) that
//!    are born and die at different ε values
//! 5. **Score each feature** by cross-agent agreement (how many agents saw it),
//!    persistence (how long it survives), and proximity (how close agents' features are)
//! 6. Output a **[ConfidenceMap](confidence::ConfidenceMap)** classifying features as
//!    Confirmed, Provisional, or Rejected
//!
//! # Quick Example
//!
//! ```
//! use topo_merge::{BeliefState, TopologicalMerger};
//!
//! let a = BeliefState::from_coords(&[vec![0.0, 0.0], vec![1.0, 0.0]])
//!     .unwrap().with_agent_id("sensor-a");
//! let b = BeliefState::from_coords(&[vec![0.1, 0.1], vec![1.1, 0.1]])
//!     .unwrap().with_agent_id("sensor-b");
//!
//! let merger = TopologicalMerger::new();
//! let result = merger.merge(&[a, b]).unwrap();
//! println!("Merged {} points, {} persistent features",
//!     result.merged_belief.len(), result.merged_diagram.features.len());
//! println!("Average confidence: {:.3}", result.confidence.average_score());
//! ```
//!
//! # Modules
//!
//! - [`belief`] — Weighted point clouds representing agent beliefs
//! - [`distance`] — Pairwise distance matrices (Euclidean, Cosine)
//! - [`rips`] — Vietoris-Rips complex construction and filtration
//! - [`persistence`] — Persistent homology computation (H₀ and H₁)
//! - [`confidence`] — Feature confidence scoring (Confirmed / Provisional / Rejected)
//! - [`merge`] — Top-level merger combining all steps

pub mod belief;
pub mod confidence;
pub mod distance;
pub mod merge;
pub mod persistence;
pub mod rips;

pub use belief::{BeliefState, WeightedPoint};
pub use confidence::{ConfidenceMap, ConfidenceLevel, FeatureConfidence};
pub use distance::{DistanceMatrix, Metric};
pub use merge::{TopologicalMerger, MergeResult, MergeParams};
pub use persistence::{PersistenceDiagram, PersistenceFeature, Barcode};
pub use rips::VietorisRipsComplex;
