//! # topo-merge
//!
//! Topological merging of agent belief states using persistent homology.
//!
//! When multiple agents observe the same system, they form individual "belief states"
//! (point clouds in some metric space). This crate uses topological data analysis
//! (persistent homology) to merge these beliefs: compute the Vietoris-Rips complex
//! of the union, find persistent features, and output a merged belief with confidence
//! scores derived from persistence.

pub mod belief;
pub mod confidence;
pub mod distance;
pub mod merge;
pub mod persistence;
pub mod rips;

pub use belief::BeliefState;
pub use confidence::{ConfidenceMap, ConfidenceLevel};
pub use distance::{DistanceMatrix, Metric};
pub use merge::{TopologicalMerger, MergeResult};
pub use persistence::{PersistenceDiagram, PersistenceFeature, Barcode};
pub use rips::VietorisRipsComplex;
