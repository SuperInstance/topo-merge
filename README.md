# topo-merge

**Topological merging of agent belief states — persistent homology for multi-agent consensus.**

[![crates.io](https://img.shields.io/crates/v/topo-merge.svg)](https://crates.io/crates/topo-merge)
[![docs.rs](https://docs.rs/topo-merge/badge.svg)](https://docs.rs/topo-merge)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

When multiple agents observe the same system, they form individual "belief states" — point clouds in some metric space. `topo-merge` uses **topological data analysis (persistent homology)** to merge these beliefs, capturing structural features that simple averaging completely misses.

## Why Topology?

Consider three agents observing a system. Two agents see a ring-shaped cluster; one sees a blob. Averaging the points produces... a blob. The ring structure — a genuine topological feature — is destroyed.

**Persistent homology** solves this. It detects features (connected components, loops, voids) at multiple scales and measures how long they survive as the scale changes. Features that persist across scales are signal, not noise. Features observed by multiple agents at similar scales are *consensus*.

This matters because:
- **Averages collapse topology.** The average of a ring is its center — zero-dimensional information lost.
- **Topology captures connectivity and holes.** If your agents detect a gap in sensor coverage, that's a 1-dimensional feature (a loop), not a cluster center.
- **Persistence separates signal from noise.** Short-lived features are artifacts; long-lived features are real structure.

## The Merge Algorithm

```
1. Take N belief states (point clouds with weights)
2. Compute the union of all points
3. Build a distance matrix (Euclidean or cosine)
4. Construct a Vietoris-Rips filtration:
   - At radius ε=0, every point is its own connected component
   - As ε grows, nearby points connect via edges
   - At larger ε, triangles (2-simplices) fill in
5. Compute persistent homology:
   - H₀: When do connected components merge? (birth = 0, death = merge distance)
   - H₁: When do loops form and fill in? (birth = cycle edge, death = filling triangle)
6. Compare each feature against individual agents' diagrams
7. Score features by:
   - How many agents observed it (consensus)
   - How persistent it is (death − birth)
   - How closely agents agree on the feature's scale (proximity)
8. Output: merged belief + persistence diagram + confidence scores
```

### Confidence Scoring

Each merged feature gets a confidence score (0–1):

| Component | Weight | Description |
|-----------|--------|-------------|
| Agent agreement | 40% | Fraction of agents that observed a similar feature |
| Persistence | 40% | How long the feature survives relative to the longest-lived feature |
| Proximity | 20% | How closely agents' individual features match in birth/death time |

Features are thresholded into three levels:
- **Confirmed** (≥0.6): Strong consensus, high persistence
- **Provisional** (≥0.3): Some evidence, moderate persistence
- **Rejected** (<0.3): Weak evidence or ephemeral

## Quick Start

```rust
use topo_merge::{BeliefState, TopologicalMerger, MergeParams, Metric};

// Two agents observing the same system
let agent_a = BeliefState::from_coords(&[
    vec![0.0, 0.0],
    vec![1.0, 0.0],
    vec![0.5, 0.866],
])?.with_agent_id("a");

let agent_b = BeliefState::from_coords(&[
    vec![0.1, 0.1],
    vec![1.1, 0.1],
    vec![0.6, 0.9],
])?.with_agent_id("b");

// Merge with default parameters
let merger = TopologicalMerger::new();
let result = merger.merge(&[agent_a, agent_b])?;

// Inspect the merged belief
println!("Merged {} points from {} agents",
    result.merged_belief.len(),
    result.agent_count);

// Check which features are confirmed
for score in result.confidence.confirmed() {
    println!("Confirmed H{} feature: birth={:.2}, death={:.2}",
        score.feature.dim,
        score.feature.birth,
        score.feature.death);
}

// The persistence diagram shows the topological structure
println!("Persistence: {}", result.merged_diagram);
```

## Architecture

| Module | Responsibility |
|--------|---------------|
| `belief` | `BeliefState` — weighted point cloud in Rⁿ with centroid, spread, union |
| `distance` | `DistanceMatrix` — upper-triangle storage, Euclidean and cosine metrics |
| `rips` | `VietorisRipsComplex` — simplicial complex at radius ε, filtration builder |
| `persistence` | `PersistenceDiagram` — H₀/H₁ computation, barcode, bottleneck distance |
| `confidence` | `ConfidenceMap` — feature-to-confidence scoring with confirmed/provisional/rejected |
| `merge` | `TopologicalMerger` — orchestrates the full pipeline from beliefs to scored consensus |

## Persistence Diagrams Explained

A persistence diagram plots features as points in (birth, death) space:

- **Points near the diagonal** (death ≈ birth) are short-lived — likely noise
- **Points far from the diagonal** are persistent — likely real structure
- **Points at death = ∞** survive forever (the "essential" component)

The **barcode** representation shows the same information as horizontal bars from birth to death. Longer bars = more significant features.

### Bottleneck Distance

The **bottleneck distance** measures how different two persistence diagrams are. It finds the optimal matching between features and reports the maximum distance. Identical observations → distance 0. Completely different structure → large distance.

This is useful for detecting when agents disagree about the underlying topology.

## API Overview

```rust
// Belief states
let bs = BeliefState::from_coords(&[vec![0.0], vec![1.0]])?;
let centroid = bs.centroid()?;
let spread = bs.spread()?;
let union = bs.union(&other)?;

// Distance matrices
let dm = DistanceMatrix::compute(&coords, Metric::Euclidean)?;
let dm = DistanceMatrix::compute(&coords, Metric::Cosine)?;

// Rips complexes
let rips = VietorisRipsComplex::build(&dm, epsilon);
let filtration = VietorisRipsComplex::filtration(&dm);

// Persistent homology
let diagram = PersistenceDiagram::compute(&dm);
let barcode = diagram.barcode();
let distance = diagram.bottleneck_distance(&other_diagram);

// Merging
let result = TopologicalMerger::new().merge(&[bs1, bs2, bs3])?;
let result = TopologicalMerger::with_params(MergeParams {
    metric: Metric::Cosine,
    consensus_threshold: 2,
}).merge(&beliefs)?;
```

## Properties

- **Pure safe Rust** — no unsafe blocks, no external math libraries
- **Serializable** — all core types implement `serde::Serialize`/`Deserialize`
- **O(n²) space** for distance matrix and Rips complex (suitable for hundreds to low thousands of points)
- **H₀** computed exactly via union-find
- **H₁** computed via cycle detection in the filtered complex

## Limitations

- H₁ computation is simplified — for large complexes, consider specialized TDA libraries
- No H₂+ (higher-dimensional homology) — the Rips complex is truncated at 2-simplices
- O(n²) memory — not suitable for very large point clouds (>10K points)
- Bottleneck distance uses greedy matching, not the optimal Hungarian algorithm

## License

MIT
