# topo-merge

**Topological merging of agent belief states using persistent homology.**

When multiple autonomous agents observe the same system — sensors tracking a target, robots mapping a room, nodes reaching consensus — each forms its own "belief state": a point cloud encoding what it thinks reality looks like. The question is: *how do you merge them without picking favorites?*

`topo-merge` answers this with a mathematical insight from topological data analysis (TDA): **the shape of consensus is encoded in persistent homology**. Features that survive across scales — connected components that refuse to merge, loops that persist — are the real structure shared by all agents. Noise and hallucinations die early.

## The Insight: Shape Agreement > Point Agreement

Most multi-agent fusion averages coordinates. That works when agents agree on a reference frame, but breaks under rotation, scaling, or partial observation.

`topo-merge` ignores *where* points are and asks *what shape they form*. It builds a [Vietoris-Rips complex](https://en.wikipedia.org/wiki/Vietoris%E2%80%93Rips_complex) — a simplicial complex that grows as you increase a distance threshold ε. As ε rises, isolated points merge into edges, edges into triangles, and the "persistence" of each feature (how long it survives before merging with something bigger) measures its significance.

Features seen by many agents with high persistence get a **Confirmed** confidence level. Features that only one agent sees or that die quickly get **Rejected**. This gives you a principled, noise-robust merge — no voting hacks, no ad-hoc thresholds.

```
                    ε = 0           ε = 0.5          ε = 1.0
                    · · ·           ·-· ·            ·-·-·
  Agent A:          · ·             ·-·              ·-·
  Agent B:           · ·             ·-·              ·-·

  Merge:            5 components    3 components     1 component
                    H₀ features:    4 die, 1 lives   ← persistence!
```

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Agent A     │     │  Agent B     │     │  Agent C     │
│  BeliefState │     │  BeliefState │     │  BeliefState │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                     │
       └──────────┬─────────┴─────────────────────┘
                  │  union()
                  ▼
       ┌─────────────────────┐
       │  Union BeliefState  │  ← all points combined
       └──────────┬──────────┘
                  │  DistanceMatrix::compute()
                  ▼
       ┌─────────────────────┐
       │  Distance Matrix    │  ← pairwise Euclidean / Cosine
       └──────────┬──────────┘
                  │
         ┌────────┴────────┐
         ▼                 ▼
┌─────────────────┐  ┌─────────────────┐
│  Merged         │  │  Individual     │
│  PersistenceDiagram  │  PersistenceDiagrams  │
│  (from union)   │  │  (per agent)    │
└────────┬────────┘  └────────┬────────┘
         │                    │
         └────────┬───────────┘
                  │  ConfidenceMap::build()
                  ▼
       ┌─────────────────────┐
       │  ConfidenceMap      │
       │  Confirmed ✓        │
       │  Provisional ~      │
       │  Rejected ✗         │
       └─────────────────────┘
```

## Quick Start

```toml
[dependencies]
topo-merge = "0.1"
```

```rust
use topo_merge::{BeliefState, TopologicalMerger, MergeParams};

fn main() {
    // Three agents observe the same system in 2D
    let agent_a = BeliefState::from_coords(&[
        vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0],
    ]).unwrap().with_agent_id("sensor-1");

    let agent_b = BeliefState::from_coords(&[
        vec![0.1, 0.1], vec![1.1, 0.1], vec![0.1, 1.1],
    ]).unwrap().with_agent_id("sensor-2");

    let agent_c = BeliefState::from_coords(&[
        vec![5.0, 5.0],  // outlier — only agent C sees this
    ]).unwrap().with_agent_id("sensor-3");

    // Merge with default parameters
    let merger = TopologicalMerger::new();
    let result = merger.merge(&[agent_a, agent_b, agent_c]).unwrap();

    println!("Merged {} points from {} agents",
        result.merged_belief.len(), result.agent_count);
    println!("Persistence: {}", result.merged_diagram);
    println!("Confidence: {}", result.confidence);

    // Check which features are real consensus
    for fc in result.confidence.confirmed() {
        println!("✓ H{} feature [birth={:.2}, death={:.2}] score={:.3}",
            fc.feature.dim, fc.feature.birth, fc.feature.death, fc.score);
    }
}
```

## Tutorial

### Weighted Beliefs

Not all observations are equal. Agents can assign weights to their points — a high-confidence sensor reading gets more weight than a noisy one.

```rust
use topo_merge::{BeliefState, WeightedPoint};

let belief = BeliefState::new(vec![
    WeightedPoint::new(vec![0.0, 0.0], 10.0),  // high confidence
    WeightedPoint::new(vec![1.0, 0.0], 1.0),   // low confidence
]).unwrap().with_agent_id("lidar");
```

Weights propagate through the merge: `total_weight()` and `centroid()` are weighted, so trusted observations pull the consensus toward them.

### Choosing a Metric

By default, `topo-merge` uses Euclidean distance. For high-dimensional or direction-sensitive data, switch to Cosine distance:

```rust
use topo_merge::{TopologicalMerger, MergeParams, distance::Metric};

let params = MergeParams {
    metric: Metric::Cosine,
    consensus_threshold: 2,
};
let merger = TopologicalMerger::with_params(params);
```

Cosine distance (1 − cos θ) treats vectors of different magnitudes but the same direction as close — ideal for embedding spaces.

### Reading the Persistence Diagram

A `PersistenceDiagram` contains features of dimension 0 (connected components) and 1 (loops). Each feature has a *birth* ε and a *death* ε:

| Field | Meaning |
|-------|---------|
| `dim: 0` | A connected component appears then merges with another |
| `dim: 1` | A loop appears then fills in (becomes a triangle) |
| `birth` | ε threshold where the feature first appears |
| `death` | ε threshold where the feature disappears |
| `death = ∞` | The feature never dies — it's a permanent part of the shape |

**Rule of thumb:** High persistence (death − birth) = significant structure. Low persistence = noise.

### Confidence Scoring

`ConfidenceMap` scores each merged feature on a 0–1 scale using three factors:

| Factor | Weight | Meaning |
|--------|--------|---------|
| Agent agreement | 40% | Fraction of agents that observed this feature |
| Persistence | 40% | How long the feature survives relative to the max |
| Proximity | 20% | How closely agents' individual features match |

The combined score maps to a level:

| Level | Score Range | Action |
|-------|-------------|--------|
| **Confirmed** | ≥ 0.6 | Trust this feature — it's real |
| **Provisional** | 0.3–0.6 | Worth considering — weak evidence |
| **Rejected** | < 0.3 | Likely noise — ignore |

### The Vietoris-Rips Complex

Under the hood, `topo-merge` builds a [Vietoris-Rips complex](https://en.wikipedia.org/wiki/Vietoris%E2%80%93Rips_complex) at each ε threshold. You can inspect the filtration directly:

```rust
use topo_merge::{BeliefState, distance::{DistanceMatrix, Metric}};
use topo_merge::rips::VietorisRipsComplex;

let coords = &[vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
let dm = DistanceMatrix::compute(coords, Metric::Euclidean).unwrap();

// Full filtration: one complex per unique distance
let filtration = VietorisRipsComplex::filtration(&dm);
for rips in &filtration {
    println!("ε={:.2}: {} vertices, {} edges, {} triangles, {} components",
        rips.epsilon, rips.vertices.len(), rips.edges.len(),
        rips.triangles.len(), rips.connected_components());
}

// Or build at a specific radius
let rips = VietorisRipsComplex::build(&dm, 1.0);
println!("Euler characteristic: {}", rips.euler_characteristic());
```

### Comparing Topologies: Bottleneck Distance

The bottleneck distance between two persistence diagrams measures how different their topological signatures are. Identical observations → distance 0.

```rust
let dist = diagram_a.bottleneck_distance(&diagram_b);
println!("Topological distance: {:.4}", dist);
```

## API Reference

### Core Types

| Type | Module | Description |
|------|--------|-------------|
| `BeliefState` | `belief` | Weighted point cloud representing an agent's belief |
| `WeightedPoint` | `belief` | A single point with coordinates and a weight |
| `DistanceMatrix` | `distance` | Upper-triangle pairwise distance storage |
| `Metric` | `distance` | `Euclidean` or `Cosine` distance |
| `VietorisRipsComplex` | `rips` | Simplicial complex at a given ε threshold |
| `PersistenceDiagram` | `persistence` | Collection of persistent features (H₀ and H₁) |
| `PersistenceFeature` | `persistence` | A single feature with birth, death, and dimension |
| `Barcode` | `persistence` | Sorted barcode representation |
| `ConfidenceMap` | `confidence` | Feature → confidence score mapping |
| `FeatureConfidence` | `confidence` | Scored feature with level (Confirmed/Provisional/Rejected) |
| `TopologicalMerger` | `merge` | Main entry point: merges belief states |
| `MergeResult` | `merge` | Output of a merge: belief + diagram + confidence |
| `MergeParams` | `merge` | Configurable merge parameters |

### Key Methods

```rust
// BeliefState
BeliefState::from_coords(&[...])          // Quick construction
BeliefState::new(points)?.with_agent_id()  // Full construction
belief.centroid()?                         // Weighted centroid
belief.spread()?                           // RMS distance from centroid
belief.union(&other)?                      // Combine two beliefs

// TopologicalMerger
let merger = TopologicalMerger::new();
let result = merger.merge(&[a, b, c])?;    // Full merge with persistence
let quick = merger.quick_merge(&[a, b])?;  // Fast union-only merge

// PersistenceDiagram
diagram.features_by_dim(0)                 // H₀ features
diagram.h0_count()                         // Number of components
diagram.h1_count()                         // Number of loops
diagram.most_persistent()                  // Highest persistence feature
diagram.bottleneck_distance(&other)        // Compare topologies

// ConfidenceMap
map.confirmed()                            // Features with score ≥ 0.6
map.provisional()                          // Features with score 0.3–0.6
map.rejected()                             // Features with score < 0.3
map.average_score()                        // Mean confidence across features

// DistanceMatrix
dm.get(i, j)                               // Distance between points i and j
dm.sorted_distances()                      // All distances sorted ascending
dm.max_distance()                          // Largest pairwise distance
```

## Ecosystem Role

`topo-merge` is the **topological consensus layer** in the SuperInstance ecosystem:

- **Input:** Agent belief states from [`constraint-schedule`](https://github.com/SuperInstance/constraint-schedule) task assignments or direct sensor feeds
- **Output:** Merged beliefs with confidence scores, consumed by downstream decision-making
- **Complementary to:** Statistical averaging (use `topo-merge` when you need *structural* agreement, not just numerical)

In a multi-agent system, `topo-merge` answers the question: *"Do these agents see the same shape, even if they disagree on exact coordinates?"* That's the right question when agents have different sensor modalities, reference frames, or partial observability.

## Serde Support

All major types implement `Serialize` and `Deserialize`. Persistence features with infinite death serialize as `null` in JSON.

```rust
let json = serde_json::to_string(&result)?;
let restored: MergeResult = serde_json::from_str(&json)?;
```

## License

MIT
