//! Persistent homology computation (H0 and H1) for filtrations.

use crate::distance::DistanceMatrix;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

/// A persistent feature: born at some epsilon, dies at another.
#[derive(Debug, Clone, PartialEq)]
pub struct PersistenceFeature {
    /// Dimension of the feature (0 = connected component, 1 = loop).
    pub dim: usize,
    /// Epsilon value where the feature appears.
    pub birth: f64,
    /// Epsilon value where the feature dies (f64::INFINITY if it persists forever).
    pub death: f64,
}

impl Serialize for PersistenceFeature {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("PersistenceFeature", 3)?;
        state.serialize_field("dim", &self.dim)?;
        state.serialize_field("birth", &self.birth)?;
        if self.death == f64::INFINITY {
            state.serialize_field("death", &None::<f64>)?;
        } else {
            state.serialize_field("death", &Some(self.death))?;
        }
        state.end()
    }
}

impl<'de> Deserialize<'de> for PersistenceFeature {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Helper {
            dim: usize,
            birth: f64,
            death: Option<f64>,
        }
        let h = Helper::deserialize(deserializer)?;
        Ok(PersistenceFeature {
            dim: h.dim,
            birth: h.birth,
            death: h.death.unwrap_or(f64::INFINITY),
        })
    }
}

impl PersistenceFeature {
    /// Persistence = death - birth. Higher means more significant.
    pub fn persistence(&self) -> f64 {
        self.death - self.birth
    }

    /// Whether this feature persists to infinity (never dies).
    pub fn is_infinite(&self) -> bool {
        self.death == f64::INFINITY
    }
}

/// A persistence diagram: collection of persistent features.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PersistenceDiagram {
    /// The features.
    pub features: Vec<PersistenceFeature>,
}

/// Barcode representation of persistence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Barcode {
    /// Bars sorted by birth time.
    pub bars: Vec<PersistenceFeature>,
}

impl PersistenceDiagram {
    /// Create an empty diagram.
    pub fn empty() -> Self {
        Self { features: vec![] }
    }

    /// Compute persistence from a distance matrix.
    ///
    /// H0: Connected components merge as epsilon increases.
    /// H1: Loops form and fill in as epsilon increases.
    pub fn compute(dm: &DistanceMatrix) -> Self {
        let n = dm.n;
        if n <= 1 {
            // Single point: one component born at 0, never dies
            return Self {
                features: vec![PersistenceFeature {
                    dim: 0,
                    birth: 0.0,
                    death: f64::INFINITY,
                }],
            };
        }

        let mut features = Vec::new();

        // === H0: Union-Find persistence ===
        // Sort edges by distance
        let mut edges: Vec<(usize, usize, f64)> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                edges.push((i, j, dm.get(i, j)));
            }
        }
        edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        // Union-find with birth times
        let mut parent: Vec<usize> = (0..n).collect();
        let mut rank: Vec<usize> = vec![0; n];
        let birth_time: Vec<f64> = vec![0.0; n]; // each component born at 0

        fn find(parent: &mut Vec<usize>, x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }

        let mut _components = n;

        for (a, b, dist) in &edges {
            let ra = find(&mut parent, *a);
            let rb = find(&mut parent, *b);
            if ra != rb {
                // The younger component (higher birth) dies
                // Both born at 0, so the one with higher rank survives
                let (survivor, dying) = if rank[ra] >= rank[rb] {
                    (ra, rb)
                } else {
                    (rb, ra)
                };
                features.push(PersistenceFeature {
                    dim: 0,
                    birth: birth_time[dying],
                    death: *dist,
                });
                parent[dying] = survivor;
                if rank[ra] == rank[rb] {
                    rank[survivor] += 1;
                }
                _components -= 1;
            }
        }

        // Remaining components persist to infinity
        for i in 0..n {
            if find(&mut parent, i) == i {
                if features.iter().any(|f| f.dim == 0 && f.death == f64::INFINITY) {
                    continue; // only one infinite component
                }
                // Find the root with lowest index
                let mut roots: Vec<usize> = (0..n).filter(|&j| find(&mut parent, j) == j).collect();
                roots.sort();
                // First root is the survivor
                for &root in &roots {
                    let already = features.iter().any(|f| f.dim == 0 && f.death == f64::INFINITY);
                    if !already {
                        features.push(PersistenceFeature {
                            dim: 0,
                            birth: birth_time[root],
                            death: f64::INFINITY,
                        });
                    } else {
                        // This shouldn't happen if we track correctly
                    }
                }
                break;
            }
        }

        // Ensure exactly one infinite H0 feature
        let inf_count = features.iter().filter(|f| f.dim == 0 && f.death == f64::INFINITY).count();
        if inf_count == 0 {
            features.push(PersistenceFeature {
                dim: 0,
                birth: 0.0,
                death: f64::INFINITY,
            });
        }

        // === H1: Loop detection ===
        // A loop forms when an edge connects two already-connected vertices.
        // The loop "dies" when a triangle fills it in.
        // Simplified: track cycle-creating edges as H1 births.
        // For proper H1, we'd need a boundary matrix reduction.
        // Here we implement a simplified version for small complexes.

        if n >= 3 {
            let h1_features = compute_h1(&edges, dm);
            features.extend(h1_features);
        }

        Self { features }
    }

    /// Get features of a specific dimension.
    pub fn features_by_dim(&self, dim: usize) -> Vec<&PersistenceFeature> {
        self.features.iter().filter(|f| f.dim == dim).collect()
    }

    /// Number of H0 features (connected components over the filtration).
    pub fn h0_count(&self) -> usize {
        self.features.iter().filter(|f| f.dim == 0).count()
    }

    /// Number of H1 features (loops).
    pub fn h1_count(&self) -> usize {
        self.features.iter().filter(|f| f.dim == 1).count()
    }

    /// Convert to barcode representation.
    pub fn barcode(&self) -> Barcode {
        let mut bars = self.features.clone();
        bars.sort_by(|a, b| {
            a.birth
                .partial_cmp(&b.birth)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Barcode { bars }
    }

    /// Bottleneck distance to another diagram.
    ///
    /// The bottleneck distance is the minimum over all matchings of the
    /// maximum distance between matched features.
    pub fn bottleneck_distance(&self, other: &PersistenceDiagram) -> f64 {
        let _all_features: Vec<&PersistenceFeature> = self.features.iter().chain(other.features.iter()).collect();

        // For each dimension separately
        let mut max_dist = 0.0_f64;
        for dim in 0..=1 {
            let fa: Vec<&PersistenceFeature> = self.features.iter().filter(|f| f.dim == dim).collect();
            let fb: Vec<&PersistenceFeature> = other.features.iter().filter(|f| f.dim == dim).collect();

            if fa.is_empty() && fb.is_empty() {
                continue;
            }

            // Compute the optimal matching via greedy approach for small diagrams
            // For production, use Hungarian algorithm. Here we use a simplified approach:
            // Match features greedily by closeness, unmatched features contribute
            // their distance to the diagonal.
            let dist = greedy_bottleneck(&fa, &fb);
            max_dist = max_dist.max(dist);
        }

        max_dist
    }

    /// Most persistent feature (highest persistence value).
    pub fn most_persistent(&self) -> Option<&PersistenceFeature> {
        self.features
            .iter()
            .filter(|f| f.death != f64::INFINITY)
            .max_by(|a, b| {
                a.persistence()
                    .partial_cmp(&b.persistence())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

/// Compute H1 features from the edge list.
/// A simplified approach: detect when edges close cycles.
fn compute_h1(edges: &[(usize, usize, f64)], dm: &DistanceMatrix) -> Vec<PersistenceFeature> {
    let n = dm.n;
    let mut features = Vec::new();

    // Build adjacency incrementally to detect cycle-creating edges
    let mut adj: Vec<std::collections::HashSet<usize>> = vec![std::collections::HashSet::new(); n];
    let mut uf_parent: Vec<usize> = (0..n).collect();

    fn uf_find(parent: &mut Vec<usize>, x: usize) -> usize {
        if parent[x] != x {
            parent[x] = uf_find(parent, parent[x]);
        }
        parent[x]
    }

    for &(a, b, dist) in edges {
        adj[a].insert(b);
        adj[b].insert(a);

        let ra = uf_find(&mut uf_parent, a);
        let rb = uf_find(&mut uf_parent, b);

        if ra == rb {
            // This edge creates a cycle — H1 birth
            // Death: when a triangle fills this cycle
            // Simplified: check if a triangle already exists at this distance
            let death = find_triangle_death(a, b, dist, &adj, dm);
            features.push(PersistenceFeature {
                dim: 1,
                birth: dist,
                death,
            });
        } else {
            uf_parent[ra] = rb;
        }
    }

    features
}

/// Find when a triangle kills a loop containing edge (a, b).
fn find_triangle_death(
    a: usize,
    b: usize,
    _birth_dist: f64,
    adj: &[std::collections::HashSet<usize>],
    dm: &DistanceMatrix,
) -> f64 {
    // A triangle (a, b, c) exists when c is adjacent to both a and b
    let common_neighbors: Vec<usize> = adj[a]
        .iter()
        .filter(|&&c| c != b && adj[b].contains(&c))
        .copied()
        .collect();

    if common_neighbors.is_empty() {
        f64::INFINITY // loop never dies
    } else {
        // Death is the maximum edge length in the smallest enclosing triangle
        common_neighbors
            .iter()
            .map(|&c| {
                let d_ac = dm.get(a, c);
                let d_bc = dm.get(b, c);
                d_ac.max(d_bc)
            })
            .fold(f64::INFINITY, f64::min)
    }
}

/// Greedy bottleneck distance approximation.
fn greedy_bottleneck(a: &[&PersistenceFeature], b: &[&PersistenceFeature]) -> f64 {
    let mut max_dist = 0.0_f64;

    // Diagonal distance for a point (birth, death) is (death - birth) / sqrt(2)
    let diag_dist = |f: &PersistenceFeature| {
        if f.death == f64::INFINITY {
            f64::INFINITY
        } else {
            (f.death - f.birth) / std::f64::consts::SQRT_2
        }
    };

    let point_dist = |f1: &PersistenceFeature, f2: &PersistenceFeature| -> f64 {
        let b = (f1.birth - f2.birth).abs();
        let d = if f1.death == f64::INFINITY && f2.death == f64::INFINITY {
            0.0
        } else if f1.death == f64::INFINITY || f2.death == f64::INFINITY {
            f64::INFINITY
        } else {
            (f1.death - f2.death).abs()
        };
        (b * b + d * d).sqrt()
    };

    if a.is_empty() {
        return b.iter().map(|f| diag_dist(f)).fold(0.0_f64, f64::max);
    }
    if b.is_empty() {
        return a.iter().map(|f| diag_dist(f)).fold(0.0_f64, f64::max);
    }

    // Greedy matching: for each feature in a, find closest in b
    let mut used = vec![false; b.len()];
    for &fa in a {
        let mut best_dist = f64::INFINITY;
        let mut best_j = 0;
        for (j, &fb) in b.iter().enumerate() {
            if used[j] {
                continue;
            }
            let d = point_dist(fa, fb);
            if d < best_dist {
                best_dist = d;
                best_j = j;
            }
        }
        if best_dist < f64::INFINITY && !b.is_empty() {
            used[best_j] = true;
            max_dist = max_dist.max(best_dist);
        } else {
            max_dist = max_dist.max(diag_dist(fa));
        }
    }
    // Unmatched b features
    for (j, &fb) in b.iter().enumerate() {
        if !used[j] {
            max_dist = max_dist.max(diag_dist(fb));
        }
    }

    max_dist
}

impl fmt::Display for PersistenceDiagram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PersistenceDiagram(H0={}, H1={})", self.h0_count(), self.h1_count())
    }
}

impl fmt::Display for PersistenceFeature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let death_str = if self.death == f64::INFINITY {
            "∞".to_string()
        } else {
            format!("{:.3}", self.death)
        };
        write!(
            f,
            "H{}({:.3} → {}, pers={:.3})",
            self.dim,
            self.birth,
            death_str,
            self.persistence()
        )
    }
}

impl fmt::Display for Barcode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Barcode({} bars):", self.bars.len())?;
        for bar in &self.bars {
            writeln!(f, "  {}", bar)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::Metric;

    #[test]
    fn test_single_point() {
        let dm = DistanceMatrix::compute(&[vec![0.0, 0.0]], Metric::Euclidean).unwrap();
        let pd = PersistenceDiagram::compute(&dm);
        assert_eq!(pd.h0_count(), 1);
        assert!(pd.features[0].is_infinite());
    }

    #[test]
    fn test_two_points() {
        let dm = DistanceMatrix::compute(&[vec![0.0], vec![1.0]], Metric::Euclidean).unwrap();
        let pd = PersistenceDiagram::compute(&dm);
        assert_eq!(pd.h0_count(), 2);
        // One component dies at dist=1.0, one persists
        let h0: Vec<_> = pd.features_by_dim(0);
        let finite: Vec<_> = h0.iter().filter(|f| !f.is_infinite()).collect();
        assert_eq!(finite.len(), 1);
        assert!((finite[0].death - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_three_points_line() {
        let dm = DistanceMatrix::compute(
            &[vec![0.0], vec![1.0], vec![2.0]],
            Metric::Euclidean,
        )
        .unwrap();
        let pd = PersistenceDiagram::compute(&dm);
        let h0 = pd.features_by_dim(0);
        assert_eq!(h0.len(), 3);
        // Two finite components die, one persists
        let finite: Vec<_> = h0.iter().filter(|f| !f.is_infinite()).collect();
        assert_eq!(finite.len(), 2);
    }

    #[test]
    fn test_triangle_h0() {
        let dm = DistanceMatrix::compute(
            &[vec![0.0, 0.0], vec![1.0, 0.0], vec![0.5, 0.866]],
            Metric::Euclidean,
        )
        .unwrap();
        let pd = PersistenceDiagram::compute(&dm);
        assert_eq!(pd.h0_count(), 3); // 2 finite + 1 infinite
    }

    #[test]
    fn test_empty_diagram() {
        let pd = PersistenceDiagram::empty();
        assert_eq!(pd.features.len(), 0);
    }

    #[test]
    fn test_persistence_value() {
        let f = PersistenceFeature {
            dim: 0,
            birth: 1.0,
            death: 3.0,
        };
        assert!((f.persistence() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_infinite_persistence() {
        let f = PersistenceFeature {
            dim: 0,
            birth: 0.0,
            death: f64::INFINITY,
        };
        assert!(f.is_infinite());
        assert_eq!(f.persistence(), f64::INFINITY);
    }

    #[test]
    fn test_barcode() {
        let dm = DistanceMatrix::compute(&[vec![0.0], vec![1.0]], Metric::Euclidean).unwrap();
        let pd = PersistenceDiagram::compute(&dm);
        let bc = pd.barcode();
        assert!(!bc.bars.is_empty());
        // Bars should be sorted by birth
        for i in 1..bc.bars.len() {
            assert!(bc.bars[i].birth >= bc.bars[i - 1].birth);
        }
    }

    #[test]
    fn test_most_persistent() {
        let dm = DistanceMatrix::compute(
            &[vec![0.0], vec![1.0], vec![10.0]],
            Metric::Euclidean,
        )
        .unwrap();
        let pd = PersistenceDiagram::compute(&dm);
        let mp = pd.most_persistent().unwrap();
        // Most persistent finite feature should be the merge of the close pair
        assert!(mp.persistence() > 0.0);
    }

    #[test]
    fn test_bottleneck_identical() {
        let dm = DistanceMatrix::compute(&[vec![0.0], vec![1.0]], Metric::Euclidean).unwrap();
        let pd1 = PersistenceDiagram::compute(&dm);
        let pd2 = PersistenceDiagram::compute(&dm);
        let dist = pd1.bottleneck_distance(&pd2);
        assert!(dist < 1e-10, "identical diagrams should have zero bottleneck distance, got {}", dist);
    }

    #[test]
    fn test_display_diagram() {
        let dm = DistanceMatrix::compute(&[vec![0.0], vec![1.0]], Metric::Euclidean).unwrap();
        let pd = PersistenceDiagram::compute(&dm);
        let s = format!("{}", pd);
        assert!(s.contains("H0="));
    }

    #[test]
    fn test_display_feature() {
        let f = PersistenceFeature {
            dim: 0,
            birth: 1.0,
            death: 2.0,
        };
        let s = format!("{}", f);
        assert!(s.contains("H0"));
    }

    #[test]
    fn test_display_barcode() {
        let dm = DistanceMatrix::compute(&[vec![0.0], vec![1.0]], Metric::Euclidean).unwrap();
        let pd = PersistenceDiagram::compute(&dm);
        let bc = pd.barcode();
        let s = format!("{}", bc);
        assert!(s.contains("bars"));
    }

    #[test]
    fn test_serialize_deserialize() {
        let dm = DistanceMatrix::compute(&[vec![0.0], vec![1.0]], Metric::Euclidean).unwrap();
        let pd = PersistenceDiagram::compute(&dm);
        let json = serde_json::to_string(&pd).unwrap();
        let pd2: PersistenceDiagram = serde_json::from_str(&json).unwrap();
        assert_eq!(pd, pd2);
    }
}
