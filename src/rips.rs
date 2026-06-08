//! Vietoris-Rips complex construction from distance matrices.

use crate::distance::DistanceMatrix;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A simplex (vertex, edge, or triangle) in the Rips complex.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Simplex {
    /// 0-simplex (vertex).
    Vertex(usize),
    /// 1-simplex (edge).
    Edge(usize, usize),
    /// 2-simplex (triangle).
    Triangle(usize, usize, usize),
}

impl Simplex {
    /// The dimension of the simplex (0, 1, or 2).
    pub fn dim(&self) -> usize {
        match self {
            Simplex::Vertex(_) => 0,
            Simplex::Edge(_, _) => 1,
            Simplex::Triangle(_, _, _) => 2,
        }
    }

    /// The vertex indices in the simplex.
    pub fn vertices(&self) -> Vec<usize> {
        match self {
            Simplex::Vertex(v) => vec![*v],
            Simplex::Edge(a, b) => vec![*a, *b],
            Simplex::Triangle(a, b, c) => vec![*a, *b, *c],
        }
    }

    /// The filtration value (epsilon) at which this simplex first appears.
    pub fn birth_radius(&self, dm: &DistanceMatrix) -> f64 {
        match self {
            Simplex::Vertex(_) => 0.0,
            Simplex::Edge(a, b) => dm.get(*a, *b),
            Simplex::Triangle(a, b, c) => {
                let d_ab = dm.get(*a, *b);
                let d_ac = dm.get(*a, *c);
                let d_bc = dm.get(*b, *c);
                d_ab.max(d_ac).max(d_bc)
            }
        }
    }
}

/// Vietoris-Rips complex at a given radius epsilon.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VietorisRipsComplex {
    /// The radius parameter epsilon.
    pub epsilon: f64,
    /// Vertices (all points are always included).
    pub vertices: Vec<usize>,
    /// Edges where distance <= epsilon.
    pub edges: Vec<(usize, usize)>,
    /// Triangles where all pairwise distances <= epsilon.
    pub triangles: Vec<(usize, usize, usize)>,
    /// Number of points in the original point cloud.
    pub n_points: usize,
}

impl VietorisRipsComplex {
    /// Build the Rips complex from a distance matrix at radius epsilon.
    pub fn build(dm: &DistanceMatrix, epsilon: f64) -> Self {
        let n = dm.n;
        let vertices: Vec<usize> = (0..n).collect();

        // Find edges
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if dm.get(i, j) <= epsilon {
                    edges.push((i, j));
                }
            }
        }

        // Find triangles: for each edge (i,j), check if any k forms a triangle
        let mut triangles = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if dm.get(i, j) > epsilon {
                    continue;
                }
                for k in (j + 1)..n {
                    if dm.get(i, k) <= epsilon && dm.get(j, k) <= epsilon {
                        triangles.push((i, j, k));
                    }
                }
            }
        }

        Self {
            epsilon,
            vertices,
            edges,
            triangles,
            n_points: n,
        }
    }

    /// Build a filtration: Rips complexes at multiple epsilon values.
    ///
    /// Uses all unique distances as threshold values, plus 0.0.
    pub fn filtration(dm: &DistanceMatrix) -> Vec<Self> {
        let mut epsilons = vec![0.0];
        let mut dists: Vec<f64> = dm.data.to_vec();
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        dists.dedup();
        epsilons.extend(dists);

        epsilons.into_iter().map(|eps| Self::build(dm, eps)).collect()
    }

    /// Number of simplices total.
    pub fn simplex_count(&self) -> usize {
        self.vertices.len() + self.edges.len() + self.triangles.len()
    }

    /// All simplices in the complex.
    pub fn simplices(&self) -> Vec<Simplex> {
        let mut s = Vec::new();
        for &v in &self.vertices {
            s.push(Simplex::Vertex(v));
        }
        for &(a, b) in &self.edges {
            s.push(Simplex::Edge(a, b));
        }
        for &(a, b, c) in &self.triangles {
            s.push(Simplex::Triangle(a, b, c));
        }
        s
    }

    /// Euler characteristic: V - E + F.
    pub fn euler_characteristic(&self) -> i32 {
        self.vertices.len() as i32 - self.edges.len() as i32 + self.triangles.len() as i32
    }

    /// Number of connected components (for H0 computation).
    /// Uses union-find.
    pub fn connected_components(&self) -> usize {
        if self.vertices.is_empty() {
            return 0;
        }
        let n = self.n_points;
        let mut parent: Vec<usize> = (0..n).collect();

        fn find(parent: &mut Vec<usize>, x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }

        for &(a, b) in &self.edges {
            let ra = find(&mut parent, a);
            let rb = find(&mut parent, b);
            if ra != rb {
                parent[ra] = rb;
            }
        }

        let mut roots = std::collections::HashSet::new();
        for i in 0..n {
            roots.insert(find(&mut parent, i));
        }
        roots.len()
    }
}

impl fmt::Display for VietorisRipsComplex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Rips(eps={:.3}, V={}, E={}, T={}, components={})",
            self.epsilon,
            self.vertices.len(),
            self.edges.len(),
            self.triangles.len(),
            self.connected_components()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::Metric;

    fn triangle_dm() -> DistanceMatrix {
        DistanceMatrix::compute(
            &[vec![0.0, 0.0], vec![1.0, 0.0], vec![0.5, 0.866]],
            Metric::Euclidean,
        )
        .unwrap()
    }

    #[test]
    fn test_rips_zero_epsilon() {
        let dm = triangle_dm();
        let rips = VietorisRipsComplex::build(&dm, 0.0);
        assert_eq!(rips.vertices.len(), 3);
        assert!(rips.edges.is_empty());
        assert!(rips.triangles.is_empty());
        assert_eq!(rips.connected_components(), 3);
    }

    #[test]
    fn test_rips_small_epsilon() {
        let dm = triangle_dm();
        let rips = VietorisRipsComplex::build(&dm, 0.9);
        assert_eq!(rips.vertices.len(), 3);
        // All sides of equilateral triangle with side 1.0 are > 0.9? Let's check
        // Actually the side length is 1.0, so at eps=0.9, no edges
        assert!(rips.edges.is_empty());
    }

    #[test]
    fn test_rips_full_triangle() {
        let dm = triangle_dm();
        let rips = VietorisRipsComplex::build(&dm, 1.5);
        assert_eq!(rips.vertices.len(), 3);
        assert_eq!(rips.edges.len(), 3);
        assert_eq!(rips.triangles.len(), 1);
        assert_eq!(rips.connected_components(), 1);
    }

    #[test]
    fn test_rips_two_components() {
        let coords = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![10.0, 0.0],
            vec![11.0, 0.0],
        ];
        let dm = DistanceMatrix::compute(&coords, Metric::Euclidean).unwrap();
        let rips = VietorisRipsComplex::build(&dm, 1.5);
        // Two clusters, each connected internally
        assert_eq!(rips.edges.len(), 2);
        assert_eq!(rips.connected_components(), 2);
    }

    #[test]
    fn test_filtration() {
        let dm = triangle_dm();
        let filt = VietorisRipsComplex::filtration(&dm);
        assert!(filt.len() >= 2);
        // Filtration should be non-decreasing in simplex count
        for i in 1..filt.len() {
            assert!(filt[i].epsilon >= filt[i - 1].epsilon);
        }
    }

    #[test]
    fn test_euler_characteristic() {
        let dm = triangle_dm();
        let rips = VietorisRipsComplex::build(&dm, 1.5);
        // V=3, E=3, F=1 => euler = 1
        assert_eq!(rips.euler_characteristic(), 1);
    }

    #[test]
    fn test_empty_point_cloud() {
        let dm = DistanceMatrix::compute(&[], Metric::Euclidean).unwrap();
        let rips = VietorisRipsComplex::build(&dm, 1.0);
        assert!(rips.vertices.is_empty());
        assert!(rips.edges.is_empty());
        assert_eq!(rips.connected_components(), 0);
    }

    #[test]
    fn test_single_point() {
        let dm = DistanceMatrix::compute(&[vec![0.0, 0.0]], Metric::Euclidean).unwrap();
        let rips = VietorisRipsComplex::build(&dm, 1.0);
        assert_eq!(rips.vertices.len(), 1);
        assert!(rips.edges.is_empty());
        assert_eq!(rips.connected_components(), 1);
    }

    #[test]
    fn test_simplex_dim() {
        assert_eq!(Simplex::Vertex(0).dim(), 0);
        assert_eq!(Simplex::Edge(0, 1).dim(), 1);
        assert_eq!(Simplex::Triangle(0, 1, 2).dim(), 2);
    }

    #[test]
    fn test_simplex_vertices() {
        let s = Simplex::Triangle(1, 2, 3);
        assert_eq!(s.vertices(), vec![1, 2, 3]);
    }

    #[test]
    fn test_birth_radius() {
        let dm = triangle_dm();
        let edge = Simplex::Edge(0, 1);
        let r = edge.birth_radius(&dm);
        assert!(r > 0.0);
        assert!(r <= 1.1);
    }

    #[test]
    fn test_simplices_method() {
        let dm = triangle_dm();
        let rips = VietorisRipsComplex::build(&dm, 1.5);
        let simplices = rips.simplices();
        assert_eq!(simplices.len(), 7); // 3V + 3E + 1T
    }

    #[test]
    fn test_display() {
        let dm = triangle_dm();
        let rips = VietorisRipsComplex::build(&dm, 1.5);
        let s = format!("{}", rips);
        assert!(s.contains("V=3"));
        assert!(s.contains("E=3"));
    }
}
