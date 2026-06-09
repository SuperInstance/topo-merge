//! A guided walkthrough of topological belief merging.
//!
//! Run with: cargo run --example tutorial

use topo_merge::{
    BeliefState, ConfidenceLevel, DistanceMatrix, MergeParams, Metric,
    PersistenceDiagram, TopologicalMerger, VietorisRipsComplex, WeightedPoint,
};

fn main() {
    println!("=== topo-merge Tutorial ===\n");

    // --- Step 1: Create belief states for three agents ---
    println!("Step 1: Creating belief states for three sensors\n");

    let sensor_a = BeliefState::from_coords(&[
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
    ])
    .unwrap()
    .with_agent_id("lidar-front");

    let sensor_b = BeliefState::from_coords(&[
        vec![0.1, 0.1],
        vec![1.1, 0.0],
        vec![0.0, 1.1],
    ])
    .unwrap()
    .with_agent_id("lidar-rear");

    // Agent C is a noisy camera with one outlier
    let sensor_c = BeliefState::new(vec![
        WeightedPoint::new(vec![0.05, 0.05], 5.0),
        WeightedPoint::new(vec![1.0, 0.05], 5.0),
        WeightedPoint::new(vec![0.05, 1.0], 5.0),
        WeightedPoint::new(vec![10.0, 10.0], 1.0), // outlier
    ])
    .unwrap()
    .with_agent_id("camera");

    println!("  {}", sensor_a);
    println!("  {}", sensor_b);
    println!("  {}\n", sensor_c);

    // --- Step 2: Quick merge (union only, no topology) ---
    println!("Step 2: Quick merge (union only)\n");
    let merger = TopologicalMerger::new();
    let quick = merger.quick_merge(&[sensor_a.clone(), sensor_b.clone()]).unwrap();
    println!("  Quick merged: {} points\n", quick.len());

    // --- Step 3: Full topological merge ---
    println!("Step 3: Full topological merge with persistence\n");
    let result = merger
        .merge(&[sensor_a.clone(), sensor_b.clone(), sensor_c.clone()])
        .unwrap();

    println!("  {}", result);
    println!();

    // --- Step 4: Inspect the persistence diagram ---
    println!("Step 4: Persistence diagram\n");
    let diagram = &result.merged_diagram;
    println!("  H0 features (components): {}", diagram.h0_count());
    println!("  H1 features (loops):      {}", diagram.h1_count());

    for f in &diagram.features {
        println!("  {}", f);
    }
    println!();

    // --- Step 5: Read confidence scores ---
    println!("Step 5: Confidence scoring\n");
    let confidence = &result.confidence;

    println!(
        "  Average confidence: {:.3}",
        confidence.average_score()
    );
    println!(
        "  Confirmed:   {} features",
        confidence.confirmed().len()
    );
    println!(
        "  Provisional: {} features",
        confidence.provisional().len()
    );
    println!(
        "  Rejected:    {} features\n",
        confidence.rejected().len()
    );

    for fc in &confidence.scores {
        let icon = match fc.level {
            ConfidenceLevel::Confirmed => "✓",
            ConfidenceLevel::Provisional => "~",
            ConfidenceLevel::Rejected => "✗",
        };
        println!(
            "  {} H{} [{:.2}, {:.2}] score={:.3} agents={}/{}",
            icon,
            fc.feature.dim,
            fc.feature.birth,
            if fc.feature.death == f64::INFINITY {
                999.99
            } else {
                fc.feature.death
            },
            fc.score,
            fc.agent_count,
            fc.total_agents
        );
    }
    println!();

    // --- Step 6: Explore the Vietoris-Rips filtration ---
    println!("Step 6: Vietoris-Rips filtration\n");
    let coords: Vec<Vec<f64>> = sensor_a.points.iter().map(|p| p.coords.clone()).collect();
    let dm = DistanceMatrix::compute(&coords, Metric::Euclidean).unwrap();
    let filtration = VietorisRipsComplex::filtration(&dm);

    for rips in &filtration {
        println!(
            "  ε={:.3}: V={} E={} T={} components={} euler={}",
            rips.epsilon,
            rips.vertices.len(),
            rips.edges.len(),
            rips.triangles.len(),
            rips.connected_components(),
            rips.euler_characteristic()
        );
    }
    println!();

    // --- Step 7: Weighted beliefs ---
    println!("Step 7: Weighted beliefs\n");
    let heavy = BeliefState::new(vec![
        WeightedPoint::new(vec![0.0, 0.0], 100.0),
        WeightedPoint::new(vec![1.0, 0.0], 1.0),
    ])
    .unwrap()
    .with_agent_id("trusted");
    let centroid = heavy.centroid().unwrap();
    println!("  Centroid of weighted belief: ({:.3}, {:.3})", centroid[0], centroid[1]);
    println!("  (Pulled toward the high-weight point at origin)\n");

    // --- Step 8: Cosine metric for embeddings ---
    println!("Step 8: Cosine distance metric\n");
    let params = MergeParams {
        metric: Metric::Cosine,
        consensus_threshold: 2,
    };
    let cos_merger = TopologicalMerger::with_params(params);

    let embed_a = BeliefState::from_coords(&[vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]])
        .unwrap()
        .with_agent_id("embed-a");
    let embed_b = BeliefState::from_coords(&[vec![2.0, 0.0, 0.0], vec![0.0, 2.0, 0.0]])
        .unwrap()
        .with_agent_id("embed-b");

    let cos_result = cos_merger.merge(&[embed_a, embed_b]).unwrap();
    println!(
        "  Cosine merge: {} features, avg confidence {:.3}\n",
        cos_result.merged_diagram.features.len(),
        cos_result.confidence.average_score()
    );

    // --- Step 9: Bottleneck distance ---
    println!("Step 9: Comparing topologies with bottleneck distance\n");
    let dm1 = DistanceMatrix::compute(&[vec![0.0], vec![1.0]], Metric::Euclidean).unwrap();
    let dm2 = DistanceMatrix::compute(&[vec![0.0], vec![2.0]], Metric::Euclidean).unwrap();
    let pd1 = PersistenceDiagram::compute(&dm1);
    let pd2 = PersistenceDiagram::compute(&dm2);
    println!("  Bottleneck distance: {:.4}", pd1.bottleneck_distance(&pd2));
    println!("  (Non-zero: different topological signatures)\n");

    println!("=== Tutorial complete! ===");
}
