//! Advanced usage: multi-sensor fusion with topological filtering.
//!
//! Demonstrates a real-world scenario where three sensors observe a target
//! with varying noise levels and partial observability. The topological merge
//! identifies which features are consensus vs. noise.
//!
//! Run with: cargo run --example advanced

use topo_merge::{
    BeliefState, MergeParams, Metric, TopologicalMerger, WeightedPoint,
};

fn main() {
    println!("=== Advanced: Multi-Sensor Fusion ===\n");

    // Scenario: Three sensors tracking a moving object in 3D space.
    // Sensor A (LIDAR): accurate, 10 points around the true trajectory
    // Sensor B (RADAR): moderately noisy, 8 points with some drift
    // Sensor C (Camera): very noisy, 5 points with several outliers

    let true_positions: Vec<Vec<f64>> = (0..10)
        .map(|i| {
            let t = i as f64 * 0.5;
            vec![t.cos(), t.sin(), t * 0.1]
        })
        .collect();

    // Sensor A: high confidence, accurate observations
    let sensor_a_points: Vec<WeightedPoint> = true_positions
        .iter()
        .map(|p| WeightedPoint::new(p.clone(), 10.0))
        .collect();
    let sensor_a = BeliefState::new(sensor_a_points)
        .unwrap()
        .with_agent_id("lidar");

    // Sensor B: moderate confidence, slightly drifted observations
    let sensor_b_points: Vec<WeightedPoint> = true_positions
        .iter()
        .take(8)
        .map(|p| {
            let drifted = vec![p[0] + 0.2, p[1] - 0.1, p[2] + 0.05];
            WeightedPoint::new(drifted, 5.0)
        })
        .collect();
    let sensor_b = BeliefState::new(sensor_b_points)
        .unwrap()
        .with_agent_id("radar");

    // Sensor C: low confidence, noisy with outliers
    let sensor_c_points: Vec<WeightedPoint> = true_positions
        .iter()
        .take(5)
        .enumerate()
        .flat_map(|(i, p)| {
            let mut pts = vec![WeightedPoint::new(
                vec![p[0] + 0.5, p[1] + 0.3, p[2]],
                2.0,
            )];
            // Add outliers for some observations
            if i % 2 == 0 {
                pts.push(WeightedPoint::new(
                    vec![p[0] + 10.0, p[1] + 10.0, p[2] + 10.0],
                    1.0,
                ));
            }
            pts
        })
        .collect();
    let sensor_c = BeliefState::new(sensor_c_points)
        .unwrap()
        .with_agent_id("camera");

    println!("Sensor A (LIDAR):  {} points, weights ~10.0", sensor_a.len());
    println!("Sensor B (RADAR):  {} points, weights ~5.0", sensor_b.len());
    println!("Sensor C (Camera): {} points, weights ~1-2.0", sensor_c.len());
    println!();

    // Merge with Euclidean metric
    let params = MergeParams {
        metric: Metric::Euclidean,
        consensus_threshold: 2,
    };
    let merger = TopologicalMerger::with_params(params);
    let result = merger
        .merge(&[sensor_a.clone(), sensor_b.clone(), sensor_c.clone()])
        .unwrap();

    println!("--- Merge Results ---");
    println!("Total merged points: {}", result.merged_belief.len());
    println!("Agents:              {}", result.agent_count);
    println!(
        "Total weight:        {:.1}",
        result.merged_belief.total_weight()
    );
    println!();

    // Analyze the persistence diagram
    println!("--- Persistence Diagram ---");
    let diag = &result.merged_diagram;
    println!("H0 features (components): {}", diag.h0_count());
    println!("H1 features (loops):      {}", diag.h1_count());

    if let Some(mp) = diag.most_persistent() {
        println!(
            "Most persistent feature: H{} [{:.2}, {:.2}] pers={:.2}",
            mp.dim,
            mp.birth,
            if mp.death == f64::INFINITY {
                f64::INFINITY
            } else {
                mp.death
            },
            mp.persistence()
        );
    }
    println!();

    // Analyze confidence
    println!("--- Confidence Analysis ---");
    let conf = &result.confidence;
    println!("Average confidence: {:.3}", conf.average_score());

    println!("\nConfirmed features (score >= 0.6):");
    for fc in conf.confirmed() {
        println!(
            "  ✓ H{} [{:.2}, {:.2}] score={:.3} seen by {}/{} agents",
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

    println!("\nProvisional features (score 0.3-0.6):");
    for fc in conf.provisional() {
        println!(
            "  ~ H{} [{:.2}, {:.2}] score={:.3} seen by {}/{} agents",
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

    println!("\nRejected features (score < 0.3):");
    for fc in conf.rejected() {
        println!(
            "  ✗ H{} [{:.2}, {:.2}] score={:.3} seen by {}/{} agents",
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

    // Filter: keep only confirmed features for downstream processing
    println!("\n--- Filtering ---");
    let confirmed_count = conf.confirmed().len();
    let total = conf.scores.len();
    println!(
        "Keeping {}/{} features ({}% pass rate)",
        confirmed_count,
        total,
        if total > 0 {
            confirmed_count * 100 / total
        } else {
            0
        }
    );

    // Compare with a merge using Cosine metric (for embedding-space scenarios)
    println!("\n--- Comparison: Cosine vs Euclidean ---");
    let cos_params = MergeParams {
        metric: Metric::Cosine,
        consensus_threshold: 2,
    };
    let cos_merger = TopologicalMerger::with_params(cos_params);
    let cos_result = cos_merger
        .merge(&[sensor_a, sensor_b, sensor_c])
        .unwrap();

    println!(
        "Euclidean: {} features, avg confidence {:.3}",
        result.merged_diagram.features.len(),
        result.confidence.average_score()
    );
    println!(
        "Cosine:    {} features, avg confidence {:.3}",
        cos_result.merged_diagram.features.len(),
        cos_result.confidence.average_score()
    );

    println!("\n=== Advanced demo complete! ===");
}
