use std::time::Instant;

use flock_core::challenger::FsChallenger;
use flock_prover::mhot::{
    merkle_membership::{ContentMeta, MhotMembershipInput, mhot_node_to_route_witness},
    native_witness::MhotNodeWitness,
    sound_multiproof::{prove_sound_multiproof, verify_sound_multiproof},
};
use serde::Deserialize;

#[derive(Deserialize)]
struct NodeData {
    children: Vec<Vec<u8>>,
    selected_child: usize,
    extraction_masks: [u64; 4],
    sparse_partial_keys: Vec<u32>,
    child_leaf_counts: Vec<u32>,
}

#[derive(Deserialize)]
struct PathData {
    nodes: Vec<NodeData>,
}

#[derive(Deserialize)]
struct ExportData {
    paths: Vec<PathData>,
    native_single_proof_total_bytes: usize,
    native_multi_proof_bytes: usize,
    native_verify_seq_ms: f64,
    native_verify_par_ms: f64,
    native_verify_multi_ms: f64,
}

fn node_data_to_input(nd: &NodeData) -> MhotMembershipInput {
    let children: Vec<[u8; 32]> = nd.children.iter().map(|c| {
        let mut arr = [0u8; 32];
        arr.copy_from_slice(c);
        arr
    }).collect();

    let node = MhotNodeWitness {
        children,
        selected_child: nd.selected_child,
    };
    let route_witness = mhot_node_to_route_witness(&node);
    let content = ContentMeta {
        extraction_masks: nd.extraction_masks,
        sparse_partial_keys: nd.sparse_partial_keys.clone(),
        child_leaf_counts: nd.child_leaf_counts.clone(),
    };
    MhotMembershipInput { node, route_witness, content }
}

#[test]
fn sound_vs_native_benchmark() {
    eprintln!();
    eprintln!("=== MHOT Membership Proof: Native vs Flock Sound (1M-key SHA-256 tree) ===");
    eprintln!("{:>6} {:>10} {:>10} {:>8} {:>10} {:>10} {:>10} {:>8}",
        "N", "native_KB", "flock_KB", "ratio", "nv_ms", "fp_ms", "fv_ms", "unique");
    eprintln!("{}", "-".repeat(80));

    for &n in &[1, 16, 256] {
        let filename = format!("/tmp/mhot_export_n{}.json", n);
        let json = match std::fs::read_to_string(&filename) {
            Ok(j) => j,
            Err(_) => {
                eprintln!("{:>6}  (skipped — {} not found)", n, filename);
                continue;
            }
        };
        let data: ExportData = serde_json::from_str(&json).unwrap();

        let paths: Vec<Vec<MhotMembershipInput>> = data.paths.iter()
            .map(|p| p.nodes.iter().map(node_data_to_input).collect())
            .collect();

        let mut ch = FsChallenger::new(b"sound-vs-native");
        let t0 = Instant::now();
        let proof = prove_sound_multiproof(&paths, &mut ch);
        let prove_ms = t0.elapsed().as_secs_f64() * 1e3;

        let root_u = proof.path_mapping.node_indices[0][0];
        let root = proof.chain_content_hashes[root_u];
        let mut chv = FsChallenger::new(b"sound-vs-native");
        let t1 = Instant::now();
        verify_sound_multiproof(&proof, &root, &mut chv).expect("must verify");
        let verify_ms = t1.elapsed().as_secs_f64() * 1e3;

        let flock_bytes = proof.proof_size_bytes();
        let native_kb = data.native_single_proof_total_bytes as f64 / 1024.0;
        let flock_kb = flock_bytes as f64 / 1024.0;
        let ratio = flock_kb / native_kb;
        let unique = proof.merkle_leaves.len();

        eprintln!("{:>6} {:>10.1} {:>10.1} {:>7.1}x {:>10.2} {:>10.1} {:>10.1} {:>8}",
            n, native_kb, flock_kb, ratio,
            data.native_verify_seq_ms, prove_ms, verify_ms, unique);
    }
}
