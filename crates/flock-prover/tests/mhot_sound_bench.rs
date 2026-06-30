use std::time::Instant;

use flock_core::challenger::FsChallenger;
use flock_prover::mhot::{
    merkle_membership::{
        compute_content_hash, MhotMembershipInput, ContentMeta,
    },
    multiproof::{prove_mhot_multiproof, verify_mhot_multiproof, MhotPathInput},
    native_witness::MhotNodeWitness,
    ref_witness::build_ref_witness,
    route_f32::{self as route, RouteF32Witness},
    schedule::MhotHashSchedule,
    sound_multiproof::{prove_sound_multiproof, verify_sound_multiproof},
};
use flock_prover::r1cs_hashes::sha2::sha256_compress;

fn synthetic_content(nc: usize) -> ContentMeta {
    ContentMeta {
        extraction_masks: [0x1F; 4],
        sparse_partial_keys: vec![0; nc],
        child_leaf_counts: vec![1; nc],
    }
}

fn words_to_bytes(w: &[u32; 8]) -> [u8; 32] {
    let mut h = [0u8; 32];
    for i in 0..8 {
        h[4 * i..4 * i + 4].copy_from_slice(&w[i].to_be_bytes());
    }
    h
}

fn node_content_hash_bytes(node: &MhotNodeWitness, content: &ContentMeta) -> [u8; 32] {
    use flock_prover::mhot::native_witness::mhot_node_to_sha256_merkle;
    let w = mhot_node_to_sha256_merkle(node);
    let merkle_root = words_to_bytes(&w.native_root);
    compute_content_hash(content, &merkle_root)
}

fn linked_inputs(fanouts: &[usize]) -> Vec<MhotMembershipInput> {
    let depth = fanouts.len();
    let mut inputs_rev = Vec::with_capacity(depth);

    let leaf_children: Vec<[u8; 32]> = (0..fanouts[depth - 1])
        .map(|i| {
            let mut h = [0u8; 32];
            h[0] = i as u8;
            h[1] = 0xAA;
            h
        })
        .collect();
    let leaf_node = MhotNodeWitness {
        children: leaf_children,
        selected_child: 0,
    };
    let leaf_content = synthetic_content(leaf_node.children.len());
    let mut child_content_hash = node_content_hash_bytes(&leaf_node, &leaf_content);
    inputs_rev.push(MhotMembershipInput::from_node(leaf_node));

    for level in (0..depth - 1).rev() {
        let fanout = fanouts[level];
        let selected = 1.min(fanout - 1);
        let mut children: Vec<[u8; 32]> = (0..fanout)
            .map(|i| {
                let mut h = [0u8; 32];
                h[0] = (level * 31 + i * 17) as u8;
                h[1] = level as u8;
                h
            })
            .collect();
        children[selected] = child_content_hash;
        let node = MhotNodeWitness { children, selected_child: selected };
        let content = synthetic_content(node.children.len());
        child_content_hash = node_content_hash_bytes(&node, &content);
        inputs_rev.push(MhotMembershipInput::from_node(node));
    }
    inputs_rev.reverse();
    inputs_rev
}

fn make_multiproof_input(fanouts: &[usize], seed: u64) -> MhotPathInput {
    let sched = MhotHashSchedule::from_fanouts(fanouts);
    let hash_witness = build_ref_witness(&sched, seed);
    let route_witnesses: Vec<RouteF32Witness> = sched
        .fanouts
        .iter()
        .enumerate()
        .map(|(node, &fanout)| {
            let mut key = [false; route::KEY_BITS];
            let mut mask = [false; route::KEY_BITS];
            key[0] = (node & 1) != 0;
            key[1] = true;
            mask[0] = true;
            mask[1] = true;
            let children: Vec<[bool; route::DIGEST_BITS]> = (0..fanout.min(route::FANOUT))
                .map(|c| std::array::from_fn(|b| ((node * 31 + c * 17 + b) & 1) != 0))
                .collect();
            RouteF32Witness::new_padded(key, mask, &children, fanout.min(route::FANOUT))
        })
        .collect();
    MhotPathInput { schedule: sched, hash_witness, route_witnesses }
}

#[test]
fn sound_membership_benchmark() {
    let fanouts = &[8, 4, 2];
    eprintln!();
    eprintln!("=== Sound MHOT Membership Benchmark ===");
    eprintln!("{:>8} {:>12} {:>12} {:>12} {:>12} {:>10}",
        "n_paths", "prove_ms", "verify_ms", "total_ms", "per_path", "unique");
    eprintln!("{}", "-".repeat(72));

    for &n_paths in &[1, 4, 16] {
        let path = linked_inputs(fanouts);
        let paths: Vec<Vec<MhotMembershipInput>> =
            (0..n_paths).map(|_| path.clone()).collect();

        let mut ch = FsChallenger::new(b"sound-bench");
        let t0 = Instant::now();
        let proof = prove_sound_multiproof(&paths, &mut ch);
        let prove_ms = t0.elapsed().as_secs_f64() * 1e3;

        let root = proof.content_proofs[proof.path_mapping.node_indices[0][0]].content_hash;
        let mut chv = FsChallenger::new(b"sound-bench");
        let t1 = Instant::now();
        verify_sound_multiproof(&proof, &root, &mut chv).expect("must verify");
        let verify_ms = t1.elapsed().as_secs_f64() * 1e3;

        let total = prove_ms + verify_ms;
        let unique = proof.hash_proofs.len();
        eprintln!("{:>8} {:>12.1} {:>12.1} {:>12.1} {:>11.3} {:>10}",
            n_paths, prove_ms, verify_ms, total, total / n_paths as f64, unique);
    }

    eprintln!();
    eprintln!("=== Flock Multiproof (benchmark tool, not sound) ===");
    eprintln!("{:>8} {:>12} {:>12} {:>12} {:>12}",
        "n_paths", "prove_ms", "verify_ms", "total_ms", "per_path");
    eprintln!("{}", "-".repeat(60));

    for &n_paths in &[1, 4, 16, 256, 4096, 16384] {
        let paths: Vec<MhotPathInput> =
            (0..n_paths).map(|_| make_multiproof_input(fanouts, 42)).collect();

        let t0 = Instant::now();
        let proof = prove_mhot_multiproof(&paths);
        let prove_ms = t0.elapsed().as_secs_f64() * 1e3;

        let t1 = Instant::now();
        verify_mhot_multiproof(&proof).expect("must verify");
        let verify_ms = t1.elapsed().as_secs_f64() * 1e3;

        let total = prove_ms + verify_ms;
        eprintln!("{:>8} {:>12.1} {:>12.1} {:>12.1} {:>11.3}",
            n_paths, prove_ms, verify_ms, total, total / n_paths as f64);
    }
}
