use flock_core::challenger::FsChallenger;
use flock_prover::mhot::{
    merkle_membership::{
        compute_content_hash, ContentMeta, MhotMembershipError, MhotMembershipInput,
    },
    native_witness::{MhotNodeWitness, mhot_node_to_sha256_merkle},
    sound_multiproof::{prove_sound_multiproof, verify_sound_multiproof},
};

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
    let w = mhot_node_to_sha256_merkle(node, false);
    let merkle_root = words_to_bytes(&w.native_root);
    compute_content_hash(content, &merkle_root)
}

/// Public root = content_hash of the root node (chain_content_hashes was
/// deleted; the verifier recomputes content_hash natively).
fn root_of(input: &MhotMembershipInput) -> [u32; 8] {
    let b = node_content_hash_bytes(&input.node, &input.content);
    let mut w = [0u32; 8];
    for i in 0..8 {
        w[i] = u32::from_be_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
    }
    w
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

#[test]
fn sound_multiproof_1_path() {
    let path = linked_inputs(&[8, 4, 2]);
    let paths = vec![path];
    let mut ch = FsChallenger::new(b"smp-1path");
    let proof = prove_sound_multiproof(&paths, &mut ch);
    assert_eq!(proof.n_paths, 1);
    assert_eq!(proof.merkle_shifts.len(), 3);
    let root = root_of(&paths[0][0]);
    let mut chv = FsChallenger::new(b"smp-1path");
    let res = verify_sound_multiproof(&proof, &root, &mut chv);
    match &res {
        Ok(()) => println!("  verify OK"),
        Err(e) => println!("  verify FAILED: {:?}", e),
    }
    res.expect("single path must verify");

    let size = proof.proof_size_bytes();
    println!("  1-path proof size: {} bytes ({:.1} KB)", size, size as f64 / 1024.0);
}

#[test]
fn sound_multiproof_4_paths_shared() {
    let path = linked_inputs(&[8, 4, 2]);
    let paths = vec![path.clone(), path.clone(), path.clone(), path.clone()];
    let mut ch = FsChallenger::new(b"smp-4shared");
    let proof = prove_sound_multiproof(&paths, &mut ch);
    assert_eq!(proof.n_paths, 4);
    assert_eq!(
        proof.merkle_shifts.len(), 3,
        "4 identical paths should dedup to 3 unique nodes"
    );
    assert_eq!(proof.n_routes, 3);

    let root = root_of(&paths[0][0]);
    let mut chv = FsChallenger::new(b"smp-4shared");
    verify_sound_multiproof(&proof, &root, &mut chv).expect("4 shared paths must verify");

    let size = proof.proof_size_bytes();
    println!("  4-shared proof size: {} bytes ({:.1} KB)", size, size as f64 / 1024.0);
}

#[test]
fn sound_multiproof_tampered_content_hash() {
    let path = linked_inputs(&[8, 4, 2]);
    let paths = vec![path];
    let mut ch = FsChallenger::new(b"smp-tamper-ch");
    let mut proof = prove_sound_multiproof(&paths, &mut ch);
    let root = root_of(&paths[0][0]);
    // content_hash is now verifier-computed from content_metas; tampering a
    // non-root node's metadata changes its computed content_hash and breaks the
    // cross-node binding to the parent's authenticated selected leaf.
    proof.content_metas[1].child_leaf_counts[0] ^= 1;
    let mut chv = FsChallenger::new(b"smp-tamper-ch");
    match verify_sound_multiproof(&proof, &root, &mut chv) {
        Err(MhotMembershipError::CrossNodeBinding { .. }) => {}
        Err(MhotMembershipError::RootMismatch { .. }) => {}
        other => panic!("tampered content_meta must be rejected, got {other:?}"),
    }
}

#[test]
fn sound_multiproof_wrong_root() {
    let path = linked_inputs(&[8, 4, 2]);
    let paths = vec![path];
    let mut ch = FsChallenger::new(b"smp-wrong-root");
    let proof = prove_sound_multiproof(&paths, &mut ch);
    let mut root = root_of(&paths[0][0]);
    root[0] ^= 1;
    let mut chv = FsChallenger::new(b"smp-wrong-root");
    match verify_sound_multiproof(&proof, &root, &mut chv) {
        Err(MhotMembershipError::RootMismatch { .. }) => {}
        other => panic!("wrong root must fail, got {other:?}"),
    }
}
