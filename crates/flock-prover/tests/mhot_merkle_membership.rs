use flock_core::challenger::FsChallenger;
use flock_prover::mhot::{
    merkle_membership::{
        prove_node_merkle, verify_node_merkle, prove_path_merkle, verify_path_merkle,
        MhotMembershipError,
    },
    native_witness::{MhotNodeWitness, mhot_node_to_sha256_merkle},
};

fn make_node(fanout: usize, selected: usize) -> MhotNodeWitness {
    let children: Vec<[u8; 32]> = (0..fanout)
        .map(|i| {
            let mut h = [0u8; 32];
            h[0] = i as u8;
            h[1] = (fanout as u8).wrapping_mul(7);
            blake3::hash(&h).as_bytes().clone()
        })
        .collect();
    MhotNodeWitness { children, selected_child: selected }
}

#[test]
fn single_node_roundtrip() {
    let node = make_node(8, 3);
    let mut ch = FsChallenger::new(b"test");
    let proof = prove_node_merkle(&node, &mut ch);
    let mut ch2 = FsChallenger::new(b"test");
    verify_node_merkle(&proof, &mut ch2).expect("single node must verify");
}

#[test]
fn various_fanouts() {
    for fanout in [2, 3, 4, 8, 16, 22, 32] {
        for selected in [0, fanout / 2, fanout - 1] {
            let node = make_node(fanout, selected);
            let mut ch = FsChallenger::new(b"test-various");
            let proof = prove_node_merkle(&node, &mut ch);
            let mut ch2 = FsChallenger::new(b"test-various");
            verify_node_merkle(&proof, &mut ch2)
                .unwrap_or_else(|e| panic!("fanout={fanout} selected={selected}: {e:?}"));
        }
    }
}

#[test]
fn multi_node_independent_proves() {
    let nodes = vec![
        make_node(8, 2),
        make_node(4, 1),
        make_node(2, 0),
    ];
    let mut ch = FsChallenger::new(b"test-path");
    let proofs = prove_path_merkle(&nodes, &mut ch);
    assert_eq!(proofs.len(), 3);
    let mut ch2 = FsChallenger::new(b"test-path");
    for p in &proofs {
        verify_node_merkle(p, &mut ch2).expect("each node must verify independently");
    }
}

#[test]
fn multi_node_unlinked_fails_binding() {
    let nodes = vec![make_node(8, 2), make_node(4, 1)];
    let mut ch = FsChallenger::new(b"test-unlinked");
    let proofs = prove_path_merkle(&nodes, &mut ch);
    let mut ch2 = FsChallenger::new(b"test-unlinked");
    match verify_path_merkle(&proofs, &mut ch2) {
        Err(MhotMembershipError::CrossNodeBinding { .. }) => {}
        other => panic!("unlinked nodes should fail binding, got {other:?}"),
    }
}

#[test]
fn tampered_leaf_fails() {
    let node = make_node(8, 3);
    let mut ch = FsChallenger::new(b"test-tamper");
    let mut proof = prove_node_merkle(&node, &mut ch);
    proof.leaf[0] ^= 1;
    let mut ch2 = FsChallenger::new(b"test-tamper");
    verify_node_merkle(&proof, &mut ch2)
        .expect_err("tampered leaf must fail");
}

#[test]
fn tampered_root_fails() {
    let node = make_node(8, 3);
    let mut ch = FsChallenger::new(b"test-tamper-root");
    let mut proof = prove_node_merkle(&node, &mut ch);
    proof.root[0] ^= 1;
    let mut ch2 = FsChallenger::new(b"test-tamper-root");
    verify_node_merkle(&proof, &mut ch2)
        .expect_err("tampered root must fail");
}

#[test]
fn tampered_commitment_fails() {
    let node = make_node(8, 3);
    let mut ch = FsChallenger::new(b"test-tamper-commit");
    let mut proof = prove_node_merkle(&node, &mut ch);
    proof.commitment.root[0] ^= 0xFF;
    let mut ch2 = FsChallenger::new(b"test-tamper-commit");
    verify_node_merkle(&proof, &mut ch2)
        .expect_err("tampered commitment must fail");
}

// --- cross-node binding ---

fn compute_native_root(children: &[[u8; 32]]) -> [u8; 32] {
    use flock_prover::r1cs_hashes::sha2::{SHA256_IV, sha256_compress};
    fn bytes_to_words(h: &[u8; 32]) -> [u32; 8] {
        let mut w = [0u32; 8];
        for i in 0..8 {
            w[i] = u32::from_be_bytes([h[4*i], h[4*i+1], h[4*i+2], h[4*i+3]]);
        }
        w
    }
    fn words_to_bytes(w: &[u32; 8]) -> [u8; 32] {
        let mut h = [0u8; 32];
        for i in 0..8 {
            h[4*i..4*i+4].copy_from_slice(&w[i].to_be_bytes());
        }
        h
    }
    let padded_len = children.len().next_power_of_two();
    let mut hashes: Vec<[u32; 8]> = children.iter().map(|c| bytes_to_words(c)).collect();
    hashes.resize(padded_len, [0u32; 8]);
    while hashes.len() > 1 {
        let mut next = Vec::new();
        for pair in hashes.chunks(2) {
            let mut m = [0u32; 16];
            m[..8].copy_from_slice(&pair[0]);
            m[8..].copy_from_slice(&pair[1]);
            next.push(sha256_compress(&SHA256_IV, &m));
        }
        hashes = next;
    }
    words_to_bytes(&hashes[0])
}

fn words_to_bytes_helper(w: &[u32; 8]) -> [u8; 32] {
    let mut h = [0u8; 32];
    for i in 0..8 {
        h[4*i..4*i+4].copy_from_slice(&w[i].to_be_bytes());
    }
    h
}

fn flock_root_bytes(node: &MhotNodeWitness) -> [u8; 32] {
    let w = mhot_node_to_sha256_merkle(node);
    use flock_prover::r1cs_hashes::sha2::{SHA256_IV, sha256_compress, min_n_blocks_log};
    let n_real = w.compressions.len();
    let mut compressions = w.compressions;
    let mut b_bits = w.b_bits;
    let needed = 1usize << min_n_blocks_log(n_real);
    let mut current = if compressions.is_empty() {
        [0u32; 8]
    } else {
        let (iv, m) = &compressions[compressions.len() - 1];
        sha256_compress(iv, m)
    };
    while compressions.len() < needed {
        let mut m = [0u32; 16];
        m[..8].copy_from_slice(&current);
        compressions.push((SHA256_IV, m));
        b_bits.push(false);
        current = sha256_compress(&SHA256_IV, &m);
    }
    words_to_bytes_helper(&current)
}

fn build_linked_path(fanouts: &[usize]) -> Vec<MhotNodeWitness> {
    let depth = fanouts.len();
    let mut nodes_rev = Vec::with_capacity(depth);

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
    let mut child_flock_root = flock_root_bytes(&leaf_node);
    nodes_rev.push(leaf_node);

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
        children[selected] = child_flock_root;
        let node = MhotNodeWitness {
            children: children,
            selected_child: selected,
        };
        child_flock_root = flock_root_bytes(&node);
        nodes_rev.push(node);
    }
    nodes_rev.reverse();
    nodes_rev
}

#[test]
fn cross_node_binding_valid_path() {
    let nodes = build_linked_path(&[8, 4, 2]);
    let mut ch = FsChallenger::new(b"test-binding");
    let proofs = prove_path_merkle(&nodes, &mut ch);
    let mut ch2 = FsChallenger::new(b"test-binding");
    verify_path_merkle(&proofs, &mut ch2)
        .expect("linked path with matching digests must verify");
}

#[test]
fn cross_node_binding_mismatch_fails() {
    let mut nodes = build_linked_path(&[8, 4, 2]);
    nodes[1].children[0] = [0xFF; 32];
    let mut ch = FsChallenger::new(b"test-binding-bad");
    let proofs = prove_path_merkle(&nodes, &mut ch);
    let mut ch2 = FsChallenger::new(b"test-binding-bad");
    match verify_path_merkle(&proofs, &mut ch2) {
        Err(MhotMembershipError::CrossNodeBinding { .. }) => {}
        other => panic!("expected CrossNodeBinding error, got {other:?}"),
    }
}
