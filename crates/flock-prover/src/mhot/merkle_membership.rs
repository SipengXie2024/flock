use crate::r1cs_hashes::sha2::{
    Compression, MerklePathProof, MerklePathVerifyError, SHA256_IV,
    Sha256HybridSetup, min_n_blocks_log, sha256_compress,
};
use flock_core::challenger::Challenger;
use flock_core::pcs::Commitment;

use super::native_witness::{MhotNodeWitness, mhot_node_to_sha256_merkle};

/// Proof for a single MHOT node's in-node binary Merkle path.
pub struct NodeMerkleProof {
    pub proof: MerklePathProof,
    pub commitment: Commitment,
    /// The selected child hash (the leaf of this in-node Merkle path).
    pub leaf: [u32; 8],
    /// Chain root after padding (what the Flock protocol verifies).
    pub root: [u32; 8],
    /// The real MHOT in-node Merkle root (for cross-node binding).
    pub native_root: [u32; 8],
    pub b_bits: Vec<bool>,
    pub n_real_compressions: usize,
}

/// Prove the in-node binary Merkle path for a single MHOT node.
///
/// The node's children form a binary Merkle tree. This function extracts the
/// path from the selected child to the root, pads it to the minimum power-of-2
/// length (at least 8), and produces a Flock SHA-256 Merkle path proof.
pub fn prove_node_merkle<Ch: Challenger>(
    node: &MhotNodeWitness,
    challenger: &mut Ch,
) -> NodeMerkleProof {
    let w = mhot_node_to_sha256_merkle(node);
    let n_real = w.compressions.len();

    let mut compressions = w.compressions;
    let mut b_bits = w.b_bits.clone();
    let needed = 1usize << min_n_blocks_log(n_real);
    let padded_root = pad_to_needed(&mut compressions, &mut b_bits, needed);

    let setup = Sha256HybridSetup::new(needed);
    let (proof, commitment) = setup.prove_merkle_path(&compressions, &b_bits, challenger);
    NodeMerkleProof {
        proof,
        commitment,
        leaf: w.leaf,
        root: padded_root,
        native_root: w.native_root,
        b_bits: w.b_bits,
        n_real_compressions: n_real,
    }
}

/// Verify a single MHOT node's in-node Merkle path proof.
pub fn verify_node_merkle<Ch: Challenger>(
    proof: &NodeMerkleProof,
    challenger: &mut Ch,
) -> Result<(), MerklePathVerifyError> {
    let needed = 1usize << min_n_blocks_log(proof.n_real_compressions);
    let setup = Sha256HybridSetup::new(needed);
    let mut b_bits = proof.b_bits.clone();
    b_bits.resize(needed, false);
    setup.verify_merkle_path(
        &proof.commitment,
        &proof.proof,
        &proof.leaf,
        &proof.root,
        &b_bits,
        challenger,
    )
}

/// Prove in-node Merkle paths for a sequence of MHOT nodes (one proof per node).
///
/// Each node gets an independent proof. Cross-node linking (child's content
/// hash == parent's selected leaf) is handled at a higher protocol level.
pub fn prove_path_merkle<Ch: Challenger>(
    nodes: &[MhotNodeWitness],
    challenger: &mut Ch,
) -> Vec<NodeMerkleProof> {
    nodes.iter().map(|n| prove_node_merkle(n, challenger)).collect()
}

/// Verify in-node Merkle paths for a sequence of MHOT nodes.
pub fn verify_path_merkle<Ch: Challenger>(
    proofs: &[NodeMerkleProof],
    challenger: &mut Ch,
) -> Result<(), MerklePathVerifyError> {
    for p in proofs {
        verify_node_merkle(p, challenger)?;
    }
    Ok(())
}

/// Pad compressions and b_bits to `needed` slots with dummy identity
/// compressions that extend the Merkle chain. Returns the final chain root.
fn pad_to_needed(
    compressions: &mut Vec<Compression>,
    b_bits: &mut Vec<bool>,
    needed: usize,
) -> [u32; 8] {
    let last_output = if compressions.is_empty() {
        [0u32; 8]
    } else {
        let (iv, m) = &compressions[compressions.len() - 1];
        sha256_compress(iv, m)
    };

    if compressions.len() >= needed {
        return last_output;
    }

    let mut current = last_output;
    while compressions.len() < needed {
        let sibling = [0u32; 8];
        let mut m = [0u32; 16];
        m[..8].copy_from_slice(&current);
        m[8..].copy_from_slice(&sibling);
        compressions.push((SHA256_IV, m));
        b_bits.push(false);
        current = sha256_compress(&SHA256_IV, &m);
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mhot::native_witness::MhotNodeWitness;
    use flock_core::challenger::FsChallenger;

    fn make_random_children(n: usize, seed: u64) -> Vec<[u8; 32]> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                let mut h = [0u8; 32];
                for b in h.iter_mut() {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    *b = s as u8;
                }
                h
            })
            .collect()
    }

    #[test]
    fn single_node_fanout8_roundtrip() {
        let node = MhotNodeWitness {
            children: make_random_children(8, 0xABCD_1234),
            selected_child: 5,
        };
        let mut ch = FsChallenger::new(b"mhot-node-merkle-1");
        let proof = prove_node_merkle(&node, &mut ch);

        let mut chv = FsChallenger::new(b"mhot-node-merkle-1");
        verify_node_merkle(&proof, &mut chv)
            .expect("single node roundtrip must verify");
    }

    #[test]
    fn three_node_path_roundtrip() {
        let nodes = vec![
            MhotNodeWitness {
                children: make_random_children(8, 0x1111),
                selected_child: 2,
            },
            MhotNodeWitness {
                children: make_random_children(4, 0x2222),
                selected_child: 1,
            },
            MhotNodeWitness {
                children: make_random_children(16, 0x3333),
                selected_child: 9,
            },
        ];
        let mut ch = FsChallenger::new(b"mhot-path-merkle-3");
        let proofs = prove_path_merkle(&nodes, &mut ch);
        assert_eq!(proofs.len(), 3);

        let mut chv = FsChallenger::new(b"mhot-path-merkle-3");
        verify_path_merkle(&proofs, &mut chv)
            .expect("3-node path roundtrip must verify");
    }

    #[test]
    fn rejects_wrong_leaf() {
        let node = MhotNodeWitness {
            children: make_random_children(8, 0xBAD_CAFE),
            selected_child: 0,
        };
        let mut ch = FsChallenger::new(b"mhot-wrong-leaf");
        let mut proof = prove_node_merkle(&node, &mut ch);
        proof.leaf[0] ^= 1;

        let mut chv = FsChallenger::new(b"mhot-wrong-leaf");
        let res = verify_node_merkle(&proof, &mut chv);
        assert!(res.is_err(), "verifier must reject tampered leaf");
    }

    #[test]
    fn rejects_wrong_root() {
        let node = MhotNodeWitness {
            children: make_random_children(8, 0xDEAD_F00D),
            selected_child: 3,
        };
        let mut ch = FsChallenger::new(b"mhot-wrong-root");
        let mut proof = prove_node_merkle(&node, &mut ch);
        proof.root[7] ^= 0xFFFF_FFFF;

        let mut chv = FsChallenger::new(b"mhot-wrong-root");
        let res = verify_node_merkle(&proof, &mut chv);
        assert!(res.is_err(), "verifier must reject tampered root");
    }
}
