//! Full MHOT membership proof: Levels 1-3 integrated with cross-node
//! binding and N-path multipoint support.
//!
//! Level 1: In-node binary Merkle path (prove_node_merkle / verify_node_merkle)
//! Level 2: Content hash H(masks||keys||merkle_root||counts) via SHA-256 chain
//! Level 3: Cross-node binding — content_hash[node_i] == leaf[node_{i-1}]
//!
//! The multipoint proof is N independent paths, each doing Levels 1-3.

use crate::r1cs_hashes::sha2::MerklePathVerifyError;
use flock_core::challenger::Challenger;

use super::merkle_membership::{
    NodeMerkleProof, prove_node_merkle, verify_node_merkle,
};
use super::mhot_to_sha256::content_hash_to_sha256_chain;
use super::native_witness::MhotNodeWitness;

/// Full witness for a single MHOT node: the child hashes for Level 1,
/// plus the semantic fields for Level 2 content hash.
#[derive(Clone, Debug)]
pub struct MhotNodeFullWitness {
    pub children: Vec<[u8; 32]>,
    pub selected_child: usize,
    pub extraction_masks: [u64; 4],
    pub sparse_keys: Vec<u32>,
    pub counts: Vec<u32>,
}

/// Proof for a single MHOT membership path (all levels).
pub struct MhotPathProof {
    pub node_proofs: Vec<NodeMerkleProof>,
    /// Content hash (8 x u32 words) for each node. These are the Level 2
    /// outputs that the verifier uses for cross-node binding.
    pub content_hashes: Vec<[u32; 8]>,
    /// The native MHOT Merkle root (as 32 raw bytes) for each node, used as
    /// input to the content hash preimage. Exposed so the verifier can
    /// recompute content hashes.
    pub native_merkle_roots: Vec<[u8; 32]>,
    /// The full witness data for each node, needed by the verifier to
    /// recompute content hashes from the public preimage fields.
    pub node_witnesses: Vec<MhotNodeFullWitness>,
}

/// Multipoint membership proof: N independent paths.
pub struct MhotMultipointProof {
    pub path_proofs: Vec<MhotPathProof>,
}

/// Errors from membership verification.
#[derive(Debug)]
pub enum MembershipVerifyError {
    MerklePath {
        path_idx: usize,
        node_idx: usize,
        inner: MerklePathVerifyError,
    },
    CrossNodeMismatch {
        path_idx: usize,
        parent_node_idx: usize,
    },
    ContentHashMismatch {
        path_idx: usize,
        node_idx: usize,
    },
}

impl std::fmt::Display for MembershipVerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MerklePath {
                path_idx,
                node_idx,
                inner,
            } => write!(
                f,
                "path {path_idx} node {node_idx}: Merkle path verify failed: {inner:?}"
            ),
            Self::CrossNodeMismatch {
                path_idx,
                parent_node_idx,
            } => write!(
                f,
                "path {path_idx}: content_hash of node {} != leaf of node {parent_node_idx}",
                parent_node_idx + 1,
            ),
            Self::ContentHashMismatch {
                path_idx,
                node_idx,
            } => write!(
                f,
                "path {path_idx} node {node_idx}: recomputed content hash does not match proof"
            ),
        }
    }
}

fn native_root_to_bytes(root: &[u32; 8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, &w) in root.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&w.to_be_bytes());
    }
    out
}

fn words_to_bytes(words: &[u32; 8]) -> [u8; 32] {
    native_root_to_bytes(words)
}

/// Prove a single MHOT membership path (Levels 1+2+3).
///
/// Each node in the path gets:
/// - Level 1: in-node binary Merkle path proof
/// - Level 2: content hash computed from semantic fields
/// - Level 3: cross-node binding verified at prove time (panic on mismatch)
pub fn prove_path<Ch: Challenger>(
    nodes: &[MhotNodeFullWitness],
    challenger: &mut Ch,
) -> MhotPathProof {
    assert!(!nodes.is_empty(), "path must have at least one node");

    let mut node_proofs: Vec<NodeMerkleProof> = Vec::with_capacity(nodes.len());
    let mut content_hashes = Vec::with_capacity(nodes.len());
    let mut native_merkle_roots = Vec::with_capacity(nodes.len());

    for (i, node) in nodes.iter().enumerate() {
        let merkle_node = MhotNodeWitness {
            children: node.children.clone(),
            selected_child: node.selected_child,
        };
        let proof = prove_node_merkle(&merkle_node, challenger);

        let native_root_bytes = native_root_to_bytes(&proof.native_root);
        let (_, content_hash) = content_hash_to_sha256_chain(
            &node.extraction_masks,
            &node.sparse_keys,
            &native_root_bytes,
            &node.counts,
        );

        // Level 3: cross-node binding at prove time.
        // For node i > 0: content_hash[i] must equal leaf[i-1].
        // The "leaf" of node i-1 is the selected child hash in node i-1's
        // in-node Merkle tree, which should be the content hash of node i.
        if i > 0 {
            let parent_leaf = node_proofs[i - 1].leaf;
            let content_hash_bytes = words_to_bytes(&content_hash);
            let parent_leaf_bytes = words_to_bytes(&parent_leaf);
            assert_eq!(
                content_hash_bytes, parent_leaf_bytes,
                "cross-node binding failed at prove time: content_hash of node {i} \
                 != leaf of node {} (parent). This means the children array of \
                 node {} does not contain node {i}'s content hash at position {}.",
                i - 1,
                i - 1,
                nodes[i - 1].selected_child,
            );
        }

        node_proofs.push(proof);
        content_hashes.push(content_hash);
        native_merkle_roots.push(native_root_bytes);
    }

    MhotPathProof {
        node_proofs,
        content_hashes,
        native_merkle_roots,
        node_witnesses: nodes.to_vec(),
    }
}

/// Verify a single MHOT membership path (Levels 1+2+3).
///
/// Checks:
/// 1. Each node's in-node Merkle path proof verifies (Level 1)
/// 2. Each node's content hash matches the recomputed value (Level 2)
/// 3. For consecutive nodes i, i+1: content_hash[i+1] == leaf[i] (Level 3)
pub fn verify_path<Ch: Challenger>(
    proof: &MhotPathProof,
    challenger: &mut Ch,
) -> Result<(), MembershipVerifyError> {
    let path_idx = 0;
    verify_path_inner(proof, path_idx, challenger)
}

fn verify_path_inner<Ch: Challenger>(
    proof: &MhotPathProof,
    path_idx: usize,
    challenger: &mut Ch,
) -> Result<(), MembershipVerifyError> {
    for (i, node_proof) in proof.node_proofs.iter().enumerate() {
        // Level 1: verify in-node Merkle path
        verify_node_merkle(node_proof, challenger).map_err(|inner| {
            MembershipVerifyError::MerklePath {
                path_idx,
                node_idx: i,
                inner,
            }
        })?;

        // Level 2: recompute content hash and check
        let w = &proof.node_witnesses[i];
        let (_, recomputed) = content_hash_to_sha256_chain(
            &w.extraction_masks,
            &w.sparse_keys,
            &proof.native_merkle_roots[i],
            &w.counts,
        );
        if recomputed != proof.content_hashes[i] {
            return Err(MembershipVerifyError::ContentHashMismatch {
                path_idx,
                node_idx: i,
            });
        }

        // Level 3: cross-node binding
        if i > 0 {
            let parent_leaf = proof.node_proofs[i - 1].leaf;
            if proof.content_hashes[i] != parent_leaf {
                return Err(MembershipVerifyError::CrossNodeMismatch {
                    path_idx,
                    parent_node_idx: i - 1,
                });
            }
        }
    }

    Ok(())
}

/// Prove N independent MHOT membership paths (multipoint).
pub fn prove_multipoint<Ch: Challenger>(
    paths: &[Vec<MhotNodeFullWitness>],
    challenger: &mut Ch,
) -> MhotMultipointProof {
    assert!(!paths.is_empty(), "multipoint needs at least one path");

    let path_proofs = paths
        .iter()
        .map(|nodes| prove_path(nodes, challenger))
        .collect();

    MhotMultipointProof { path_proofs }
}

/// Verify N independent MHOT membership paths (multipoint).
pub fn verify_multipoint<Ch: Challenger>(
    proof: &MhotMultipointProof,
    challenger: &mut Ch,
) -> Result<(), MembershipVerifyError> {
    for (path_idx, path_proof) in proof.path_proofs.iter().enumerate() {
        verify_path_inner(path_proof, path_idx, challenger)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use flock_core::challenger::FsChallenger;

    fn make_random_hash(seed: &mut u64) -> [u8; 32] {
        let mut h = [0u8; 32];
        for b in h.iter_mut() {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            *b = *seed as u8;
        }
        h
    }

    fn make_random_masks(seed: &mut u64) -> [u64; 4] {
        let mut masks = [0u64; 4];
        for m in masks.iter_mut() {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            *m = *seed;
        }
        masks
    }

    fn make_random_u32_vec(n: usize, seed: &mut u64) -> Vec<u32> {
        (0..n)
            .map(|_| {
                *seed ^= *seed << 13;
                *seed ^= *seed >> 7;
                *seed ^= *seed << 17;
                *seed as u32
            })
            .collect()
    }

    /// Build a consistent multi-node path where node i+1's content hash
    /// is placed as the selected child in node i's children array.
    ///
    /// We build bottom-up: start from the deepest node (leaf), compute its
    /// content hash, then place that hash in the parent's children array.
    fn make_consistent_path(
        n_nodes: usize,
        fanouts: &[usize],
        seed: u64,
    ) -> Vec<MhotNodeFullWitness> {
        assert_eq!(fanouts.len(), n_nodes);
        let mut rng = seed;

        // Build nodes bottom-up
        let mut nodes: Vec<MhotNodeFullWitness> = Vec::with_capacity(n_nodes);

        // Start with the deepest (leaf-level) node
        for level in (0..n_nodes).rev() {
            let fanout = fanouts[level];
            let selected = if level < n_nodes - 1 {
                // parent: selected_child points to the child node
                (rng as usize) % fanout
            } else {
                // deepest node: arbitrary selection
                0
            };

            let mut children: Vec<[u8; 32]> = (0..fanout)
                .map(|_| make_random_hash(&mut rng))
                .collect();
            let masks = make_random_masks(&mut rng);
            let sparse_keys = make_random_u32_vec(fanout, &mut rng);
            let counts = make_random_u32_vec(fanout, &mut rng);

            // If this is a parent (not the deepest), the child we just built
            // needs its content hash placed in this node's children array.
            if level < n_nodes - 1 {
                let child_node = &nodes[0]; // the last node we built (child)
                // Compute the child's content hash using its native Merkle root
                let child_merkle_node = MhotNodeWitness {
                    children: child_node.children.clone(),
                    selected_child: child_node.selected_child,
                };
                let child_witness =
                    super::super::native_witness::mhot_node_to_sha256_merkle(&child_merkle_node, false);
                let child_native_root_bytes = native_root_to_bytes(&child_witness.native_root);
                let (_, child_content_hash) = content_hash_to_sha256_chain(
                    &child_node.extraction_masks,
                    &child_node.sparse_keys,
                    &child_native_root_bytes,
                    &child_node.counts,
                );
                let child_content_bytes = words_to_bytes(&child_content_hash);
                children[selected] = child_content_bytes;
            }

            nodes.insert(
                0,
                MhotNodeFullWitness {
                    children,
                    selected_child: selected,
                    extraction_masks: masks,
                    sparse_keys,
                    counts,
                },
            );
        }

        nodes
    }

    #[test]
    fn single_node_path_roundtrip() {
        let nodes = make_consistent_path(1, &[8], 0xAAAA_BBBB);
        let mut ch = FsChallenger::new(b"membership-single-1");
        let proof = prove_path(&nodes, &mut ch);
        assert_eq!(proof.node_proofs.len(), 1);
        assert_eq!(proof.content_hashes.len(), 1);

        let mut chv = FsChallenger::new(b"membership-single-1");
        verify_path(&proof, &mut chv).expect("single-node path must verify");
    }

    #[test]
    fn three_node_path_roundtrip() {
        let nodes = make_consistent_path(3, &[8, 4, 16], 0x1234_5678);
        let mut ch = FsChallenger::new(b"membership-3node");
        let proof = prove_path(&nodes, &mut ch);
        assert_eq!(proof.node_proofs.len(), 3);

        let mut chv = FsChallenger::new(b"membership-3node");
        verify_path(&proof, &mut chv).expect("3-node path must verify");
    }

    #[test]
    fn cross_node_binding_rejects_mismatch() {
        let nodes = make_consistent_path(3, &[8, 4, 16], 0xDEAD_BEEF);
        let mut ch = FsChallenger::new(b"membership-cross-reject");
        let mut proof = prove_path(&nodes, &mut ch);

        // Tamper node 1's witness so its recomputed content hash changes
        // (Level 2 still passes with this new hash), but it no longer
        // matches node 0's leaf (Level 3 fails).
        proof.node_witnesses[1].extraction_masks[0] ^= 1;
        let w = &proof.node_witnesses[1];
        let (_, new_hash) = content_hash_to_sha256_chain(
            &w.extraction_masks,
            &w.sparse_keys,
            &proof.native_merkle_roots[1],
            &w.counts,
        );
        proof.content_hashes[1] = new_hash;

        let mut chv = FsChallenger::new(b"membership-cross-reject");
        let err = verify_path(&proof, &mut chv);
        match err {
            Err(MembershipVerifyError::CrossNodeMismatch {
                path_idx: 0,
                parent_node_idx: 0,
            }) => {}
            other => panic!(
                "expected CrossNodeMismatch at parent_node_idx=0, got {:?}",
                other.err()
            ),
        }
    }

    #[test]
    fn content_hash_tamper_rejects() {
        let nodes = make_consistent_path(2, &[4, 8], 0xCAFE_BABE);
        let mut ch = FsChallenger::new(b"membership-content-tamper");
        let proof = prove_path(&nodes, &mut ch);

        // Tamper: flip a bit in node 0's extraction_masks
        let mut tampered = proof;
        tampered.node_witnesses[0].extraction_masks[0] ^= 1;

        let mut chv = FsChallenger::new(b"membership-content-tamper");
        let err = verify_path(&tampered, &mut chv);
        match err {
            Err(MembershipVerifyError::ContentHashMismatch {
                path_idx: 0,
                node_idx: 0,
            }) => {}
            other => panic!(
                "expected ContentHashMismatch at node_idx=0, got {:?}",
                other.err()
            ),
        }
    }

    #[test]
    fn multipoint_4_paths_roundtrip() {
        let paths: Vec<Vec<MhotNodeFullWitness>> = (0..4)
            .map(|i| {
                make_consistent_path(3, &[8, 4, 16], 0x1000_0000 + i)
            })
            .collect();

        let mut ch = FsChallenger::new(b"multipoint-4paths");
        let proof = prove_multipoint(&paths, &mut ch);
        assert_eq!(proof.path_proofs.len(), 4);

        let mut chv = FsChallenger::new(b"multipoint-4paths");
        verify_multipoint(&proof, &mut chv)
            .expect("4-path multipoint must verify");
    }

    #[test]
    fn multipoint_cross_node_reject() {
        let paths: Vec<Vec<MhotNodeFullWitness>> = (0..2)
            .map(|i| {
                make_consistent_path(2, &[4, 8], 0x2000_0000 + i)
            })
            .collect();

        let mut ch = FsChallenger::new(b"multipoint-cross-reject");
        let mut proof = prove_multipoint(&paths, &mut ch);

        // Tamper path 1, node 1's witness so recomputed content hash is
        // internally consistent but mismatches node 0's leaf.
        proof.path_proofs[1].node_witnesses[1].extraction_masks[0] ^= 1;
        let w = &proof.path_proofs[1].node_witnesses[1];
        let (_, new_hash) = content_hash_to_sha256_chain(
            &w.extraction_masks,
            &w.sparse_keys,
            &proof.path_proofs[1].native_merkle_roots[1],
            &w.counts,
        );
        proof.path_proofs[1].content_hashes[1] = new_hash;

        let mut chv = FsChallenger::new(b"multipoint-cross-reject");
        let err = verify_multipoint(&proof, &mut chv);
        match err {
            Err(MembershipVerifyError::CrossNodeMismatch {
                path_idx: 1,
                parent_node_idx: 0,
            }) => {}
            other => panic!(
                "expected CrossNodeMismatch at path_idx=1, parent_node_idx=0, got {:?}",
                other.err()
            ),
        }
    }
}
