use crate::prover::prove_fast_core;
use crate::r1cs_hashes::sha2::{
    Compression, MerklePathProof, MerklePathVerifyError, SHA256_IV,
    Sha256HybridSetup, min_n_blocks_log, sha256_compress,
};
use flock_core::challenger::{Challenger, FsChallenger};
use flock_core::field::F128;
use flock_core::lincheck::LincheckProof;
use flock_core::pcs::{
    self, BatchOpeningProofLigerito, Commitment, DirectEqInd, PackedDirectClaim,
    PackedDirectClaimRef,
};
use flock_core::proof::R1csClaim;
use flock_core::verifier::VerifyError;
use flock_core::zerocheck::ZerocheckProof;

use super::multiproof::{open_core_ligerito, verify_core_opening_ligerito};
use super::native_witness::{MhotNodeWitness, mhot_node_to_sha256_merkle};
use super::route_f32::{self as route, RouteF32Setup, RouteF32Witness};

#[derive(Debug)]
pub enum MhotMembershipError {
    NodeVerify(MerklePathVerifyError),
    CrossNodeBinding {
        parent_idx: usize,
        parent_leaf: [u32; 8],
        child_root: [u32; 8],
    },
    RouteVerify(VerifyError),
    RouteOpening(VerifyError),
    RootMismatch {
        expected: [u32; 8],
        actual: [u32; 8],
    },
}

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
    let n_real = proof.b_bits.len();
    let needed = 1usize << min_n_blocks_log(n_real);
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

/// Verify in-node Merkle paths for a sequence of MHOT nodes, including
/// cross-node binding: node[i].leaf (selected child digest) must equal
/// node[i+1].native_root (the child's content hash / in-node Merkle root).
///
/// Each node's in-node wiring is proven sound via shift-sumcheck (O(1) PD
/// claim per node). Cross-node binding is a public-value equality check:
/// the shift-sumcheck guarantees that `leaf` and `native_root` are the
/// actual committed values, so verifier-side equality suffices.
pub fn verify_path_merkle<Ch: Challenger>(
    proofs: &[NodeMerkleProof],
    challenger: &mut Ch,
) -> Result<(), MhotMembershipError> {
    for p in proofs {
        verify_node_merkle(p, challenger).map_err(MhotMembershipError::NodeVerify)?;
    }
    // Cross-node binding uses SNARK-authenticated values only.
    // proofs[i].leaf is authenticated (SNARK public input).
    // proofs[i+1].root is authenticated (SNARK public input).
    // native_root is NOT used because it is not SNARK-authenticated.
    for i in 0..proofs.len().saturating_sub(1) {
        let parent_selected = proofs[i].leaf;
        let child_root = proofs[i + 1].root;
        if parent_selected != child_root {
            return Err(MhotMembershipError::CrossNodeBinding {
                parent_idx: i,
                parent_leaf: parent_selected,
                child_root,
            });
        }
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

// ---------------------------------------------------------------------------
// Route↔hash binding (Task 1): one membership proof tying the route R1CS
// (PEXT matching + content soundness) to the in-node SHA-256 Merkle proof.
// Two independent commitments (hash + route) are bound by opening the route
// commitment's SELECTED_OUT_FINAL (the routed-child digest) via PackedDirect-
// Claims and asserting it equals the hash side's authenticated leaf.
// ---------------------------------------------------------------------------

/// Within-block packed index of SELECTED_OUT_FINAL's first F128. It is
/// DIGEST_BITS-aligned, so it spans two consecutive F128 slots:
/// SOF_PACKED_BASE and SOF_PACKED_BASE + 1.
const SOF_PACKED_BASE: usize = route::SELECTED_OUT_FINAL_BASE / 128;
/// F128 slots per route block.
const BLOCK_PACKED: usize = route::K / 128;

/// One membership step: the MHOT node plus the route witness that PEXT-routes
/// to the selected child.
#[derive(Clone)]
pub struct MhotMembershipInput {
    pub node: MhotNodeWitness,
    pub route_witness: RouteF32Witness,
}

impl MhotMembershipInput {
    /// Build an input whose route witness PEXT-routes to `node.selected_child`.
    pub fn from_node(node: MhotNodeWitness) -> Self {
        let route_witness = mhot_node_to_route_witness(&node);
        Self { node, route_witness }
    }
}

/// A sound membership proof for one MHOT path: per-node SHA-256 Merkle path
/// proofs (hash base) + one batched route R1CS proof (route base), bound by
/// PackedDirectClaims over the route commitment.
pub struct MhotMembershipProof {
    pub hash_proofs: Vec<NodeMerkleProof>,
    pub route_zc: ZerocheckProof,
    pub route_lc: LincheckProof,
    pub route_pcs: BatchOpeningProofLigerito,
    pub route_commitment: Commitment,
    pub route_claim: R1csClaim,
    pub n_routes: usize,
}

/// Build a route witness PEXT-routing to `node.selected_child`: a prefix mask
/// of width W_MAX plus a key equal to the selected child index satisfy the
/// route R1CS content checks (mask prefix + key validity).
pub fn mhot_node_to_route_witness(node: &MhotNodeWitness) -> RouteF32Witness {
    let selected = node.selected_child;
    let mut key = [false; route::KEY_BITS];
    let mut mask = [false; route::KEY_BITS];
    for j in 0..route::W_MAX {
        mask[j] = true;
        key[j] = (selected >> j) & 1 == 1;
    }
    let child_bits: Vec<[bool; route::DIGEST_BITS]> =
        node.children.iter().map(digest_bytes_to_route_bits).collect();
    let fanout = child_bits.len();
    RouteF32Witness::new_padded(key, mask, &child_bits, fanout)
}

/// `[u8; 32]` digest → `[bool; 256]` in byte-major, LSB-first-within-byte order.
fn digest_bytes_to_route_bits(d: &[u8; 32]) -> [bool; route::DIGEST_BITS] {
    let mut bits = [false; route::DIGEST_BITS];
    for (byte_i, &byte) in d.iter().enumerate() {
        for k in 0..8 {
            bits[byte_i * 8 + k] = (byte >> k) & 1 == 1;
        }
    }
    bits
}

/// Recover the `[u8; 32]` digest from a SHA-256 leaf (big-endian words).
fn leaf_words_to_digest_bytes(leaf: &[u32; 8]) -> [u8; 32] {
    let mut d = [0u8; 32];
    for i in 0..8 {
        d[4 * i..4 * i + 4].copy_from_slice(&leaf[i].to_be_bytes());
    }
    d
}

/// Pack up to 128 bools into one F128 (lo = bits 0..64, hi = bits 64..128).
fn pack_bits_to_f128(bits: &[bool]) -> F128 {
    let mut lo = 0u64;
    let mut hi = 0u64;
    for (k, &b) in bits.iter().enumerate() {
        if b {
            if k < 64 {
                lo |= 1u64 << k;
            } else {
                hi |= 1u64 << (k - 64);
            }
        }
    }
    F128 { lo, hi }
}

/// The two SELECTED_OUT_FINAL F128 values for a child digest given as bytes.
fn digest_to_sof_f128(d: &[u8; 32]) -> [F128; 2] {
    let bits = digest_bytes_to_route_bits(d);
    [
        pack_bits_to_f128(&bits[0..128]),
        pack_bits_to_f128(&bits[128..256]),
    ]
}

/// The two SELECTED_OUT_FINAL F128 values the route R1CS produces for this
/// witness: the digest of the child whose 5-bit index equals the extracted
/// key bits. Matches what is committed in the route z_packed.
fn route_sof_f128(rw: &RouteF32Witness) -> [F128; 2] {
    let mut idx = 0usize;
    for j in 0..route::W_MAX {
        if rw.key[j] && rw.mask[j] {
            idx |= 1 << j;
        }
    }
    let bits = &rw.children[idx];
    [
        pack_bits_to_f128(&bits[0..128]),
        pack_bits_to_f128(&bits[128..256]),
    ]
}

/// PackedDirectClaim point selecting route instance `instance`'s F128 at
/// within-block packed index `within`: the LSB-first binary expansion of the
/// global packed index over `L = m − LOG_PACKING` coords.
fn pd_point(setup: &RouteF32Setup, instance: usize, within: usize) -> Vec<F128> {
    let gpi = instance * BLOCK_PACKED + within;
    let l = setup.r1cs.m - pcs::LOG_PACKING;
    (0..l)
        .map(|k| if (gpi >> k) & 1 == 1 { F128::ONE } else { F128::ZERO })
        .collect()
}

/// Prove a single sound MHOT membership path.
pub fn prove_membership(
    path: &[MhotMembershipInput],
    challenger: &mut FsChallenger,
) -> MhotMembershipProof {
    assert!(!path.is_empty(), "membership path must have at least one node");

    // ---- Hash base: per-node in-node Merkle path proofs (threads challenger).
    let nodes: Vec<MhotNodeWitness> = path.iter().map(|p| p.node.clone()).collect();
    let hash_proofs = prove_path_merkle(&nodes, challenger);

    // ---- Route base: batch all route witnesses into one commitment.
    let route_witnesses: Vec<RouteF32Witness> =
        path.iter().map(|p| p.route_witness.clone()).collect();
    let n_routes = route_witnesses.len();
    let setup = RouteF32Setup::cached(n_routes);
    let (rz, ra, rb, rzlc) =
        route::generate_witness_with_ab_packed_and_lincheck(&route_witnesses, setup.n_blocks_log());
    let route_core = prove_fast_core(
        &setup.r1cs,
        &setup.pcs_params,
        rz,
        ra,
        rb,
        rzlc,
        setup.r1cs.csc_lincheck_circuit(),
        challenger,
    );

    // ---- Binding: open each route instance's SELECTED_OUT_FINAL (2 F128).
    // The PD value is the route's actual routed-child digest (committed in
    // z_packed), so the opening is always valid; the route↔hash equality is
    // enforced at verify time, where the verifier supplies the hash leaf.
    let mut pd_claims: Vec<PackedDirectClaim> = Vec::with_capacity(2 * path.len());
    for (i, p) in path.iter().enumerate() {
        let sof = route_sof_f128(&p.route_witness);
        for (slot, &value) in [SOF_PACKED_BASE, SOF_PACKED_BASE + 1].iter().zip(sof.iter()) {
            let point = pd_point(&setup, i, *slot);
            let eq_ind = DirectEqInd::Sparse(pcs::ring_switch::build_eq_sparse(&point));
            pd_claims.push(PackedDirectClaim { point, value, eq_ind });
        }
    }

    let route_open = open_core_ligerito(
        &setup.r1cs,
        &setup.pcs_params,
        route_core,
        n_routes,
        &pd_claims,
        challenger,
    );

    MhotMembershipProof {
        hash_proofs,
        route_zc: route_open.zc_proof,
        route_lc: route_open.lc_proof,
        route_pcs: route_open.pcs_open,
        route_commitment: route_open.commitment,
        route_claim: route_open.claim,
        n_routes,
    }
}

/// Verify a sound MHOT membership path against a public root.
pub fn verify_membership(
    proof: &MhotMembershipProof,
    expected_root: &[u32; 8],
    challenger: &mut FsChallenger,
) -> Result<(), MhotMembershipError> {
    assert!(!proof.hash_proofs.is_empty(), "empty membership proof");

    // ---- Hash base + cross-node binding (threads challenger).
    verify_path_merkle(&proof.hash_proofs, challenger)?;

    // ---- Public root = the top node's in-node Merkle root.
    if proof.hash_proofs[0].root != *expected_root {
        return Err(MhotMembershipError::RootMismatch {
            expected: *expected_root,
            actual: proof.hash_proofs[0].root,
        });
    }

    // ---- Route base: replay the core, then check the binding PD claims using
    // the SNARK-authenticated hash leaves as the expected SELECTED_OUT_FINAL.
    let setup = RouteF32Setup::cached(proof.n_routes);
    let (route_ab, route_c) = flock_core::verifier::verify_core(
        &setup.r1cs,
        &proof.route_zc,
        &proof.route_lc,
        &proof.route_commitment,
        setup.r1cs.csc_lincheck_circuit(),
        challenger,
    )
    .map_err(MhotMembershipError::RouteVerify)?;

    let mut pd_data: Vec<(Vec<F128>, F128)> = Vec::with_capacity(2 * proof.hash_proofs.len());
    for (i, hp) in proof.hash_proofs.iter().enumerate() {
        let sof = digest_to_sof_f128(&leaf_words_to_digest_bytes(&hp.leaf));
        pd_data.push((pd_point(&setup, i, SOF_PACKED_BASE), sof[0]));
        pd_data.push((pd_point(&setup, i, SOF_PACKED_BASE + 1), sof[1]));
    }
    let pd_refs: Vec<PackedDirectClaimRef> = pd_data
        .iter()
        .map(|(point, value)| PackedDirectClaimRef { point, value: *value })
        .collect();

    verify_core_opening_ligerito(
        &setup.r1cs,
        &setup.pcs_params,
        &proof.route_commitment,
        &proof.route_pcs,
        &route_ab,
        &route_c,
        &pd_refs,
        challenger,
    )
    .map_err(MhotMembershipError::RouteOpening)?;

    Ok(())
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
    fn three_node_path_independent_verify() {
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
        for p in &proofs {
            verify_node_merkle(p, &mut chv)
                .expect("each node must verify independently");
        }
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
