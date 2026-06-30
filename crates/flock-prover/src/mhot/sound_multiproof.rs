use std::collections::HashMap;

use crate::prover::prove_fast_core;
use crate::r1cs_hashes::sha2::{SHA256_IV, Sha256HybridSetup, sha256_compress};
use flock_core::challenger::{Challenger, FsChallenger};
use flock_core::field::F128;
use flock_core::lincheck::LincheckProof;
use flock_core::pcs::{
    self, BatchOpeningProofLigerito, Commitment, DirectEqInd, PackedDirectClaim,
    PackedDirectClaimRef,
};
use flock_core::zerocheck::ZerocheckProof;

use super::merkle_membership::{
    ContentChainProof, MhotMembershipError, MhotMembershipInput,
    NodeMerkleProof, SOF_PACKED_BASE, build_content_hash_chain,
    digest_to_sof_f128, fork_content_challenger, leaf_words_to_digest_bytes,
    pd_point, prove_node_merkle, route_sof_f128, verify_node_merkle,
};
use super::multiproof::{open_core_ligerito, verify_core_opening_ligerito};
use super::route_f32::{self as route, RouteF32Setup};

#[derive(serde::Serialize)]
pub struct PathMapping {
    pub node_indices: Vec<Vec<usize>>,
}

#[derive(serde::Serialize)]
pub struct SoundMultiproof {
    pub hash_proofs: Vec<NodeMerkleProof>,
    pub content_proofs: Vec<ContentChainProof>,

    pub route_zc: ZerocheckProof,
    pub route_lc: LincheckProof,
    pub route_pcs: BatchOpeningProofLigerito,
    pub route_commitment: Commitment,
    pub n_routes: usize,

    pub n_paths: usize,
    pub path_depths: Vec<usize>,
    pub path_mapping: PathMapping,
}

impl SoundMultiproof {
    pub fn proof_size_bytes(&self) -> usize {
        bincode::serialized_size(self).unwrap_or(0) as usize
    }
}

fn node_identity(input: &MhotMembershipInput) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(input.node.children.len() as u32).to_le_bytes());
    for child in &input.node.children {
        bytes.extend_from_slice(child);
    }
    bytes.extend_from_slice(&(input.node.selected_child as u32).to_le_bytes());
    for &mask in &input.content.extraction_masks {
        bytes.extend_from_slice(&mask.to_le_bytes());
    }
    bytes.extend_from_slice(&(input.content.sparse_partial_keys.len() as u32).to_le_bytes());
    for &key in &input.content.sparse_partial_keys {
        bytes.extend_from_slice(&key.to_le_bytes());
    }
    bytes.extend_from_slice(&(input.content.child_leaf_counts.len() as u32).to_le_bytes());
    for &count in &input.content.child_leaf_counts {
        bytes.extend_from_slice(&count.to_le_bytes());
    }
    bytes
}

pub fn prove_sound_multiproof(
    paths: &[Vec<MhotMembershipInput>],
    challenger: &mut FsChallenger,
) -> SoundMultiproof {
    assert!(!paths.is_empty(), "need at least one path");
    for path in paths {
        assert!(!path.is_empty(), "each path must have at least one node");
    }

    // -- Dedup --
    let mut identity_map: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut unique_nodes: Vec<MhotMembershipInput> = Vec::new();
    let mut node_indices_per_path: Vec<Vec<usize>> = Vec::with_capacity(paths.len());
    let mut path_depths: Vec<usize> = Vec::with_capacity(paths.len());

    for path in paths {
        let mut indices = Vec::with_capacity(path.len());
        for input in path {
            let key = node_identity(input);
            let idx = identity_map.entry(key).or_insert_with(|| {
                let i = unique_nodes.len();
                unique_nodes.push(input.clone());
                i
            });
            indices.push(*idx);
        }
        path_depths.push(path.len());
        node_indices_per_path.push(indices);
    }

    let u = unique_nodes.len();

    // -- Hash base: per unique node --
    let hash_proofs: Vec<NodeMerkleProof> = unique_nodes
        .iter()
        .map(|input| prove_node_merkle(&input.node, challenger))
        .collect();

    // -- Content chain: per unique node (forked challengers) --
    let content_proofs: Vec<ContentChainProof> = unique_nodes
        .iter()
        .zip(hash_proofs.iter())
        .enumerate()
        .map(|(idx, (input, hp))| {
            let merkle_root_bytes = leaf_words_to_digest_bytes(&hp.native_root);
            let (compressions, content_hash, cv_last, n_real) =
                build_content_hash_chain(&input.content, &merkle_root_bytes);
            let n = compressions.len();
            let setup = Sha256HybridSetup::cached(n);
            let mut chain_ch = fork_content_challenger(challenger, idx);
            let (proof, commitment) = setup.prove_chain(&compressions, &mut chain_ch);
            challenger.observe_bytes(&commitment.root);
            ContentChainProof {
                proof, commitment, content_hash, cv_last,
                n_compressions: n, n_real_compressions: n_real,
            }
        })
        .collect();

    // -- Route base: batch all unique route witnesses --
    let route_witnesses: Vec<route::RouteF32Witness> =
        unique_nodes.iter().map(|input| input.route_witness.clone()).collect();
    let n_routes = route_witnesses.len();
    let setup = RouteF32Setup::cached(n_routes);
    let (rz, ra, rb, rzlc) =
        route::generate_witness_with_ab_packed_and_lincheck(&route_witnesses, setup.n_blocks_log());
    let route_core = prove_fast_core(
        &setup.r1cs,
        &setup.pcs_params,
        rz, ra, rb, rzlc,
        setup.r1cs.csc_lincheck_circuit(),
        challenger,
    );

    // -- PD claims: bind route SELECTED_OUT_FINAL to hash leaf --
    let mut pd_claims: Vec<PackedDirectClaim> = Vec::with_capacity(2 * u);
    for (i, input) in unique_nodes.iter().enumerate() {
        let sof = route_sof_f128(&input.route_witness);
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

    SoundMultiproof {
        hash_proofs,
        content_proofs,
        route_zc: route_open.zc_proof,
        route_lc: route_open.lc_proof,
        route_pcs: route_open.pcs_open,
        route_commitment: route_open.commitment,
        n_routes,
        n_paths: paths.len(),
        path_depths,
        path_mapping: PathMapping { node_indices: node_indices_per_path },
    }
}

pub fn verify_sound_multiproof(
    proof: &SoundMultiproof,
    expected_root: &[u32; 8],
    challenger: &mut FsChallenger,
) -> Result<(), MhotMembershipError> {
    let u = proof.hash_proofs.len();
    if u == 0 || u != proof.content_proofs.len() {
        return Err(MhotMembershipError::RootMismatch {
            expected: *expected_root,
            actual: [0; 8],
        });
    }

    // -- Hash base --
    for hp in &proof.hash_proofs {
        verify_node_merkle(hp, challenger).map_err(MhotMembershipError::NodeVerify)?;
    }

    // -- Content chain --
    for (i, cp) in proof.content_proofs.iter().enumerate() {
        let setup = Sha256HybridSetup::cached(cp.n_compressions);
        let mut chain_ch = fork_content_challenger(challenger, i);
        setup
            .verify_chain(&cp.commitment, &cp.proof, &SHA256_IV, &cp.cv_last, &mut chain_ch)
            .map_err(|e| MhotMembershipError::ContentChainVerify(i, e))?;
        challenger.observe_bytes(&cp.commitment.root);

        if cp.n_real_compressions > cp.n_compressions {
            return Err(MhotMembershipError::ContentHashMismatch { node_idx: i });
        }
        let n_pad = cp.n_compressions - cp.n_real_compressions;
        let mut expected_cv = cp.content_hash;
        for _ in 0..n_pad {
            expected_cv = sha256_compress(&expected_cv, &[0u32; 16]);
        }
        if expected_cv != cp.cv_last {
            return Err(MhotMembershipError::ContentHashMismatch { node_idx: i });
        }
    }

    // -- Cross-node binding (per path) --
    for (p, indices) in proof.path_mapping.node_indices.iter().enumerate() {
        if p >= proof.path_depths.len() || indices.len() != proof.path_depths[p] {
            return Err(MhotMembershipError::RootMismatch {
                expected: *expected_root, actual: [0; 8],
            });
        }
        for i in 0..indices.len().saturating_sub(1) {
            let parent_u = indices[i];
            let child_u = indices[i + 1];
            if parent_u >= u || child_u >= u {
                return Err(MhotMembershipError::RootMismatch {
                    expected: *expected_root, actual: [0; 8],
                });
            }
            let parent_leaf = proof.hash_proofs[parent_u].leaf;
            let child_content = proof.content_proofs[child_u].content_hash;
            if parent_leaf != child_content {
                return Err(MhotMembershipError::CrossNodeBinding {
                    parent_idx: parent_u,
                    parent_leaf,
                    child_root: child_content,
                });
            }
        }
    }

    // -- Root check: every path's top node must match expected_root --
    for indices in &proof.path_mapping.node_indices {
        if indices.is_empty() {
            return Err(MhotMembershipError::RootMismatch {
                expected: *expected_root, actual: [0; 8],
            });
        }
        let root_u = indices[0];
        if root_u >= u {
            return Err(MhotMembershipError::RootMismatch {
                expected: *expected_root, actual: [0; 8],
            });
        }
        if proof.content_proofs[root_u].content_hash != *expected_root {
            return Err(MhotMembershipError::RootMismatch {
                expected: *expected_root,
                actual: proof.content_proofs[root_u].content_hash,
            });
        }
    }

    // -- Route base --
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

    // -- PD claims: verify route <-> hash binding --
    let mut pd_data: Vec<(Vec<F128>, F128)> = Vec::with_capacity(2 * u);
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
