use std::collections::HashMap;

fn vmrss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("VmRSS:"))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<f64>().ok()))
        .unwrap_or(0.0) / 1024.0
}

use crate::chain::{ChainShiftProof, prove_chain_shift, verify_chain_shift};
use crate::merkle_path::{MerklePathShiftProof, prove_merkle_path_shift, verify_merkle_path_shift};
use crate::prover::prove_fast_core_with_block_count;
use crate::r1cs_hashes::chain_common::{
    ChainFold, assemble_chain_claim_at_offset, build_chain_claim_point_at_offset,
    fold_in_out_range,
};
use crate::r1cs_hashes::merkle_path_common::{
    MerklePathFold, assemble_merkle_path_claim_at_offset, build_merkle_claim_point_at_offset,
    fold_all_slots_range,
};
use crate::r1cs_hashes::sha2::{
    CHAIN_LAYOUT, Compression, MERKLE_LAYOUT, SHA256_IV, Sha256HybridSetup,
    generate_witness_with_ab_packed_and_lincheck, min_n_blocks_log, sha256_compress,
};
use flock_core::challenger::{Challenger, FsChallenger};
use flock_core::field::F128;
use flock_core::lincheck::LincheckProof;
use flock_core::pcs::{
    self, BatchOpeningProofLigerito, Commitment, PackedDirectClaim, PackedDirectClaimRef,
};
use flock_core::zerocheck::ZerocheckProof;

use super::merkle_membership::{
    MhotMembershipError, MhotMembershipInput, SOF_PACKED_BASE,
    build_content_hash_chain, digest_to_sof_f128,
    leaf_words_to_digest_bytes, pad_to_needed, pd_point, route_sof_f128,
};
use super::multiproof::{fork_pcs_challenger, open_core_ligerito, verify_core_opening_ligerito};
use super::native_witness::mhot_node_to_sha256_merkle;
use super::route_f32::{self as route, RouteF32Setup};

#[derive(serde::Serialize)]
pub struct PathMapping {
    pub node_indices: Vec<Vec<usize>>,
}

#[derive(serde::Serialize)]
pub struct SoundMultiproof {
    // Merkle base (separate commitment for in-node merkle compressions)
    pub merkle_zc: ZerocheckProof,
    pub merkle_lc: LincheckProof,
    pub merkle_pcs: BatchOpeningProofLigerito,
    pub merkle_commitment: Commitment,

    // Chain base (separate commitment for content-hash chain compressions)
    pub chain_zc: ZerocheckProof,
    pub chain_lc: LincheckProof,
    pub chain_pcs: BatchOpeningProofLigerito,
    pub chain_commitment: Commitment,

    // Per-node merkle shift proofs
    pub merkle_shifts: Vec<MerklePathShiftProof>,
    pub merkle_leaves: Vec<[u32; 8]>,
    pub merkle_roots: Vec<[u32; 8]>,
    pub merkle_native_roots: Vec<[u32; 8]>,
    pub merkle_b_bits: Vec<Vec<bool>>,

    // Per-node chain shift proofs
    pub chain_shifts: Vec<ChainShiftProof>,
    pub chain_content_hashes: Vec<[u32; 8]>,
    pub chain_cv_lasts: Vec<[u32; 8]>,
    pub chain_n_compressions: Vec<usize>,
    pub chain_n_real: Vec<usize>,

    // Block layout metadata (verifier needs these for PD claim assembly)
    pub merkle_block_offsets: Vec<usize>,
    pub merkle_block_counts: Vec<usize>,
    pub chain_block_offsets: Vec<usize>,
    pub n_log_merkle: usize,
    pub n_log_chain: usize,

    // Route base (unchanged)
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

struct AlignedAllocation {
    offsets: Vec<usize>,
    n_real: usize,
    n_total: usize,
    n_log: usize,
}

fn allocate_blocks_aligned(block_counts: &[usize]) -> AlignedAllocation {
    let u = block_counts.len();
    let mut indices: Vec<usize> = (0..u).collect();
    indices.sort_by(|&a, &b| block_counts[b].cmp(&block_counts[a]));

    let mut offsets = vec![0usize; u];
    let mut cursor = 0usize;
    for &i in &indices {
        let align = block_counts[i];
        cursor = (cursor + align - 1) & !(align - 1);
        offsets[i] = cursor;
        cursor += block_counts[i];
    }
    let n_real = cursor;
    let min_n = 1usize << (22 - crate::r1cs_hashes::sha2::K_LOG);
    let n_total = cursor.max(min_n).next_power_of_two();
    let n_log = n_total.trailing_zeros() as usize;
    AlignedAllocation { offsets, n_real, n_total, n_log }
}

fn build_comp_vec(
    data: &[(Vec<Compression>, usize)],
    n_total: usize,
) -> Vec<Compression> {
    let mut comps: Vec<Compression> = vec![(SHA256_IV, [0; 16]); n_total];
    for (comp_list, offset) in data {
        for (j, comp) in comp_list.iter().enumerate() {
            comps[*offset + j] = *comp;
        }
    }
    comps
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
    eprintln!("[mem] after dedup ({} unique): {:.0} MB", u, vmrss_mb());

    // -- Per-node: compute merkle + chain compressions and block counts --
    let mut merkle_data: Vec<(Vec<Compression>, Vec<bool>, [u32; 8], [u32; 8], [u32; 8])> =
        Vec::with_capacity(u);
    let mut chain_data: Vec<(Vec<Compression>, [u32; 8], [u32; 8], usize)> = Vec::with_capacity(u);
    let mut merkle_block_counts: Vec<usize> = Vec::with_capacity(u);
    let mut chain_block_counts: Vec<usize> = Vec::with_capacity(u);

    for input in &unique_nodes {
        let w = mhot_node_to_sha256_merkle(&input.node);
        let n_real_merkle = w.compressions.len();
        let mut compressions = w.compressions;
        let mut b_bits = w.b_bits.clone();
        let needed = 1usize << min_n_blocks_log(n_real_merkle);
        let padded_root = pad_to_needed(&mut compressions, &mut b_bits, needed);
        merkle_data.push((compressions, b_bits, w.leaf, padded_root, w.native_root));
        merkle_block_counts.push(needed);

        let merkle_root_bytes = leaf_words_to_digest_bytes(&w.native_root);
        let (chain_comps, content_hash, cv_last, n_real) =
            build_content_hash_chain(&input.content, &merkle_root_bytes);
        chain_data.push((chain_comps.clone(), content_hash, cv_last, n_real));
        chain_block_counts.push(chain_comps.len());
    }

    // -- Pass 1: Merkle commitment --
    let merkle_alloc = allocate_blocks_aligned(&merkle_block_counts);
    eprintln!("[mem] merkle alloc (n_log={}, {} blocks): {:.0} MB",
        merkle_alloc.n_log, merkle_alloc.n_total, vmrss_mb());

    let merkle_comp_data: Vec<(Vec<Compression>, usize)> = (0..u)
        .map(|i| (merkle_data[i].0.clone(), merkle_alloc.offsets[i]))
        .collect();
    let merkle_comps = build_comp_vec(&merkle_comp_data, merkle_alloc.n_total);
    drop(merkle_comp_data);

    let (mz, ma, mb, mzlc) =
        generate_witness_with_ab_packed_and_lincheck(&merkle_comps, merkle_alloc.n_log);
    drop(merkle_comps);

    let merkle_setup = Sha256HybridSetup::cached(merkle_alloc.n_total);
    let merkle_core = prove_fast_core_with_block_count(
        &merkle_setup.r1cs, &merkle_setup.pcs_params,
        mz, ma, mb, mzlc,
        merkle_setup.r1cs.csc_lincheck_circuit(),
        Some(merkle_alloc.n_real), challenger,
    );
    eprintln!("[mem] after merkle prove_fast_core: {:.0} MB", vmrss_mb());

    let merkle_tau_pos = challenger.sample_f128_vec(MERKLE_LAYOUT.tau_pos_len());
    let merkle_fold = MerklePathFold::new(&MERKLE_LAYOUT, merkle_tau_pos);

    let mut merkle_shifts = Vec::with_capacity(u);
    let mut merkle_pd_claims: Vec<PackedDirectClaim> = Vec::with_capacity(u);

    for i in 0..u {
        let slots = fold_all_slots_range(
            &MERKLE_LAYOUT, &merkle_core.z_packed, &merkle_fold,
            merkle_alloc.offsets[i], merkle_block_counts[i],
        );
        let n_inst = merkle_block_counts[i];
        let mut b_bits_padded = merkle_data[i].1.clone();
        b_bits_padded.resize(n_inst, false);

        let (shift_proof, shift_claims) = prove_merkle_path_shift(
            0,
            &slots[MERKLE_LAYOUT.x_l_slot as usize],
            &slots[MERKLE_LAYOUT.x_r_slot as usize],
            &slots[MERKLE_LAYOUT.z_slot as usize],
            &slots[MERKLE_LAYOUT.other_slot() as usize],
            &b_bits_padded,
            MERKLE_LAYOUT.slot_layout(),
            challenger,
        );
        let pd = assemble_merkle_path_claim_at_offset(
            &MERKLE_LAYOUT, &merkle_fold, &shift_claims,
            merkle_alloc.offsets[i], merkle_block_counts[i], merkle_alloc.n_log,
        );
        merkle_shifts.push(shift_proof);
        merkle_pd_claims.push(pd);
    }

    let mut merkle_pcs_ch = fork_pcs_challenger(challenger, b"merkle");
    let merkle_open = open_core_ligerito(
        &merkle_setup.r1cs, &merkle_setup.pcs_params,
        merkle_core, merkle_alloc.n_real, &merkle_pd_claims, &mut merkle_pcs_ch,
    );
    eprintln!("[mem] after merkle PCS open: {:.0} MB", vmrss_mb());

    // -- Pass 2: Chain commitment --
    let chain_alloc = allocate_blocks_aligned(&chain_block_counts);
    eprintln!("[mem] chain alloc (n_log={}, {} blocks): {:.0} MB",
        chain_alloc.n_log, chain_alloc.n_total, vmrss_mb());

    let chain_comp_data: Vec<(Vec<Compression>, usize)> = (0..u)
        .map(|i| (chain_data[i].0.clone(), chain_alloc.offsets[i]))
        .collect();
    let chain_comps = build_comp_vec(&chain_comp_data, chain_alloc.n_total);
    drop(chain_comp_data);

    let (cz, ca, cb, czlc) =
        generate_witness_with_ab_packed_and_lincheck(&chain_comps, chain_alloc.n_log);
    drop(chain_comps);

    let chain_setup = Sha256HybridSetup::cached(chain_alloc.n_total);
    let chain_core = prove_fast_core_with_block_count(
        &chain_setup.r1cs, &chain_setup.pcs_params,
        cz, ca, cb, czlc,
        chain_setup.r1cs.csc_lincheck_circuit(),
        Some(chain_alloc.n_real), challenger,
    );
    eprintln!("[mem] after chain prove_fast_core: {:.0} MB", vmrss_mb());

    let chain_tau_pos = challenger.sample_f128_vec(CHAIN_LAYOUT.tau_pos_len());
    let chain_fold = ChainFold::new(&CHAIN_LAYOUT, chain_tau_pos);

    let mut chain_shifts = Vec::with_capacity(u);
    let mut chain_pd_claims: Vec<PackedDirectClaim> = Vec::with_capacity(u);

    for i in 0..u {
        let (in_vals, out_vals) = fold_in_out_range(
            &CHAIN_LAYOUT, &chain_core.z_packed, &chain_fold,
            chain_alloc.offsets[i], chain_block_counts[i],
        );
        let (shift_proof, shift_claims) = prove_chain_shift(&in_vals, &out_vals, challenger);
        let pd = assemble_chain_claim_at_offset(
            &CHAIN_LAYOUT, &chain_fold, &shift_claims,
            chain_alloc.offsets[i], chain_block_counts[i], chain_alloc.n_log,
        );
        chain_shifts.push(shift_proof);
        chain_pd_claims.push(pd);
    }

    let mut chain_pcs_ch = fork_pcs_challenger(challenger, b"chain");
    let chain_open = open_core_ligerito(
        &chain_setup.r1cs, &chain_setup.pcs_params,
        chain_core, chain_alloc.n_real, &chain_pd_claims, &mut chain_pcs_ch,
    );
    eprintln!("[mem] after chain PCS open: {:.0} MB", vmrss_mb());

    // -- Pass 3: Route base (unchanged) --
    let route_witnesses: Vec<route::RouteF32Witness> =
        unique_nodes.iter().map(|input| input.route_witness.clone()).collect();
    let n_routes = route_witnesses.len();
    let route_setup = RouteF32Setup::cached(n_routes);
    let (rz, ra, rb, rzlc) =
        route::generate_witness_with_ab_packed_and_lincheck(&route_witnesses, route_setup.n_blocks_log());
    let route_core = prove_fast_core_with_block_count(
        &route_setup.r1cs, &route_setup.pcs_params,
        rz, ra, rb, rzlc,
        route_setup.r1cs.csc_lincheck_circuit(),
        None, challenger,
    );

    let mut route_pd_claims: Vec<PackedDirectClaim> = Vec::with_capacity(2 * u);
    for (i, input) in unique_nodes.iter().enumerate() {
        let sof = route_sof_f128(&input.route_witness);
        for (slot, &value) in [SOF_PACKED_BASE, SOF_PACKED_BASE + 1].iter().zip(sof.iter()) {
            let point = pd_point(&route_setup, i, *slot);
            let eq_ind = pcs::DirectEqInd::Sparse(pcs::ring_switch::build_eq_sparse(&point));
            route_pd_claims.push(PackedDirectClaim { point, value, eq_ind });
        }
    }

    let mut route_pcs_ch = fork_pcs_challenger(challenger, b"route");
    let route_open = open_core_ligerito(
        &route_setup.r1cs, &route_setup.pcs_params,
        route_core, n_routes, &route_pd_claims, &mut route_pcs_ch,
    );
    eprintln!("[mem] after route proof: {:.0} MB", vmrss_mb());

    SoundMultiproof {
        merkle_zc: merkle_open.zc_proof,
        merkle_lc: merkle_open.lc_proof,
        merkle_pcs: merkle_open.pcs_open,
        merkle_commitment: merkle_open.commitment,

        chain_zc: chain_open.zc_proof,
        chain_lc: chain_open.lc_proof,
        chain_pcs: chain_open.pcs_open,
        chain_commitment: chain_open.commitment,

        merkle_shifts,
        merkle_leaves: (0..u).map(|i| merkle_data[i].2).collect(),
        merkle_roots: (0..u).map(|i| merkle_data[i].3).collect(),
        merkle_native_roots: (0..u).map(|i| merkle_data[i].4).collect(),
        merkle_b_bits: (0..u).map(|i| merkle_data[i].1.clone()).collect(),

        chain_shifts,
        chain_content_hashes: (0..u).map(|i| chain_data[i].1).collect(),
        chain_cv_lasts: (0..u).map(|i| chain_data[i].2).collect(),
        chain_n_compressions: chain_block_counts.clone(),
        chain_n_real: (0..u).map(|i| chain_data[i].3).collect(),

        merkle_block_offsets: merkle_alloc.offsets,
        merkle_block_counts,
        chain_block_offsets: chain_alloc.offsets,
        n_log_merkle: merkle_alloc.n_log,
        n_log_chain: chain_alloc.n_log,

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
    let u = proof.merkle_shifts.len();
    if u == 0 || u != proof.chain_shifts.len() {
        return Err(MhotMembershipError::RootMismatch {
            expected: *expected_root,
            actual: [0; 8],
        });
    }

    // -- Verify merkle core --
    let merkle_n_total = 1usize << proof.n_log_merkle;
    let merkle_setup = Sha256HybridSetup::cached(merkle_n_total);
    let (merkle_ab, merkle_c) = flock_core::verifier::verify_core(
        &merkle_setup.r1cs,
        &proof.merkle_zc, &proof.merkle_lc, &proof.merkle_commitment,
        merkle_setup.r1cs.csc_lincheck_circuit(),
        challenger,
    ).map_err(MhotMembershipError::NodeVerify2)?;

    // -- Merkle shifts --
    let merkle_tau_pos = challenger.sample_f128_vec(MERKLE_LAYOUT.tau_pos_len());
    let merkle_fold = MerklePathFold::new(&MERKLE_LAYOUT, merkle_tau_pos);
    let mut merkle_pd_refs_data: Vec<(Vec<F128>, F128)> = Vec::with_capacity(u);

    for i in 0..u {
        let n_inst = proof.merkle_block_counts[i];
        let inst_log = n_inst.trailing_zeros() as usize;
        let leaf_phys = crate::r1cs_hashes::sha2::hash_to_phys_bits(&proof.merkle_leaves[i]);
        let leaf_r = merkle_fold.fold_public_phys(&leaf_phys);
        let root_phys = crate::r1cs_hashes::sha2::hash_to_phys_bits(&proof.merkle_roots[i]);
        let root_r = merkle_fold.fold_public_phys(&root_phys);

        let mut b_bits_padded = proof.merkle_b_bits[i].clone();
        b_bits_padded.resize(n_inst, false);

        let claims = verify_merkle_path_shift(
            0, &proof.merkle_shifts[i], &[leaf_r], root_r,
            &b_bits_padded, inst_log, MERKLE_LAYOUT.slot_layout(), challenger,
        ).map_err(|e| MhotMembershipError::NodeVerify2(
            flock_core::verifier::VerifyError::Wiring(format!("merkle shift {i}: {e:?}"))
        ))?;

        let point = build_merkle_claim_point_at_offset(
            &MERKLE_LAYOUT, &merkle_fold, &claims,
            proof.merkle_block_offsets[i], proof.merkle_block_counts[i],
            proof.n_log_merkle,
        );
        merkle_pd_refs_data.push((point, claims.value));
    }

    // -- Merkle PCS verify (forked challenger) --
    let merkle_pd_refs: Vec<PackedDirectClaimRef> = merkle_pd_refs_data
        .iter().map(|(p, v)| PackedDirectClaimRef { point: p, value: *v }).collect();

    let mut merkle_pcs_ch = fork_pcs_challenger(challenger, b"merkle");
    verify_core_opening_ligerito(
        &merkle_setup.r1cs, &merkle_setup.pcs_params,
        &proof.merkle_commitment, &proof.merkle_pcs,
        &merkle_ab, &merkle_c, &merkle_pd_refs, &mut merkle_pcs_ch,
    ).map_err(MhotMembershipError::NodeVerify2)?;

    // -- Verify chain core --
    let chain_n_total = 1usize << proof.n_log_chain;
    let chain_setup = Sha256HybridSetup::cached(chain_n_total);
    let (chain_ab, chain_c) = flock_core::verifier::verify_core(
        &chain_setup.r1cs,
        &proof.chain_zc, &proof.chain_lc, &proof.chain_commitment,
        chain_setup.r1cs.csc_lincheck_circuit(),
        challenger,
    ).map_err(MhotMembershipError::NodeVerify2)?;

    // -- Chain shifts --
    let chain_tau_pos = challenger.sample_f128_vec(CHAIN_LAYOUT.tau_pos_len());
    let chain_fold = ChainFold::new(&CHAIN_LAYOUT, chain_tau_pos);
    let mut chain_pd_refs_data: Vec<(Vec<F128>, F128)> = Vec::with_capacity(u);

    for i in 0..u {
        let n_inst = proof.chain_n_compressions[i];
        let inst_log = n_inst.trailing_zeros() as usize;
        let iv_phys = crate::r1cs_hashes::sha2::cv_to_phys_bits(&SHA256_IV);
        let x0_r = chain_fold.fold_public_phys(&iv_phys);
        let cv_last_phys = crate::r1cs_hashes::sha2::cv_to_phys_bits(&proof.chain_cv_lasts[i]);
        let xlast_r = chain_fold.fold_public_phys(&cv_last_phys);

        let claims = verify_chain_shift(
            &proof.chain_shifts[i], x0_r, xlast_r, inst_log, challenger,
        ).map_err(|e| MhotMembershipError::NodeVerify2(
            flock_core::verifier::VerifyError::Wiring(format!("chain shift {i}: {e:?}"))
        ))?;

        let point = build_chain_claim_point_at_offset(
            &CHAIN_LAYOUT, &chain_fold, &claims,
            proof.chain_block_offsets[i], proof.chain_n_compressions[i],
            proof.n_log_chain,
        );
        chain_pd_refs_data.push((point, claims.value));

        if proof.chain_n_real[i] > n_inst {
            return Err(MhotMembershipError::ContentHashMismatch { node_idx: i });
        }
        let n_pad = n_inst - proof.chain_n_real[i];
        let mut expected_cv = proof.chain_content_hashes[i];
        for _ in 0..n_pad {
            expected_cv = sha256_compress(&expected_cv, &[0u32; 16]);
        }
        if expected_cv != proof.chain_cv_lasts[i] {
            return Err(MhotMembershipError::ContentHashMismatch { node_idx: i });
        }
    }

    // -- Chain PCS verify (forked challenger) --
    let chain_pd_refs: Vec<PackedDirectClaimRef> = chain_pd_refs_data
        .iter().map(|(p, v)| PackedDirectClaimRef { point: p, value: *v }).collect();

    let mut chain_pcs_ch = fork_pcs_challenger(challenger, b"chain");
    verify_core_opening_ligerito(
        &chain_setup.r1cs, &chain_setup.pcs_params,
        &proof.chain_commitment, &proof.chain_pcs,
        &chain_ab, &chain_c, &chain_pd_refs, &mut chain_pcs_ch,
    ).map_err(MhotMembershipError::NodeVerify2)?;

    // -- Cross-node binding (per path) --
    for (p, indices) in proof.path_mapping.node_indices.iter().enumerate() {
        if p >= proof.path_depths.len() || indices.len() != proof.path_depths[p] {
            return Err(MhotMembershipError::RootMismatch {
                expected: *expected_root, actual: [0; 8],
            });
        }
        for j in 0..indices.len().saturating_sub(1) {
            let parent_u = indices[j];
            let child_u = indices[j + 1];
            if parent_u >= u || child_u >= u {
                return Err(MhotMembershipError::RootMismatch {
                    expected: *expected_root, actual: [0; 8],
                });
            }
            let parent_leaf = proof.merkle_leaves[parent_u];
            let child_content = proof.chain_content_hashes[child_u];
            if parent_leaf != child_content {
                return Err(MhotMembershipError::CrossNodeBinding {
                    parent_idx: parent_u,
                    parent_leaf,
                    child_root: child_content,
                });
            }
        }
    }

    // -- Root check --
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
        if proof.chain_content_hashes[root_u] != *expected_root {
            return Err(MhotMembershipError::RootMismatch {
                expected: *expected_root,
                actual: proof.chain_content_hashes[root_u],
            });
        }
    }

    // -- Route verify core --
    let route_setup = RouteF32Setup::cached(proof.n_routes);
    let (route_ab, route_c) = flock_core::verifier::verify_core(
        &route_setup.r1cs,
        &proof.route_zc, &proof.route_lc, &proof.route_commitment,
        route_setup.r1cs.csc_lincheck_circuit(),
        challenger,
    ).map_err(MhotMembershipError::RouteVerify)?;

    // -- Route PCS verify (forked challenger) --
    let mut route_pd_data: Vec<(Vec<F128>, F128)> = Vec::with_capacity(2 * u);
    for (i, leaf) in proof.merkle_leaves.iter().enumerate() {
        let sof = digest_to_sof_f128(&leaf_words_to_digest_bytes(leaf));
        route_pd_data.push((pd_point(&route_setup, i, SOF_PACKED_BASE), sof[0]));
        route_pd_data.push((pd_point(&route_setup, i, SOF_PACKED_BASE + 1), sof[1]));
    }
    let route_pd_refs: Vec<PackedDirectClaimRef> = route_pd_data
        .iter().map(|(p, v)| PackedDirectClaimRef { point: p, value: *v }).collect();

    let mut route_pcs_ch = fork_pcs_challenger(challenger, b"route");
    verify_core_opening_ligerito(
        &route_setup.r1cs, &route_setup.pcs_params,
        &proof.route_commitment, &proof.route_pcs,
        &route_ab, &route_c, &route_pd_refs, &mut route_pcs_ch,
    ).map_err(MhotMembershipError::RouteOpening)?;

    Ok(())
}
