use crate::chain::{ChainShiftProof, verify_chain_shift};
use crate::merkle_path::{MerklePathShiftProof, verify_merkle_path_shift};
use crate::r1cs_hashes::chain_common::{
    ChainFold, build_chain_claim_point_at_offset,
};
use crate::r1cs_hashes::merkle_path_common::{
    MerklePathFold, build_merkle_claim_point_at_offset,
};
use crate::r1cs_hashes::sha2::{
    CHAIN_LAYOUT, MERKLE_LAYOUT, SHA256_IV, Sha256HybridSetup, sha256_compress,
};
use flock_core::challenger::{Challenger, FsChallenger};
use flock_core::field::F128;
use flock_core::lincheck::LincheckProof;
use flock_core::pcs::{
    BatchOpeningProofLigerito, Commitment, PackedDirectClaimRef,
};
use flock_core::zerocheck::ZerocheckProof;

use super::merkle_membership::{
    ContentMeta, MhotMembershipError, PathEntry, SOF_PACKED_BASE, compute_content_hash, compute_dense_key, digest_to_slot_f128,
    digest_to_sof_f128, leaf_content_hash, leaf_words_to_digest_bytes, merkle_slot_pd_point, pd_point, recompute_native_root, search_in_sparse_keys,
};
use super::multiproof::{fork_pcs_challenger, verify_core_opening_ligerito};
use super::route_f32::RouteF32Setup;

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
    pub merkle_b_bits: Vec<Vec<bool>>,
    // Per-node authenticated in-node Merkle siblings (real depth), the true
    // depth-0 side bit, and content metadata — the verifier recomputes the
    // committed tree's native_root and binds it to content_hash (soundness).
    pub merkle_siblings: Vec<Vec<[u32; 8]>>,
    pub merkle_leaf_is_right: Vec<bool>,
    pub content_metas: Vec<ContentMeta>,

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


mod prove;
pub use prove::prove_sound_multiproof;
pub(crate) use prove::bytes_to_words;
#[cfg(test)]
use prove::prove_sound_multiproof_impl;

/// In-node binary-Merkle depth is `ceil(log2(fanout))`; fanout is capped at 32
/// (native `dense_key` is `u32`), so depth never exceeds 5. Reject anything far
/// beyond that so the `1 << d` / `1 << bit_pos` shifts below cannot overflow on
/// a malformed proof.
const MAX_INNODE_DEPTH: usize = 32;

/// The true in-node side bit at depth `d` for node `i` (LSB-at-depth-0). Depth 0
/// carries the real leaf side in `merkle_leaf_is_right` (the protocol forces
/// `b_bits[0]=false`); deeper levels use the public `b_bits`. Single source of
/// truth for the side-bit convention — the native_root recompute and the entry
/// routing check both read it, so a convention change (e.g. Route 2) updates
/// both at once.
#[inline]
fn authenticated_side(proof: &SoundMultiproof, i: usize, d: usize) -> bool {
    if d == 0 {
        proof.merkle_leaf_is_right[i]
    } else {
        proof.merkle_b_bits[i].get(d).copied().unwrap_or(false)
    }
}

/// The authenticated selected-child index of node `i`, reconstructed from its
/// side bits (the same bits whose recompute is pinned to `content_hash`).
fn selected_index(proof: &SoundMultiproof, i: usize) -> usize {
    let depth = proof.merkle_siblings[i].len();
    (0..depth).fold(0usize, |acc, d| {
        acc | ((authenticated_side(proof, i, d) as usize) << d)
    })
}

/// Structural verify of a batched MHOT multiproof.
///
/// SOUNDNESS NOTE: this checks only that the proof is a well-formed node chain
/// rooted at `expected_root`. It does NOT bind any specific `(key, value)` as a
/// member — the route witness key is private. For the membership statement
/// ("these entries are members under the root") call
/// [`verify_sound_multiproof_with_entries`], which is the intended public API;
/// this function exists for internal/structural checks and benchmarking.
pub fn verify_sound_multiproof(
    proof: &SoundMultiproof,
    expected_root: &[u32; 8],
    challenger: &mut FsChallenger,
) -> Result<(), MhotMembershipError> {
    let u = proof.merkle_shifts.len();
    // Every per-node Vec is indexed at [0..u) below; a malformed proof with a
    // short Vec must be rejected cleanly, not panic (verifier DoS hardening —
    // load-bearing once SoundMultiproof gains a Deserialize path).
    if u == 0
        || u != proof.chain_shifts.len()
        || u != proof.merkle_siblings.len()
        || u != proof.merkle_leaf_is_right.len()
        || u != proof.content_metas.len()
        || u != proof.merkle_b_bits.len()
        || u != proof.merkle_block_counts.len()
        || u != proof.merkle_leaves.len()
        || u != proof.merkle_roots.len()
        || u != proof.chain_content_hashes.len()
        || u != proof.chain_cv_lasts.len()
        || u != proof.chain_n_compressions.len()
        || u != proof.chain_n_real.len()
        || u != proof.chain_block_offsets.len()
        || u != proof.merkle_block_offsets.len()
    {
        return Err(MhotMembershipError::RootMismatch {
            expected: *expected_root,
            actual: [0; 8],
        });
    }
    for i in 0..u {
        let meta = &proof.content_metas[i];
        let mask_bits: u32 = meta.extraction_masks.iter().map(|m| m.count_ones()).sum();
        if mask_bits > 32
            || meta.sparse_partial_keys.len() != meta.child_leaf_counts.len()
            || meta.sparse_partial_keys.len() > 32
            || proof.merkle_siblings[i].len() > MAX_INNODE_DEPTH
        {
            return Err(MhotMembershipError::RootMismatch {
                expected: *expected_root,
                actual: [0; 8],
            });
        }
    }

    let vt = std::env::var_os("VERIFY_BREAKDOWN").is_some();
    let t0 = std::time::Instant::now();

    // -- Verify merkle core --
    let merkle_n_total = 1usize << proof.n_log_merkle;
    let merkle_setup = Sha256HybridSetup::cached(merkle_n_total);
    let (merkle_ab, merkle_c) = flock_core::verifier::verify_core(
        &merkle_setup.r1cs,
        &proof.merkle_zc, &proof.merkle_lc, &proof.merkle_commitment,
        merkle_setup.r1cs.csc_lincheck_circuit(),
        challenger,
    ).map_err(MhotMembershipError::NodeVerify2)?;

    let t1 = std::time::Instant::now();
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

        // Sibling PD refs (must mirror the prover's per-node claim order). A wrong
        // slot (from a bad b_bit) opens a different committed value → PCS rejects.
        let block_base = proof.merkle_block_offsets[i];
        for (d, sib) in proof.merkle_siblings[i].iter().enumerate() {
            let is_right = proof.merkle_b_bits[i].get(d).copied().unwrap_or(false);
            let slot = if d == 0 || !is_right {
                MERKLE_LAYOUT.x_r_slot as usize
            } else {
                MERKLE_LAYOUT.x_l_slot as usize
            };
            let vals = digest_to_slot_f128(sib);
            for within in 0..2 {
                let point =
                    merkle_slot_pd_point(merkle_setup.r1cs.m, block_base + d, slot, within);
                merkle_pd_refs_data.push((point, vals[within]));
            }
        }
    }

    let t2 = std::time::Instant::now();
    // -- Merkle PCS verify (forked challenger) --
    let merkle_pd_refs: Vec<PackedDirectClaimRef> = merkle_pd_refs_data
        .iter().map(|(p, v)| PackedDirectClaimRef { point: p, value: *v }).collect();

    let mut merkle_pcs_ch = fork_pcs_challenger(challenger, b"merkle");
    verify_core_opening_ligerito(
        &merkle_setup.r1cs, &merkle_setup.pcs_params,
        &proof.merkle_commitment, &proof.merkle_pcs,
        &merkle_ab, &merkle_c, &merkle_pd_refs, &mut merkle_pcs_ch,
    ).map_err(MhotMembershipError::NodeVerify2)?;

    // -- Bind committed merkle tree ↔ content_hash: recompute each node's true
    //    native_root from its now-authenticated siblings and check content_hash
    //    was actually built over that root. Closes the forgery where a fake
    //    in-node tree is authenticated while content_hash commits a different
    //    (real) root — the two were never compared before. --
    for i in 0..u {
        let siblings = &proof.merkle_siblings[i];
        let depth = siblings.len();
        let sides: Vec<bool> = (0..depth).map(|d| authenticated_side(proof, i, d)).collect();
        let recomputed = recompute_native_root(&proof.merkle_leaves[i], siblings, &sides);
        let expected_ch = compute_content_hash(
            &proof.content_metas[i],
            &leaf_words_to_digest_bytes(&recomputed),
        );
        if bytes_to_words(&expected_ch) != proof.chain_content_hashes[i] {
            return Err(MhotMembershipError::NativeRootMismatch { node_idx: i });
        }
    }

    let t3 = std::time::Instant::now();
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

    let t4 = std::time::Instant::now();
    // -- Chain PCS verify (forked challenger) --
    let chain_pd_refs: Vec<PackedDirectClaimRef> = chain_pd_refs_data
        .iter().map(|(p, v)| PackedDirectClaimRef { point: p, value: *v }).collect();

    let mut chain_pcs_ch = fork_pcs_challenger(challenger, b"chain");
    verify_core_opening_ligerito(
        &chain_setup.r1cs, &chain_setup.pcs_params,
        &proof.chain_commitment, &proof.chain_pcs,
        &chain_ab, &chain_c, &chain_pd_refs, &mut chain_pcs_ch,
    ).map_err(MhotMembershipError::NodeVerify2)?;

    let t5 = std::time::Instant::now();
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

    let t6 = std::time::Instant::now();
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

    let t7 = std::time::Instant::now();
    if vt {
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
        eprintln!("[verify] merkle_core={:.1}ms merkle_shifts={:.1}ms merkle_pcs+recompute={:.1}ms chain_core+shifts={:.1}ms chain_pcs={:.1}ms cross+root={:.1}ms route={:.1}ms total={:.1}ms",
            ms(t1-t0), ms(t2-t1), ms(t3-t2), ms(t4-t3), ms(t5-t4), ms(t6-t5), ms(t7-t6), ms(t7-t0));
    }
    Ok(())
}

/// Full-statement verify: the structural checks of [`verify_sound_multiproof`]
/// plus native entry binding — for each path, the public (key, value) must
/// route through every authenticated node (HOT routing re-run by the verifier
/// on the authenticated ContentMetas, matched against the in-node position the
/// sibling recompute walked), the terminal node's selected child must be a leaf,
/// and that authenticated leaf must equal the entry's leaf hash. This makes the
/// proven statement equal to native's: "each (key, value) in `entries` is a
/// member under `expected_root`".
///
/// FIAT-SHAMIR NOTE: `entries` are bound purely by these verifier-side native
/// checks and are intentionally NOT absorbed into the transcript — sound today
/// because every input to the checks (content_metas, b_bits, leaf_is_right,
/// siblings, leaves) is transitively authenticated against `expected_root`. If
/// these checks ever move in-protocol (Route 2 / RLC-batched public claims),
/// `entries` and `path_mapping` MUST be absorbed BEFORE any batching coefficient
/// is sampled, or a Jolt-style unfaithful-claims under-binding reopens forgeries.
pub fn verify_sound_multiproof_with_entries(
    proof: &SoundMultiproof,
    expected_root: &[u32; 8],
    entries: &[PathEntry],
    challenger: &mut FsChallenger,
) -> Result<(), MhotMembershipError> {
    // Cheap precondition before the ~O(U) structural verify.
    let n_paths = proof.path_mapping.node_indices.len();
    if entries.len() != n_paths {
        return Err(MhotMembershipError::EntryCountMismatch {
            n_entries: entries.len(),
            n_paths,
        });
    }

    verify_sound_multiproof(proof, expected_root, challenger)?;

    // Authenticated selected-child index per unique node (depends only on u_i).
    let selected: Vec<usize> = (0..proof.merkle_shifts.len())
        .map(|i| selected_index(proof, i))
        .collect();

    for (p, (indices, entry)) in proof
        .path_mapping
        .node_indices
        .iter()
        .zip(entries.iter())
        .enumerate()
    {
        for (level, &u_i) in indices.iter().enumerate() {
            let meta = &proof.content_metas[u_i];
            let dense = compute_dense_key(&entry.key, &meta.extraction_masks);
            let matched = search_in_sparse_keys(dense, &meta.sparse_partial_keys);
            if matched != selected[u_i] {
                return Err(MhotMembershipError::RoutingMismatch {
                    path_idx: p,
                    level,
                    matched,
                    selected: selected[u_i],
                });
            }
        }

        let last_u = *indices.last().expect("path verified non-empty");
        // The terminal's selected child must be a LEAF, else a truncated path
        // could claim an internal node as the member: leaf and internal content
        // preimages share SHA-256 with no domain tag, so a crafted (key, value)
        // can byte-collide with an internal node's content_hash. A leaf subtree
        // has exactly one leaf; internal children have ≥2. `.get()` also guards
        // the degenerate empty-fanout node (selected has no counts entry).
        if proof.content_metas[last_u].child_leaf_counts.get(selected[last_u]) != Some(&1) {
            return Err(MhotMembershipError::EntryLeafMismatch { path_idx: p });
        }
        let expected_leaf = bytes_to_words(&leaf_content_hash(entry));
        if proof.merkle_leaves[last_u] != expected_leaf {
            return Err(MhotMembershipError::EntryLeafMismatch { path_idx: p });
        }
    }
    Ok(())
}


#[cfg(test)]
mod forgery_tests;
