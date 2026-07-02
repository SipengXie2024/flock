//! Proving path for the sound MHOT multiproof (split from `mod.rs`).

use std::collections::HashMap;

fn vmrss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("VmRSS:"))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<f64>().ok()))
        .unwrap_or(0.0) / 1024.0
}

pub(crate) fn bytes_to_words(b: &[u8; 32]) -> [u32; 8] {
    let mut w = [0u32; 8];
    for i in 0..8 {
        w[i] = u32::from_be_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
    }
    w
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

use crate::merkle_path::prove_merkle_path_shift;
use crate::prover::prove_fast_core_with_block_count;
use crate::r1cs_hashes::merkle_path_common::{
    MerklePathFold, assemble_merkle_path_claim_at_offset,
    fold_all_slots_range,
};
use crate::r1cs_hashes::sha2::{
    Compression, MERKLE_LAYOUT, SHA256_IV, Sha256HybridSetup,
    generate_witness_with_ab_packed_and_lincheck, min_n_blocks_log,
};
use flock_core::challenger::{Challenger, FsChallenger};
use flock_core::pcs::{
    self, PackedDirectClaim,
};

use crate::mhot::merkle_membership::{
    MhotMembershipInput, SOF_PACKED_BASE, digest_to_slot_f128,
    merkle_slot_pd_point, pad_to_needed, pd_point, route_sof_f128,
};
use crate::mhot::multiproof::{fork_pcs_challenger, open_core_ligerito};
use crate::mhot::native_witness::mhot_node_to_sha256_merkle;
use crate::mhot::route_f32::{self as route, RouteF32Setup};

use super::{PathMapping, SoundMultiproof};

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

    // -- Per-node: compute in-node merkle compressions and block counts --
    let mut merkle_data: Vec<(Vec<Compression>, Vec<bool>, [u32; 8], [u32; 8], [u32; 8])> =
        Vec::with_capacity(u);
    let mut merkle_block_counts: Vec<usize> = Vec::with_capacity(u);
    let mut merkle_siblings: Vec<Vec<[u32; 8]>> = Vec::with_capacity(u);
    let mut merkle_sib_slots: Vec<Vec<usize>> = Vec::with_capacity(u);

    for (_node_idx, input) in unique_nodes.iter().enumerate() {
        let w = mhot_node_to_sha256_merkle(&input.node);
        let n_real_merkle = w.compressions.len();
        let mut compressions = w.compressions;
        let mut b_bits = w.b_bits.clone();
        let needed = 1usize << min_n_blocks_log(n_real_merkle);
        let padded_root = pad_to_needed(&mut compressions, &mut b_bits, needed);

        // Per-level siblings (values + committed slots) for the verifier's
        // native_root recompute. Only the real depth carries siblings; the
        // padding blocks are zero-sibling.
        let selected = input.node.selected_child;
        let mut node_sibs: Vec<[u32; 8]> = Vec::with_capacity(n_real_merkle);
        let mut node_slots: Vec<usize> = Vec::with_capacity(n_real_merkle);
        for d in 0..n_real_merkle {
            let m = &compressions[d].1;
            let real_side = (selected >> d) & 1 == 1;
            // Sibling is in X_R (m[8..16]) at d=0 (leaf forced left) and whenever
            // the real side is left; in X_L (m[0..8]) when the real side is right.
            let (src, slot): (&[u32], usize) = if d == 0 || !real_side {
                (&m[8..16], MERKLE_LAYOUT.x_r_slot as usize)
            } else {
                (&m[0..8], MERKLE_LAYOUT.x_l_slot as usize)
            };
            let mut sib = [0u32; 8];
            sib.copy_from_slice(src);
            node_sibs.push(sib);
            node_slots.push(slot);
        }
        merkle_siblings.push(node_sibs);
        merkle_sib_slots.push(node_slots);

        merkle_data.push((compressions, b_bits, w.leaf, padded_root, w.native_root));
        merkle_block_counts.push(needed);
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

    // Setup before witness: the R1CS/params build transients must not stack
    // on top of the four live witness buffers (cached() is pure — no
    // challenger interaction, so the swap is transcript-neutral).
    let merkle_setup = Sha256HybridSetup::cached(merkle_alloc.n_total);
    let (mz, ma, mb, mzlc) =
        generate_witness_with_ab_packed_and_lincheck(&merkle_comps, merkle_alloc.n_log);
    drop(merkle_comps);
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

        // Sibling PD claims: pin each committed sibling slot to its claimed value
        // so the verifier's native_root recompute uses authenticated siblings.
        let block_base = merkle_alloc.offsets[i];
        for (d, (sib, &slot)) in
            merkle_siblings[i].iter().zip(merkle_sib_slots[i].iter()).enumerate()
        {
            let vals = digest_to_slot_f128(sib);
            for within in 0..2 {
                let point =
                    merkle_slot_pd_point(merkle_setup.r1cs.m, block_base + d, slot, within);
                let eq_ind = pcs::DirectEqInd::Sparse(pcs::ring_switch::build_eq_sparse(&point));
                merkle_pd_claims.push(PackedDirectClaim { point, value: vals[within], eq_ind });
            }
        }
    }

    let mut merkle_pcs_ch = fork_pcs_challenger(challenger, b"merkle");
    let merkle_open = open_core_ligerito(
        &merkle_setup.r1cs, &merkle_setup.pcs_params,
        merkle_core, merkle_alloc.n_real, &merkle_pd_claims, &mut merkle_pcs_ch,
    );
    eprintln!("[mem] after merkle PCS open: {:.0} MB", vmrss_mb());

    // -- Pass 2: Route base (content_hash is verifier-recomputed, no chain SNARK) --
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

        merkle_shifts,
        merkle_leaves: (0..u).map(|i| merkle_data[i].2).collect(),
        merkle_roots: (0..u).map(|i| merkle_data[i].3).collect(),
        merkle_b_bits: (0..u).map(|i| merkle_data[i].1.clone()).collect(),
        merkle_siblings,
        merkle_leaf_is_right: (0..u)
            .map(|i| unique_nodes[i].node.selected_child & 1 == 1)
            .collect(),
        content_metas: (0..u).map(|i| unique_nodes[i].content.clone()).collect(),

        merkle_block_offsets: merkle_alloc.offsets,
        merkle_block_counts,
        n_log_merkle: merkle_alloc.n_log,

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

