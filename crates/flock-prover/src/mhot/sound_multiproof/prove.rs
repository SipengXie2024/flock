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

/// The canonical deterministic dummy node filling padding slots `[u, P)` so the
/// global shift runs over the full `2^m`-node cube. Both prover and verifier
/// build the IDENTICAL chain (a fanout-2 node with zero-digest children,
/// native-order, padded to 8 blocks), so padding cannot be a prover-chosen
/// cancellation reservoir. Returns `(compressions[8], b_bits[8], leaf digest,
/// padded-root digest)`.
pub(crate) fn dummy_padding_chain() -> (Vec<Compression>, Vec<bool>, [u32; 8], [u32; 8]) {
    let node = crate::mhot::native_witness::MhotNodeWitness {
        children: vec![[0u8; 32], [0u8; 32]],
        selected_child: 0,
    };
    let w = mhot_node_to_sha256_merkle(&node, true);
    let mut compressions = w.compressions;
    let mut b_bits = w.b_bits;
    let padded_root = pad_to_needed(&mut compressions, &mut b_bits, super::MERKLE_BLOCKS_PER_NODE);
    (compressions, b_bits, w.leaf, padded_root)
}

pub(crate) fn bytes_to_words(b: &[u8; 32]) -> [u32; 8] {
    let mut w = [0u32; 8];
    for i in 0..8 {
        w[i] = u32::from_be_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
    }
    w
}

/// Identity of the PHYSICAL node: children + content metadata, WITHOUT the
/// selected child. Two queries that traverse the same tree node but diverge
/// to different children share one physical identity (and hence one wire
/// copy of the tree-determined data: content_meta, native/padded root,
/// block count) while keeping distinct per-pair data (leaf, b_bits, shift).
fn phys_identity(input: &MhotMembershipInput) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(input.node.children.len() as u32).to_le_bytes());
    for child in &input.node.children {
        bytes.extend_from_slice(child);
    }
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

/// Identity of a (physical node, selected child) PAIR — the unit the shift
/// proofs, leaves, b_bits and route witnesses are keyed by.
fn pair_identity(input: &MhotMembershipInput) -> Vec<u8> {
    let mut bytes = phys_identity(input);
    bytes.extend_from_slice(&(input.node.selected_child as u32).to_le_bytes());
    bytes
}

use crate::merkle_path::prove_merkle_path_shift;
use crate::prover::prove_fast_core_with_block_count;
use crate::r1cs_hashes::merkle_path_common::{
    MerklePathFold, assemble_merkle_path_claim, fold_all_slots,
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
    ContentMeta, MhotMembershipInput, SOF_PACKED_BASE, pad_to_needed, pd_point, route_sof_f128,
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

/// Uniform-8 block allocation (E2): every node's chain is exactly 8 blocks
/// (see `MERKLE_BLOCKS_PER_NODE`), so `off_i = i·8` and the committed cube is
/// `P × 8` blocks where `P = n_total/8` is a power of two. Real nodes fill
/// `[0, u)`; `[u, P)` are deterministic dummy padding nodes.
fn allocate_blocks_uniform(u: usize) -> AlignedAllocation {
    let block = super::MERKLE_BLOCKS_PER_NODE;
    let offsets: Vec<usize> = (0..u).map(|i| i * block).collect();
    let min_n = 1usize << (22 - crate::r1cs_hashes::sha2::K_LOG);
    let n_total = (u * block).max(min_n).next_power_of_two();
    let n_log = n_total.trailing_zeros() as usize;
    // Padding slots [u, P) are filled with VALID dummy chains (not zero rows),
    // so ALL n_total blocks are real compressions for the zerocheck — n_real =
    // n_total (no zero-padding region). The global shift needs every node.
    AlignedAllocation { offsets, n_real: n_total, n_total, n_log }
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

    // -- Two-level dedup: pairs (node, selected-child) and physical nodes --
    let mut pair_map: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut phys_map: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut unique_nodes: Vec<MhotMembershipInput> = Vec::new();
    let mut pair_phys: Vec<usize> = Vec::new();
    let mut n_phys = 0usize;
    let mut node_indices_per_path: Vec<Vec<usize>> = Vec::with_capacity(paths.len());
    let mut path_depths: Vec<usize> = Vec::with_capacity(paths.len());

    for path in paths {
        let mut indices = Vec::with_capacity(path.len());
        for input in path {
            let key = pair_identity(input);
            let idx = *pair_map.entry(key).or_insert_with(|| {
                let p = *phys_map.entry(phys_identity(input)).or_insert_with(|| {
                    let p = n_phys;
                    n_phys += 1;
                    p
                });
                let i = unique_nodes.len();
                unique_nodes.push(input.clone());
                pair_phys.push(p);
                i
            });
            indices.push(idx);
        }
        path_depths.push(path.len());
        node_indices_per_path.push(indices);
    }

    let u = unique_nodes.len();
    eprintln!(
        "[mem] after dedup ({} pairs, {} physical): {:.0} MB",
        u, n_phys, vmrss_mb()
    );

    // -- Per-node: compute in-node merkle compressions and block counts --
    // native_order=true: the chain IS the true in-node tree path, so the shift
    // authenticates the padded native root directly (no siblings needed for the
    // verifier — it binds native_root via the pad-forward check instead).
    let mut merkle_data: Vec<(Vec<Compression>, Vec<bool>, [u32; 8], [u32; 8], [u32; 8])> =
        Vec::with_capacity(u);
    let mut merkle_block_counts: Vec<usize> = Vec::with_capacity(u);

    for input in unique_nodes.iter() {
        let w = mhot_node_to_sha256_merkle(&input.node, true);
        let n_real_merkle = w.compressions.len();
        let mut compressions = w.compressions;
        let mut b_bits = w.b_bits.clone();
        let needed = 1usize << min_n_blocks_log(n_real_merkle);
        let padded_root = pad_to_needed(&mut compressions, &mut b_bits, needed);

        merkle_data.push((compressions, b_bits, w.leaf, padded_root, w.native_root));
        merkle_block_counts.push(needed);
    }

    // Uniform-8 invariant (E2 layout): depth ≤ 5 (fanout ≤ 32) floored to 8 by
    // min_n_blocks_log, so every node's chain is exactly 8 blocks. The global
    // sumcheck's `off_i = i·8` node×8×slot layout depends on this.
    assert!(
        merkle_block_counts
            .iter()
            .all(|&c| c == super::MERKLE_BLOCKS_PER_NODE),
        "non-uniform merkle block counts break the E2 uniform-8 layout"
    );

    // Tree-determined (physical) fields, once per physical node. Computed here
    // (not at the end) so FS Step 0 can absorb them before the shift's η=τ_p.
    let mut phys_roots: Vec<[u32; 8]> = vec![[0; 8]; n_phys];
    let mut phys_native_roots: Vec<[u32; 8]> = vec![[0; 8]; n_phys];
    let mut phys_block_counts: Vec<usize> = vec![0; n_phys];
    let mut phys_metas: Vec<Option<ContentMeta>> = vec![None; n_phys];
    for i in 0..u {
        let p = pair_phys[i];
        if phys_metas[p].is_none() {
            phys_roots[p] = merkle_data[i].3;
            phys_native_roots[p] = merkle_data[i].4;
            phys_block_counts[p] = merkle_block_counts[i];
            phys_metas[p] = Some(unique_nodes[i].content.clone());
        } else {
            debug_assert_eq!(phys_roots[p], merkle_data[i].3);
            debug_assert_eq!(phys_native_roots[p], merkle_data[i].4);
            debug_assert_eq!(phys_block_counts[p], merkle_block_counts[i]);
        }
    }
    let phys_metas: Vec<ContentMeta> = phys_metas
        .into_iter()
        .map(|m| m.expect("every phys referenced"))
        .collect();
    let leaves_vec: Vec<[u32; 8]> = (0..u).map(|i| merkle_data[i].2).collect();
    let b_bits_vec: Vec<Vec<bool>> = (0..u).map(|i| merkle_data[i].1.clone()).collect();

    // -- Pass 1: Merkle commitment --
    let merkle_alloc = allocate_blocks_uniform(u);
    let block = super::MERKLE_BLOCKS_PER_NODE;
    let p_nodes = merkle_alloc.n_total / block; // 2^m, real [0,u) + padding [u,P)
    eprintln!("[mem] merkle alloc (n_log={}, {} blocks, {} nodes): {:.0} MB",
        merkle_alloc.n_log, merkle_alloc.n_total, p_nodes, vmrss_mb());

    // Padding nodes [u, P) get the deterministic dummy chain so the global
    // shift runs over the whole power-of-two node cube (a valid chain, not the
    // build_comp_vec IV-filler which satisfies the R1CS but NOT the shift link).
    let (dummy_comps, dummy_b, _dummy_leaf, _dummy_root) = dummy_padding_chain();
    let mut merkle_comp_data: Vec<(Vec<Compression>, usize)> = (0..u)
        .map(|i| (merkle_data[i].0.clone(), merkle_alloc.offsets[i]))
        .collect();
    for node in u..p_nodes {
        merkle_comp_data.push((dummy_comps.clone(), node * block));
    }
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

    // FS Step 0: bind all public IO the global shift consumes BEFORE η=τ_p.
    // n_routes == u (route_witnesses.len()); mirrors the verifier's absorb.
    super::absorb_public_io(
        challenger,
        &leaves_vec,
        &b_bits_vec,
        &pair_phys,
        &phys_roots,
        merkle_alloc.n_log,
        u,
    );

    let merkle_tau_pos = challenger.sample_f128_vec(MERKLE_LAYOUT.tau_pos_len());
    let merkle_fold = MerklePathFold::new(&MERKLE_LAYOUT, merkle_tau_pos);

    // ONE global shift over the full P-node cube. Fold every block's slots at
    // once (whole witness), then prove_merkle_path_shift(path_log = log2(P)):
    // the node dimension is batched via τ_p = η, each node carrying its own
    // root/leaf. b_bits is N-major (node N's 8 side bits at [N·8, N·8+8)).
    let slots = fold_all_slots(&MERKLE_LAYOUT, &merkle_core.z_packed, &merkle_fold);
    let mut b_bits_full = vec![false; merkle_alloc.n_total];
    for i in 0..u {
        for (y, &bit) in merkle_data[i].1.iter().enumerate() {
            b_bits_full[i * block + y] = bit;
        }
    }
    for node in u..p_nodes {
        for (y, &bit) in dummy_b.iter().enumerate() {
            b_bits_full[node * block + y] = bit;
        }
    }
    let path_log = merkle_alloc.n_log - block.trailing_zeros() as usize; // log2(P)
    let (merkle_shift, shift_claims) = prove_merkle_path_shift(
        path_log,
        &slots[MERKLE_LAYOUT.x_l_slot as usize],
        &slots[MERKLE_LAYOUT.x_r_slot as usize],
        &slots[MERKLE_LAYOUT.z_slot as usize],
        &slots[MERKLE_LAYOUT.other_slot() as usize],
        &b_bits_full,
        MERKLE_LAYOUT.slot_layout(),
        challenger,
    );
    let merkle_pd = assemble_merkle_path_claim(&MERKLE_LAYOUT, &merkle_fold, &shift_claims);

    let mut merkle_pcs_ch = fork_pcs_challenger(challenger, b"merkle");
    let merkle_open = open_core_ligerito(
        &merkle_setup.r1cs, &merkle_setup.pcs_params,
        merkle_core, merkle_alloc.n_real, std::slice::from_ref(&merkle_pd), &mut merkle_pcs_ch,
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

        merkle_shift,
        merkle_leaves: leaves_vec,
        merkle_roots: phys_roots,
        merkle_b_bits: b_bits_vec,
        merkle_native_roots: phys_native_roots,
        content_metas: phys_metas,
        pair_phys,

        merkle_block_counts: phys_block_counts,
        n_log_merkle: merkle_alloc.n_log,

        route_zc: route_open.zc_proof,
        route_lc: route_open.lc_proof,
        route_pcs: route_open.pcs_open,
        route_commitment: route_open.commitment,
        n_routes,

        path_depths,
        path_mapping: PathMapping { node_indices: node_indices_per_path },
    }
}

