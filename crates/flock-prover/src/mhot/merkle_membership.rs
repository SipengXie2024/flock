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
use flock_core::verifier::VerifyError;
use flock_core::zerocheck::ZerocheckProof;

use super::multiproof::{open_core_ligerito, verify_core_opening_ligerito};
use super::native_witness::{MhotNodeWitness, mhot_node_to_sha256_merkle};
use super::route_f32::{self as route, RouteF32Setup, RouteF32Witness};

#[derive(Debug)]
pub enum MhotMembershipError {
    NodeVerify(MerklePathVerifyError),
    NodeVerify2(VerifyError),
    CrossNodeBinding {
        parent_idx: usize,
        parent_leaf: [u32; 8],
        child_root: [u32; 8],
    },
    RouteVerify(VerifyError),
    RouteOpening(VerifyError),
    ContentChainVerify(usize, ChainVerifyError),
    ContentHashMismatch { node_idx: usize },
    RootMismatch {
        expected: [u32; 8],
        actual: [u32; 8],
    },
    /// Number of public entries does not match the number of proven paths.
    EntryCountMismatch { n_entries: usize, n_paths: usize },
    /// The public key does not route to the authenticated child position at
    /// some node on its path (native HOT routing re-run by the verifier).
    RoutingMismatch {
        path_idx: usize,
        level: usize,
        matched: usize,
        selected: usize,
    },
    /// The path's terminal authenticated leaf is not the hash of the public
    /// (key, value) entry.
    EntryLeafMismatch { path_idx: usize },
}

/// Proof for a single MHOT node's in-node binary Merkle path.
#[derive(serde::Serialize)]
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

    let setup = Sha256HybridSetup::cached(needed);
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
    let setup = Sha256HybridSetup::cached(needed);
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
pub(crate) fn pad_to_needed(
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
// Content hash chain: SHA-256(masks ‖ keys ‖ merkle_root ‖ counts) proved as
// a sequential chain via chain_common. Authenticates node identity.
// ---------------------------------------------------------------------------

use crate::r1cs_hashes::sha2::ChainVerifyError;
use crate::r1cs_hashes::chain_common::ChainProofLigerito;

/// Content metadata needed to compute the native content_hash for one node.
#[derive(Clone, Debug, serde::Serialize)]
pub struct ContentMeta {
    pub extraction_masks: [u64; 4],
    pub sparse_partial_keys: Vec<u32>,
    pub child_leaf_counts: Vec<u32>,
}

/// Build the SHA-256 Merkle-Damgard chain for content_hash.
///
/// Returns `(compressions, content_hash, cv_last, n_real)`.
pub fn build_content_hash_chain(
    meta: &ContentMeta,
    merkle_root: &[u8; 32],
) -> (Vec<Compression>, [u32; 8], [u32; 8], usize) {
    let mut data = Vec::new();
    for &mask in &meta.extraction_masks {
        data.extend_from_slice(&mask.to_le_bytes());
    }
    for &key in &meta.sparse_partial_keys {
        data.extend_from_slice(&key.to_le_bytes());
    }
    data.extend_from_slice(merkle_root);
    for &count in &meta.child_leaf_counts {
        data.extend_from_slice(&count.to_le_bytes());
    }

    let data_len_bits = (data.len() as u64) * 8;
    data.push(0x80);
    while data.len() % 64 != 56 {
        data.push(0);
    }
    data.extend_from_slice(&data_len_bits.to_be_bytes());
    assert_eq!(data.len() % 64, 0);

    let n_real = data.len() / 64;
    let n_padded = n_real.max(8).next_power_of_two();

    let mut compressions = Vec::with_capacity(n_padded);
    let mut cv = SHA256_IV;
    for block in data.chunks_exact(64) {
        let mut m = [0u32; 16];
        for (i, chunk) in block.chunks_exact(4).enumerate() {
            m[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        compressions.push((cv, m));
        cv = sha256_compress(&cv, &m);
    }
    let content_hash = cv;

    while compressions.len() < n_padded {
        let m = [0u32; 16];
        compressions.push((cv, m));
        cv = sha256_compress(&cv, &m);
    }

    (compressions, content_hash, cv, n_real)
}

/// Compute content_hash natively (for verifier-side checks). Matches
/// `mhot-verify/src/proof.rs::compute_node_content_hash`.
pub fn compute_content_hash(meta: &ContentMeta, merkle_root: &[u8; 32]) -> [u8; 32] {
    let (_, cv, _, _) = build_content_hash_chain(meta, merkle_root);
    let mut out = [0u8; 32];
    for i in 0..8 {
        out[4 * i..4 * i + 4].copy_from_slice(&cv[i].to_be_bytes());
    }
    out
}

pub use super::entry_binding::{
    PathEntry, compute_dense_key, leaf_content_hash, search_in_sparse_keys,
};

// ---------------------------------------------------------------------------
// Route↔hash binding: PackedDirectClaims open the route commitment's
// SELECTED_OUT_FINAL and assert it equals the hash side's authenticated leaf.
// ---------------------------------------------------------------------------

/// Within-block packed index of SELECTED_OUT_FINAL's first F128. It is
/// DIGEST_BITS-aligned, so it spans two consecutive F128 slots:
/// SOF_PACKED_BASE and SOF_PACKED_BASE + 1.
pub(crate) const SOF_PACKED_BASE: usize = route::SELECTED_OUT_FINAL_BASE / 128;
pub(crate) const BLOCK_PACKED: usize = route::K / 128;

/// One membership step: the MHOT node, route witness, and content metadata.
#[derive(Clone)]
pub struct MhotMembershipInput {
    pub node: MhotNodeWitness,
    pub route_witness: RouteF32Witness,
    pub content: ContentMeta,
}

impl MhotMembershipInput {
    /// Build an input with synthetic content metadata (for tests).
    pub fn from_node(node: MhotNodeWitness) -> Self {
        let route_witness = mhot_node_to_route_witness(&node);
        let nc = node.children.len();
        let content = ContentMeta {
            extraction_masks: [0x1F; 4],
            sparse_partial_keys: vec![0; nc],
            child_leaf_counts: vec![1; nc],
        };
        Self { node, route_witness, content }
    }
}

/// Per-node content hash chain proof.
#[derive(serde::Serialize)]
pub struct ContentChainProof {
    pub proof: ChainProofLigerito,
    pub commitment: Commitment,
    pub content_hash: [u32; 8],
    pub cv_last: [u32; 8],
    pub n_compressions: usize,
    pub n_real_compressions: usize,
}

/// A sound membership proof for one MHOT path: per-node SHA-256 Merkle path
/// proofs (hash base) + per-node content hash chain proofs + one batched
/// route R1CS proof (route base), bound by PackedDirectClaims.
pub struct MhotMembershipProof {
    pub hash_proofs: Vec<NodeMerkleProof>,
    pub content_proofs: Vec<ContentChainProof>,
    pub route_zc: ZerocheckProof,
    pub route_lc: LincheckProof,
    pub route_pcs: BatchOpeningProofLigerito,
    pub route_commitment: Commitment,
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
pub(crate) fn digest_bytes_to_route_bits(d: &[u8; 32]) -> [bool; route::DIGEST_BITS] {
    let mut bits = [false; route::DIGEST_BITS];
    for (byte_i, &byte) in d.iter().enumerate() {
        for k in 0..8 {
            bits[byte_i * 8 + k] = (byte >> k) & 1 == 1;
        }
    }
    bits
}

/// Recover the `[u8; 32]` digest from a SHA-256 leaf (big-endian words).
pub(crate) fn leaf_words_to_digest_bytes(leaf: &[u32; 8]) -> [u8; 32] {
    let mut d = [0u8; 32];
    for i in 0..8 {
        d[4 * i..4 * i + 4].copy_from_slice(&leaf[i].to_be_bytes());
    }
    d
}

/// Pack up to 128 bools into one F128 (lo = bits 0..64, hi = bits 64..128).
pub(crate) fn pack_bits_to_f128(bits: &[bool]) -> F128 {
    assert!(bits.len() <= 128, "pack_bits_to_f128: slice length {} > 128", bits.len());
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
pub(crate) fn digest_to_sof_f128(d: &[u8; 32]) -> [F128; 2] {
    let bits = digest_bytes_to_route_bits(d);
    [
        pack_bits_to_f128(&bits[0..128]),
        pack_bits_to_f128(&bits[128..256]),
    ]
}

/// The two SELECTED_OUT_FINAL F128 values the route R1CS produces for this
/// witness: the digest of the child whose 5-bit index equals the extracted
/// key bits. Matches what is committed in the route z_packed.
pub(crate) fn route_sof_f128(rw: &RouteF32Witness) -> [F128; 2] {
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

pub(crate) fn fork_content_challenger(parent: &FsChallenger, node_idx: usize) -> FsChallenger {
    let mut ch = parent.clone();
    ch.observe_label(b"mhot-content-chain-fork-v0");
    ch.observe_bytes(&(node_idx as u64).to_le_bytes());
    ch
}

/// PackedDirectClaim point selecting route instance `instance`'s F128 at
/// within-block packed index `within`: the LSB-first binary expansion of the
/// global packed index over `L = m − LOG_PACKING` coords.
pub(crate) fn pd_point(setup: &RouteF32Setup, instance: usize, within: usize) -> Vec<F128> {
    let gpi = instance * BLOCK_PACKED + within;
    let l = setup.r1cs.m - pcs::LOG_PACKING;
    (0..l)
        .map(|k| if (gpi >> k) & 1 == 1 { F128::ONE } else { F128::ZERO })
        .collect()
}

/// Recompute the real in-node binary Merkle tree root from the selected leaf,
/// the per-level siblings, and the per-level REAL side bits (true = leaf/current
/// is the right child at that level). Mirrors `native_witness`'s tree order; used
/// by the verifier to bind the committed merkle tree to content_hash's root.
pub(crate) fn recompute_native_root(
    leaf: &[u32; 8],
    siblings: &[[u32; 8]],
    sides: &[bool],
) -> [u32; 8] {
    assert_eq!(siblings.len(), sides.len(), "siblings/sides length mismatch");
    let mut current = *leaf;
    for (sibling, &is_right) in siblings.iter().zip(sides.iter()) {
        let mut m = [0u32; 16];
        if is_right {
            m[..8].copy_from_slice(sibling);
            m[8..].copy_from_slice(&current);
        } else {
            m[..8].copy_from_slice(&current);
            m[8..].copy_from_slice(sibling);
        }
        current = crate::r1cs_hashes::sha2::sha256_compress(
            &crate::r1cs_hashes::sha2::SHA256_IV,
            &m,
        );
    }
    current
}

/// PackedDirectClaim point selecting the F128 at `within ∈ {0,1}` of `slot`'s
/// 256-bit region in merkle compression block `block_index`, over
/// `L = m − LOG_PACKING` coords. A 256-bit slot occupies 2 packed F128, so slot
/// `s` has packed base `s·2`; a block occupies `2^(K_LOG−LOG_PACKING)` packed.
pub(crate) fn merkle_slot_pd_point(
    m_merkle: usize,
    block_index: usize,
    slot: usize,
    within: usize,
) -> Vec<F128> {
    let block_packed = 1usize << (crate::r1cs_hashes::sha2::K_LOG - pcs::LOG_PACKING);
    let gpi = block_index * block_packed + slot * 2 + within;
    let l = m_merkle - pcs::LOG_PACKING;
    (0..l)
        .map(|k| if (gpi >> k) & 1 == 1 { F128::ONE } else { F128::ZERO })
        .collect()
}

/// The two F128 values a child digest occupies in a merkle message slot
/// (word-major `cv_to_phys_bits` order, matching how message words are committed).
pub(crate) fn digest_to_slot_f128(d: &[u32; 8]) -> [F128; 2] {
    let bits = crate::r1cs_hashes::sha2::cv_to_phys_bits(d);
    [pack_bits_to_f128(&bits[0..128]), pack_bits_to_f128(&bits[128..256])]
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

    // ---- Content hash chain: per-node SHA-256(masks ‖ keys ‖ merkle_root ‖ counts).
    // Each chain uses a forked challenger (observe commitment in main to bind),
    // so the chain's internal challenges don't shift the main challenger state.
    let content_proofs: Vec<ContentChainProof> = path
        .iter()
        .zip(hash_proofs.iter())
        .enumerate()
        .map(|(idx, (input, hp))| {
            let merkle_root_bytes = leaf_words_to_digest_bytes(&hp.native_root);
            let (mut compressions, content_hash, cv_last, n_real) =
                build_content_hash_chain(&input.content, &merkle_root_bytes);
            let min_standalone = 1usize << (22 - crate::r1cs_hashes::sha2::K_LOG);
            let n = compressions.len().max(min_standalone).next_power_of_two();
            let mut cv = cv_last;
            while compressions.len() < n {
                let m = [0u32; 16];
                compressions.push((cv, m));
                cv = sha256_compress(&cv, &m);
            }
            let cv_last_extended = cv;
            let setup = Sha256HybridSetup::cached(n);
            let mut chain_ch = fork_content_challenger(challenger, idx);
            let (proof, commitment) = setup.prove_chain(&compressions, &mut chain_ch);
            challenger.observe_bytes(&commitment.root);
            ContentChainProof {
                proof, commitment, content_hash, cv_last: cv_last_extended,
                n_compressions: n, n_real_compressions: n_real,
            }
        })
        .collect();

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
        content_proofs,
        route_zc: route_open.zc_proof,
        route_lc: route_open.lc_proof,
        route_pcs: route_open.pcs_open,
        route_commitment: route_open.commitment,
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
    assert_eq!(
        proof.hash_proofs.len(),
        proof.content_proofs.len(),
        "hash_proofs and content_proofs must have the same length"
    );

    // ---- Hash base: verify each node's in-node Merkle path.
    for hp in &proof.hash_proofs {
        verify_node_merkle(hp, challenger).map_err(MhotMembershipError::NodeVerify)?;
    }

    // ---- Content hash chain: verify each node's chain (forked challenger).
    // After chain verify (which authenticates cv_last), check that applying
    // the deterministic padding from content_hash reproduces cv_last. This
    // authenticates content_hash via SHA-256 collision resistance.
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

    // Cross-node binding: parent's hash leaf == child's content_hash.
    // content_hash is authenticated by the chain proof (SHA256_IV → cv_last
    // passes through content_hash; chain shift-sumcheck guarantees it).
    for i in 0..proof.hash_proofs.len().saturating_sub(1) {
        let parent_leaf = proof.hash_proofs[i].leaf;
        let child_content_hash = proof.content_proofs[i + 1].content_hash;
        if parent_leaf != child_content_hash {
            return Err(MhotMembershipError::CrossNodeBinding {
                parent_idx: i,
                parent_leaf: parent_leaf,
                child_root: child_content_hash,
            });
        }
    }

    // ---- Public root = the top node's content_hash (native node identity).
    if proof.content_proofs[0].content_hash != *expected_root {
        return Err(MhotMembershipError::RootMismatch {
            expected: *expected_root,
            actual: proof.content_proofs[0].content_hash,
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
