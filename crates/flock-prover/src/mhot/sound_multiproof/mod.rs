use crate::merkle_path::{MerklePathShiftProof, verify_merkle_path_shift};
use crate::r1cs_hashes::merkle_path_common::{
    MerklePathFold, build_merkle_claim_point,
};
use crate::r1cs_hashes::sha2::{
    MERKLE_LAYOUT, SHA256_IV, Sha256HybridSetup, sha256_compress,
};
use flock_core::challenger::{Challenger, FsChallenger};
use flock_core::field::F128;
use flock_core::lincheck::LincheckProof;
use flock_core::pcs::{
    BatchOpeningProofLigerito, Commitment, PackedDirectClaimRef,
};
use flock_core::zerocheck::ZerocheckProof;

use super::merkle_membership::{
    ContentMeta, MhotMembershipError, PathEntry, SOF_PACKED_BASE, compute_content_hash,
    compute_dense_key, digest_to_sof_f128, leaf_content_hash, leaf_words_to_digest_bytes,
    pd_point, search_in_sparse_keys,
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

    // ONE global shift-sumcheck over the full `node × 8 × slot` cube (E2):
    // `m + 5` rounds batch all u pairs (+ deterministic padding to 2^m nodes)
    // via τ_p = η, replacing the u per-pair shifts (~2.7MB → ~m·48B).
    pub merkle_shift: MerklePathShiftProof,

    // Per-PAIR data (u = number of unique (node, selected-child) pairs):
    // the shift authenticates each pair's selected leaf, so leaf and side bits
    // are keyed by pair. Block offsets are derived (`off_i = i·8`, uniform-8).
    pub merkle_leaves: Vec<[u32; 8]>,
    pub merkle_b_bits: Vec<Vec<bool>>,
    /// pair → physical-node index into the per-physical vectors below. Two
    /// queries traversing the same tree node toward different children share
    /// one physical entry (tree-determined data) but keep distinct pair data.
    pub pair_phys: Vec<usize>,

    // Per-PHYSICAL-node data (tree-determined, independent of the selected
    // child; deduplicating these is the E1 wire shrink).
    /// The PADDED chain root the shift authenticates. Physical: padding
    /// continues from the native root, so it is a function of the tree alone.
    pub merkle_roots: Vec<[u32; 8]>,
    /// Real (pre-padding) in-node tree root. The verifier pad-forwards this
    /// n_pad times, asserts it reaches merkle_roots[p] (which the shifts
    /// authenticate), then computes content_hash natively over it — binding
    /// the committed tree to the parent leaf / public root.
    pub merkle_native_roots: Vec<[u32; 8]>,
    pub content_metas: Vec<ContentMeta>,
    pub merkle_block_counts: Vec<usize>,

    pub n_log_merkle: usize,

    // Route base (unchanged)
    pub route_zc: ZerocheckProof,
    pub route_lc: LincheckProof,
    pub route_pcs: BatchOpeningProofLigerito,
    pub route_commitment: Commitment,
    pub n_routes: usize,

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

/// In-node binary tree depth from the authenticated fanout: `ceil(log2(fanout))`.
/// The fanout ≤ 32 gate bounds this at 5, so the `1 << d` shifts below cannot
/// overflow. NOT derivable from `b_bits.len()` (padded to the block count).
#[inline]
fn innode_depth(meta: &ContentMeta) -> usize {
    meta.sparse_partial_keys.len().next_power_of_two().trailing_zeros() as usize
}

/// Upper bound on a node's merkle chain block count. Honest chains are always
/// padded to 8 (fanout ≤ 32 ⇒ depth ≤ 5 ⇒ `min_n_blocks_log` floors at 8); 64
/// leaves protocol headroom. The bound (with the power-of-two gate) keeps the
/// pad-forward loop and `b_bits` resize bounded and makes the shift's
/// `b_bits.len() == 1 << inst_log` assert unreachable on a malformed proof —
/// verifier DoS hardening, latent while `SoundMultiproof` is Serialize-only.
const MAX_MERKLE_BLOCKS: usize = 64;

/// Every honest node's in-node chain is exactly this many blocks. The shift
/// authenticates the SELECTED LEAF's PATH to the root — `depth` compressions,
/// where `depth = log2(next_pow2(fanout)) ≤ 5` (fanout ≤ 32). `min_n_blocks_log`
/// floors the pad at 8, and depth ≤ 5 < 8, so the count is a uniform 8 for
/// EVERY node regardless of fanout. E2's global sumcheck requires this uniform
/// `node × 8 × slot` layout (`off_i = i·8`); a non-uniform count is rejected.
pub(crate) const MERKLE_BLOCKS_PER_NODE: usize = 8;

/// Absolute cap on `n_log_merkle`, checked BEFORE any `1 << n_log_merkle`
/// (shift-overflow) or `Sha256HybridSetup::cached(1 << n_log_merkle)`
/// (attacker-sized allocation). Honest values: 17 @N=4096, 18 @N=8192,
/// 19 @N=16384 — 20 leaves one doubling of headroom. Defense sits in three
/// layers: this cap, the canonical-recompute gate below (pins n_log to the
/// offsets+counts-derived value), and the verify path not prewarming the
/// prover pool.
const MAX_N_LOG_MERKLE: usize = 20;

/// Cap on a public entry's value length: the verifier hashes each value once
/// (leaf_content_hash), so unbounded values make verify cost attacker-chosen.
/// Native MHOT values are ≤ a few hundred bytes; 1 MiB is generous headroom.
const MAX_ENTRY_VALUE_LEN: usize = 1 << 20;

/// The true in-node side bit at depth `d` for node `i` (LSB-at-depth-0). Every
/// depth including 0 carries the REAL side in the public `b_bits` (native-order
/// chains; the forced-b0 convention is gone). Single source of truth for the
/// side-bit convention — the pad-forward binding and the entry routing check
/// both consume the same bits the merkle shift authenticated.
#[inline]
fn authenticated_side(proof: &SoundMultiproof, i: usize, d: usize) -> bool {
    proof.merkle_b_bits[i].get(d).copied().unwrap_or(false)
}

/// FS Step 0 (E2 global batching): absorb the public IO that defines the global
/// shift's per-node error vector `e_N` BEFORE the shift challenge (τ_p = η) is
/// sampled. The batching's cross-node-cancellation resistance (Schwartz-Zippel,
/// ~m/|F|) needs `e_N` FIXED before η, or a char-2 prover could pick these after
/// seeing η to force `Σ_N eq(η,N)·e_N = 0`.
///
/// `e_N` depends ONLY on the committed trace `G` (fixed by the commitment) and
/// the shift's public inputs: the padded `roots` R(η), `leaves` L(η), side bits
/// `b_bits` B(N,Y), and `pair_phys` (which routes root_N = roots[pair_phys[N]]),
/// plus `n_log_merkle` (the domain size P) and `n_routes`. So ONLY those are
/// absorbed. `native_roots`, `content_metas`, `path_mapping`, and `entries` feed
/// DETERMINISTIC downstream checks (pad-forward, cross-node, routing, terminal
/// leaf) that consume no challenge — they are bound to `expected_root` by those
/// equalities regardless of η, so they stay outside Step 0 (and their tamper
/// tests keep exercising those specific checks). Prover and verifier build
/// byte-identical input; a divergence fails the honest roundtrip immediately.
pub(crate) fn absorb_public_io<Ch: Challenger>(
    ch: &mut Ch,
    leaves: &[[u32; 8]],
    b_bits: &[Vec<bool>],
    pair_phys: &[usize],
    roots: &[[u32; 8]],
    n_log_merkle: usize,
    n_routes: usize,
) {
    let mut buf: Vec<u8> = Vec::new();
    let u64le = |b: &mut Vec<u8>, v: usize| b.extend_from_slice(&(v as u64).to_le_bytes());
    let digest = |b: &mut Vec<u8>, d: &[u32; 8]| {
        for w in d {
            b.extend_from_slice(&w.to_le_bytes());
        }
    };
    u64le(&mut buf, leaves.len());
    for l in leaves {
        digest(&mut buf, l);
    }
    u64le(&mut buf, b_bits.len());
    for bb in b_bits {
        u64le(&mut buf, bb.len());
        for &bit in bb {
            buf.push(bit as u8);
        }
    }
    u64le(&mut buf, pair_phys.len());
    for &p in pair_phys {
        u64le(&mut buf, p);
    }
    u64le(&mut buf, roots.len());
    for r in roots {
        digest(&mut buf, r);
    }
    u64le(&mut buf, n_log_merkle);
    u64le(&mut buf, n_routes);
    ch.observe_label(b"mhot-sound-public-io-v0");
    ch.observe_bytes(&buf);
}

/// The authenticated selected-child index of pair `i`, reconstructed from its
/// side bits (the same bits the shift consumed as the tree order).
fn selected_index(proof: &SoundMultiproof, i: usize) -> usize {
    let depth = innode_depth(&proof.content_metas[proof.pair_phys[i]]);
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
    let u = proof.merkle_leaves.len();
    let n_phys = proof.merkle_roots.len();
    // Every per-pair Vec is indexed at [0..u) and every per-physical Vec at
    // [0..n_phys) below; a malformed proof with a short Vec must be rejected
    // cleanly, not panic (verifier DoS hardening — load-bearing once
    // SoundMultiproof gains a Deserialize path).
    if u == 0
        || u != proof.merkle_b_bits.len()
        || u != proof.pair_phys.len()
        || n_phys == 0
        || n_phys != proof.merkle_native_roots.len()
        || n_phys != proof.content_metas.len()
        || n_phys != proof.merkle_block_counts.len()
    {
        return Err(MhotMembershipError::RootMismatch {
            expected: *expected_root,
            actual: [0; 8],
        });
    }
    // -- Wire-validity gates: every gate here runs BEFORE the value it checks
    //    reaches a cached()/setup/pool allocation or a shift expression, so a
    //    malformed proof is rejected at wire-comparison cost. The n_log cap
    //    must precede any `1 << n_log_merkle`.
    if proof.n_log_merkle > MAX_N_LOG_MERKLE {
        return Err(MhotMembershipError::MalformedProof {
            reason: "n_log_merkle over absolute cap",
        });
    }
    if proof.n_routes != u {
        return Err(MhotMembershipError::MalformedProof {
            reason: "n_routes != unique pair count",
        });
    }
    // n_phys ≤ u by construction (every physical node is created inside pair
    // creation, prove.rs). Without this bound, junk physical nodes never
    // referenced by any pair still cost the verifier a per-phys gate + a
    // pad-forward loop (≤64 SHA-256 each) apiece — O(n_phys) work decoupled
    // from u. Latent while Serialize-only; the amplification is cheap (arrays,
    // not proof objects) once a Deserialize path exists.
    if n_phys > u {
        return Err(MhotMembershipError::MalformedProof {
            reason: "n_phys exceeds pair count",
        });
    }
    for p in 0..n_phys {
        let meta = &proof.content_metas[p];
        let mask_bits: u32 = meta.extraction_masks.iter().map(|m| m.count_ones()).sum();
        let fanout = meta.sparse_partial_keys.len();
        // fanout ∈ [2, 32]: internal nodes always have ≥2 children (the prover
        // asserts this) and native dense_key is u32. The upper gate bounds
        // innode_depth at 5; the block-count gate keeps n_pad from underflowing
        // in the pad-forward check below.
        if mask_bits > 32
            || fanout != meta.child_leaf_counts.len()
            || fanout < 2
            || fanout > 32
            || innode_depth(meta) > proof.merkle_block_counts[p]
            || !proof.merkle_block_counts[p].is_power_of_two()
            || proof.merkle_block_counts[p] > MAX_MERKLE_BLOCKS
        {
            return Err(MhotMembershipError::RootMismatch {
                expected: *expected_root,
                actual: [0; 8],
            });
        }
        // Uniform-8 layout (E2 prerequisite): every honest chain is exactly 8
        // blocks (depth ≤ 5 floored to 8). A non-uniform count would break the
        // `off_i = i·8` node×8×slot layout the global sumcheck commits to.
        if proof.merkle_block_counts[p] != MERKLE_BLOCKS_PER_NODE {
            return Err(MhotMembershipError::MalformedProof {
                reason: "non-uniform merkle block count (expected 8)",
            });
        }
    }
    for i in 0..u {
        // Every pair must reference a valid physical node BEFORE any indexed
        // use of pair_phys[i] below.
        if proof.pair_phys[i] >= n_phys {
            return Err(MhotMembershipError::MalformedProof {
                reason: "pair_phys index out of range",
            });
        }
        // Every honest chain is padded to exactly 8 side bits. Bound the length
        // BEFORE FS Step-0 (absorb_public_io) iterates every bit — otherwise a
        // single oversized b_bits vector (the shift itself truncates at 8, so
        // it is the ONLY unbounded consumer) hangs/OOMs the verifier. Latent
        // while Serialize-only; a per-pair DoS gate all the same.
        if proof.merkle_b_bits[i].len() != MERKLE_BLOCKS_PER_NODE {
            return Err(MhotMembershipError::MalformedProof {
                reason: "b_bits length != 8",
            });
        }
    }
    // Canonical-recompute gate (uniform-8): offsets are DERIVED (`off_i = i·8`),
    // not carried, so n_log_merkle must be EXACTLY `next_pow2(max(8u, min_n))`.
    // This pins the setup allocation size to u; a wrong n_log is rejected before
    // `Sha256HybridSetup::cached` sizes an allocation from it.
    {
        let min_n = 1usize << (22 - crate::r1cs_hashes::sha2::K_LOG);
        let n_real = u * MERKLE_BLOCKS_PER_NODE;
        if 1usize << proof.n_log_merkle != n_real.max(min_n).next_power_of_two() {
            return Err(MhotMembershipError::MalformedProof {
                reason: "n_log_merkle != canonical allocation size",
            });
        }
    }

    let vt = std::env::var_os("VERIFY_BREAKDOWN").is_some();
    let t0 = std::time::Instant::now();

    // -- Verify merkle core --
    let merkle_n_total = 1usize << proof.n_log_merkle;
    let merkle_setup = Sha256HybridSetup::cached_verify(merkle_n_total);
    let (merkle_ab, merkle_c) = flock_core::verifier::verify_core(
        &merkle_setup.r1cs,
        &proof.merkle_zc, &proof.merkle_lc, &proof.merkle_commitment,
        merkle_setup.r1cs.csc_lincheck_circuit(),
        challenger,
    ).map_err(MhotMembershipError::NodeVerify2)?;

    let t1 = std::time::Instant::now();
    // -- FS Step 0: bind all public IO the global shift consumes BEFORE η=τ_p --
    absorb_public_io(
        challenger,
        &proof.merkle_leaves,
        &proof.merkle_b_bits,
        &proof.pair_phys,
        &proof.merkle_roots,
        proof.n_log_merkle,
        proof.n_routes,
    );
    // -- ONE global merkle shift over the full P-node cube (E2) --
    let merkle_tau_pos = challenger.sample_f128_vec(MERKLE_LAYOUT.tau_pos_len());
    let merkle_fold = MerklePathFold::new(&MERKLE_LAYOUT, merkle_tau_pos);

    let block = MERKLE_BLOCKS_PER_NODE;
    let p_nodes = merkle_n_total / block; // 2^m
    let fold_digest = |d: &[u32; 8]| -> F128 {
        merkle_fold.fold_public_phys(&crate::r1cs_hashes::sha2::hash_to_phys_bits(d))
    };
    // Per-node public leaf / root, folded. Real nodes [0,u); padding [u,P) get
    // the deterministic dummy chain the prover committed — the verifier rebuilds
    // it identically, so padding cannot be a prover-chosen cancellation lever.
    let (_dummy_comps, dummy_b, dummy_leaf, dummy_root) = prove::dummy_padding_chain();
    let dummy_leaf_r = fold_digest(&dummy_leaf);
    let dummy_root_r = fold_digest(&dummy_root);
    let mut leaf_evals = Vec::with_capacity(p_nodes);
    let mut root_evals = Vec::with_capacity(p_nodes);
    let mut b_bits_full = vec![false; merkle_n_total];
    for i in 0..u {
        leaf_evals.push(fold_digest(&proof.merkle_leaves[i]));
        root_evals.push(fold_digest(&proof.merkle_roots[proof.pair_phys[i]]));
        for (y, &bit) in proof.merkle_b_bits[i].iter().enumerate() {
            if y < block {
                b_bits_full[i * block + y] = bit;
            }
        }
    }
    for node in u..p_nodes {
        leaf_evals.push(dummy_leaf_r);
        root_evals.push(dummy_root_r);
        for (y, &bit) in dummy_b.iter().enumerate() {
            b_bits_full[node * block + y] = bit;
        }
    }

    let path_log = proof.n_log_merkle - block.trailing_zeros() as usize;
    let claims = verify_merkle_path_shift(
        path_log,
        &proof.merkle_shift,
        &leaf_evals,
        &root_evals,
        &b_bits_full,
        proof.n_log_merkle,
        MERKLE_LAYOUT.slot_layout(),
        challenger,
    )
    .map_err(|e| {
        MhotMembershipError::NodeVerify2(flock_core::verifier::VerifyError::Wiring(format!(
            "global merkle shift: {e:?}"
        )))
    })?;

    let t2 = std::time::Instant::now();
    // -- Merkle PCS verify (forked challenger): ONE global PD claim --
    let point = build_merkle_claim_point(&MERKLE_LAYOUT, &merkle_fold, &claims);
    let merkle_pd_refs = [PackedDirectClaimRef { point: &point, value: claims.value }];

    let mut merkle_pcs_ch = fork_pcs_challenger(challenger, b"merkle");
    verify_core_opening_ligerito(
        &merkle_setup.r1cs, &merkle_setup.pcs_params,
        &proof.merkle_commitment, &proof.merkle_pcs,
        &merkle_ab, &merkle_c, &merkle_pd_refs, &mut merkle_pcs_ch,
    ).map_err(MhotMembershipError::NodeVerify2)?;

    // -- Node identity: authenticate each PHYSICAL node's real (pre-padding)
    //    tree root via the pad-forward check — padding merkle_native_roots[p]
    //    forward n_pad times (mirroring pad_to_needed byte-for-byte: current
    //    in X_L, zero sibling, SHA256_IV) must reach the shift-authenticated
    //    padded root merkle_roots[p]. The depth (hence n_pad) derives from the
    //    authenticated fanout, so a chain of the wrong tree depth cannot pass.
    //    content_hash is then COMPUTED natively over the authenticated
    //    native_root; committing a fake in-node tree or tampering
    //    content_metas changes it and breaks the binding below (cross-node
    //    parent-leaf / public root). Runs once per physical node — pairs
    //    sharing a node share ONE root and ONE content_hash by construction,
    //    which is what makes the mixed-root forgery (two pairs of one node
    //    carrying different roots) structurally impossible. --
    let mut content_hashes: Vec<[u32; 8]> = Vec::with_capacity(n_phys);
    for p in 0..n_phys {
        let n_pad = proof.merkle_block_counts[p] - innode_depth(&proof.content_metas[p]);
        let mut padded = proof.merkle_native_roots[p];
        for _ in 0..n_pad {
            let mut m = [0u32; 16];
            m[..8].copy_from_slice(&padded);
            padded = sha256_compress(&SHA256_IV, &m);
        }
        if padded != proof.merkle_roots[p] {
            return Err(MhotMembershipError::NativeRootMismatch { node_idx: p });
        }
        let ch = compute_content_hash(
            &proof.content_metas[p],
            &leaf_words_to_digest_bytes(&proof.merkle_native_roots[p]),
        );
        content_hashes.push(bytes_to_words(&ch));
    }

    let t3 = std::time::Instant::now();
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
            let child_content = content_hashes[proof.pair_phys[child_u]];
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
        if content_hashes[proof.pair_phys[root_u]] != *expected_root {
            return Err(MhotMembershipError::RootMismatch {
                expected: *expected_root,
                actual: content_hashes[proof.pair_phys[root_u]],
            });
        }
    }

    // -- Route verify core --
    let route_setup = RouteF32Setup::cached_verify(proof.n_routes);
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
        eprintln!("[verify] merkle_core={:.1}ms merkle_shifts={:.1}ms merkle_pcs+padfwd={:.1}ms cross+root+route={:.1}ms total={:.1}ms",
            ms(t1-t0), ms(t2-t1), ms(t3-t2), ms(t7-t3), ms(t7-t0));
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
/// FIAT-SHAMIR NOTE (updated for E2 global batching): the shift moved
/// in-protocol — the global sumcheck batches all P nodes via a challenge
/// (η = τ_p) that folds the per-node padded roots R(η) and leaves L(η). So the
/// shift's public IO that defines the batched error vector — padded `roots`,
/// `leaves`, `b_bits`, `pair_phys`, `n_log_merkle`, `n_routes` — IS now absorbed
/// into the transcript BEFORE η, by [`absorb_public_io`] (FS Step 0), on both
/// sides byte-identically. Without it a char-2 prover could pick those after
/// seeing η to cancel a forged node (`Σ_N eq(η,N)·e_N = 0`).
///
/// Everything else stays bound by DETERMINISTIC verifier equalities that consume
/// no challenge, so it needs no absorption: `entries` (routing on authenticated
/// content_metas + terminal leaf hash), `merkle_native_roots` (pad-forward to
/// the shift-authenticated padded root), `content_metas` / `path_mapping`
/// (cross-node + root checks). Each is transitively authenticated against
/// `expected_root`. If ANY of those checks later moves in-protocol (RLC-batched
/// public claims), its inputs MUST likewise be absorbed before the batching
/// coefficient, or unfaithful-claims under-binding reopens forgeries.
pub fn verify_sound_multiproof_with_entries(
    proof: &SoundMultiproof,
    expected_root: &[u32; 8],
    entries: &[PathEntry],
    challenger: &mut FsChallenger,
) -> Result<(), MhotMembershipError> {
    // Cheap preconditions before the ~O(U) structural verify.
    let n_paths = proof.path_mapping.node_indices.len();
    if entries.len() != n_paths {
        return Err(MhotMembershipError::EntryCountMismatch {
            n_entries: entries.len(),
            n_paths,
        });
    }
    if entries.iter().any(|e| e.value.len() > MAX_ENTRY_VALUE_LEN) {
        return Err(MhotMembershipError::MalformedProof {
            reason: "entry value over length cap",
        });
    }

    verify_sound_multiproof(proof, expected_root, challenger)?;

    // Authenticated selected-child index per unique pair (depends only on u_i).
    let selected: Vec<usize> = (0..proof.merkle_leaves.len())
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
            let meta = &proof.content_metas[proof.pair_phys[u_i]];
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
        if proof.content_metas[proof.pair_phys[last_u]]
            .child_leaf_counts
            .get(selected[last_u])
            != Some(&1)
        {
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
