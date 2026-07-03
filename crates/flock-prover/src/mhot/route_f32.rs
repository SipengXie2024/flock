// CURRENT SOLUTION for adaptive span: fixed FANOUT=32 worst-case scan.
//
// Benchmarked against two alternatives (2026-06-27, 5-agent workflow):
//   Route 1 (this): F=32 fixed, ~80ms prove+verify, zero soundness gaps
//   Route 2 (multi-bucket): 3 R1CS sizes, 4530ms total — 56x slower due to
//     Ligerito m=22 padding floor taxed 3x. REJECTED.
//   Route 3a (single-child atom, route_single.rs): ~71ms, but 3 soundness
//     gaps (child_index forgeable, no uniqueness, found/selected chain
//     unbound). KEPT AS BACKUP — revisit when wiring sumcheck lands.
//
// F=32 wastes ~1.45x at avg fanout 22, but route is not the bottleneck
// (keccak3 K=131072 >> route K=32768). The waste is invisible in benchmarks.
use crate::prover::{prove_fast_core, quirky_x_outer_full, ProveCore};
use crate::r1cs_hashes::common::{
    build_block_r1cs_with_matrices, drive_witness_packed_and_lincheck, or_bit_at,
};
use flock_core::challenger::FsChallenger;
use flock_core::field::F128;
use flock_core::pcs::{self, Commitment, PcsParams};
use flock_core::proof::R1csClaim;
use flock_core::r1cs::{BlockR1cs, SparseBinaryMatrix};
use flock_core::verifier::VerifyError;
use flock_core::{lincheck, zerocheck};

pub const K_LOG: usize = 15;
pub const K: usize = 1 << K_LOG;
pub const K_SKIP: usize = 6;
pub const U64_PER_BLOCK: usize = K / 64;

pub const KEY_BITS: usize = 256;
pub const DIGEST_BITS: usize = 256;
pub const FANOUT: usize = 32;
pub const W_MAX: usize = 5;

pub const KEY_BASE: usize = 0;
pub const Z_CONST_POS: usize = KEY_BASE + KEY_BITS;
pub const SELECTED_IN_BASE: usize = Z_CONST_POS + 1;
pub const MASK_BASE: usize = SELECTED_IN_BASE + DIGEST_BITS;
pub const CHILDREN_BASE: usize = MASK_BASE + KEY_BITS;

// Per-child layout:
//   [0..256)      child digest bits
//   [256..261)    extracted bits (W_MAX=5)
//   [261..264)    match intermediates: p01, p23, p0123
//   [264]         match_result (= p0123 AND eq_4)
//   [265]         take
//   [266]         found_state
//   [267..523)    mux_delta (256 bits)
// stride = 523

pub const CHILD_DIGEST_OFFSET: usize = 0;
pub const EXTRACTED_OFFSET: usize = CHILD_DIGEST_OFFSET + DIGEST_BITS;
pub const MATCH_P01_OFFSET: usize = EXTRACTED_OFFSET + W_MAX;
pub const MATCH_P23_OFFSET: usize = MATCH_P01_OFFSET + 1;
pub const MATCH_P0123_OFFSET: usize = MATCH_P23_OFFSET + 1;
pub const MATCH_RESULT_OFFSET: usize = MATCH_P0123_OFFSET + 1;
pub const TAKE_OFFSET: usize = MATCH_RESULT_OFFSET + 1;
pub const FOUND_STATE_OFFSET: usize = TAKE_OFFSET + 1;
pub const MUX_DELTA_OFFSET: usize = FOUND_STATE_OFFSET + 1;
pub const CHILD_STRIDE: usize = MUX_DELTA_OFFSET + DIGEST_BITS;

pub const FOUND_OUT_FINAL_POS: usize = CHILDREN_BASE + FANOUT * CHILD_STRIDE;
// Content soundness: mask must be a prefix (contiguous 1s from bit 0),
// key bits above mask width must be 0. Placed right after the found flag so
// the routed-child digest output can go last at an aligned offset.
const MASK_CHECK_BASE: usize = FOUND_OUT_FINAL_POS + 1;
const N_MASK_CHECKS: usize = KEY_BITS - 1; // 255
const MASK_AND_BASE: usize = MASK_CHECK_BASE + N_MASK_CHECKS;
const N_MASK_AND: usize = N_MASK_CHECKS; // 255 (1 linear + 254 mul)
const KEY_CHECK_BASE: usize = MASK_AND_BASE + N_MASK_AND;
const N_KEY_CHECKS: usize = KEY_BITS; // 256
const KEY_AND_BASE: usize = KEY_CHECK_BASE + N_KEY_CHECKS;
const N_KEY_AND: usize = N_KEY_CHECKS; // 256 (1 linear + 255 mul)
const ALL_OK_POS: usize = KEY_AND_BASE + N_KEY_AND;
// SELECTED_OUT_FINAL (256-bit routed-child digest) sits at a DIGEST_BITS-aligned
// offset so one PackedDirectClaim extracts it as a single aligned slot for the
// route↔hash binding in merkle_membership.rs.
pub const SELECTED_OUT_FINAL_BASE: usize =
    ((ALL_OK_POS + 1 + DIGEST_BITS - 1) / DIGEST_BITS) * DIGEST_BITS;
pub const USEFUL_BITS: usize = SELECTED_OUT_FINAL_BASE + DIGEST_BITS;

const ROUTE_MASK_POSITIONS: [usize; W_MAX] = [0, 1, 2, 3, 4];
const TRANSCRIPT_LABEL: &[u8] = b"mhot-route-f32-v0";
const MIN_LIGERITO_M: usize = 22;
const MIN_LIGERITO_INSTANCES: usize = 1 << (MIN_LIGERITO_M - K_LOG);

#[inline]
pub const fn child_base(child: usize) -> usize {
    CHILDREN_BASE + child * CHILD_STRIDE
}

#[inline]
pub const fn child_digest_pos(child: usize, bit: usize) -> usize {
    child_base(child) + CHILD_DIGEST_OFFSET + bit
}

#[inline]
pub const fn extracted_pos(child: usize, bit: usize) -> usize {
    child_base(child) + EXTRACTED_OFFSET + bit
}

#[inline]
pub const fn match_p01_pos(child: usize) -> usize {
    child_base(child) + MATCH_P01_OFFSET
}

#[inline]
pub const fn match_p23_pos(child: usize) -> usize {
    child_base(child) + MATCH_P23_OFFSET
}

#[inline]
pub const fn match_p0123_pos(child: usize) -> usize {
    child_base(child) + MATCH_P0123_OFFSET
}

#[inline]
pub const fn match_result_pos(child: usize) -> usize {
    child_base(child) + MATCH_RESULT_OFFSET
}

#[inline]
pub const fn take_pos(child: usize) -> usize {
    child_base(child) + TAKE_OFFSET
}

#[inline]
pub const fn found_state_pos(child: usize) -> usize {
    child_base(child) + FOUND_STATE_OFFSET
}

#[inline]
pub const fn mux_delta_pos(child: usize, bit: usize) -> usize {
    child_base(child) + MUX_DELTA_OFFSET + bit
}

#[inline]
pub const fn selected_in_pos(bit: usize) -> usize {
    SELECTED_IN_BASE + bit
}

#[inline]
pub const fn mask_pos(bit: usize) -> usize {
    MASK_BASE + bit
}

#[inline]
pub const fn selected_out_final_pos(bit: usize) -> usize {
    SELECTED_OUT_FINAL_BASE + bit
}

pub fn build_matrices_f32() -> (SparseBinaryMatrix, SparseBinaryMatrix) {
    let mut a_rows: Vec<Vec<usize>> = vec![Vec::new(); K];
    let mut b_rows: Vec<Vec<usize>> = vec![Vec::new(); K];

    input_emit(&mut a_rows, &mut b_rows, KEY_BASE, KEY_BITS);
    input_emit(&mut a_rows, &mut b_rows, MASK_BASE, KEY_BITS);

    for child in 0..FANOUT {
        input_emit(
            &mut a_rows,
            &mut b_rows,
            child_base(child) + CHILD_DIGEST_OFFSET,
            DIGEST_BITS,
        );

        for j in 0..W_MAX {
            let key_bit = KEY_BASE + ROUTE_MASK_POSITIONS[j];
            let mask_bit = mask_pos(ROUTE_MASK_POSITIONS[j]);
            set_mul(
                &mut a_rows,
                &mut b_rows,
                extracted_pos(child, j),
                vec![key_bit],
                vec![mask_bit],
            );
        }

        // 5-bit match reduction via binary tree:
        // p01 = eq(bit0) * eq(bit1)
        let eq0 = eq_expr(child, 0);
        let eq1 = eq_expr(child, 1);
        set_mul(&mut a_rows, &mut b_rows, match_p01_pos(child), eq0, eq1);

        // p23 = eq(bit2) * eq(bit3)
        let eq2 = eq_expr(child, 2);
        let eq3 = eq_expr(child, 3);
        set_mul(&mut a_rows, &mut b_rows, match_p23_pos(child), eq2, eq3);

        // p0123 = p01 * p23
        set_mul(
            &mut a_rows,
            &mut b_rows,
            match_p0123_pos(child),
            vec![match_p01_pos(child)],
            vec![match_p23_pos(child)],
        );

        // match_result = p0123 * eq(bit4)
        let eq4 = eq_expr(child, 4);
        set_mul(
            &mut a_rows,
            &mut b_rows,
            match_result_pos(child),
            vec![match_p0123_pos(child)],
            eq4,
        );

        set_mul(
            &mut a_rows,
            &mut b_rows,
            take_pos(child),
            vec![match_result_pos(child)],
            not_found_expr(child),
        );

        set_linear(
            &mut a_rows,
            &mut b_rows,
            found_state_pos(child),
            found_out_expr(child),
        );

        for bit in 0..DIGEST_BITS {
            let mut rhs = selected_state_expr(child, bit);
            rhs.push(child_digest_pos(child, bit));
            set_mul(
                &mut a_rows,
                &mut b_rows,
                mux_delta_pos(child, bit),
                vec![take_pos(child)],
                rhs,
            );
        }
    }

    set_linear(
        &mut a_rows,
        &mut b_rows,
        FOUND_OUT_FINAL_POS,
        found_in_expr(FANOUT),
    );
    for bit in 0..DIGEST_BITS {
        set_linear(
            &mut a_rows,
            &mut b_rows,
            selected_out_final_pos(bit),
            selected_state_expr(FANOUT, bit),
        );
    }

    // --- Content soundness: mask prefix + key validity ---

    // Mask prefix: check[i] = NOT(mask[i]) AND mask[i+1] for i in 0..254
    for i in 0..N_MASK_CHECKS {
        set_mul(
            &mut a_rows, &mut b_rows,
            MASK_CHECK_BASE + i,
            vec![mask_pos(i), Z_CONST_POS],
            vec![mask_pos(i + 1)],
        );
    }

    // AND-chain of NOT(check[i]): mask_ok = NOT(c0) AND NOT(c1) AND ...
    // mask_and[0] = NOT(check[0]) — linear
    set_linear(&mut a_rows, &mut b_rows,
        MASK_AND_BASE, vec![MASK_CHECK_BASE, Z_CONST_POS]);
    // mask_and[i] = mask_and[i-1] AND NOT(check[i])
    for i in 1..N_MASK_AND {
        set_mul(
            &mut a_rows, &mut b_rows,
            MASK_AND_BASE + i,
            vec![MASK_AND_BASE + i - 1],
            vec![MASK_CHECK_BASE + i, Z_CONST_POS],
        );
    }

    // Key validity: check[i] = key[i] AND NOT(mask[i]) for i in 0..255
    for i in 0..N_KEY_CHECKS {
        set_mul(
            &mut a_rows, &mut b_rows,
            KEY_CHECK_BASE + i,
            vec![KEY_BASE + i],
            vec![mask_pos(i), Z_CONST_POS],
        );
    }

    // AND-chain of NOT(check[i]): key_ok = NOT(c0) AND NOT(c1) AND ...
    set_linear(&mut a_rows, &mut b_rows,
        KEY_AND_BASE, vec![KEY_CHECK_BASE, Z_CONST_POS]);
    for i in 1..N_KEY_AND {
        set_mul(
            &mut a_rows, &mut b_rows,
            KEY_AND_BASE + i,
            vec![KEY_AND_BASE + i - 1],
            vec![KEY_CHECK_BASE + i, Z_CONST_POS],
        );
    }

    // all_ok = mask_ok AND key_ok
    let mask_ok_pos = MASK_AND_BASE + N_MASK_AND - 1;
    let key_ok_pos = KEY_AND_BASE + N_KEY_AND - 1;
    set_mul(&mut a_rows, &mut b_rows, ALL_OK_POS, vec![mask_ok_pos], vec![key_ok_pos]);

    // Final assert: found AND all_ok = 1 (via const_pin at Z_CONST_POS)
    set_mul(
        &mut a_rows, &mut b_rows,
        Z_CONST_POS,
        vec![FOUND_OUT_FINAL_POS],
        vec![ALL_OK_POS],
    );

    let to_mat = |rows| SparseBinaryMatrix {
        num_rows: K,
        num_cols: K,
        rows,
    };
    (to_mat(a_rows), to_mat(b_rows))
}

fn input_emit(a_rows: &mut [Vec<usize>], b_rows: &mut [Vec<usize>], base: usize, len: usize) {
    for bit in 0..len {
        let pos = base + bit;
        set_linear(a_rows, b_rows, pos, vec![pos]);
    }
}

fn set_linear(a_rows: &mut [Vec<usize>], b_rows: &mut [Vec<usize>], row: usize, expr: Vec<usize>) {
    set_mul(a_rows, b_rows, row, expr, vec![Z_CONST_POS]);
}

fn set_mul(
    a_rows: &mut [Vec<usize>],
    b_rows: &mut [Vec<usize>],
    row: usize,
    a: Vec<usize>,
    b: Vec<usize>,
) {
    a_rows[row] = a;
    b_rows[row] = b;
}

pub fn build_block_r1cs_f32(n_blocks_log: usize) -> BlockR1cs {
    let (a_0, b_0) = build_matrices_f32();
    build_block_r1cs_with_matrices(
        n_blocks_log,
        K_LOG,
        K_SKIP,
        USEFUL_BITS,
        a_0,
        b_0,
        Some(Z_CONST_POS),
    )
}

#[derive(Clone, Debug)]
pub struct RouteF32Setup {
    pub n_instances: usize,
    pub setup_n_instances: usize,
    pub r1cs: BlockR1cs,
    pub pcs_params: PcsParams,
}

static SETUP_CACHE: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<usize, std::sync::Arc<RouteF32Setup>>>,
> = std::sync::OnceLock::new();

impl RouteF32Setup {
    pub fn cached(n_instances: usize) -> std::sync::Arc<Self> {
        let setup = Self::cached_inner(n_instances, true);
        // Re-warm after a scratch::clear() (or after cached_verify built this
        // entry without prewarming): O(1) via the pool watermark when warm.
        flock_core::scratch::prewarm_prover(setup.r1cs.m);
        setup
    }

    /// [`Self::cached`] minus the prover-pool prewarm. Verification never
    /// touches the scratch pool, so a cold-cache verifier must not fault the
    /// prove buffer set just to build the route R1CS.
    pub fn cached_verify(n_instances: usize) -> std::sync::Arc<Self> {
        Self::cached_inner(n_instances, false)
    }

    fn cached_inner(n_instances: usize, prewarm: bool) -> std::sync::Arc<Self> {
        let cache = SETUP_CACHE.get_or_init(|| std::sync::Mutex::new(Default::default()));
        let setup_n = route_setup_n_instances(n_instances);
        let mut map = cache.lock().unwrap();
        std::sync::Arc::clone(
            map.entry(setup_n)
                .or_insert_with(|| std::sync::Arc::new(Self::new_opt(n_instances, prewarm))),
        )
    }

    /// Drop every cached setup. R1CS matrices are size-keyed and never
    /// evicted otherwise — an N-sweep accumulates them across sizes and
    /// contaminates peak-memory numbers.
    pub fn clear_setup_cache() {
        if let Some(cache) = SETUP_CACHE.get() {
            cache.lock().unwrap().clear();
        }
    }

    pub fn new(n_instances: usize) -> Self {
        Self::new_opt(n_instances, true)
    }

    fn new_opt(n_instances: usize, prewarm: bool) -> Self {
        assert!(n_instances >= 1, "n_instances must be >= 1");
        let setup_n_instances = route_setup_n_instances(n_instances);
        let n_blocks_log = setup_n_instances.trailing_zeros() as usize;
        let r1cs = build_block_r1cs_f32(n_blocks_log);
        r1cs.csc_lincheck_circuit();
        if prewarm {
            flock_core::scratch::prewarm_prover(r1cs.m);
        }
        let pcs_params = PcsParams {
            m: r1cs.m,
            log_inv_rate: 1,
            log_batch_size: 6,
            profile: pcs::ligerito::LigeritoProfile::Fast,
        };
        Self {
            n_instances,
            setup_n_instances,
            r1cs,
            pcs_params,
        }
    }

    pub fn n_blocks_log(&self) -> usize {
        self.r1cs.m - self.r1cs.k_log
    }
}

#[derive(Clone, Debug)]
pub struct RouteF32Witness {
    pub key: [bool; KEY_BITS],
    pub mask: [bool; KEY_BITS],
    pub children: Vec<[bool; DIGEST_BITS]>,
}

impl RouteF32Witness {
    pub fn new(
        key: [bool; KEY_BITS],
        mask: [bool; KEY_BITS],
        children: Vec<[bool; DIGEST_BITS]>,
    ) -> Self {
        assert_eq!(children.len(), FANOUT, "F_route F32 fanout is fixed at 32");
        Self {
            key,
            mask,
            children,
        }
    }

    pub fn new_padded(
        key: [bool; KEY_BITS],
        mask: [bool; KEY_BITS],
        effective_children: &[[bool; DIGEST_BITS]],
        effective_fanout: usize,
    ) -> Self {
        assert!(
            effective_fanout <= FANOUT,
            "effective_fanout {effective_fanout} > FANOUT {FANOUT}"
        );
        assert_eq!(
            effective_children.len(),
            effective_fanout,
            "children.len() must match effective_fanout"
        );
        let mut children = Vec::with_capacity(FANOUT);
        children.extend_from_slice(effective_children);
        for _ in effective_fanout..FANOUT {
            children.push([false; DIGEST_BITS]);
        }
        Self {
            key,
            mask,
            children,
        }
    }
}

pub struct RouteF32Proof {
    pub zc_proof: zerocheck::ZerocheckProof,
    pub lc_proof: lincheck::LincheckProof,
    pub pcs_open: pcs::BatchOpeningProofLigerito,
    pub commitment: Commitment,
    pub claim: R1csClaim,
    pub n_instances: usize,
    pub setup_n_instances: usize,
}

pub fn generate_witness_with_ab_packed_and_lincheck(
    witnesses: &[RouteF32Witness],
    n_blocks_log: usize,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
    let padding = padding_route_witness();
    drive_witness_packed_and_lincheck(
        witnesses,
        Some(&padding),
        n_blocks_log,
        K_LOG,
        |witness, z_u64, a_u64, b_u64| {
            fill_block_witness_f32(
                &witness.key,
                &witness.mask,
                &witness.children,
                z_u64,
                a_u64,
                b_u64,
            );
        },
    )
}

pub fn prove_route_f32(setup: &RouteF32Setup, witnesses: &[RouteF32Witness]) -> RouteF32Proof {
    assert_eq!(
        witnesses.len(),
        setup.n_instances,
        "witness count must match RouteF32Setup::n_instances"
    );
    let (z_packed, a_packed, b_packed, z_lincheck) =
        generate_witness_with_ab_packed_and_lincheck(witnesses, setup.n_blocks_log());
    prove_route_from_parts(setup, z_packed, a_packed, b_packed, z_lincheck)
}

pub fn verify_route_f32(
    setup: &RouteF32Setup,
    proof: &RouteF32Proof,
) -> Result<R1csClaim, VerifyError> {
    assert_eq!(
        proof.n_instances, setup.n_instances,
        "logical route instance count mismatch"
    );
    assert_eq!(
        proof.setup_n_instances, setup.setup_n_instances,
        "setup route instance count mismatch"
    );

    let mut challenger = FsChallenger::new(TRANSCRIPT_LABEL);
    let lc_circuit = setup.r1cs.csc_lincheck_circuit();
    let (ab, c) = flock_core::verifier::verify_core(
        &setup.r1cs,
        &proof.zc_proof,
        &proof.lc_proof,
        &proof.commitment,
        lc_circuit,
        &mut challenger,
    )?;

    let z_skips = [ab.point.z_skip, c.point.z_skip];
    let values = [ab.value, c.value];
    let ab_x_outer = quirky_x_outer_full(&ab.point);
    let c_x_outer = quirky_x_outer_full(&c.point);
    let x_outers = [ab_x_outer.as_slice(), c_x_outer.as_slice()];
    let log_n = setup.r1cs.m - pcs::LOG_PACKING;
    let lig_config = pcs::ligerito::verifier_config_for(
        log_n,
        setup.pcs_params.log_batch_size,
        setup.pcs_params.profile,
    )
    .expect("Ligerito default verifier config");
    pcs::verify_opening_batch_ligerito_mixed(
        &proof.commitment,
        &values,
        &z_skips,
        &x_outers,
        &[],
        &proof.pcs_open,
        &lig_config,
        &mut challenger,
    )
    .map_err(VerifyError::PcsAb)?;

    Ok(R1csClaim { ab, c })
}

pub fn fill_block_witness_f32(
    key: &[bool; KEY_BITS],
    mask: &[bool; KEY_BITS],
    children: &[[bool; DIGEST_BITS]],
    z_u64: &mut [u64],
    a_u64: &mut [u64],
    b_u64: &mut [u64],
) {
    assert_eq!(children.len(), FANOUT, "F_route F32 fanout is fixed at 32");
    assert_eq!(z_u64.len(), U64_PER_BLOCK);
    assert_eq!(a_u64.len(), U64_PER_BLOCK);
    assert_eq!(b_u64.len(), U64_PER_BLOCK);

    z_u64.fill(0);
    a_u64.fill(0);
    b_u64.fill(0);

    set_zab(Z_CONST_POS, true, false, true, z_u64, a_u64, b_u64);

    for bit in 0..KEY_BITS {
        write_input(KEY_BASE + bit, key[bit], z_u64, a_u64, b_u64);
        write_input(mask_pos(bit), mask[bit], z_u64, a_u64, b_u64);
    }

    let mut found = false;
    let mut selected = [false; DIGEST_BITS];

    for (child, child_digest) in children.iter().enumerate() {
        for bit in 0..DIGEST_BITS {
            write_input(
                child_digest_pos(child, bit),
                child_digest[bit],
                z_u64,
                a_u64,
                b_u64,
            );
        }

        let mut extracted = [false; W_MAX];
        for j in 0..W_MAX {
            let pos = ROUTE_MASK_POSITIONS[j];
            extracted[j] = key[pos] & mask[pos];
            set_zab(
                extracted_pos(child, j),
                extracted[j],
                key[pos],
                mask[pos],
                z_u64,
                a_u64,
                b_u64,
            );
        }

        // 5-bit match via binary tree:
        let bit_eq_0 = extracted[0] ^ child_index_bit(child, 0) ^ true;
        let bit_eq_1 = extracted[1] ^ child_index_bit(child, 1) ^ true;
        let bit_eq_2 = extracted[2] ^ child_index_bit(child, 2) ^ true;
        let bit_eq_3 = extracted[3] ^ child_index_bit(child, 3) ^ true;
        let bit_eq_4 = extracted[4] ^ child_index_bit(child, 4) ^ true;

        let p01 = bit_eq_0 & bit_eq_1;
        set_zab(
            match_p01_pos(child),
            p01,
            bit_eq_0,
            bit_eq_1,
            z_u64,
            a_u64,
            b_u64,
        );

        let p23 = bit_eq_2 & bit_eq_3;
        set_zab(
            match_p23_pos(child),
            p23,
            bit_eq_2,
            bit_eq_3,
            z_u64,
            a_u64,
            b_u64,
        );

        let p0123 = p01 & p23;
        set_zab(
            match_p0123_pos(child),
            p0123,
            p01,
            p23,
            z_u64,
            a_u64,
            b_u64,
        );

        let matches = p0123 & bit_eq_4;
        set_zab(
            match_result_pos(child),
            matches,
            p0123,
            bit_eq_4,
            z_u64,
            a_u64,
            b_u64,
        );

        let take = matches & !found;
        set_zab(take_pos(child), take, matches, !found, z_u64, a_u64, b_u64);

        let found_out = found ^ take;
        write_linear(found_state_pos(child), found_out, z_u64, a_u64, b_u64);

        for bit in 0..DIGEST_BITS {
            let mux_rhs = child_digest[bit] ^ selected[bit];
            let delta = take & mux_rhs;
            set_zab(
                mux_delta_pos(child, bit),
                delta,
                take,
                mux_rhs,
                z_u64,
                a_u64,
                b_u64,
            );
            selected[bit] ^= delta;
        }
        found = found_out;
    }

    write_linear(FOUND_OUT_FINAL_POS, found, z_u64, a_u64, b_u64);

    for (bit, value) in selected.iter().copied().enumerate() {
        write_linear(selected_out_final_pos(bit), value, z_u64, a_u64, b_u64);
    }

    // --- Content soundness witness ---

    // Mask prefix checks: check[i] = NOT(mask[i]) AND mask[i+1]
    for i in 0..N_MASK_CHECKS {
        let check = !mask[i] & mask[i + 1];
        set_zab(MASK_CHECK_BASE + i, check, !mask[i], mask[i + 1], z_u64, a_u64, b_u64);
    }

    // AND-chain: mask_and[0] = NOT(check[0])
    let mut mask_ok = {
        let c0 = !mask[0] & mask[1];
        let v = !c0;
        write_linear(MASK_AND_BASE, v, z_u64, a_u64, b_u64);
        v
    };
    for i in 1..N_MASK_AND {
        let check = !mask[i] & mask[i + 1];
        let not_check = !check;
        let v = mask_ok & not_check;
        set_zab(MASK_AND_BASE + i, v, mask_ok, not_check, z_u64, a_u64, b_u64);
        mask_ok = v;
    }

    // Key validity checks: check[i] = key[i] AND NOT(mask[i])
    for i in 0..N_KEY_CHECKS {
        let check = key[i] & !mask[i];
        set_zab(KEY_CHECK_BASE + i, check, key[i], !mask[i], z_u64, a_u64, b_u64);
    }

    // AND-chain: key_and[0] = NOT(check[0])
    let mut key_ok = {
        let c0 = key[0] & !mask[0];
        let v = !c0;
        write_linear(KEY_AND_BASE, v, z_u64, a_u64, b_u64);
        v
    };
    for i in 1..N_KEY_AND {
        let check = key[i] & !mask[i];
        let not_check = !check;
        let v = key_ok & not_check;
        set_zab(KEY_AND_BASE + i, v, key_ok, not_check, z_u64, a_u64, b_u64);
        key_ok = v;
    }

    // all_ok = mask_ok AND key_ok
    let all_ok = mask_ok & key_ok;
    set_zab(ALL_OK_POS, all_ok, mask_ok, key_ok, z_u64, a_u64, b_u64);

    // Final assert: found AND all_ok = 1 (via const_pin)
    set_zab(Z_CONST_POS, true, found, all_ok, z_u64, a_u64, b_u64);
}

fn prove_route_from_parts(
    setup: &RouteF32Setup,
    z_packed: Vec<F128>,
    a_packed: Vec<F128>,
    b_packed: Vec<F128>,
    z_lincheck: Vec<u8>,
) -> RouteF32Proof {
    let mut challenger = FsChallenger::new(TRANSCRIPT_LABEL);
    let lc_circuit = setup.r1cs.csc_lincheck_circuit();
    let core = prove_fast_core(
        &setup.r1cs,
        &setup.pcs_params,
        z_packed,
        a_packed,
        b_packed,
        z_lincheck,
        lc_circuit,
        &mut challenger,
    );

    let ProveCore {
        zc_proof,
        lc_proof,
        ab,
        c,
        commitment,
        prover_data,
        z_packed,
        s_hat_v_ab,
        s_hat_v_c,
    } = core;

    let log_n = setup.r1cs.m - pcs::LOG_PACKING;
    let lig_config = pcs::ligerito::prover_config_for(
        log_n,
        setup.pcs_params.log_batch_size,
        setup.pcs_params.profile,
    )
    .expect("Ligerito default prover config");

    let padding = zerocheck::PaddingSpec {
        k_log: setup.r1cs.k_log,
        useful_bits_per_block: setup.r1cs.useful_bits,
            n_real_blocks: None,
    };
    let ab_x_outer = quirky_x_outer_full(&ab.point);
    let c_x_outer = quirky_x_outer_full(&c.point);
    let pre_ab = s_hat_v_ab.as_deref();
    let pre_c = Some(s_hat_v_c.as_slice());
    let pcs_open = pcs::open_batch_mixed_ligerito_with_precomputed_s_hat_v(
        z_packed,
        &prover_data,
        &commitment,
        &[ab_x_outer.as_slice(), c_x_outer.as_slice()],
        &[pre_ab, pre_c],
        &[],
        &padding,
        &lig_config,
        &mut challenger,
    );

    let claim = R1csClaim { ab, c };

    RouteF32Proof {
        zc_proof,
        lc_proof,
        pcs_open,
        commitment,
        claim,
        n_instances: setup.n_instances,
        setup_n_instances: setup.setup_n_instances,
    }
}

fn route_setup_n_instances(n_instances: usize) -> usize {
    n_instances.max(MIN_LIGERITO_INSTANCES).next_power_of_two()
}

fn padding_route_witness() -> RouteF32Witness {
    let key = [false; KEY_BITS];
    let mut mask = [false; KEY_BITS];
    for j in 0..W_MAX {
        mask[ROUTE_MASK_POSITIONS[j]] = true;
    }
    RouteF32Witness::new(key, mask, vec![[false; DIGEST_BITS]; FANOUT])
}

fn eq_expr(child: usize, bit: usize) -> Vec<usize> {
    if child_index_bit(child, bit) {
        vec![extracted_pos(child, bit)]
    } else {
        vec![Z_CONST_POS, extracted_pos(child, bit)]
    }
}

fn not_found_expr(child: usize) -> Vec<usize> {
    let mut expr = vec![Z_CONST_POS];
    expr.extend(found_in_expr(child));
    expr
}

fn found_out_expr(child: usize) -> Vec<usize> {
    let mut expr = found_in_expr(child);
    expr.push(take_pos(child));
    expr
}

fn found_in_expr(child: usize) -> Vec<usize> {
    if child == 0 {
        Vec::new()
    } else {
        vec![found_state_pos(child - 1)]
    }
}

fn selected_state_expr(child: usize, bit: usize) -> Vec<usize> {
    let mut expr = vec![selected_in_pos(bit)];
    for prev_child in 0..child {
        expr.push(mux_delta_pos(prev_child, bit));
    }
    expr
}

#[inline]
fn child_index_bit(child: usize, bit: usize) -> bool {
    ((child >> bit) & 1) != 0
}

fn write_input(pos: usize, value: bool, z_u64: &mut [u64], a_u64: &mut [u64], b_u64: &mut [u64]) {
    set_zab(pos, value, value, true, z_u64, a_u64, b_u64);
}

fn write_linear(pos: usize, value: bool, z_u64: &mut [u64], a_u64: &mut [u64], b_u64: &mut [u64]) {
    set_zab(pos, value, value, true, z_u64, a_u64, b_u64);
}

fn set_zab(
    pos: usize,
    z: bool,
    a: bool,
    b: bool,
    z_u64: &mut [u64],
    a_u64: &mut [u64],
    b_u64: &mut [u64],
) {
    if z {
        or_bit_at(z_u64, pos);
    }
    if a {
        or_bit_at(a_u64, pos);
    }
    if b {
        or_bit_at(b_u64, pos);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn layout_constants_consistent_f32() {
        assert_eq!(K, 32768);
        assert_eq!(K_SKIP, 6);
        assert_eq!(KEY_BASE, 0);
        assert_eq!(Z_CONST_POS, 256);
        assert_eq!(SELECTED_IN_BASE, 257);
        assert_eq!(MASK_BASE, 513);
        assert_eq!(CHILDREN_BASE, 769);
        assert_eq!(CHILD_STRIDE, 523);
        assert_eq!(FOUND_OUT_FINAL_POS, 769 + 32 * 523);
        assert_eq!(FOUND_OUT_FINAL_POS, 17505);
        assert_eq!(SELECTED_OUT_FINAL_BASE, 18688);
        assert_eq!(
            SELECTED_OUT_FINAL_BASE % DIGEST_BITS,
            0,
            "SELECTED_OUT_FINAL must be 256-bit aligned for PD-claim binding"
        );
        assert_eq!(USEFUL_BITS, 18944);
        assert!(USEFUL_BITS <= K, "USEFUL_BITS {USEFUL_BITS} > K {K}");

        let (a_0, b_0) = build_matrices_f32();
        let a_nnz: usize = a_0.rows.iter().map(Vec::len).sum();
        let b_nnz: usize = b_0.rows.iter().map(Vec::len).sum();
        let nonempty_rows = a_0
            .rows
            .iter()
            .zip(b_0.rows.iter())
            .filter(|(a, b)| !a.is_empty() || !b.is_empty())
            .count();
        eprintln!(
            "F_route_f32 layout: useful_bits={USEFUL_BITS}, nonempty_rows={nonempty_rows}, a_nnz={a_nnz}, b_nnz={b_nnz}, total_nnz={}",
            a_nnz + b_nnz
        );
    }

    #[test]
    fn witness_satisfies_r1cs_f32_full() {
        let r1cs = build_block_r1cs_f32(3);
        let z = one_block_witness_bool_full();
        assert_eq!(z.len(), r1cs.n());
        assert!(r1cs.satisfies(&z), "R1CS not satisfied for fanout-32 witness");
    }

    #[test]
    fn witness_satisfies_r1cs_f32_padded_8() {
        let r1cs = build_block_r1cs_f32(3);
        let z = one_block_witness_bool_padded(8);
        assert_eq!(z.len(), r1cs.n());
        assert!(
            r1cs.satisfies(&z),
            "R1CS not satisfied for fanout-8 padded to 32"
        );
    }

    #[test]
    fn prove_verify_route_f32_roundtrip() {
        let (elapsed, _) = time_it(|| {
            let setup = RouteF32Setup::new(1);
            assert_eq!(setup.setup_n_instances, MIN_LIGERITO_INSTANCES);
            assert_eq!(setup.r1cs.m, MIN_LIGERITO_M);
            let witness = sample_route_witness_full();
            let proof = prove_route_f32(&setup, &[witness]);
            let claim = verify_route_f32(&setup, &proof)
                .unwrap_or_else(|err| panic!("route_f32 verifier rejected: {err:?}"));
            assert_eq!(claim, proof.claim);
        });
        eprintln!(
            "prove_verify_route_f32_roundtrip elapsed: {elapsed:?}, setup_n_instances={MIN_LIGERITO_INSTANCES}, m={MIN_LIGERITO_M}"
        );
    }

    #[test]
    fn bench_route_f32_effective_fanouts() {
        let effective_fanouts = [4, 8, 16, 22, 32];
        println!("\n=== Route F32 Benchmark (FANOUT=32, W_MAX=5) ===");
        println!("USEFUL_BITS = {USEFUL_BITS}");

        let (a_0, b_0) = build_matrices_f32();
        let a_nnz: usize = a_0.rows.iter().map(Vec::len).sum();
        let b_nnz: usize = b_0.rows.iter().map(Vec::len).sum();
        println!("A_nnz = {a_nnz}, B_nnz = {b_nnz}, total_nnz = {}", a_nnz + b_nnz);
        println!();

        let setup = RouteF32Setup::new(1);

        for &eff in &effective_fanouts {
            let witness = sample_route_witness_padded(eff);
            let t_prove = Instant::now();
            let proof = prove_route_f32(&setup, &[witness]);
            let prove_ms = t_prove.elapsed().as_secs_f64() * 1000.0;

            let t_verify = Instant::now();
            let claim = verify_route_f32(&setup, &proof)
                .unwrap_or_else(|err| panic!("route_f32 verifier rejected (eff={eff}): {err:?}"));
            let verify_ms = t_verify.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(claim, proof.claim);

            println!(
                "effective_fanout={eff:>2}: prove={prove_ms:>8.1}ms  verify={verify_ms:>6.1}ms"
            );
        }
        println!("=== End Benchmark ===\n");
    }

    fn one_block_witness_bool_full() -> Vec<bool> {
        let witness = sample_route_witness_full();
        witness_to_bool_vec(&witness)
    }

    fn one_block_witness_bool_padded(effective_fanout: usize) -> Vec<bool> {
        let witness = sample_route_witness_padded(effective_fanout);
        witness_to_bool_vec(&witness)
    }

    fn witness_to_bool_vec(witness: &RouteF32Witness) -> Vec<bool> {
        let mut z_u64 = vec![0u64; U64_PER_BLOCK];
        let mut a_u64 = vec![0u64; U64_PER_BLOCK];
        let mut b_u64 = vec![0u64; U64_PER_BLOCK];
        fill_block_witness_f32(
            &witness.key,
            &witness.mask,
            &witness.children,
            &mut z_u64,
            &mut a_u64,
            &mut b_u64,
        );

        let mut z = vec![false; K * (1 << 3)];
        for bit in 0..K {
            z[bit] = ((z_u64[bit >> 6] >> (bit & 63)) & 1) != 0;
        }
        z
    }

    fn sample_route_witness_full() -> RouteF32Witness {
        let mut key = [false; KEY_BITS];
        let mut mask = [false; KEY_BITS];
        // child index = 0b10110 = 22, so key bits [0..5] = [0,1,1,0,1]
        key[0] = false;
        key[1] = true;
        key[2] = true;
        key[3] = false;
        key[4] = true;
        for j in 0..W_MAX {
            mask[ROUTE_MASK_POSITIONS[j]] = true;
        }

        let children: Vec<[bool; DIGEST_BITS]> = (0..FANOUT)
            .map(|child| std::array::from_fn(|bit| ((child * 17 + bit) & 1) != 0))
            .collect();
        RouteF32Witness::new(key, mask, children)
    }

    fn sample_route_witness_padded(effective_fanout: usize) -> RouteF32Witness {
        let mut key = [false; KEY_BITS];
        let mut mask = [false; KEY_BITS];
        let w = (effective_fanout as f64).log2().ceil() as usize;
        let target_child = effective_fanout / 2;
        for j in 0..w.min(W_MAX) {
            let bit_val = ((target_child >> j) & 1) != 0;
            key[ROUTE_MASK_POSITIONS[j]] = bit_val;
            mask[ROUTE_MASK_POSITIONS[j]] = true;
        }
        for j in w..W_MAX {
            mask[ROUTE_MASK_POSITIONS[j]] = true;
        }

        let effective_children: Vec<[bool; DIGEST_BITS]> = (0..effective_fanout)
            .map(|child| std::array::from_fn(|bit| ((child * 17 + bit) & 1) != 0))
            .collect();
        RouteF32Witness::new_padded(key, mask, &effective_children, effective_fanout)
    }

    fn time_it<T>(f: impl FnOnce() -> T) -> (Duration, T) {
        let start = Instant::now();
        let output = f();
        (start.elapsed(), output)
    }
}
