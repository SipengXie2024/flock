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
pub const FANOUT: usize = 4;
pub const W_MAX: usize = 2;

pub const KEY_BASE: usize = 0;
pub const Z_CONST_POS: usize = KEY_BASE + KEY_BITS;
pub const SELECTED_IN_BASE: usize = Z_CONST_POS + 1;
pub const MASK_BASE: usize = SELECTED_IN_BASE + DIGEST_BITS;
pub const CHILDREN_BASE: usize = MASK_BASE + KEY_BITS;

pub const CHILD_DIGEST_OFFSET: usize = 0;
pub const EXTRACTED_OFFSET: usize = CHILD_DIGEST_OFFSET + DIGEST_BITS;
pub const MATCH_RESULT_OFFSET: usize = EXTRACTED_OFFSET + W_MAX;
pub const TAKE_OFFSET: usize = MATCH_RESULT_OFFSET + 1;
pub const FOUND_STATE_OFFSET: usize = TAKE_OFFSET + 1;
pub const MUX_DELTA_OFFSET: usize = FOUND_STATE_OFFSET + 1;
pub const CHILD_STRIDE: usize = MUX_DELTA_OFFSET + DIGEST_BITS;

pub const FOUND_OUT_FINAL_POS: usize = CHILDREN_BASE + FANOUT * CHILD_STRIDE;
pub const SELECTED_OUT_FINAL_BASE: usize = FOUND_OUT_FINAL_POS + 1;
pub const USEFUL_BITS: usize = SELECTED_OUT_FINAL_BASE + DIGEST_BITS;

const ROUTE_MASK_POSITIONS: [usize; W_MAX] = [0, 1];
const TRANSCRIPT_LABEL: &[u8] = b"mhot-route-v0";
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

pub fn build_matrices() -> (SparseBinaryMatrix, SparseBinaryMatrix) {
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

        let eq0 = eq_expr(child, 0);
        let eq1 = eq_expr(child, 1);
        set_mul(&mut a_rows, &mut b_rows, match_result_pos(child), eq0, eq1);

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
    set_mul(
        &mut a_rows,
        &mut b_rows,
        Z_CONST_POS,
        vec![FOUND_OUT_FINAL_POS],
        vec![Z_CONST_POS],
    );

    for bit in 0..DIGEST_BITS {
        set_linear(
            &mut a_rows,
            &mut b_rows,
            selected_out_final_pos(bit),
            selected_state_expr(FANOUT, bit),
        );
    }

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

pub fn build_block_r1cs(n_blocks_log: usize) -> BlockR1cs {
    let (a_0, b_0) = build_matrices();
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
pub struct RouteSetup {
    pub n_instances: usize,
    pub setup_n_instances: usize,
    pub r1cs: BlockR1cs,
    pub pcs_params: PcsParams,
}

impl RouteSetup {
    pub fn new(n_instances: usize) -> Self {
        assert!(n_instances >= 1, "n_instances must be >= 1");
        let setup_n_instances = route_setup_n_instances(n_instances);
        let n_blocks_log = setup_n_instances.trailing_zeros() as usize;
        let r1cs = build_block_r1cs(n_blocks_log);
        r1cs.csc_lincheck_circuit();
        flock_core::scratch::prewarm_prover(r1cs.m);
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
pub struct RouteWitness {
    pub key: [bool; KEY_BITS],
    pub mask: [bool; KEY_BITS],
    pub children: Vec<[bool; DIGEST_BITS]>,
}

impl RouteWitness {
    pub fn new(
        key: [bool; KEY_BITS],
        mask: [bool; KEY_BITS],
        children: Vec<[bool; DIGEST_BITS]>,
    ) -> Self {
        assert_eq!(children.len(), FANOUT, "F_route PoC fanout is fixed at 4");
        Self {
            key,
            mask,
            children,
        }
    }
}

pub struct RouteProof {
    pub zc_proof: zerocheck::ZerocheckProof,
    pub lc_proof: lincheck::LincheckProof,
    pub pcs_open: pcs::BatchOpeningProofLigerito,
    pub commitment: Commitment,
    pub claim: R1csClaim,
    pub n_instances: usize,
    pub setup_n_instances: usize,
}

pub fn generate_witness_with_ab_packed_and_lincheck(
    witnesses: &[RouteWitness],
    n_blocks_log: usize,
) -> (
    Vec<flock_core::field::F128>,
    Vec<flock_core::field::F128>,
    Vec<flock_core::field::F128>,
    Vec<u8>,
) {
    let padding = padding_route_witness();
    drive_witness_packed_and_lincheck(
        witnesses,
        Some(&padding),
        n_blocks_log,
        K_LOG,
        |witness, z_u64, a_u64, b_u64| {
            fill_block_witness(
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

pub fn prove_route(setup: &RouteSetup, witnesses: &[RouteWitness]) -> RouteProof {
    assert_eq!(
        witnesses.len(),
        setup.n_instances,
        "witness count must match RouteSetup::n_instances"
    );
    let (z_packed, a_packed, b_packed, z_lincheck) =
        generate_witness_with_ab_packed_and_lincheck(witnesses, setup.n_blocks_log());
    prove_route_from_parts(setup, z_packed, a_packed, b_packed, z_lincheck)
}

pub fn verify_route(setup: &RouteSetup, proof: &RouteProof) -> Result<R1csClaim, VerifyError> {
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

pub fn fill_block_witness(
    key: &[bool; KEY_BITS],
    mask: &[bool; KEY_BITS],
    children: &[[bool; DIGEST_BITS]],
    z_u64: &mut [u64],
    a_u64: &mut [u64],
    b_u64: &mut [u64],
) {
    assert_eq!(children.len(), FANOUT, "F_route PoC fanout is fixed at 4");
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

        let bit_eq_0 = extracted[0] ^ child_index_bit(child, 0) ^ true;
        let bit_eq_1 = extracted[1] ^ child_index_bit(child, 1) ^ true;
        let matches = bit_eq_0 & bit_eq_1;
        set_zab(
            match_result_pos(child),
            matches,
            bit_eq_0,
            bit_eq_1,
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
    set_zab(Z_CONST_POS, true, found, true, z_u64, a_u64, b_u64);

    for (bit, value) in selected.iter().copied().enumerate() {
        write_linear(selected_out_final_pos(bit), value, z_u64, a_u64, b_u64);
    }
}

fn prove_route_from_parts(
    setup: &RouteSetup,
    z_packed: Vec<F128>,
    a_packed: Vec<F128>,
    b_packed: Vec<F128>,
    z_lincheck: Vec<u8>,
) -> RouteProof {
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

    RouteProof {
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

fn padding_route_witness() -> RouteWitness {
    let key = [false; KEY_BITS];
    let mut mask = [false; KEY_BITS];
    mask[0] = true;
    mask[1] = true;
    RouteWitness::new(key, mask, vec![[false; DIGEST_BITS]; FANOUT])
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
    fn layout_constants_consistent() {
        assert_eq!(K, 32768);
        assert_eq!(K_SKIP, 6);
        assert_eq!(KEY_BASE, 0);
        assert_eq!(Z_CONST_POS, 256);
        assert_eq!(SELECTED_IN_BASE, 257);
        assert_eq!(MASK_BASE, 513);
        assert_eq!(CHILDREN_BASE, 769);
        assert_eq!(CHILD_STRIDE, 517);
        assert_eq!(FOUND_OUT_FINAL_POS, 2837);
        assert_eq!(SELECTED_OUT_FINAL_BASE, 2838);
        assert_eq!(USEFUL_BITS, 3094);
        assert!(USEFUL_BITS <= K);

        let (a_0, b_0) = build_matrices();
        let a_nnz: usize = a_0.rows.iter().map(Vec::len).sum();
        let b_nnz: usize = b_0.rows.iter().map(Vec::len).sum();
        let nonempty_rows = a_0
            .rows
            .iter()
            .zip(b_0.rows.iter())
            .filter(|(a, b)| !a.is_empty() || !b.is_empty())
            .count();
        eprintln!(
            "F_route layout: useful_bits={USEFUL_BITS}, nonempty_rows={nonempty_rows}, a_nnz={a_nnz}, b_nnz={b_nnz}, total_nnz={}",
            a_nnz + b_nnz
        );
        assert_eq!(nonempty_rows, 2838);
        assert_eq!(a_nnz, 3867);
        assert_eq!(b_nnz, 5403);

        for child in 0..FANOUT {
            assert_eq!(child_digest_pos(child, 0), child_base(child));
            assert_eq!(extracted_pos(child, 0), child_base(child) + 256);
            assert_eq!(match_result_pos(child), child_base(child) + 258);
            assert_eq!(take_pos(child), child_base(child) + 259);
            assert_eq!(found_state_pos(child), child_base(child) + 260);
            assert_eq!(mux_delta_pos(child, 0), child_base(child) + 261);
            assert_eq!(mux_delta_pos(child, 255), child_base(child + 1) - 1);
        }
    }

    #[test]
    fn witness_satisfies_r1cs() {
        let r1cs = build_block_r1cs(3);
        let z = one_block_witness_bool();
        assert_eq!(z.len(), r1cs.n());
        assert!(r1cs.satisfies(&z));
    }

    #[test]
    fn all_zero_witness_rejected() {
        let r1cs = build_block_r1cs(3);
        let z = vec![false; r1cs.n()];

        assert!(
            r1cs.satisfies(&z),
            "BlockR1cs::satisfies is matrix-only and does not enforce const_pin"
        );
        assert!(!satisfies_with_const_pin(&r1cs, &z));
    }

    #[test]
    fn wrong_key_witness_rejected() {
        let r1cs = build_block_r1cs(3);
        let mut z = one_block_witness_bool();
        assert!(r1cs.satisfies(&z));

        z[KEY_BASE] ^= true;
        assert!(!r1cs.satisfies(&z));
    }

    #[test]
    fn prove_verify_route_roundtrip_smoke() {
        let (elapsed, _) = time_it(|| {
            let setup = RouteSetup::new(1);
            assert_eq!(setup.setup_n_instances, MIN_LIGERITO_INSTANCES);
            assert_eq!(setup.r1cs.m, MIN_LIGERITO_M);
            let witness = sample_route_witness();
            let proof = prove_route(&setup, &[witness]);
            let claim = verify_route(&setup, &proof)
                .unwrap_or_else(|err| panic!("route verifier rejected: {err:?}"));
            assert_eq!(claim, proof.claim);
        });
        eprintln!(
            "prove_verify_route_roundtrip_smoke elapsed: {elapsed:?}, setup_n_instances={MIN_LIGERITO_INSTANCES}, m={MIN_LIGERITO_M}"
        );
    }

    #[test]
    fn wrong_key_prove_verify_rejected() {
        let (elapsed, _) = time_it(|| {
            let setup = RouteSetup::new(1);
            let witness = sample_route_witness();
            let (mut z_packed, a_packed, b_packed, mut z_lincheck) =
                generate_witness_with_ab_packed_and_lincheck(&[witness], setup.n_blocks_log());

            flip_packed_bit(&mut z_packed, KEY_BASE);
            flip_lincheck_bit(&mut z_lincheck, 0, KEY_BASE);

            let proof = prove_route_from_parts(&setup, z_packed, a_packed, b_packed, z_lincheck);
            let result = verify_route(&setup, &proof);
            assert!(
                result.is_err(),
                "verifier must reject a trace with a tampered key bit"
            );
        });
        eprintln!("wrong_key_prove_verify_rejected elapsed: {elapsed:?}");
    }

    fn one_block_witness_bool() -> Vec<bool> {
        let witness = sample_route_witness();

        let mut z_u64 = vec![0u64; U64_PER_BLOCK];
        let mut a_u64 = vec![0u64; U64_PER_BLOCK];
        let mut b_u64 = vec![0u64; U64_PER_BLOCK];
        fill_block_witness(
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

    fn sample_route_witness() -> RouteWitness {
        let mut key = [false; KEY_BITS];
        let mut mask = [false; KEY_BITS];
        key[0] = false;
        key[1] = true;
        mask[0] = true;
        mask[1] = true;

        let children: Vec<[bool; DIGEST_BITS]> = (0..FANOUT)
            .map(|child| std::array::from_fn(|bit| ((child * 17 + bit) & 1) != 0))
            .collect();
        RouteWitness::new(key, mask, children)
    }

    fn flip_packed_bit(z_packed: &mut [F128], global_bit: usize) {
        let packed_idx = global_bit / 128;
        let local = global_bit % 128;
        if local < 64 {
            z_packed[packed_idx].lo ^= 1u64 << local;
        } else {
            z_packed[packed_idx].hi ^= 1u64 << (local - 64);
        }
    }

    fn flip_lincheck_bit(z_lincheck: &mut [u8], block: usize, inner: usize) {
        let byte_idx = block / 8;
        let bit = block % 8;
        z_lincheck[byte_idx * K + inner] ^= 1u8 << bit;
    }

    fn satisfies_with_const_pin(r1cs: &BlockR1cs, z: &[bool]) -> bool {
        if !r1cs.satisfies(z) {
            return false;
        }
        match r1cs.const_pin {
            Some(pin) => (0..r1cs.n_outer()).all(|block| z[block * r1cs.k() + pin]),
            None => true,
        }
    }

    fn time_it<T>(f: impl FnOnce() -> T) -> (Duration, T) {
        let start = Instant::now();
        let output = f();
        (start.elapsed(), output)
    }
}
