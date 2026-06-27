// BACKUP SOLUTION for adaptive span: one R1CS block per single child match.
//
// Advantages over route_f32: K_LOG=11 (vs 15), NNZ=3630 (vs 178K), 76% util.
// At large batch sizes (2^14+ paths), smaller K may outperform fixed F=32.
//
// NOT the current default because of 3 soundness gaps:
//   1. child_index is a witness — malicious prover can forge it
//   2. No uniqueness constraint — same child_index reusable across blocks
//   3. found_in/selected_in chain relies on external glue (not in-block R1CS)
// All three need a wiring sumcheck or lookup argument to close.
//
// Revisit when: (a) wiring sumcheck lands, or (b) benchmarks at 2^16 batch
// show route_f32's NNZ becoming a bottleneck.
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

pub const K_LOG: usize = 11;
pub const K: usize = 1 << K_LOG;
pub const K_SKIP: usize = 6;
pub const U64_PER_BLOCK: usize = K / 64;

pub const KEY_BITS: usize = 256;
pub const DIGEST_BITS: usize = 256;
pub const W_MAX: usize = 5;
pub const MATCH_AUX_COUNT: usize = W_MAX - 2;

pub const KEY_BASE: usize = 0;
pub const MASK_BASE: usize = KEY_BASE + KEY_BITS;
pub const Z_CONST_POS: usize = MASK_BASE + KEY_BITS;
pub const CHILD_DIGEST_BASE: usize = Z_CONST_POS + 1;
pub const CHILD_INDEX_BASE: usize = CHILD_DIGEST_BASE + DIGEST_BITS;
pub const EXTRACTED_BASE: usize = CHILD_INDEX_BASE + W_MAX;
pub const MATCH_AUX_BASE: usize = EXTRACTED_BASE + W_MAX;
pub const MATCH_RESULT_POS: usize = MATCH_AUX_BASE + MATCH_AUX_COUNT;
pub const TAKE_POS: usize = MATCH_RESULT_POS + 1;
pub const FOUND_IN_POS: usize = TAKE_POS + 1;
pub const FOUND_OUT_POS: usize = FOUND_IN_POS + 1;
pub const SELECTED_IN_BASE: usize = FOUND_OUT_POS + 1;
pub const MUX_DELTA_BASE: usize = SELECTED_IN_BASE + DIGEST_BITS;
pub const SELECTED_OUT_BASE: usize = MUX_DELTA_BASE + DIGEST_BITS;
pub const USEFUL_BITS: usize = SELECTED_OUT_BASE + DIGEST_BITS;

const ROUTE_MASK_POSITIONS: [usize; W_MAX] = [0, 1, 2, 3, 4];
const TRANSCRIPT_LABEL: &[u8] = b"mhot-route-single-v0";
const MIN_LIGERITO_M: usize = 22;
const MIN_LIGERITO_INSTANCES: usize = 1 << (MIN_LIGERITO_M - K_LOG);

fn eq_expr(bit: usize) -> Vec<usize> {
    vec![Z_CONST_POS, EXTRACTED_BASE + bit, CHILD_INDEX_BASE + bit]
}

pub fn build_matrices_single() -> (SparseBinaryMatrix, SparseBinaryMatrix) {
    let mut a_rows: Vec<Vec<usize>> = vec![Vec::new(); K];
    let mut b_rows: Vec<Vec<usize>> = vec![Vec::new(); K];

    input_emit(&mut a_rows, &mut b_rows, KEY_BASE, KEY_BITS);
    input_emit(&mut a_rows, &mut b_rows, MASK_BASE, KEY_BITS);
    input_emit(&mut a_rows, &mut b_rows, CHILD_DIGEST_BASE, DIGEST_BITS);
    input_emit(&mut a_rows, &mut b_rows, CHILD_INDEX_BASE, W_MAX);
    input_emit(&mut a_rows, &mut b_rows, FOUND_IN_POS, 1);
    input_emit(&mut a_rows, &mut b_rows, SELECTED_IN_BASE, DIGEST_BITS);

    for j in 0..W_MAX {
        let key_bit = KEY_BASE + ROUTE_MASK_POSITIONS[j];
        let mask_bit = MASK_BASE + ROUTE_MASK_POSITIONS[j];
        set_mul(
            &mut a_rows,
            &mut b_rows,
            EXTRACTED_BASE + j,
            vec![key_bit],
            vec![mask_bit],
        );
    }

    // match = AND(eq(0), eq(1), eq(2), eq(3), eq(4))
    // Chain: aux0 = eq(0)*eq(1), aux1 = aux0*eq(2), aux2 = aux1*eq(3),
    //        match = aux2*eq(4)
    set_mul(
        &mut a_rows,
        &mut b_rows,
        MATCH_AUX_BASE,
        eq_expr(0),
        eq_expr(1),
    );
    set_mul(
        &mut a_rows,
        &mut b_rows,
        MATCH_AUX_BASE + 1,
        vec![MATCH_AUX_BASE],
        eq_expr(2),
    );
    set_mul(
        &mut a_rows,
        &mut b_rows,
        MATCH_AUX_BASE + 2,
        vec![MATCH_AUX_BASE + 1],
        eq_expr(3),
    );
    set_mul(
        &mut a_rows,
        &mut b_rows,
        MATCH_RESULT_POS,
        vec![MATCH_AUX_BASE + 2],
        eq_expr(4),
    );

    // take = match_result * (1 + found_in)
    set_mul(
        &mut a_rows,
        &mut b_rows,
        TAKE_POS,
        vec![MATCH_RESULT_POS],
        vec![Z_CONST_POS, FOUND_IN_POS],
    );

    // found_out = found_in + take (linear)
    set_linear(
        &mut a_rows,
        &mut b_rows,
        FOUND_OUT_POS,
        vec![FOUND_IN_POS, TAKE_POS],
    );

    // mux_delta[bit] = take * (child_digest[bit] + selected_in[bit])
    for bit in 0..DIGEST_BITS {
        set_mul(
            &mut a_rows,
            &mut b_rows,
            MUX_DELTA_BASE + bit,
            vec![TAKE_POS],
            vec![CHILD_DIGEST_BASE + bit, SELECTED_IN_BASE + bit],
        );
    }

    // selected_out[bit] = selected_in[bit] + mux_delta[bit] (linear)
    for bit in 0..DIGEST_BITS {
        set_linear(
            &mut a_rows,
            &mut b_rows,
            SELECTED_OUT_BASE + bit,
            vec![SELECTED_IN_BASE + bit, MUX_DELTA_BASE + bit],
        );
    }

    // const pin: Z_CONST * Z_CONST = Z_CONST (already implicitly done via
    // the pin mechanism, but we also need found_out to be constrained to
    // equal 1 at the end of the full chain -- that's done at the caller
    // level, not in the per-block R1CS).
    //
    // Actually, we need a self-consistency constraint for Z_CONST_POS.
    // Route.rs uses: set_mul(Z_CONST, [FOUND_OUT_FINAL], [Z_CONST])
    // which forces FOUND_OUT_FINAL=1. But here each block is independent,
    // so we just need Z_CONST * Z_CONST = Z_CONST for the const pin.
    // The const_pin mechanism handles this.

    set_mul(
        &mut a_rows,
        &mut b_rows,
        Z_CONST_POS,
        vec![Z_CONST_POS],
        vec![Z_CONST_POS],
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

fn set_linear(
    a_rows: &mut [Vec<usize>],
    b_rows: &mut [Vec<usize>],
    row: usize,
    expr: Vec<usize>,
) {
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
    let (a_0, b_0) = build_matrices_single();
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
pub struct RouteSingleSetup {
    pub n_instances: usize,
    pub setup_n_instances: usize,
    pub r1cs: BlockR1cs,
    pub pcs_params: PcsParams,
}

impl RouteSingleSetup {
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
pub struct RouteSingleWitness {
    pub key: [bool; KEY_BITS],
    pub mask: [bool; KEY_BITS],
    pub child_digest: [bool; DIGEST_BITS],
    pub child_index: usize,
    pub found_in: bool,
    pub selected_in: [bool; DIGEST_BITS],
}

pub struct RouteSingleProof {
    pub zc_proof: zerocheck::ZerocheckProof,
    pub lc_proof: lincheck::LincheckProof,
    pub pcs_open: pcs::BatchOpeningProofLigerito,
    pub commitment: Commitment,
    pub claim: R1csClaim,
    pub n_instances: usize,
    pub setup_n_instances: usize,
}

pub fn fill_block_witness_single(
    witness: &RouteSingleWitness,
    z_u64: &mut [u64],
    a_u64: &mut [u64],
    b_u64: &mut [u64],
) {
    assert_eq!(z_u64.len(), U64_PER_BLOCK);
    assert_eq!(a_u64.len(), U64_PER_BLOCK);
    assert_eq!(b_u64.len(), U64_PER_BLOCK);

    z_u64.fill(0);
    a_u64.fill(0);
    b_u64.fill(0);

    set_zab(Z_CONST_POS, true, true, true, z_u64, a_u64, b_u64);

    for bit in 0..KEY_BITS {
        write_input(KEY_BASE + bit, witness.key[bit], z_u64, a_u64, b_u64);
        write_input(MASK_BASE + bit, witness.mask[bit], z_u64, a_u64, b_u64);
    }

    for bit in 0..DIGEST_BITS {
        write_input(
            CHILD_DIGEST_BASE + bit,
            witness.child_digest[bit],
            z_u64,
            a_u64,
            b_u64,
        );
    }

    for j in 0..W_MAX {
        let idx_bit = ((witness.child_index >> j) & 1) != 0;
        write_input(CHILD_INDEX_BASE + j, idx_bit, z_u64, a_u64, b_u64);
    }

    write_input(FOUND_IN_POS, witness.found_in, z_u64, a_u64, b_u64);

    for bit in 0..DIGEST_BITS {
        write_input(
            SELECTED_IN_BASE + bit,
            witness.selected_in[bit],
            z_u64,
            a_u64,
            b_u64,
        );
    }

    let mut extracted = [false; W_MAX];
    for j in 0..W_MAX {
        let pos = ROUTE_MASK_POSITIONS[j];
        extracted[j] = witness.key[pos] & witness.mask[pos];
        set_zab(
            EXTRACTED_BASE + j,
            extracted[j],
            witness.key[pos],
            witness.mask[pos],
            z_u64,
            a_u64,
            b_u64,
        );
    }

    let mut bit_eq = [false; W_MAX];
    for j in 0..W_MAX {
        let idx_bit = ((witness.child_index >> j) & 1) != 0;
        bit_eq[j] = !(extracted[j] ^ idx_bit);
    }

    // match chain: aux0 = eq0*eq1, aux1 = aux0*eq2, aux2 = aux1*eq3,
    // match = aux2*eq4
    let aux0 = bit_eq[0] & bit_eq[1];
    set_zab(
        MATCH_AUX_BASE,
        aux0,
        bit_eq[0],
        bit_eq[1],
        z_u64,
        a_u64,
        b_u64,
    );
    let aux1 = aux0 & bit_eq[2];
    set_zab(
        MATCH_AUX_BASE + 1,
        aux1,
        aux0,
        bit_eq[2],
        z_u64,
        a_u64,
        b_u64,
    );
    let aux2 = aux1 & bit_eq[3];
    set_zab(
        MATCH_AUX_BASE + 2,
        aux2,
        aux1,
        bit_eq[3],
        z_u64,
        a_u64,
        b_u64,
    );
    let matches = aux2 & bit_eq[4];
    set_zab(
        MATCH_RESULT_POS,
        matches,
        aux2,
        bit_eq[4],
        z_u64,
        a_u64,
        b_u64,
    );

    let take = matches & !witness.found_in;
    set_zab(
        TAKE_POS,
        take,
        matches,
        !witness.found_in,
        z_u64,
        a_u64,
        b_u64,
    );

    let found_out = witness.found_in ^ take;
    write_linear(FOUND_OUT_POS, found_out, z_u64, a_u64, b_u64);

    for bit in 0..DIGEST_BITS {
        let mux_rhs = witness.child_digest[bit] ^ witness.selected_in[bit];
        let delta = take & mux_rhs;
        set_zab(
            MUX_DELTA_BASE + bit,
            delta,
            take,
            mux_rhs,
            z_u64,
            a_u64,
            b_u64,
        );
        let sel_out = witness.selected_in[bit] ^ delta;
        write_linear(SELECTED_OUT_BASE + bit, sel_out, z_u64, a_u64, b_u64);
    }
}

pub fn generate_witness_with_ab_packed_and_lincheck(
    witnesses: &[RouteSingleWitness],
    n_blocks_log: usize,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
    let padding = padding_witness();
    drive_witness_packed_and_lincheck(
        witnesses,
        Some(&padding),
        n_blocks_log,
        K_LOG,
        |witness, z_u64, a_u64, b_u64| {
            fill_block_witness_single(witness, z_u64, a_u64, b_u64);
        },
    )
}

pub fn prove_route_single(
    setup: &RouteSingleSetup,
    witnesses: &[RouteSingleWitness],
) -> RouteSingleProof {
    assert_eq!(
        witnesses.len(),
        setup.n_instances,
        "witness count must match RouteSingleSetup::n_instances"
    );
    let (z_packed, a_packed, b_packed, z_lincheck) =
        generate_witness_with_ab_packed_and_lincheck(witnesses, setup.n_blocks_log());
    prove_from_parts(setup, z_packed, a_packed, b_packed, z_lincheck)
}

pub fn verify_route_single(
    setup: &RouteSingleSetup,
    proof: &RouteSingleProof,
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

fn prove_from_parts(
    setup: &RouteSingleSetup,
    z_packed: Vec<F128>,
    a_packed: Vec<F128>,
    b_packed: Vec<F128>,
    z_lincheck: Vec<u8>,
) -> RouteSingleProof {
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

    RouteSingleProof {
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

fn padding_witness() -> RouteSingleWitness {
    let key = [false; KEY_BITS];
    let mut mask = [false; KEY_BITS];
    for j in 0..W_MAX {
        mask[ROUTE_MASK_POSITIONS[j]] = true;
    }
    RouteSingleWitness {
        key,
        mask,
        child_digest: [false; DIGEST_BITS],
        child_index: 0,
        found_in: false,
        selected_in: [false; DIGEST_BITS],
    }
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

fn write_input(
    pos: usize,
    value: bool,
    z_u64: &mut [u64],
    a_u64: &mut [u64],
    b_u64: &mut [u64],
) {
    set_zab(pos, value, value, true, z_u64, a_u64, b_u64);
}

fn write_linear(
    pos: usize,
    value: bool,
    z_u64: &mut [u64],
    a_u64: &mut [u64],
    b_u64: &mut [u64],
) {
    set_zab(pos, value, value, true, z_u64, a_u64, b_u64);
}

pub fn build_node_witnesses(
    key: [bool; KEY_BITS],
    mask: [bool; KEY_BITS],
    children: &[[bool; DIGEST_BITS]],
) -> Vec<RouteSingleWitness> {
    let fanout = children.len();
    let mut witnesses = Vec::with_capacity(fanout);
    let mut found = false;
    let mut selected = [false; DIGEST_BITS];

    for (child_idx, child_digest) in children.iter().enumerate() {
        witnesses.push(RouteSingleWitness {
            key,
            mask,
            child_digest: *child_digest,
            child_index: child_idx,
            found_in: found,
            selected_in: selected,
        });

        let mut extracted = [false; W_MAX];
        for j in 0..W_MAX {
            let pos = ROUTE_MASK_POSITIONS[j];
            extracted[j] = key[pos] & mask[pos];
        }

        let mut bit_eq = [false; W_MAX];
        for j in 0..W_MAX {
            let idx_bit = ((child_idx >> j) & 1) != 0;
            bit_eq[j] = !(extracted[j] ^ idx_bit);
        }
        let matches = bit_eq.iter().all(|&b| b);

        let take = matches & !found;
        let found_out = found ^ take;

        if take {
            selected = *child_digest;
        }
        found = found_out;
    }

    witnesses
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn layout_constants_consistent() {
        assert_eq!(K, 2048);
        assert_eq!(KEY_BASE, 0);
        assert_eq!(MASK_BASE, 256);
        assert_eq!(Z_CONST_POS, 512);
        assert_eq!(CHILD_DIGEST_BASE, 513);
        assert_eq!(CHILD_INDEX_BASE, 769);
        assert_eq!(EXTRACTED_BASE, 774);
        assert_eq!(MATCH_AUX_BASE, 779);
        assert_eq!(MATCH_RESULT_POS, 782);
        assert_eq!(TAKE_POS, 783);
        assert_eq!(FOUND_IN_POS, 784);
        assert_eq!(FOUND_OUT_POS, 785);
        assert_eq!(SELECTED_IN_BASE, 786);
        assert_eq!(MUX_DELTA_BASE, 1042);
        assert_eq!(SELECTED_OUT_BASE, 1298);
        assert_eq!(USEFUL_BITS, 1554);
        assert!(USEFUL_BITS <= K);

        let (a_0, b_0) = build_matrices_single();
        let a_nnz: usize = a_0.rows.iter().map(Vec::len).sum();
        let b_nnz: usize = b_0.rows.iter().map(Vec::len).sum();
        let nonempty_rows = a_0
            .rows
            .iter()
            .zip(b_0.rows.iter())
            .filter(|(a, b)| !a.is_empty() || !b.is_empty())
            .count();
        eprintln!(
            "route_single layout: useful_bits={USEFUL_BITS}, K={K}, \
             nonempty_rows={nonempty_rows}, a_nnz={a_nnz}, b_nnz={b_nnz}, \
             total_nnz={}",
            a_nnz + b_nnz
        );
    }

    #[test]
    fn witness_satisfies_r1cs() {
        let r1cs = build_block_r1cs(3);
        let witness = sample_single_witness();
        let z = one_block_witness_bool(&witness);
        assert_eq!(z.len(), r1cs.n());

        let a = r1cs.apply_a(&z);
        let b = r1cs.apply_b(&z);
        let c = r1cs.apply_c(&z);
        let mut fail_count = 0;
        for row in 0..K {
            if (a[row] & b[row]) != c[row] {
                if fail_count < 20 {
                    eprintln!(
                        "row {row}: a={}, b={}, c={}, a*b={}, z={}",
                        a[row] as u8,
                        b[row] as u8,
                        c[row] as u8,
                        (a[row] & b[row]) as u8,
                        z[row] as u8
                    );
                }
                fail_count += 1;
            }
        }
        if fail_count > 0 {
            eprintln!("total failing rows: {fail_count} (only first block shown)");
        }
        assert!(r1cs.satisfies(&z), "R1CS not satisfied for matching witness");
    }

    #[test]
    fn witness_non_matching_child() {
        let r1cs = build_block_r1cs(3);
        let witness = RouteSingleWitness {
            key: {
                let mut k = [false; KEY_BITS];
                k[0] = true;
                k[1] = true;
                k
            },
            mask: {
                let mut m = [false; KEY_BITS];
                for j in 0..W_MAX {
                    m[ROUTE_MASK_POSITIONS[j]] = true;
                }
                m
            },
            child_digest: std::array::from_fn(|bit| (bit & 1) != 0),
            child_index: 7,
            found_in: false,
            selected_in: [false; DIGEST_BITS],
        };
        let z = one_block_witness_bool(&witness);
        assert!(r1cs.satisfies(&z), "R1CS not satisfied for non-matching witness");
    }

    #[test]
    fn build_node_witnesses_roundtrip() {
        let mut key = [false; KEY_BITS];
        key[0] = false;
        key[1] = true;
        key[2] = true;
        let mut mask = [false; KEY_BITS];
        for j in 0..W_MAX {
            mask[ROUTE_MASK_POSITIONS[j]] = true;
        }
        let children: Vec<[bool; DIGEST_BITS]> = (0..4)
            .map(|c| std::array::from_fn(|b| ((c * 17 + b) & 1) != 0))
            .collect();

        let witnesses = build_node_witnesses(key, mask, &children);
        assert_eq!(witnesses.len(), 4);

        let r1cs = build_block_r1cs(3);
        for (i, w) in witnesses.iter().enumerate() {
            let z = one_block_witness_bool(w);
            assert!(
                r1cs.satisfies(&z),
                "R1CS not satisfied for child {i}"
            );
        }

        let last = &witnesses[witnesses.len() - 1];
        let mut z_u64 = vec![0u64; U64_PER_BLOCK];
        let mut a_u64 = vec![0u64; U64_PER_BLOCK];
        let mut b_u64 = vec![0u64; U64_PER_BLOCK];
        fill_block_witness_single(last, &mut z_u64, &mut a_u64, &mut b_u64);
        let found_out = ((z_u64[FOUND_OUT_POS >> 6] >> (FOUND_OUT_POS & 63)) & 1) != 0;
        eprintln!("found_out after last child: {found_out}");
    }

    #[test]
    fn prove_verify_single_roundtrip_smoke() {
        let (elapsed, _) = time_it(|| {
            let setup = RouteSingleSetup::new(1);
            eprintln!(
                "setup: n_instances=1, setup_n_instances={}, m={}, K_LOG={K_LOG}",
                setup.setup_n_instances,
                setup.r1cs.m
            );
            let witness = sample_single_witness();
            let proof = prove_route_single(&setup, &[witness]);
            let claim = verify_route_single(&setup, &proof)
                .unwrap_or_else(|err| panic!("route_single verifier rejected: {err:?}"));
            assert_eq!(claim, proof.claim);
        });
        eprintln!("prove_verify_single_roundtrip_smoke elapsed: {elapsed:?}");
    }

    #[test]
    fn prove_verify_fanout_22_node() {
        let (elapsed, _) = time_it(|| {
            let fanout = 22;
            let mut key = [false; KEY_BITS];
            key[0] = false;
            key[1] = true;
            key[2] = true;
            let mut mask = [false; KEY_BITS];
            for j in 0..W_MAX {
                mask[ROUTE_MASK_POSITIONS[j]] = true;
            }
            let children: Vec<[bool; DIGEST_BITS]> = (0..fanout)
                .map(|c| std::array::from_fn(|b| ((c * 17 + b) & 1) != 0))
                .collect();
            let witnesses = build_node_witnesses(key, mask, &children);
            assert_eq!(witnesses.len(), fanout);

            let setup = RouteSingleSetup::new(fanout);
            eprintln!(
                "fanout={fanout}: setup_n_instances={}, m={}",
                setup.setup_n_instances,
                setup.r1cs.m
            );
            let proof = prove_route_single(&setup, &witnesses);
            let claim = verify_route_single(&setup, &proof)
                .unwrap_or_else(|err| panic!("fanout-{fanout} verifier rejected: {err:?}"));
            assert_eq!(claim, proof.claim);
        });
        eprintln!("prove_verify_fanout_22_node elapsed: {elapsed:?}");
    }

    #[test]
    fn benchmark_fanouts() {
        let fanouts = [4, 8, 16, 22, 32];
        println!("\n=== Route 3a (single-child-step) benchmark ===");
        println!("K_LOG={K_LOG}, K={K}, USEFUL_BITS={USEFUL_BITS}, W_MAX={W_MAX}");
        println!("MIN_LIGERITO_M={MIN_LIGERITO_M}, MIN_LIGERITO_INSTANCES={MIN_LIGERITO_INSTANCES}");
        println!(
            "{:>7} {:>10} {:>10} {:>12} {:>12} {:>8}",
            "fanout", "n_inst", "setup_n", "prove_ms", "verify_ms", "m"
        );

        for &fanout in &fanouts {
            let mut key = [false; KEY_BITS];
            key[0] = false;
            key[1] = true;
            key[2] = true;
            let mut mask = [false; KEY_BITS];
            for j in 0..W_MAX {
                mask[ROUTE_MASK_POSITIONS[j]] = true;
            }
            let children: Vec<[bool; DIGEST_BITS]> = (0..fanout)
                .map(|c| std::array::from_fn(|b| ((c * 17 + b) & 1) != 0))
                .collect();
            let witnesses = build_node_witnesses(key, mask, &children);

            let setup = RouteSingleSetup::new(fanout);

            let prove_start = Instant::now();
            let proof = prove_route_single(&setup, &witnesses);
            let prove_ms = prove_start.elapsed().as_millis();

            let verify_start = Instant::now();
            let claim = verify_route_single(&setup, &proof)
                .unwrap_or_else(|err| panic!("fanout-{fanout} verifier rejected: {err:?}"));
            let verify_ms = verify_start.elapsed().as_millis();

            assert_eq!(claim, proof.claim);
            println!(
                "{:>7} {:>10} {:>10} {:>12} {:>12} {:>8}",
                fanout,
                fanout,
                setup.setup_n_instances,
                prove_ms,
                verify_ms,
                setup.r1cs.m
            );
        }
        println!("=== end benchmark ===\n");
    }

    fn sample_single_witness() -> RouteSingleWitness {
        let mut key = [false; KEY_BITS];
        key[0] = false;
        key[1] = true;
        let mut mask = [false; KEY_BITS];
        for j in 0..W_MAX {
            mask[ROUTE_MASK_POSITIONS[j]] = true;
        }
        RouteSingleWitness {
            key,
            mask,
            child_digest: std::array::from_fn(|bit| ((2 * 17 + bit) & 1) != 0),
            child_index: 2,
            found_in: false,
            selected_in: [false; DIGEST_BITS],
        }
    }

    fn one_block_witness_bool(witness: &RouteSingleWitness) -> Vec<bool> {
        let mut z_u64 = vec![0u64; U64_PER_BLOCK];
        let mut a_u64 = vec![0u64; U64_PER_BLOCK];
        let mut b_u64 = vec![0u64; U64_PER_BLOCK];
        fill_block_witness_single(witness, &mut z_u64, &mut a_u64, &mut b_u64);

        let mut z = vec![false; K * (1 << 3)];
        for bit in 0..K {
            z[bit] = ((z_u64[bit >> 6] >> (bit & 63)) & 1) != 0;
        }
        z
    }

    fn time_it<T>(f: impl FnOnce() -> T) -> (Duration, T) {
        let start = Instant::now();
        let output = f();
        (start.elapsed(), output)
    }
}
