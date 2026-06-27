use super::ref_witness::{bytes_to_logical_state, Digest, RefWitness};
use super::route_f32::{self as route, RouteF32Setup as RouteSetup, RouteF32Witness as RouteWitness};
use super::schedule::MhotHashSchedule;
use crate::prover::{prove_fast_core, quirky_x_outer_full, ProveCore};
use crate::r1cs_hashes::keccak::{State, STATE_BITS};
use crate::r1cs_hashes::keccak3::{
    generate_witness_with_ab_packed_and_lincheck, KeccakLincheckCircuit, KeccakSetup,
};
use flock_core::challenger::{Challenger, FsChallenger};
use flock_core::lincheck::LincheckProof;
use flock_core::pcs::{self, BatchOpeningProofLigerito, Commitment, PcsParams};
use flock_core::proof::R1csClaim;
use flock_core::r1cs::BlockR1cs;
use flock_core::verifier::VerifyError;
use flock_core::zerocheck::{self, ZerocheckProof};

const TRANSCRIPT_LABEL: &[u8] = b"mhot-multi-v0";
const MIN_LIGERITO_KECCAKS: usize = 49;

pub struct MhotMultiProof {
    pub hash_zc: ZerocheckProof,
    pub hash_lc: LincheckProof,
    pub hash_pcs: BatchOpeningProofLigerito,
    pub hash_commitment: Commitment,
    pub hash_claim: R1csClaim,
    pub route_zc: ZerocheckProof,
    pub route_lc: LincheckProof,
    pub route_pcs: BatchOpeningProofLigerito,
    pub route_commitment: Commitment,
    pub route_claim: R1csClaim,
    pub expected_root: Digest,
    pub n_keccaks: usize,
    pub n_routes: usize,
}

pub fn prove_multi(
    sched: &MhotHashSchedule,
    hash_witness: &RefWitness,
    route_witnesses: &[RouteWitness],
) -> MhotMultiProof {
    assert!(
        !sched.hash_atoms.is_empty(),
        "schedule must have at least one hash atom"
    );
    assert!(
        !route_witnesses.is_empty(),
        "multi-base PoC needs at least one route witness"
    );
    assert_eq!(
        hash_witness.atom_states.len(),
        sched.hash_atoms.len(),
        "hash witness atom_states must match schedule atoms"
    );

    let mut challenger = FsChallenger::new(TRANSCRIPT_LABEL);

    let n_keccaks = sched.hash_atoms.len();
    let setup_n_keccaks = hash_setup_n_keccaks(n_keccaks);
    let mut initial_states: Vec<State> = sched
        .hash_atoms
        .iter()
        .map(|atom| bytes_to_logical_state(&hash_witness.atom_states[atom.atom_id]))
        .collect();
    initial_states.resize(setup_n_keccaks, [false; STATE_BITS]);
    let hash_setup = KeccakSetup::new(setup_n_keccaks);
    let (hash_z, hash_a, hash_b, hash_zlc) =
        generate_witness_with_ab_packed_and_lincheck(&initial_states, hash_setup.n_blocks_log());
    let hash_core = prove_fast_core(
        &hash_setup.r1cs,
        &hash_setup.pcs_params,
        hash_z,
        hash_a,
        hash_b,
        hash_zlc,
        &KeccakLincheckCircuit,
        &mut challenger,
    );

    let route_setup = RouteSetup::new(route_witnesses.len());
    let (route_z, route_a, route_b, route_zlc) =
        route::generate_witness_with_ab_packed_and_lincheck(
            route_witnesses,
            route_setup.n_blocks_log(),
        );
    let route_core = prove_fast_core(
        &route_setup.r1cs,
        &route_setup.pcs_params,
        route_z,
        route_a,
        route_b,
        route_zlc,
        route_setup.r1cs.csc_lincheck_circuit(),
        &mut challenger,
    );

    let pcs_challenger = challenger.clone();
    let mut hash_pcs_challenger = fork_pcs_challenger(&pcs_challenger, b"hash");
    let mut route_pcs_challenger = fork_pcs_challenger(&pcs_challenger, b"route");

    let hash_open = open_core_ligerito(
        &hash_setup.r1cs,
        &hash_setup.pcs_params,
        hash_core,
        &mut hash_pcs_challenger,
    );
    let route_open = open_core_ligerito(
        &route_setup.r1cs,
        &route_setup.pcs_params,
        route_core,
        &mut route_pcs_challenger,
    );

    MhotMultiProof {
        hash_zc: hash_open.zc_proof,
        hash_lc: hash_open.lc_proof,
        hash_pcs: hash_open.pcs_open,
        hash_commitment: hash_open.commitment,
        hash_claim: hash_open.claim,
        route_zc: route_open.zc_proof,
        route_lc: route_open.lc_proof,
        route_pcs: route_open.pcs_open,
        route_commitment: route_open.commitment,
        route_claim: route_open.claim,
        expected_root: hash_witness.expected_root,
        n_keccaks,
        n_routes: route_witnesses.len(),
    }
}

pub fn verify_multi(proof: &MhotMultiProof) -> Result<(), VerifyError> {
    verify_multi_with_label(proof, TRANSCRIPT_LABEL)
}

fn verify_multi_with_label(proof: &MhotMultiProof, label: &[u8]) -> Result<(), VerifyError> {
    let mut challenger = FsChallenger::new(label);

    let hash_setup = KeccakSetup::new(hash_setup_n_keccaks(proof.n_keccaks));
    let (hash_ab, hash_c) = flock_core::verifier::verify_core(
        &hash_setup.r1cs,
        &proof.hash_zc,
        &proof.hash_lc,
        &proof.hash_commitment,
        &KeccakLincheckCircuit,
        &mut challenger,
    )?;

    let route_setup = RouteSetup::new(proof.n_routes);
    let (route_ab, route_c) = flock_core::verifier::verify_core(
        &route_setup.r1cs,
        &proof.route_zc,
        &proof.route_lc,
        &proof.route_commitment,
        route_setup.r1cs.csc_lincheck_circuit(),
        &mut challenger,
    )?;

    let pcs_challenger = challenger.clone();
    let mut hash_pcs_challenger = fork_pcs_challenger(&pcs_challenger, b"hash");
    let mut route_pcs_challenger = fork_pcs_challenger(&pcs_challenger, b"route");

    verify_core_opening_ligerito(
        &hash_setup.r1cs,
        &hash_setup.pcs_params,
        &proof.hash_commitment,
        &proof.hash_pcs,
        &hash_ab,
        &hash_c,
        &mut hash_pcs_challenger,
    )?;
    verify_core_opening_ligerito(
        &route_setup.r1cs,
        &route_setup.pcs_params,
        &proof.route_commitment,
        &proof.route_pcs,
        &route_ab,
        &route_c,
        &mut route_pcs_challenger,
    )?;

    Ok(())
}

struct OpenedCore {
    zc_proof: ZerocheckProof,
    lc_proof: LincheckProof,
    pcs_open: BatchOpeningProofLigerito,
    commitment: Commitment,
    claim: R1csClaim,
}

fn open_core_ligerito(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    core: ProveCore,
    challenger: &mut FsChallenger,
) -> OpenedCore {
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

    let padding = zerocheck::PaddingSpec {
        k_log: r1cs.k_log,
        useful_bits_per_block: r1cs.useful_bits,
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
        &ligerito_prover_config(r1cs, pcs_params),
        challenger,
    );

    OpenedCore {
        zc_proof,
        lc_proof,
        pcs_open,
        commitment,
        claim: R1csClaim { ab, c },
    }
}

fn verify_core_opening_ligerito(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    commitment: &Commitment,
    pcs_open: &BatchOpeningProofLigerito,
    ab: &flock_core::proof::ZClaim,
    c: &flock_core::proof::ZClaim,
    challenger: &mut FsChallenger,
) -> Result<(), VerifyError> {
    let z_skips = [ab.point.z_skip, c.point.z_skip];
    let values = [ab.value, c.value];
    let ab_x_outer = quirky_x_outer_full(&ab.point);
    let c_x_outer = quirky_x_outer_full(&c.point);
    let x_outers = [ab_x_outer.as_slice(), c_x_outer.as_slice()];
    pcs::verify_opening_batch_ligerito_mixed(
        commitment,
        &values,
        &z_skips,
        &x_outers,
        &[],
        pcs_open,
        &ligerito_verifier_config(r1cs, pcs_params),
        challenger,
    )
    .map_err(VerifyError::PcsAb)
}

fn ligerito_prover_config(r1cs: &BlockR1cs, pcs_params: &PcsParams) -> pcs::ligerito::ProverConfig {
    let log_n = r1cs.m - pcs::LOG_PACKING;
    pcs::ligerito::prover_config_for(log_n, pcs_params.log_batch_size, pcs_params.profile)
        .expect("Ligerito default prover config")
}

fn ligerito_verifier_config(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
) -> pcs::ligerito::VerifierConfig {
    let log_n = r1cs.m - pcs::LOG_PACKING;
    pcs::ligerito::verifier_config_for(log_n, pcs_params.log_batch_size, pcs_params.profile)
        .expect("Ligerito default verifier config")
}

fn hash_setup_n_keccaks(n_keccaks: usize) -> usize {
    n_keccaks.max(MIN_LIGERITO_KECCAKS)
}

fn fork_pcs_challenger(parent: &FsChallenger, base_label: &[u8]) -> FsChallenger {
    let mut challenger = parent.clone();
    challenger.observe_label(b"mhot-multi-pcs-fork-v0");
    challenger.observe_label(base_label);
    challenger
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mhot::ref_witness::build_ref_witness;
    use std::time::Instant;

    #[test]
    fn multi_prove_verify_roundtrip() {
        let start = Instant::now();
        let proof = make_valid_multi_proof();
        verify_multi(&proof).unwrap_or_else(|err| panic!("multi verifier rejected: {err:?}"));
        eprintln!(
            "multi_prove_verify_roundtrip elapsed: {:?}, n_keccaks={}, n_routes={}",
            start.elapsed(),
            proof.n_keccaks,
            proof.n_routes
        );
    }

    #[test]
    fn multi_verify_rejects_wrong_label() {
        let proof = make_valid_multi_proof();
        let err = verify_multi_with_label(&proof, b"mhot-multi-WRONG")
            .expect_err("wrong transcript label must be rejected");
        eprintln!("multi_verify_rejects_wrong_label: {err:?}");
    }

    #[test]
    fn multi_verify_rejects_tampered_commitment() {
        let mut proof = make_valid_multi_proof();
        proof.hash_commitment.root[0] ^= 1;
        let err = verify_multi(&proof).expect_err("tampered hash commitment must be rejected");
        eprintln!("multi_verify_rejects_tampered_commitment: {err:?}");
    }

    #[test]
    fn multi_verify_rejects_swapped_base_order() {
        let mut proof = make_valid_multi_proof();
        std::mem::swap(&mut proof.hash_zc, &mut proof.route_zc);
        std::mem::swap(&mut proof.hash_lc, &mut proof.route_lc);
        std::mem::swap(&mut proof.hash_pcs, &mut proof.route_pcs);
        std::mem::swap(&mut proof.hash_commitment, &mut proof.route_commitment);
        std::mem::swap(&mut proof.hash_claim, &mut proof.route_claim);

        match std::panic::catch_unwind(|| verify_multi(&proof)) {
            Ok(Err(err)) => eprintln!("multi_verify_rejects_swapped_base_order: {err:?}"),
            Ok(Ok(())) => panic!("swapped base order must be rejected"),
            Err(_) => eprintln!("multi_verify_rejects_swapped_base_order: panic"),
        }
    }

    fn make_valid_multi_proof() -> MhotMultiProof {
        let sched = MhotHashSchedule::from_fanouts(&[4, 2]);
        let hash_witness = build_ref_witness(&sched, 42);
        let route_witnesses = route_witnesses_for_schedule(&sched);
        prove_multi(&sched, &hash_witness, &route_witnesses)
    }

    fn route_witnesses_for_schedule(sched: &MhotHashSchedule) -> Vec<RouteWitness> {
        sched
            .fanouts
            .iter()
            .enumerate()
            .map(|(node, _)| route_witness_for_node(node))
            .collect()
    }

    fn route_witness_for_node(node: usize) -> RouteWitness {
        let mut key = [false; route::KEY_BITS];
        let mut mask = [false; route::KEY_BITS];
        key[0] = (node & 1) != 0;
        key[1] = true;
        mask[0] = true;
        mask[1] = true;

        let children: Vec<[bool; route::DIGEST_BITS]> = (0..route::FANOUT)
            .map(|child| std::array::from_fn(|bit| ((node * 31 + child * 17 + bit) & 1) != 0))
            .collect();
        RouteWitness::new(key, mask, children)
    }
}
