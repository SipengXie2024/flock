use super::ref_witness::{bytes_to_logical_state, Digest, RefWitness};
use super::route_f32::{self as route, RouteF32Setup, RouteF32Witness};
use super::schedule::MhotHashSchedule;
use crate::prover::{prove_fast_core, prove_fast_core_with_block_count, quirky_x_outer_full, ProveCore};
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

const TRANSCRIPT_LABEL: &[u8] = b"mhot-multiproof-v0";
const MIN_LIGERITO_KECCAKS: usize = 49;

pub struct MhotPathInput {
    pub schedule: MhotHashSchedule,
    pub hash_witness: RefWitness,
    pub route_witnesses: Vec<RouteF32Witness>,
}

pub struct MhotMultiproof {
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
    pub root: Digest,
    pub n_keccaks: usize,
    pub n_routes: usize,
    pub n_paths: usize,
}

pub fn prove_mhot_multiproof(paths: &[MhotPathInput]) -> MhotMultiproof {
    assert!(!paths.is_empty(), "multiproof needs at least one path");

    let root = paths[0].hash_witness.expected_root;
    for (i, path) in paths.iter().enumerate().skip(1) {
        assert_eq!(
            path.hash_witness.expected_root, root,
            "path {i} root differs from path 0 — all paths must share the same root"
        );
    }

    let mut all_initial_states: Vec<State> = Vec::new();
    let mut all_route_witnesses: Vec<RouteF32Witness> = Vec::new();
    let mut total_keccaks = 0usize;

    for path in paths {
        assert_eq!(
            path.hash_witness.atom_states.len(),
            path.schedule.hash_atoms.len(),
        );
        for atom in &path.schedule.hash_atoms {
            all_initial_states.push(bytes_to_logical_state(
                &path.hash_witness.atom_states[atom.atom_id],
            ));
        }
        total_keccaks += path.schedule.hash_atoms.len();
        all_route_witnesses.extend(path.route_witnesses.iter().cloned());
    }

    let mut challenger = FsChallenger::new(TRANSCRIPT_LABEL);
    challenger.observe_bytes(&root);

    let setup_n_keccaks = total_keccaks.max(MIN_LIGERITO_KECCAKS);
    all_initial_states.resize(setup_n_keccaks, [false; STATE_BITS]);
    let hash_setup = KeccakSetup::new(setup_n_keccaks);
    let (hash_z, hash_a, hash_b, hash_zlc) =
        generate_witness_with_ab_packed_and_lincheck(&all_initial_states, hash_setup.n_blocks_log());
    let n_real_blocks = (total_keccaks + 2) / 3;
    let hash_core = prove_fast_core_with_block_count(
        &hash_setup.r1cs,
        &hash_setup.pcs_params,
        hash_z,
        hash_a,
        hash_b,
        hash_zlc,
        &KeccakLincheckCircuit,
        Some(n_real_blocks),
        &mut challenger,
    );

    let total_routes = all_route_witnesses.len();
    let route_setup = RouteF32Setup::new(total_routes);
    let (route_z, route_a, route_b, route_zlc) =
        route::generate_witness_with_ab_packed_and_lincheck(
            &all_route_witnesses,
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
        n_real_blocks,
        &mut hash_pcs_challenger,
    );
    let route_open = open_core_ligerito(
        &route_setup.r1cs,
        &route_setup.pcs_params,
        route_core,
        0,
        &mut route_pcs_challenger,
    );

    MhotMultiproof {
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
        root,
        n_keccaks: total_keccaks,
        n_routes: total_routes,
        n_paths: paths.len(),
    }
}

pub fn verify_mhot_multiproof(proof: &MhotMultiproof) -> Result<(), VerifyError> {
    let mut challenger = FsChallenger::new(TRANSCRIPT_LABEL);
    challenger.observe_bytes(&proof.root);

    let setup_n_keccaks = proof.n_keccaks.max(MIN_LIGERITO_KECCAKS);
    let hash_setup = KeccakSetup::new(setup_n_keccaks);
    let (hash_ab, hash_c) = flock_core::verifier::verify_core(
        &hash_setup.r1cs,
        &proof.hash_zc,
        &proof.hash_lc,
        &proof.hash_commitment,
        &KeccakLincheckCircuit,
        &mut challenger,
    )?;

    let route_setup = RouteF32Setup::new(proof.n_routes);
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
    n_real_blocks_hint: usize,
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
        n_real_blocks: if n_real_blocks_hint > 0 {
            Some(n_real_blocks_hint)
        } else {
            None
        },
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

fn fork_pcs_challenger(parent: &FsChallenger, base_label: &[u8]) -> FsChallenger {
    let mut challenger = parent.clone();
    challenger.observe_label(b"mhot-multiproof-pcs-fork-v0");
    challenger.observe_label(base_label);
    challenger
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mhot::ref_witness::build_ref_witness;

    fn make_path_input(fanouts: &[usize], seed: u64) -> MhotPathInput {
        let sched = MhotHashSchedule::from_fanouts(fanouts);
        let hash_witness = build_ref_witness(&sched, seed);
        let route_witnesses = make_route_witnesses(&sched);
        MhotPathInput {
            schedule: sched,
            hash_witness,
            route_witnesses,
        }
    }

    fn make_route_witnesses(sched: &MhotHashSchedule) -> Vec<RouteF32Witness> {
        sched
            .fanouts
            .iter()
            .enumerate()
            .map(|(node, &fanout)| {
                let mut key = [false; route::KEY_BITS];
                let mut mask = [false; route::KEY_BITS];
                key[0] = (node & 1) != 0;
                key[1] = true;
                mask[0] = true;
                mask[1] = true;
                let children: Vec<[bool; route::DIGEST_BITS]> = (0..fanout.min(route::FANOUT))
                    .map(|c| std::array::from_fn(|b| ((node * 31 + c * 17 + b) & 1) != 0))
                    .collect();
                RouteF32Witness::new_padded(key, mask, &children, fanout.min(route::FANOUT))
            })
            .collect()
    }

    fn make_same_root_paths(n_paths: usize, fanouts: &[usize]) -> Vec<MhotPathInput> {
        (0..n_paths)
            .map(|_| make_path_input(fanouts, 42))
            .collect()
    }

    #[test]
    fn single_path_roundtrip() {
        let paths = make_same_root_paths(1, &[4, 2]);
        let proof = prove_mhot_multiproof(&paths);
        assert_eq!(proof.n_paths, 1);
        verify_mhot_multiproof(&proof).expect("single-path multiproof must verify");
    }

    #[test]
    fn multi_path_roundtrip() {
        let paths = make_same_root_paths(4, &[4, 2]);
        let proof = prove_mhot_multiproof(&paths);
        assert_eq!(proof.n_paths, 4);
        verify_mhot_multiproof(&proof).expect("4-path multiproof must verify");
    }

    #[test]
    fn rejects_tampered_commitment() {
        let paths = make_same_root_paths(2, &[4, 2]);
        let mut proof = prove_mhot_multiproof(&paths);
        proof.hash_commitment.root[0] ^= 1;
        verify_mhot_multiproof(&proof).expect_err("tampered commitment must fail");
    }

    #[test]
    fn rejects_tampered_root() {
        let paths = make_same_root_paths(2, &[4, 2]);
        let mut proof = prove_mhot_multiproof(&paths);
        proof.root[0] ^= 1;
        verify_mhot_multiproof(&proof).expect_err("tampered root must fail");
    }

    #[test]
    #[should_panic(expected = "root differs")]
    fn rejects_mixed_roots_at_prove() {
        let mut paths = make_same_root_paths(2, &[4, 2]);
        paths[1] = make_path_input(&[4, 2], 99);
        prove_mhot_multiproof(&paths);
    }

    #[test]
    fn rejects_swapped_bases() {
        let paths = make_same_root_paths(2, &[4, 2]);
        let mut proof = prove_mhot_multiproof(&paths);
        std::mem::swap(&mut proof.hash_zc, &mut proof.route_zc);
        std::mem::swap(&mut proof.hash_lc, &mut proof.route_lc);
        std::mem::swap(&mut proof.hash_pcs, &mut proof.route_pcs);
        std::mem::swap(&mut proof.hash_commitment, &mut proof.route_commitment);
        std::mem::swap(&mut proof.hash_claim, &mut proof.route_claim);

        match std::panic::catch_unwind(|| verify_mhot_multiproof(&proof)) {
            Ok(Err(_)) => {}
            Ok(Ok(())) => panic!("swapped bases must be rejected"),
            Err(_) => {}
        }
    }
}
