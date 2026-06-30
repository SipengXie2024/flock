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

// Soundness status (multiproof = benchmark tool, NOT adversarial-sound):
// [CLOSED] Content: mask compactness + key validity enforced in route R1CS (1021 AND gates)
// [CLOSED] Topology: fanouts bound to Fiat-Shamir transcript (observe_wiring_topology)
// [OPEN]   Wiring + Binding: NOT enforced in-circuit. CPU-checked on prover side only.
//          Per-wire PackedDirectClaim was attempted but doesn't scale (O(N*wires) claims).
//          Correct solution: merkle_path_common shift-sumcheck → O(1) PD claims.
//          See prove_mhot_membership_sha256 for the sound membership proof implementation.
const TRANSCRIPT_LABEL: &[u8] = b"mhot-multiproof-v0";
const MIN_LIGERITO_KECCAKS: usize = 49;

fn observe_wiring_topology(challenger: &mut FsChallenger, all_fanouts: &[Vec<usize>]) {
    challenger.observe_label(b"mhot-wiring-topology-v0");
    challenger.observe_bytes(&(all_fanouts.len() as u64).to_le_bytes());
    for fanouts in all_fanouts {
        challenger.observe_bytes(&(fanouts.len() as u64).to_le_bytes());
        for &f in fanouts {
            challenger.observe_bytes(&(f as u64).to_le_bytes());
        }
    }
}

pub struct MhotPathInput {
    pub schedule: MhotHashSchedule,
    pub hash_witness: RefWitness,
    pub route_witnesses: Vec<RouteF32Witness>,
}

#[derive(serde::Serialize)]
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
    pub fanouts: Vec<Vec<usize>>,
}

impl MhotMultiproof {
    pub fn proof_size_bytes(&self) -> usize {
        bincode::serialized_size(self).unwrap_or(0) as usize
    }
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

    let all_fanouts: Vec<Vec<usize>> = paths.iter().map(|p| p.schedule.fanouts.clone()).collect();
    let schedules: Vec<MhotHashSchedule> = paths.iter().map(|p| p.schedule.clone()).collect();

    let total_keccaks: usize = paths.iter().map(|p| p.schedule.hash_atoms.len()).sum();
    let total_routes: usize = paths.iter().map(|p| p.route_witnesses.len()).sum();
    let mut all_initial_states = Vec::with_capacity(total_keccaks);
    let mut all_route_witnesses = Vec::with_capacity(total_routes);

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
        all_route_witnesses.extend(path.route_witnesses.iter().cloned());
    }

    let mut challenger = FsChallenger::new(TRANSCRIPT_LABEL);
    challenger.observe_bytes(&root);
    observe_wiring_topology(&mut challenger, &all_fanouts);

    let setup_n_keccaks = total_keccaks.max(MIN_LIGERITO_KECCAKS);
    all_initial_states.resize(setup_n_keccaks, [false; STATE_BITS]);
    let hash_setup = KeccakSetup::cached(setup_n_keccaks);
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
    let route_setup = RouteF32Setup::cached(total_routes);
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
        &[],
        &mut hash_pcs_challenger,
    );
    let route_open = open_core_ligerito(
        &route_setup.r1cs,
        &route_setup.pcs_params,
        route_core,
        total_routes,
        &[],
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
        fanouts: all_fanouts,
    }
}

pub fn prove_mhot_multiproof_shared(paths: &[MhotPathInput]) -> MhotMultiproof {
    assert!(!paths.is_empty(), "multiproof needs at least one path");

    let root = paths[0].hash_witness.expected_root;
    for (i, path) in paths.iter().enumerate().skip(1) {
        assert_eq!(
            path.hash_witness.expected_root, root,
            "path {i} root differs from path 0 — all paths must share the same root"
        );
    }

    let all_fanouts: Vec<Vec<usize>> = paths.iter().map(|p| p.schedule.fanouts.clone()).collect();
    let schedules: Vec<MhotHashSchedule> = paths.iter().map(|p| p.schedule.clone()).collect();

    let mut all_atom_states_full: Vec<[u8; 200]> = Vec::new();
    for path in paths {
        assert_eq!(
            path.hash_witness.atom_states.len(),
            path.schedule.hash_atoms.len(),
        );
        for atom in &path.schedule.hash_atoms {
            all_atom_states_full.push(path.hash_witness.atom_states[atom.atom_id]);
        }
    }

    let mut seen_atoms: std::collections::HashMap<[u8; 200], usize> =
        std::collections::HashMap::new();
    let mut all_initial_states: Vec<State> = Vec::new();
    let mut unique_keccaks = 0usize;

    for state_bytes in &all_atom_states_full {
        if let std::collections::hash_map::Entry::Vacant(e) = seen_atoms.entry(*state_bytes) {
            e.insert(unique_keccaks);
            all_initial_states.push(bytes_to_logical_state(state_bytes));
            unique_keccaks += 1;
        }
    }

    let mut seen_routes: std::collections::HashSet<Vec<u8>> = std::collections::HashSet::new();
    let mut all_route_witnesses: Vec<RouteF32Witness> = Vec::new();

    for path in paths {
        for rw in &path.route_witnesses {
            let key_bytes = route_witness_identity(rw);
            if seen_routes.insert(key_bytes) {
                all_route_witnesses.push(rw.clone());
            }
        }
    }

    let total_keccaks = unique_keccaks;
    let total_routes = all_route_witnesses.len();

    let mut challenger = FsChallenger::new(TRANSCRIPT_LABEL);
    challenger.observe_bytes(&root);
    observe_wiring_topology(&mut challenger, &all_fanouts);

    let setup_n_keccaks = total_keccaks.max(MIN_LIGERITO_KECCAKS);
    all_initial_states.resize(setup_n_keccaks, [false; STATE_BITS]);
    let hash_setup = KeccakSetup::cached(setup_n_keccaks);
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

    let route_setup = RouteF32Setup::cached(total_routes);
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
        &[],
        &mut hash_pcs_challenger,
    );
    let route_open = open_core_ligerito(
        &route_setup.r1cs,
        &route_setup.pcs_params,
        route_core,
        total_routes,
        &[],
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
        fanouts: all_fanouts,
    }
}

fn route_witness_identity(rw: &RouteF32Witness) -> Vec<u8> {
    let bool_to_byte = |b: bool| if b { 1u8 } else { 0u8 };
    let mut bytes = Vec::with_capacity(route::KEY_BITS + route::KEY_BITS + rw.children.len() * route::DIGEST_BITS);
    for &b in &rw.key {
        bytes.push(bool_to_byte(b));
    }
    for &b in &rw.mask {
        bytes.push(bool_to_byte(b));
    }
    for child in &rw.children {
        for &b in child {
            bytes.push(bool_to_byte(b));
        }
    }
    bytes
}

pub fn verify_mhot_multiproof(proof: &MhotMultiproof) -> Result<(), VerifyError> {
    verify_mhot_multiproof_inner(proof, false)
}

pub fn verify_mhot_multiproof_timed(proof: &MhotMultiproof) -> Result<(), VerifyError> {
    verify_mhot_multiproof_inner(proof, true)
}

fn verify_mhot_multiproof_inner(proof: &MhotMultiproof, timed: bool) -> Result<(), VerifyError> {
    use std::time::Instant;
    let t0 = Instant::now();

    let mut challenger = FsChallenger::new(TRANSCRIPT_LABEL);
    challenger.observe_bytes(&proof.root);
    observe_wiring_topology(&mut challenger, &proof.fanouts);

    let setup_n_keccaks = proof.n_keccaks.max(MIN_LIGERITO_KECCAKS);
    let hash_setup = KeccakSetup::cached(setup_n_keccaks);
    let t1 = Instant::now();

    let (hash_ab, hash_c) = flock_core::verifier::verify_core(
        &hash_setup.r1cs,
        &proof.hash_zc,
        &proof.hash_lc,
        &proof.hash_commitment,
        &KeccakLincheckCircuit,
        &mut challenger,
    )?;
    let t2 = Instant::now();

    let route_setup = RouteF32Setup::cached(proof.n_routes);
    let t2b = Instant::now();
    let (route_ab, route_c) = flock_core::verifier::verify_core(
        &route_setup.r1cs,
        &proof.route_zc,
        &proof.route_lc,
        &proof.route_commitment,
        route_setup.r1cs.csc_lincheck_circuit(),
        &mut challenger,
    )?;
    let t3 = Instant::now();

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
        &[],
        &mut hash_pcs_challenger,
    )?;
    let t4 = Instant::now();

    verify_core_opening_ligerito(
        &route_setup.r1cs,
        &route_setup.pcs_params,
        &proof.route_commitment,
        &proof.route_pcs,
        &route_ab,
        &route_c,
        &[],
        &mut route_pcs_challenger,
    )?;
    let t5 = Instant::now();

    if timed {
        eprintln!(
            "  verify breakdown: hash_setup={:.1}ms hash_vc={:.1}ms route_setup={:.1}ms route_vc={:.1}ms hash_pcs={:.1}ms route_pcs={:.1}ms total={:.1}ms",
            (t1 - t0).as_secs_f64() * 1e3,
            (t2 - t1).as_secs_f64() * 1e3,
            (t2b - t2).as_secs_f64() * 1e3,
            (t3 - t2b).as_secs_f64() * 1e3,
            (t4 - t3).as_secs_f64() * 1e3,
            (t5 - t4).as_secs_f64() * 1e3,
            (t5 - t0).as_secs_f64() * 1e3,
        );
    }

    Ok(())
}

pub(crate) struct OpenedCore {
    pub(crate) zc_proof: ZerocheckProof,
    pub(crate) lc_proof: LincheckProof,
    pub(crate) pcs_open: BatchOpeningProofLigerito,
    pub(crate) commitment: Commitment,
    pub(crate) claim: R1csClaim,
}

pub(crate) fn open_core_ligerito(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    core: ProveCore,
    n_real_blocks_hint: usize,
    packed_direct: &[flock_core::pcs::PackedDirectClaim],
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
        packed_direct,
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

pub(crate) fn verify_core_opening_ligerito(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    commitment: &Commitment,
    pcs_open: &BatchOpeningProofLigerito,
    ab: &flock_core::proof::ZClaim,
    c: &flock_core::proof::ZClaim,
    packed_direct: &[flock_core::pcs::PackedDirectClaimRef<'_>],
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
        packed_direct,
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

pub(crate) fn fork_pcs_challenger(parent: &FsChallenger, base_label: &[u8]) -> FsChallenger {
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
