use super::ref_witness::{bytes_to_logical_state, Digest, RefWitness};
use super::schedule::MhotHashSchedule;
use crate::r1cs_hashes::keccak::{State, STATE_BITS};
use crate::r1cs_hashes::keccak3::KeccakSetup;
use flock_core::challenger::FsChallenger;
use flock_core::pcs::Commitment;
use flock_core::proof::R1csClaim;

const TRANSCRIPT_LABEL: &[u8] = b"mhot-hash-only-v0";
const MIN_LIGERITO_KECCAKS: usize = 49;

/// Proof artifact from hash-only MHOT prove.
pub struct MhotHashProof {
    pub proof: flock_core::proof::R1csProofLigerito,
    pub commitment: Commitment,
    pub claim: R1csClaim,
    pub expected_root: Digest,
    pub n_keccaks: usize,
    pub setup_n_keccaks: usize,
}

/// Prove MHOT hash fold (hash atoms only, no routing/glue circuit).
///
/// This proves that every keccak-f atom in the schedule was correctly computed,
/// using Flock's keccak3 R1CS. It does NOT prove the wiring between atoms.
pub fn prove_mhot_hash_only(sched: &MhotHashSchedule, witness: &RefWitness) -> MhotHashProof {
    assert!(
        !sched.hash_atoms.is_empty(),
        "schedule must have at least one atom"
    );
    assert_eq!(
        witness.atom_states.len(),
        sched.hash_atoms.len(),
        "witness atom_states must match schedule atoms"
    );

    let n_keccaks = sched.hash_atoms.len();
    let setup_n_keccaks = ligerito_setup_n_keccaks(n_keccaks);
    let mut initial_states: Vec<State> = sched
        .hash_atoms
        .iter()
        .map(|atom| bytes_to_logical_state(&witness.atom_states[atom.atom_id]))
        .collect();
    initial_states.resize(setup_n_keccaks, [false; STATE_BITS]);

    let setup = KeccakSetup::new(setup_n_keccaks);
    let mut challenger = FsChallenger::new(TRANSCRIPT_LABEL);
    let (proof, commitment, claim) = setup.prove_fast(&initial_states, &mut challenger);

    MhotHashProof {
        proof,
        commitment,
        claim,
        expected_root: witness.expected_root,
        n_keccaks,
        setup_n_keccaks,
    }
}

/// Verify MHOT hash fold proof.
pub fn verify_mhot_hash_only(
    n_keccaks: usize,
    proof: &MhotHashProof,
) -> Result<R1csClaim, flock_core::verifier::VerifyError> {
    assert_eq!(n_keccaks, proof.n_keccaks, "logical keccak count mismatch");
    let setup_n_keccaks = ligerito_setup_n_keccaks(n_keccaks);
    assert_eq!(
        setup_n_keccaks, proof.setup_n_keccaks,
        "setup keccak count mismatch"
    );
    let setup = KeccakSetup::new(setup_n_keccaks);
    let mut challenger = FsChallenger::new(TRANSCRIPT_LABEL);
    setup.verify(&proof.commitment, &proof.proof, &mut challenger)
}

fn ligerito_setup_n_keccaks(n_keccaks: usize) -> usize {
    n_keccaks.max(MIN_LIGERITO_KECCAKS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mhot::ref_witness::{build_ref_witness, cpu_fold_root, leaf_digest};
    use crate::mhot::wide_glue::{check_wiring_cpu, compute_atom_outputs};
    use std::time::{Duration, Instant};

    #[test]
    fn prove_verify_roundtrip_smoke() {
        let (elapsed, _) = time_it(|| {
            let sched = MhotHashSchedule::from_fanouts(&[4, 2]);
            let witness = build_ref_witness(&sched, 42);

            let proof = prove_mhot_hash_only(&sched, &witness);
            let result = verify_mhot_hash_only(sched.hash_atoms.len(), &proof);
            assert!(
                result.is_ok(),
                "prove/verify roundtrip must succeed for valid witness"
            );
        });
        eprintln!("prove_verify_roundtrip_smoke elapsed: {elapsed:?}");
    }

    #[test]
    fn prove_verify_with_cpu_cross_check() {
        let (elapsed, _) = time_it(|| {
            let sched = MhotHashSchedule::from_fanouts(&[8, 4, 2]);
            let witness = build_ref_witness(&sched, 99);

            let cpu_root = cpu_fold_root(&sched, &witness.atom_states);
            assert_eq!(
                cpu_root, witness.expected_root,
                "CPU root must match witness root"
            );

            let outputs = compute_atom_outputs(&witness.atom_states);
            let leaf_digests: Vec<Digest> = {
                let total: usize = sched.fanouts.iter().sum();
                (0..total).map(|i| leaf_digest(99, i)).collect()
            };
            assert!(
                check_wiring_cpu(
                    &sched,
                    &outputs,
                    &witness.atom_states,
                    &leaf_digests,
                    &witness.expected_root
                )
                .is_ok(),
                "CPU wiring must pass"
            );

            let proof = prove_mhot_hash_only(&sched, &witness);
            let result = verify_mhot_hash_only(sched.hash_atoms.len(), &proof);
            assert!(result.is_ok(), "prove/verify must succeed");
        });
        eprintln!("prove_verify_with_cpu_cross_check elapsed: {elapsed:?}");
    }

    #[test]
    fn different_fanouts_same_shape() {
        let cases = [vec![4, 2], vec![8, 4, 2], vec![16, 8], vec![3, 7, 2]];
        let (elapsed, per_case) = time_it(|| {
            let mut per_case = Vec::new();
            for fanouts in cases {
                let case_start = Instant::now();
                let sched = MhotHashSchedule::from_fanouts(&fanouts);
                let witness = build_ref_witness(&sched, 7);
                let proof = prove_mhot_hash_only(&sched, &witness);
                let result = verify_mhot_hash_only(sched.hash_atoms.len(), &proof);
                assert!(
                    result.is_ok(),
                    "fanouts {:?} must prove/verify with same keccak3 shape",
                    fanouts
                );
                per_case.push((fanouts, case_start.elapsed()));
            }
            per_case
        });
        eprintln!("different_fanouts_same_shape elapsed: {elapsed:?}");
        for (fanouts, case_elapsed) in per_case {
            eprintln!("  fanouts {fanouts:?}: {case_elapsed:?}");
        }
    }

    fn time_it<T>(f: impl FnOnce() -> T) -> (Duration, T) {
        let start = Instant::now();
        let output = f();
        (start.elapsed(), output)
    }
}
