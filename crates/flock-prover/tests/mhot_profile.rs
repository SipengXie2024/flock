use flock_prover::mhot::{ref_witness::build_ref_witness, schedule::MhotHashSchedule};
use flock_prover::r1cs_hashes::keccak3::{generate_witness_with_ab_packed_and_lincheck, KeccakLincheckCircuit, KeccakSetup};
use flock_prover::prover::prove_fast_ligerito_timed;
use flock_core::challenger::FsChallenger;
use flock_prover::mhot::ref_witness::bytes_to_logical_state;

#[test]
fn profile_mhot_prove_phases() {
    let sched = MhotHashSchedule::from_fanouts(&[28, 24, 22, 16, 8]);
    let witness = build_ref_witness(&sched, 42);
    let n_keccaks = sched.hash_atoms.len();
    let setup_n = n_keccaks.max(49);
    let setup = KeccakSetup::new(setup_n);
    let mut initial_states: Vec<_> = sched.hash_atoms.iter()
        .map(|atom| bytes_to_logical_state(&witness.atom_states[atom.atom_id]))
        .collect();
    initial_states.resize(setup_n, [false; 1600]);
    let (z_packed, a_packed, b_packed, z_lincheck) =
        generate_witness_with_ab_packed_and_lincheck(&initial_states, setup.n_blocks_log());
    // Warmup
    {
        let mut ch = FsChallenger::new(b"warmup");
        let _ = prove_fast_ligerito_timed(&setup.r1cs, &setup.pcs_params,
            z_packed.clone(), a_packed.clone(), b_packed.clone(), z_lincheck.clone(),
            &KeccakLincheckCircuit, None, &mut ch);
    }
    // 3 timed runs
    for run in 0..3 {
        let mut ch = FsChallenger::new(b"profile");
        let t0 = std::time::Instant::now();
        let (_, _, _, t) = prove_fast_ligerito_timed(&setup.r1cs, &setup.pcs_params,
            z_packed.clone(), a_packed.clone(), b_packed.clone(), z_lincheck.clone(),
            &KeccakLincheckCircuit, None, &mut ch);
        let total = t0.elapsed().as_secs_f64();
        eprintln!("RUN {} commit={:.1}ms zerocheck={:.1}ms lincheck={:.1}ms open={:.1}ms total={:.1}ms",
            run, t.commit_s*1e3, t.zerocheck_s*1e3, t.lincheck_s*1e3, t.open_s*1e3, total*1e3);
    }
}
