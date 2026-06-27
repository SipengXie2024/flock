use flock_prover::mhot::{ref_witness::build_ref_witness, schedule::MhotHashSchedule};
use flock_prover::r1cs_hashes::keccak3::{
    generate_witness_with_ab_packed_and_lincheck, KeccakLincheckCircuit, KeccakSetup,
};
use flock_prover::mhot::ref_witness::bytes_to_logical_state;
use flock_prover::prover::prove_fast_ligerito_timed;
use flock_core::challenger::FsChallenger;

fn profile_at_scale(n_paths: usize) {
    let sched = MhotHashSchedule::from_fanouts(&[28, 24, 22, 16, 8]);
    let witness = build_ref_witness(&sched, 42);
    let atoms_per_path = sched.hash_atoms.len();

    let one_path_states: Vec<_> = sched
        .hash_atoms
        .iter()
        .map(|atom| bytes_to_logical_state(&witness.atom_states[atom.atom_id]))
        .collect();

    let total_atoms = (atoms_per_path * n_paths).max(49);
    let mut initial_states = Vec::with_capacity(total_atoms);
    for _ in 0..n_paths {
        initial_states.extend_from_slice(&one_path_states);
    }
    initial_states.resize(total_atoms, [false; 1600]);

    let setup = KeccakSetup::new(total_atoms);
    let m = setup.r1cs.m;
    let n_blocks = 1usize << setup.n_blocks_log();

    let (z_packed, a_packed, b_packed, z_lincheck) =
        generate_witness_with_ab_packed_and_lincheck(&initial_states, setup.n_blocks_log());

    // Warmup
    {
        let mut ch = FsChallenger::new(b"warmup");
        let _ = prove_fast_ligerito_timed(
            &setup.r1cs, &setup.pcs_params,
            z_packed.clone(), a_packed.clone(), b_packed.clone(), z_lincheck.clone(),
            &KeccakLincheckCircuit, None, &mut ch,
        );
    }

    // Timed run (best of 2)
    let mut best_total = f64::MAX;
    let mut best_t = None;
    for _ in 0..2 {
        let mut ch = FsChallenger::new(b"profile");
        let t0 = std::time::Instant::now();
        let (_, _, _, t) = prove_fast_ligerito_timed(
            &setup.r1cs, &setup.pcs_params,
            z_packed.clone(), a_packed.clone(), b_packed.clone(), z_lincheck.clone(),
            &KeccakLincheckCircuit, None, &mut ch,
        );
        let total = t0.elapsed().as_secs_f64();
        if total < best_total {
            best_total = total;
            best_t = Some(t);
        }
    }
    let t = best_t.unwrap();
    eprintln!(
        "paths={:>5}  atoms={:>6}  m={:>2}  blocks={:>6}  | commit={:>7.1}ms  zc={:>7.1}ms  lc={:>7.1}ms  open={:>7.1}ms  TOTAL={:>7.1}ms  | ms/path={:.2}",
        n_paths, total_atoms, m, n_blocks,
        t.commit_s * 1e3, t.zerocheck_s * 1e3, t.lincheck_s * 1e3, t.open_s * 1e3,
        best_total * 1e3, best_total * 1e3 / n_paths as f64,
    );
}

#[test]
fn profile_mhot_prove_scaling() {
    eprintln!();
    eprintln!("=== MHOT Prove Scaling (keccak3, x86 + PCLMULQDQ + AVX2 optimizations) ===");
    eprintln!("{:>5}  {:>6}  {:>2}  {:>6}  | {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  | {}",
        "paths", "atoms", "m", "blocks", "commit", "zerocheck", "lincheck", "open", "TOTAL", "ms/path");
    eprintln!("{}", "-".repeat(120));
    for &n_paths in &[1, 10, 100, 300] {
        profile_at_scale(n_paths);
    }
}
