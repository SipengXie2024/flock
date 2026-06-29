mod mhot_multiproof_tests {
    use flock_prover::mhot::{
        multiproof::{
            prove_mhot_multiproof, prove_mhot_multiproof_shared, verify_mhot_multiproof,
            MhotPathInput,
        },
        ref_witness::build_ref_witness,
        route_f32::{self as route, RouteF32Witness},
        schedule::MhotHashSchedule,
    };
    use std::time::Instant;

    fn make_path(fanouts: &[usize], seed: u64) -> MhotPathInput {
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
                let eff = fanout.min(route::FANOUT);
                let children: Vec<[bool; route::DIGEST_BITS]> = (0..eff)
                    .map(|c| std::array::from_fn(|b| ((node * 31 + c * 17 + b) & 1) != 0))
                    .collect();
                RouteF32Witness::new_padded(key, mask, &children, eff)
            })
            .collect()
    }

    fn same_root_paths(n: usize, fanouts: &[usize]) -> Vec<MhotPathInput> {
        (0..n).map(|_| make_path(fanouts, 42)).collect()
    }

    // --- positive controls ---

    #[test]
    fn single_path_roundtrip() {
        let paths = same_root_paths(1, &[4, 2]);
        let proof = prove_mhot_multiproof(&paths);
        verify_mhot_multiproof(&proof).expect("single path must verify");
    }

    #[test]
    fn four_path_roundtrip() {
        let paths = same_root_paths(4, &[8, 4, 2]);
        let proof = prove_mhot_multiproof(&paths);
        assert_eq!(proof.n_paths, 4);
        verify_mhot_multiproof(&proof).expect("4-path multiproof must verify");
    }

    #[test]
    fn various_fanout_shapes() {
        for fanouts in &[
            vec![4, 2],
            vec![8, 4, 2],
            vec![16, 8],
            vec![3, 7, 2],
        ] {
            let paths = same_root_paths(2, fanouts);
            let proof = prove_mhot_multiproof(&paths);
            verify_mhot_multiproof(&proof)
                .unwrap_or_else(|e| panic!("fanouts {fanouts:?} failed: {e:?}"));
        }
    }

    // --- negative controls ---

    #[test]
    fn neg_tampered_sibling_digest() {
        let paths = same_root_paths(2, &[4, 2]);
        let mut proof = prove_mhot_multiproof(&paths);
        proof.hash_commitment.root[3] ^= 0xFF;
        verify_mhot_multiproof(&proof)
            .expect_err("tampered sibling digest (commitment) must be rejected");
    }

    #[test]
    fn neg_wrong_root() {
        let paths = same_root_paths(2, &[4, 2]);
        let mut proof = prove_mhot_multiproof(&paths);
        proof.root[0] ^= 1;
        verify_mhot_multiproof(&proof).expect_err("wrong root must be rejected");
    }

    #[test]
    fn neg_tampered_route_commitment() {
        let paths = same_root_paths(2, &[4, 2]);
        let mut proof = prove_mhot_multiproof(&paths);
        proof.route_commitment.root[0] ^= 1;
        verify_mhot_multiproof(&proof)
            .expect_err("tampered route commitment must be rejected");
    }

    #[test]
    #[should_panic(expected = "root differs")]
    fn neg_mixed_roots_at_prove_time() {
        let mut paths = same_root_paths(2, &[4, 2]);
        paths[1] = make_path(&[4, 2], 99);
        prove_mhot_multiproof(&paths);
    }

    #[test]
    fn neg_swapped_hash_route() {
        let paths = same_root_paths(2, &[4, 2]);
        let mut proof = prove_mhot_multiproof(&paths);
        std::mem::swap(&mut proof.hash_zc, &mut proof.route_zc);
        std::mem::swap(&mut proof.hash_lc, &mut proof.route_lc);
        std::mem::swap(&mut proof.hash_pcs, &mut proof.route_pcs);
        std::mem::swap(&mut proof.hash_commitment, &mut proof.route_commitment);
        std::mem::swap(&mut proof.hash_claim, &mut proof.route_claim);
        match std::panic::catch_unwind(|| verify_mhot_multiproof(&proof)) {
            Ok(Err(_)) => {}
            Ok(Ok(())) => panic!("swapped hash/route must fail"),
            Err(_) => {}
        }
    }

    // --- scaling benchmark ---

    #[test]
    fn bench_multiproof_scaling() {
        let fanouts = &[8, 4, 2];
        for n_paths in [1, 4, 16, 64] {
            let paths = same_root_paths(n_paths, fanouts);

            let t0 = Instant::now();
            let proof = prove_mhot_multiproof(&paths);
            let prove_ms = t0.elapsed().as_millis();

            let t1 = Instant::now();
            verify_mhot_multiproof(&proof).expect("benchmark proof must verify");
            let verify_ms = t1.elapsed().as_millis();

            eprintln!(
                "BENCH n_paths={:>3} | keccaks={:>5} routes={:>4} | prove={:>6}ms verify={:>5}ms | per_path_prove={:>5}ms",
                n_paths,
                proof.n_keccaks,
                proof.n_routes,
                prove_ms,
                verify_ms,
                prove_ms / n_paths as u128,
            );
        }
    }

    #[test]
    fn flock_multiproof_full_scaling() {
        let fanouts = &[8, 4, 2];
        eprintln!();
        eprintln!("=== Flock MHOT Multiproof Scaling (keccak3, x86) ===");
        eprintln!("{:>8} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "n_paths", "prove_ms", "verify_ms", "total_ms", "per_path", "proof_KB");
        eprintln!("{}", "-".repeat(66));

        for &n_paths in &[1, 2, 4, 8, 16, 32, 64, 128, 512, 1024, 2048, 4096, 8192, 16384, 32768] {
            let paths = same_root_paths(n_paths, fanouts);

            let t0 = Instant::now();
            let proof = prove_mhot_multiproof(&paths);
            let prove_ms = t0.elapsed().as_secs_f64() * 1e3;

            let t1 = Instant::now();
            verify_mhot_multiproof(&proof).expect("must verify");
            let verify_ms = t1.elapsed().as_secs_f64() * 1e3;

            let proof_kb = proof.proof_size_bytes() as f64 / 1024.0;
            let total = prove_ms + verify_ms;
            eprintln!("{:>8} {:>10.1} {:>10.1} {:>10.1} {:>9.3} {:>9.1}",
                n_paths, prove_ms, verify_ms, total, total / n_paths as f64, proof_kb);
        }
    }

    #[test]
    fn profile_verify_16k() {
        let fanouts = &[8, 4, 2];
        let n_paths = 16384;
        let paths = same_root_paths(n_paths, fanouts);
        let proof = prove_mhot_multiproof(&paths);

        verify_mhot_multiproof(&proof).expect("must verify");

        let iterations = 100;
        let t0 = Instant::now();
        for _ in 0..iterations {
            verify_mhot_multiproof(&proof).expect("must verify");
        }
        let total_ms = t0.elapsed().as_secs_f64() * 1e3;
        eprintln!("verify 16384 paths × {iterations}: {total_ms:.0}ms total, {:.1}ms/iter",
            total_ms / iterations as f64);
    }

    #[test]
    fn profile_verify_breakdown() {
        use flock_prover::mhot::multiproof::verify_mhot_multiproof_timed;

        let fanouts = &[8, 4, 2];
        for &n_paths in &[1, 16, 256, 4096, 16384] {
            let paths = same_root_paths(n_paths, fanouts);
            let proof = prove_mhot_multiproof(&paths);

            // warmup
            verify_mhot_multiproof(&proof).expect("warmup");

            eprintln!("--- n_paths={n_paths} ---");
            verify_mhot_multiproof_timed(&proof).expect("must verify");
        }
    }

    // --- wiring topology binding ---

    #[test]
    fn wiring_binding_rejects_tampered_fanouts() {
        let paths = same_root_paths(2, &[4, 2]);
        let mut proof = prove_mhot_multiproof(&paths);
        proof.fanouts[0][0] = 5;
        verify_mhot_multiproof(&proof)
            .expect_err("tampered fanouts must be rejected");
    }

    #[test]
    fn wiring_binding_rejects_extra_path_fanouts() {
        let paths = same_root_paths(2, &[4, 2]);
        let mut proof = prove_mhot_multiproof(&paths);
        proof.fanouts.push(vec![8, 4]);
        verify_mhot_multiproof(&proof)
            .expect_err("extra path fanouts must be rejected");
    }

    #[test]
    fn wiring_binding_rejects_removed_path_fanouts() {
        let paths = same_root_paths(2, &[4, 2]);
        let mut proof = prove_mhot_multiproof(&paths);
        proof.fanouts.pop();
        verify_mhot_multiproof(&proof)
            .expect_err("removed path fanouts must be rejected");
    }

    #[test]
    fn wiring_binding_shared_rejects_tampered_fanouts() {
        let paths = same_root_paths(4, &[8, 4, 2]);
        let mut proof = prove_mhot_multiproof_shared(&paths);
        proof.fanouts[1][1] = 7;
        verify_mhot_multiproof(&proof)
            .expect_err("shared proof with tampered fanouts must be rejected");
    }

    // --- content soundness (in-circuit mask/key checks) ---

    #[test]
    fn content_non_compact_mask_verify_fails() {
        let sched = MhotHashSchedule::from_fanouts(&[4, 2]);
        let hash_witness = build_ref_witness(&sched, 42);
        let mut rws = make_route_witnesses(&sched);
        rws[0].mask = [false; 256];
        rws[0].mask[0] = true;
        rws[0].mask[2] = true;
        rws[0].key = [false; 256];
        let paths = vec![MhotPathInput { schedule: sched, hash_witness, route_witnesses: rws }];
        let proof = prove_mhot_multiproof(&paths);
        verify_mhot_multiproof(&proof)
            .expect_err("non-compact mask must fail R1CS verification");
    }

    #[test]
    fn content_key_above_mask_verify_fails() {
        let sched = MhotHashSchedule::from_fanouts(&[4, 2]);
        let hash_witness = build_ref_witness(&sched, 42);
        let mut rws = make_route_witnesses(&sched);
        let popcount = rws[0].mask.iter().filter(|&&b| b).count();
        rws[0].key[popcount + 10] = true;
        let paths = vec![MhotPathInput { schedule: sched, hash_witness, route_witnesses: rws }];
        let proof = prove_mhot_multiproof(&paths);
        verify_mhot_multiproof(&proof)
            .expect_err("key above mask must fail R1CS verification");
    }

    // --- path sharing (DAG dedup) ---

    #[test]
    fn path_sharing_reduces_atoms() {
        let fanouts = &[8, 4, 2];
        let n = 16;
        let paths = same_root_paths(n, fanouts);

        let proof_dup = prove_mhot_multiproof(&paths);
        let proof_shared = prove_mhot_multiproof_shared(&paths);

        assert!(
            proof_shared.n_keccaks < proof_dup.n_keccaks,
            "shared ({}) must have fewer keccaks than duplicated ({})",
            proof_shared.n_keccaks,
            proof_dup.n_keccaks,
        );
        assert!(
            proof_shared.n_routes < proof_dup.n_routes,
            "shared ({}) must have fewer routes than duplicated ({})",
            proof_shared.n_routes,
            proof_dup.n_routes,
        );
        assert_eq!(proof_shared.n_paths, n);

        verify_mhot_multiproof(&proof_dup).expect("duplicated proof must verify");
        verify_mhot_multiproof(&proof_shared).expect("shared proof must verify");
    }

    #[test]
    fn path_sharing_proof_is_smaller() {
        let fanouts = &[8, 4, 2];
        let n = 16;
        let paths = same_root_paths(n, fanouts);

        let proof_dup = prove_mhot_multiproof(&paths);
        let proof_shared = prove_mhot_multiproof_shared(&paths);

        assert!(
            proof_shared.n_keccaks < proof_dup.n_keccaks,
            "shared proof ({} keccaks) must have fewer atoms than duplicated ({} keccaks)",
            proof_shared.n_keccaks,
            proof_dup.n_keccaks,
        );
    }

    #[test]
    fn path_sharing_neg_tampered_root() {
        let fanouts = &[8, 4, 2];
        let paths = same_root_paths(4, fanouts);
        let mut proof = prove_mhot_multiproof_shared(&paths);
        proof.root[0] ^= 1;
        verify_mhot_multiproof(&proof).expect_err("tampered root on shared proof must fail");
    }

    #[test]
    fn path_sharing_single_path_identical() {
        let fanouts = &[8, 4, 2];
        let paths = same_root_paths(1, fanouts);

        let proof_dup = prove_mhot_multiproof(&paths);
        let proof_shared = prove_mhot_multiproof_shared(&paths);

        assert_eq!(proof_shared.n_keccaks, proof_dup.n_keccaks);
        assert_eq!(proof_shared.n_routes, proof_dup.n_routes);
        verify_mhot_multiproof(&proof_shared).expect("single-path shared must verify");
    }

    #[test]
    fn path_sharing_various_fanout_shapes() {
        for fanouts in &[
            vec![4, 2],
            vec![8, 4, 2],
            vec![16, 8],
            vec![3, 7, 2],
        ] {
            let paths = same_root_paths(4, fanouts);
            let proof = prove_mhot_multiproof_shared(&paths);
            verify_mhot_multiproof(&proof)
                .unwrap_or_else(|e| panic!("shared fanouts {fanouts:?} failed: {e:?}"));
        }
    }

    #[test]
    fn path_sharing_ab_benchmark() {
        let fanouts = &[8, 4, 2];
        eprintln!();
        eprintln!("=== Path Sharing A/B Benchmark ===");
        eprintln!(
            "{:>8} {:>12} {:>8} {:>12} {:>8} {:>8} {:>8}",
            "n_paths", "dup_ms", "dup_k", "shared_ms", "shared_k", "speedup", "k_ratio"
        );
        eprintln!("{}", "-".repeat(78));

        for &n in &[4, 16, 64, 256] {
            let paths = same_root_paths(n, fanouts);

            let t0 = Instant::now();
            let proof_dup = prove_mhot_multiproof(&paths);
            let dup_ms = t0.elapsed().as_secs_f64() * 1e3;

            let t1 = Instant::now();
            let proof_shared = prove_mhot_multiproof_shared(&paths);
            let shared_ms = t1.elapsed().as_secs_f64() * 1e3;

            verify_mhot_multiproof(&proof_dup).expect("dup must verify");
            verify_mhot_multiproof(&proof_shared).expect("shared must verify");

            eprintln!(
                "{:>8} {:>12.1} {:>8} {:>12.1} {:>8} {:>7.2}x {:>7.2}x",
                n,
                dup_ms,
                proof_dup.n_keccaks,
                shared_ms,
                proof_shared.n_keccaks,
                dup_ms / shared_ms,
                proof_dup.n_keccaks as f64 / proof_shared.n_keccaks as f64,
            );
        }
    }

}
