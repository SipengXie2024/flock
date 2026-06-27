mod mhot_multiproof_tests {
    use flock_prover::mhot::{
        multiproof::{prove_mhot_multiproof, verify_mhot_multiproof, MhotPathInput},
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
}
