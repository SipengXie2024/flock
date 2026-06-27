//! MHOT Stage-C acceptance tests: end-to-end membership and absence scenarios
//! with comprehensive negative controls.

mod mhot_acceptance {
    use flock_prover::mhot::{
        content::{check_absence, check_compact_content, check_subtree_counts},
        hash_only::{prove_mhot_hash_only, verify_mhot_hash_only},
        multi_base::{prove_multi, verify_multi},
        ref_witness::{build_ref_witness, cpu_fold_root, leaf_digest, Digest},
        route::{self, prove_route, verify_route, RouteSetup, RouteWitness},
        route_f32::{self as route32, RouteF32Setup, RouteF32Witness},
        schedule::MhotHashSchedule,
        wide_glue::{check_wiring_cpu, compute_atom_outputs},
    };

    /// End-to-end membership: hash, wiring, content, count, and prove/verify.
    #[test]
    fn membership_end_to_end() {
        let sched = MhotHashSchedule::from_fanouts(&[4, 2]);
        let witness = build_ref_witness(&sched, 42);

        let cpu_root = cpu_fold_root(&sched, &witness.atom_states);
        assert_eq!(cpu_root, witness.expected_root);

        let outputs = compute_atom_outputs(&witness.atom_states);
        let leaf_digests = make_leaf_digests(&sched, 42);
        assert!(check_wiring_cpu(
            &sched,
            &outputs,
            &witness.atom_states,
            &leaf_digests,
            &witness.expected_root,
        )
        .is_ok());

        let (key, mask) = compact_key_and_mask();
        assert!(check_compact_content(&key, &mask).is_ok());
        assert!(check_subtree_counts(&[100, 1]).is_ok());

        let proof = prove_mhot_hash_only(&sched, &witness);
        assert!(verify_mhot_hash_only(sched.hash_atoms.len(), &proof).is_ok());
    }

    /// Negative: tamper root, CPU wiring oracle rejects.
    #[test]
    fn membership_tampered_root_rejected() {
        let sched = MhotHashSchedule::from_fanouts(&[4, 2]);
        let witness = build_ref_witness(&sched, 42);
        let outputs = compute_atom_outputs(&witness.atom_states);
        let leaves = make_leaf_digests(&sched, 42);

        let mut bad_root = witness.expected_root;
        bad_root[0] ^= 1;

        assert!(
            check_wiring_cpu(&sched, &outputs, &witness.atom_states, &leaves, &bad_root).is_err()
        );
    }

    /// Negative: non-compact mask, content oracle rejects.
    #[test]
    fn membership_non_compact_mask_rejected() {
        let key = [false; 256];
        let mut mask = [false; 256];
        mask[0] = true;
        mask[2] = true;

        assert!(check_compact_content(&key, &mask).is_err());
    }

    /// Negative: bad count, count oracle rejects.
    #[test]
    fn membership_bad_count_rejected() {
        assert!(check_subtree_counts(&[100, 50, 2]).is_err());
    }

    /// Absence: valid case.
    #[test]
    fn absence_valid() {
        let mut query = [false; 256];
        let leaf = [false; 256];
        query[0] = true;

        assert!(check_absence(&query, &leaf).is_ok());
    }

    /// Absence negative: query key equals leaf key.
    #[test]
    fn absence_equal_keys_rejected() {
        let key = [false; 256];

        assert!(check_absence(&key, &key).is_err());
    }

    /// Standalone F_route roundtrip in the acceptance suite.
    #[test]
    fn route_membership_roundtrip() {
        let sched = MhotHashSchedule::from_fanouts(&[4, 2]);
        let route_witnesses = route_witnesses_for_schedule(&sched);
        let setup = RouteSetup::new(route_witnesses.len());

        let proof = prove_route(&setup, &route_witnesses);

        assert!(verify_route(&setup, &proof).is_ok());
    }

    /// Multi-base roundtrip: F_hash and F_route in one transcript.
    #[test]
    fn multi_base_membership_roundtrip() {
        let sched = MhotHashSchedule::from_fanouts(&[4, 2]);
        let witness = build_ref_witness(&sched, 7);
        let route_witnesses: Vec<RouteF32Witness> = sched.fanouts.iter().enumerate()
            .map(|(node, _)| {
                let mut key = [false; route32::KEY_BITS];
                let mut mask = [false; route32::KEY_BITS];
                key[0] = (node & 1) != 0;
                key[1] = true;
                mask[0] = true;
                mask[1] = true;
                let children: Vec<[bool; route32::DIGEST_BITS]> = (0..4)
                    .map(|c| std::array::from_fn(|b| ((node * 31 + c * 17 + b) & 1) != 0))
                    .collect();
                RouteF32Witness::new_padded(key, mask, &children, 4)
            })
            .collect();

        let proof = prove_multi(&sched, &witness, &route_witnesses);

        assert!(verify_multi(&proof).is_ok());
    }

    fn make_leaf_digests(sched: &MhotHashSchedule, seed: u64) -> Vec<Digest> {
        let total_leaves: usize = sched.fanouts.iter().sum();
        (0..total_leaves).map(|i| leaf_digest(seed, i)).collect()
    }

    fn compact_key_and_mask() -> ([bool; 256], [bool; 256]) {
        let mut key = [false; 256];
        let mut mask = [false; 256];
        mask[0] = true;
        mask[1] = true;
        key[1] = true;
        (key, mask)
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
