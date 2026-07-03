    use super::{
        bytes_to_words, prove_sound_multiproof, verify_sound_multiproof,
        verify_sound_multiproof_with_entries,
    };
    use crate::mhot::merkle_membership::{
        ContentMeta, MhotMembershipError, MhotMembershipInput, PathEntry, compute_content_hash,
        leaf_content_hash, leaf_words_to_digest_bytes, mhot_node_to_route_witness,
    };
    use crate::mhot::native_witness::{mhot_node_to_sha256_merkle, MhotNodeWitness};
    use flock_core::challenger::FsChallenger;

    /// The public root = content_hash of a node = SHA-256(content ‖ native_root).
    fn node_root(input: &MhotMembershipInput) -> [u32; 8] {
        let native = mhot_node_to_sha256_merkle(&input.node, false).native_root;
        bytes_to_words(&compute_content_hash(
            &input.content,
            &leaf_words_to_digest_bytes(&native),
        ))
    }

    /// Two-node honest path whose ContentMetas genuinely route `entry.key`:
    /// root R (selects child 0 = T) → terminal T (selects child 1 = entry leaf).
    fn honest_path_with_entry() -> (Vec<MhotMembershipInput>, PathEntry) {
        let mut key = [0u8; 32];
        key[7] = 1; // BE chunk 0 = 1 → discriminative bit 0 of T's mask is set
        let entry = PathEntry { key, value: vec![7u8; 16] };
        let leaf_ch = leaf_content_hash(&entry);

        // Terminal node T: masks[0]=0b1 → dense(key)=1; sparse [0,1] → child 1.
        let t_node = MhotNodeWitness {
            children: vec![[0x11; 32], leaf_ch],
            selected_child: 1,
        };
        let t_meta = ContentMeta {
            extraction_masks: [1, 0, 0, 0],
            sparse_partial_keys: vec![0, 1],
            child_leaf_counts: vec![1, 1],
        };
        let t_root = mhot_node_to_sha256_merkle(&t_node, false).native_root;
        let t_ch = compute_content_hash(&t_meta, &leaf_words_to_digest_bytes(&t_root));

        // Root node R: masks[0]=0b10 → dense(key)=0 (key bit 1 clear); sparse
        // [0,1] → child 0 = T.
        let r_node = MhotNodeWitness {
            children: vec![t_ch, [0x22; 32]],
            selected_child: 0,
        };
        let r_meta = ContentMeta {
            extraction_masks: [2, 0, 0, 0],
            sparse_partial_keys: vec![0, 1],
            child_leaf_counts: vec![2, 1],
        };

        let mk = |node: MhotNodeWitness, content: ContentMeta| MhotMembershipInput {
            route_witness: mhot_node_to_route_witness(&node),
            node,
            content,
        };
        (vec![mk(r_node, r_meta), mk(t_node, t_meta)], entry)
    }

    fn prove_and_root(
        paths: &[Vec<MhotMembershipInput>],
    ) -> (super::SoundMultiproof, [u32; 8]) {
        let mut ch = FsChallenger::new(b"smp-entries");
        let proof = prove_sound_multiproof(paths, &mut ch);
        // Public root = content_hash of the root node (paths[0][0]).
        let root = node_root(&paths[0][0]);
        (proof, root)
    }

    #[test]
    fn entries_binding_honest_accepts() {
        let (path, entry) = honest_path_with_entry();
        let (proof, root) = prove_and_root(&[path]);
        let mut chv = FsChallenger::new(b"smp-entries");
        verify_sound_multiproof_with_entries(&proof, &root, &[entry], &mut chv)
            .expect("honest entry must verify");
    }

    /// Domain-confusion forgery: leaf preimage (key ‖ len ‖ value) and internal
    /// content preimage (masks ‖ keys ‖ root ‖ counts) share SHA-256 with no
    /// domain tag. A prover truncates a path at node M whose authenticated
    /// selected child is an INTERNAL node J, and crafts (key, value) whose leaf
    /// preimage is byte-identical to J's content preimage — so leaf_hash == J's
    /// content_hash BY CONSTRUCTION (no hash break). Without a leaf-ness check,
    /// the non-member (key, value) is accepted. The terminal-node
    /// child_leaf_counts[selected]==1 check (leaf ⟺ subtree of exactly one leaf)
    /// rejects it.
    #[test]
    fn entries_binding_internal_node_as_leaf_rejected() {
        // Impostor INTERNAL node J, fanout 2 → content preimage = 64 + 8·2 = 80
        // bytes; a colliding leaf needs value length 80 − 40 = 40, encoded in
        // J.sparse_partial_keys[0..2] read as the u64 little-endian length.
        let j_meta = ContentMeta {
            extraction_masks: [0u64; 4],          // ⇒ leaf key = 32 zero bytes
            sparse_partial_keys: vec![40, 0],     // len(value) = 40, LE-aligned
            child_leaf_counts: vec![1, 1],
        };
        let j_root = [7u8; 32];
        let d = compute_content_hash(&j_meta, &j_root); // J's content_hash digest

        // The colliding entry: key = J.masks (zero), value = J_root ‖ J.counts.
        let mut value = Vec::with_capacity(40);
        value.extend_from_slice(&j_root);
        value.extend_from_slice(&1u32.to_le_bytes());
        value.extend_from_slice(&1u32.to_le_bytes());
        let entry = PathEntry { key: [0u8; 32], value };
        assert_eq!(
            leaf_content_hash(&entry), d,
            "test setup: leaf preimage must byte-collide with J's content preimage"
        );

        // Terminal M selects child 1 = J (internal ⇒ child_leaf_counts[1] = 2).
        // dense_key(zero key, any mask) = 0; sparse[1]=0 matches ⇒ selects 1.
        let m_node = MhotNodeWitness { children: vec![[0x33; 32], d], selected_child: 1 };
        let m_meta = ContentMeta {
            extraction_masks: [0u64; 4],
            sparse_partial_keys: vec![0xDEAD, 0],
            child_leaf_counts: vec![1, 2],
        };
        let m_root = mhot_node_to_sha256_merkle(&m_node, false).native_root;
        let m_ch = compute_content_hash(&m_meta, &leaf_words_to_digest_bytes(&m_root));

        // Root R selects child 0 = M. dense 0; sparse[0]=0 ⇒ selects 0.
        let r_node = MhotNodeWitness { children: vec![m_ch, [0x44; 32]], selected_child: 0 };
        let r_meta = ContentMeta {
            extraction_masks: [0u64; 4],
            sparse_partial_keys: vec![0, 0xBEEF],
            child_leaf_counts: vec![3, 1],
        };

        let mk = |node: MhotNodeWitness, content: ContentMeta| MhotMembershipInput {
            route_witness: mhot_node_to_route_witness(&node),
            node,
            content,
        };
        let path = vec![mk(r_node, r_meta), mk(m_node, m_meta)];
        let (proof, root) = prove_and_root(&[path]);
        let mut chv = FsChallenger::new(b"smp-entries");
        let res = verify_sound_multiproof_with_entries(&proof, &root, &[entry], &mut chv);
        assert!(
            matches!(res, Err(MhotMembershipError::EntryLeafMismatch { path_idx: 0 })),
            "internal node claimed as member leaf must be rejected, got {res:?}"
        );
    }

    /// Absent-key bit-flip forgery: flip a key bit OUTSIDE every mask — the
    /// routing is unchanged, so before the terminal-leaf check this non-member
    /// key would be accepted. The leaf hash must catch it.
    #[test]
    fn entries_binding_nondiscriminative_bitflip_rejected() {
        let (path, mut entry) = honest_path_with_entry();
        let (proof, root) = prove_and_root(&[path]);
        entry.key[31] ^= 1; // chunk 3; masks[3] == 0 everywhere on the path
        let mut chv = FsChallenger::new(b"smp-entries");
        let res = verify_sound_multiproof_with_entries(&proof, &root, &[entry], &mut chv);
        assert!(
            matches!(res, Err(MhotMembershipError::EntryLeafMismatch { path_idx: 0 })),
            "non-member key differing only outside the masks must be rejected, got {res:?}"
        );
    }

    /// A key whose discriminative bit routes to a DIFFERENT child than the
    /// proof authenticated must fail the routing re-run.
    #[test]
    fn entries_binding_wrong_routing_rejected() {
        let (path, mut entry) = honest_path_with_entry();
        let (proof, root) = prove_and_root(&[path]);
        entry.key[7] = 0; // T's dense key becomes 0 → routes to child 0, proof selected 1
        let mut chv = FsChallenger::new(b"smp-entries");
        let res = verify_sound_multiproof_with_entries(&proof, &root, &[entry], &mut chv);
        assert!(
            matches!(res, Err(MhotMembershipError::RoutingMismatch { path_idx: 0, level: 1, .. })),
            "key routing to a different child must be rejected, got {res:?}"
        );
    }

    /// Multi-path + BFS dedup: two identical paths collapse to shared unique
    /// nodes but path_mapping keeps two entries. Exercises the per-path entry
    /// loop and the shared-node `selected` reconstruction the single-path tests
    /// and the (silently-skippable) bench don't cover deterministically.
    #[test]
    fn entries_binding_multipath_dedup_accepts_and_rejects() {
        let (path, entry) = honest_path_with_entry();
        let (proof, root) = prove_and_root(&[path.clone(), path]);
        assert_eq!(proof.path_mapping.node_indices.len(), 2, "two paths");

        let mut chv = FsChallenger::new(b"smp-entries");
        verify_sound_multiproof_with_entries(
            &proof, &root, &[entry.clone(), entry.clone()], &mut chv,
        ).expect("two honest entries over shared nodes must verify");

        // Corrupting one path's entry (non-discriminative bit flip) must reject
        // only via the terminal leaf check, proving per-path binding is live.
        let mut bad = entry.clone();
        bad.key[31] ^= 1;
        let mut chv2 = FsChallenger::new(b"smp-entries");
        let res = verify_sound_multiproof_with_entries(&proof, &root, &[entry, bad], &mut chv2);
        assert!(
            matches!(res, Err(MhotMembershipError::EntryLeafMismatch { path_idx: 1 })),
            "corrupted second entry must be rejected at path_idx 1, got {res:?}"
        );
    }

    #[test]
    fn entries_binding_wrong_value_rejected() {
        let (path, mut entry) = honest_path_with_entry();
        let (proof, root) = prove_and_root(&[path]);
        entry.value = vec![8u8; 16];
        let mut chv = FsChallenger::new(b"smp-entries");
        let res = verify_sound_multiproof_with_entries(&proof, &root, &[entry], &mut chv);
        assert!(
            matches!(res, Err(MhotMembershipError::EntryLeafMismatch { path_idx: 0 })),
            "wrong value must be rejected, got {res:?}"
        );
    }

    #[test]
    fn entries_binding_count_mismatch_rejected() {
        let (path, entry) = honest_path_with_entry();
        let (proof, root) = prove_and_root(&[path]);
        let mut chv = FsChallenger::new(b"smp-entries");
        let res = verify_sound_multiproof_with_entries(
            &proof, &root, &[entry.clone(), entry], &mut chv,
        );
        assert!(
            matches!(res, Err(MhotMembershipError::EntryCountMismatch { .. })),
            "entry count mismatch must be rejected, got {res:?}"
        );
    }

    /// Forgery: a malicious prover commits a FAKE in-node tree (selected child is
    /// a non-member `L`) but claims the public root of the REAL tree. Since the
    /// verifier COMPUTES content_hash from the committed tree (not from a
    /// prover-supplied value — the chain SNARK was deleted), the fake tree's
    /// content_hash ≠ the real tree's, so the root check rejects. This is the
    /// binding the deleted chain base used to guard; it is now inherent because
    /// nothing decouples content_hash from the committed merkle tree.
    #[test]
    fn forged_fake_merkle_real_root_rejected() {
        // Real node with genuine children.
        let real_children: Vec<[u8; 32]> = (0..8)
            .map(|i| {
                let mut h = [0u8; 32];
                h[0] = i as u8;
                h[1] = 0xAA;
                h
            })
            .collect();
        let selected = 3usize;
        let real_node = MhotNodeWitness {
            children: real_children.clone(),
            selected_child: selected,
        };
        let real_input = MhotMembershipInput::from_node(real_node);
        let real_root = node_root(&real_input); // the public root of the REAL tree

        // Fake node: same shape, but the selected child is a non-member L.
        let mut fake_children = real_children;
        fake_children[selected] = [0xEE; 32];
        let fake_node = MhotNodeWitness {
            children: fake_children,
            selected_child: selected,
        };
        let fake_input = MhotMembershipInput::from_node(fake_node);

        // Prove the FAKE tree, then claim the REAL tree's public root.
        let mut ch = FsChallenger::new(b"smp-forge");
        let proof = prove_sound_multiproof(&[vec![fake_input]], &mut ch);
        let mut chv = FsChallenger::new(b"smp-forge");
        let res = verify_sound_multiproof(&proof, &real_root, &mut chv);
        assert!(
            matches!(res, Err(MhotMembershipError::RootMismatch { .. })),
            "fake in-node tree under the real tree's root must be REJECTED (verifier \
             computes content_hash over the committed fake tree); got {res:?}"
        );
    }

    fn two_child_input(selected: usize) -> MhotMembershipInput {
        let node = MhotNodeWitness {
            children: vec![[0x11; 32], [0x22; 32]],
            selected_child: selected,
        };
        MhotMembershipInput::from_node(node)
    }

    /// Route-2 positive: b_bits[0] is now the REAL depth-0 side bit (native-order
    /// chains), so a right-child-at-depth-0 selected leaf must authenticate
    /// end-to-end — production never exercised real b0 before this.
    #[test]
    fn b0_right_child_honest_accepts() {
        for selected in [0usize, 1] {
            let input = two_child_input(selected);
            let root = node_root(&input);
            let mut ch = FsChallenger::new(b"smp-b0");
            let proof = prove_sound_multiproof(&[vec![input]], &mut ch);
            assert_eq!(
                proof.merkle_b_bits[0][0],
                selected == 1,
                "depth-0 side bit must be the real side (selected={selected})"
            );
            let mut chv = FsChallenger::new(b"smp-b0");
            verify_sound_multiproof(&proof, &root, &mut chv)
                .unwrap_or_else(|e| panic!("honest selected={selected} must verify: {e:?}"));
        }
    }

    /// Route-2 soundness delta: the public depth-0 side bit is consumed by the
    /// merkle shift (tree order — the W formula routes the leaf term to X_L/X_R
    /// per B(0)) and by selected_index (entry routing position). Flipping it
    /// post-prove must be rejected: the shift's authenticated chain no longer
    /// matches the committed compressions.
    ///
    /// The "rebuild a consistent fake tree for the flipped bit" variant of this
    /// attack is a different committed tree (the leaf sits at the other index),
    /// so its native_root — and hence its verifier-computed content_hash —
    /// differs from the honest one and the public-root check rejects; that is
    /// exactly `forged_fake_merkle_real_root_rejected`.
    #[test]
    fn b0_flip_rejected() {
        for selected in [0usize, 1] {
            let input = two_child_input(selected);
            let root = node_root(&input);
            let mut ch = FsChallenger::new(b"smp-b0-flip");
            let mut proof = prove_sound_multiproof(&[vec![input]], &mut ch);
            proof.merkle_b_bits[0][0] = !proof.merkle_b_bits[0][0];
            let mut chv = FsChallenger::new(b"smp-b0-flip");
            let res = verify_sound_multiproof(&proof, &root, &mut chv);
            assert!(
                res.is_err(),
                "flipped depth-0 side bit must be rejected (selected={selected}), got Ok"
            );
        }
    }

    /// The Route-2 decoupling attack: commit a FAKE in-node tree (whose selected
    /// leaf is a non-member, honestly shift-authenticated) but supply the REAL
    /// tree's native_root, so the verifier-computed content_hash hits the real
    /// public root and the root check passes on genuine values. The pad-forward
    /// check is the ONLY thing standing between this forgery and acceptance:
    /// pad-forwarding the real native_root cannot reach the fake chain's
    /// shift-authenticated padded root (SHA-256 collision resistance).
    #[test]
    fn forged_native_root_decoupling_rejected() {
        let real_children: Vec<[u8; 32]> = (0..8)
            .map(|i| {
                let mut h = [0u8; 32];
                h[0] = i as u8;
                h[1] = 0xBB;
                h
            })
            .collect();
        let selected = 3usize;
        let real_node = MhotNodeWitness {
            children: real_children.clone(),
            selected_child: selected,
        };
        let real_input = MhotMembershipInput::from_node(real_node);
        let real_root = node_root(&real_input);
        let real_native =
            mhot_node_to_sha256_merkle(&real_input.node, false).native_root;

        let mut fake_children = real_children;
        fake_children[selected] = [0xEE; 32];
        let fake_node = MhotNodeWitness {
            children: fake_children,
            selected_child: selected,
        };
        let fake_input = MhotMembershipInput::from_node(fake_node);

        let mut ch = FsChallenger::new(b"smp-nr-decouple");
        let mut proof = prove_sound_multiproof(&[vec![fake_input]], &mut ch);
        proof.merkle_native_roots[0] = real_native;
        let mut chv = FsChallenger::new(b"smp-nr-decouple");
        let res = verify_sound_multiproof(&proof, &real_root, &mut chv);
        assert!(
            matches!(res, Err(MhotMembershipError::NativeRootMismatch { node_idx: 0 })),
            "fake tree + real native_root must fail the pad-forward check, got {res:?}"
        );
    }

    /// The pad-forward binding: merkle_native_roots[i] is prover-supplied and
    /// only trustworthy because padding it forward n_pad times must reach the
    /// shift-authenticated merkle_roots[i]. Tampering it must fail exactly there.
    #[test]
    fn tampered_native_root_rejected() {
        let input = two_child_input(1);
        let root = node_root(&input);
        let mut ch = FsChallenger::new(b"smp-nr-tamper");
        let mut proof = prove_sound_multiproof(&[vec![input]], &mut ch);
        proof.merkle_native_roots[0][0] ^= 1;
        let mut chv = FsChallenger::new(b"smp-nr-tamper");
        let res = verify_sound_multiproof(&proof, &root, &mut chv);
        assert!(
            matches!(res, Err(MhotMembershipError::NativeRootMismatch { node_idx: 0 })),
            "tampered native_root must fail the pad-forward check, got {res:?}"
        );
    }

    // ---- DoS wire-validity gates ----
    //
    // Latent while `SoundMultiproof` is Serialize-only, load-bearing the moment
    // it gains a Deserialize path. Each gate must reject with MalformedProof
    // BEFORE any cached()/allocation-driving use of the tampered value; tests
    // use honest+1-style malformations so a mis-ordered gate fails the test
    // quickly (wrong error variant) instead of OOMing the test runner.

    fn dos_proof(domain: &'static [u8]) -> (super::SoundMultiproof, [u32; 8]) {
        let input = two_child_input(1);
        let root = node_root(&input);
        let mut ch = FsChallenger::new(domain);
        let proof = prove_sound_multiproof(&[vec![input]], &mut ch);
        (proof, root)
    }

    fn expect_malformed(
        proof: &super::SoundMultiproof,
        root: &[u32; 8],
        domain: &'static [u8],
        what: &str,
    ) {
        let mut chv = FsChallenger::new(domain);
        let res = verify_sound_multiproof(proof, root, &mut chv);
        assert!(
            matches!(res, Err(MhotMembershipError::MalformedProof { .. })),
            "{what} must be rejected by a wire-validity gate, got {res:?}"
        );
    }

    /// n_log_merkle ≥ word size would make `1usize << n_log_merkle` shift-
    /// overflow (panic in debug, wrap in release) before any semantic check.
    /// The absolute cap must reject it first.
    #[test]
    fn dos_gate_n_log_overflow_rejected() {
        let (mut proof, root) = dos_proof(b"smp-dos-nlog-ovf");
        proof.n_log_merkle = 64;
        expect_malformed(&proof, &root, b"smp-dos-nlog-ovf", "n_log_merkle=64");
    }

    /// Inflated n_log_merkle drives `Sha256HybridSetup::cached(1 << n_log)` — an
    /// attacker-sized allocation. The canonical-recompute gate pins n_log to the
    /// exact value `allocate_blocks_aligned` derives from offsets+counts.
    #[test]
    fn dos_gate_n_log_inflated_rejected() {
        let (mut proof, root) = dos_proof(b"smp-dos-nlog-inf");
        proof.n_log_merkle += 1;
        expect_malformed(&proof, &root, b"smp-dos-nlog-inf", "n_log_merkle+1");
    }

    /// n_routes drives `RouteF32Setup::cached(n_routes)` (R1CS build sized by
    /// it). Honest value is exactly the unique-node count.
    #[test]
    fn dos_gate_n_routes_inflated_rejected() {
        let (mut proof, root) = dos_proof(b"smp-dos-nroutes");
        proof.n_routes += 1;
        expect_malformed(&proof, &root, b"smp-dos-nroutes", "n_routes+1");
    }

    /// Block offsets are prover-chosen wire numbers; the claim-point assembly
    /// consumes them raw. They must be aligned to their (power-of-two) count —
    /// the invariant `allocate_blocks_aligned` guarantees for honest proofs.
    #[test]
    fn dos_gate_offset_misaligned_rejected() {
        let (mut proof, root) = dos_proof(b"smp-dos-off-align");
        proof.merkle_block_offsets[0] += 1;
        expect_malformed(&proof, &root, b"smp-dos-off-align", "misaligned offset");
    }

    /// An offset past the committed range would place the PD claim outside the
    /// commitment (and lets offset+count overflow-wrap on adversarial values).
    #[test]
    fn dos_gate_offset_out_of_range_rejected() {
        let (mut proof, root) = dos_proof(b"smp-dos-off-range");
        // Aligned (2^n_log is a multiple of the count) but end > 1 << n_log.
        proof.merkle_block_offsets[0] = 1usize << proof.n_log_merkle;
        expect_malformed(&proof, &root, b"smp-dos-off-range", "out-of-range offset");
    }

    /// Every pair must reference a valid physical-node entry; an out-of-range
    /// pair_phys index must be gated before any indexed use.
    #[test]
    fn dos_gate_pair_phys_out_of_range_rejected() {
        let (mut proof, root) = dos_proof(b"smp-dos-pairphys");
        proof.pair_phys[0] = proof.merkle_roots.len();
        expect_malformed(&proof, &root, b"smp-dos-pairphys", "pair_phys out of range");
    }

    /// Uniform-8 layout gate (E2 prerequisite): every honest chain is 8 blocks.
    /// A tampered non-8 count must be rejected — the global sumcheck's
    /// off_i = i·8 node×8×slot layout has no meaning otherwise.
    #[test]
    fn dos_gate_non_uniform_block_count_rejected() {
        let (mut proof, root) = dos_proof(b"smp-dos-uniform8");
        assert!(
            proof.merkle_block_counts.iter().all(|&c| c == 8),
            "honest counts must be uniform 8"
        );
        proof.merkle_block_counts[0] = 16;
        expect_malformed(&proof, &root, b"smp-dos-uniform8", "non-uniform block count");
    }

    /// n_phys > u lets junk physical nodes (never referenced by any pair) each
    /// cost the verifier a per-phys gate + a pad-forward loop, decoupled from
    /// u. Honest n_phys ≤ u; padding the per-phys vectors past u must reject.
    #[test]
    fn dos_gate_n_phys_inflated_rejected() {
        let (mut proof, root) = dos_proof(b"smp-dos-nphys");
        // Append junk physical entries (all four per-phys vectors must stay
        // length-consistent so the earlier length gate doesn't fire first).
        let junk_meta = proof.content_metas[0].clone();
        for _ in 0..4 {
            proof.merkle_roots.push([0; 8]);
            proof.merkle_native_roots.push([0; 8]);
            proof.content_metas.push(junk_meta.clone());
            proof.merkle_block_counts.push(proof.merkle_block_counts[0]);
        }
        expect_malformed(&proof, &root, b"smp-dos-nphys", "n_phys > u");
    }

    /// E1 mixed-root surface: pointing a pair at a DIFFERENT physical node's
    /// entry swaps the padded root its shift is checked against — the shift
    /// replay of its committed chain can no longer reach that root.
    #[test]
    fn forged_pair_phys_swap_rejected() {
        let (path, _entry) = honest_path_with_entry();
        let (mut proof, root) = prove_and_root(&[path]);
        assert_eq!(proof.merkle_roots.len(), 2, "two physical nodes");
        proof.pair_phys.swap(0, 1);
        let mut chv = FsChallenger::new(b"smp-entries");
        let res = verify_sound_multiproof(&proof, &root, &mut chv);
        assert!(
            res.is_err(),
            "pair→phys table permutation must be rejected, got Ok"
        );
    }

    /// E1 positive: two paths traversing the SAME physical node toward
    /// different children share one physical entry (n_phys < u) and verify.
    #[test]
    fn shared_physical_node_two_selections_accepts() {
        let node0 = MhotNodeWitness {
            children: vec![[0x11; 32], [0x22; 32]],
            selected_child: 0,
        };
        let node1 = MhotNodeWitness {
            children: vec![[0x11; 32], [0x22; 32]],
            selected_child: 1,
        };
        let i0 = MhotMembershipInput::from_node(node0);
        let i1 = MhotMembershipInput::from_node(node1);
        let root = node_root(&i0);
        let mut ch = FsChallenger::new(b"smp-shared-phys");
        let proof = prove_sound_multiproof(&[vec![i0], vec![i1]], &mut ch);
        assert_eq!(proof.merkle_shifts.len(), 2, "two pairs");
        assert_eq!(proof.merkle_roots.len(), 1, "one shared physical node");
        assert_eq!(proof.pair_phys, vec![0, 0]);
        let mut chv = FsChallenger::new(b"smp-shared-phys");
        verify_sound_multiproof(&proof, &root, &mut chv)
            .expect("two selections of one physical node must verify");
    }

    /// M1 acceptance gate: proof bytes must be INDEPENDENT of scratch-pool
    /// and setup-cache state. The pool hands out UNINITIALIZED recycled
    /// buffers (write-before-read contract) — any read-before-write bug, or
    /// any prewarm-set change that alters behavior, shows up here as a byte
    /// diff between a cold-pool prove and a dirty-pool prove of the same
    /// input. Also the regression gate for prewarm right-sizing.
    #[test]
    fn proof_bytes_independent_of_pool_state() {
        let (path, _entry) = honest_path_with_entry();
        let paths = vec![path];

        flock_core::scratch::clear();
        crate::r1cs_hashes::sha2::Sha256HybridSetup::clear_setup_cache();
        crate::mhot::route_f32::RouteF32Setup::clear_setup_cache();
        let mut ch1 = FsChallenger::new(b"smp-pool-ab");
        let p1 = prove_sound_multiproof(&paths, &mut ch1);
        let b1 = bincode::serialize(&p1).expect("serialize");

        // Dirty the pool with a different prove (recycled buffers now hold
        // that prove's stale contents), then re-prove the same input.
        let other = two_child_input(0);
        let mut chx = FsChallenger::new(b"smp-pool-dirty");
        let _ = prove_sound_multiproof(&[vec![other]], &mut chx);

        let mut ch2 = FsChallenger::new(b"smp-pool-ab");
        let p2 = prove_sound_multiproof(&paths, &mut ch2);
        let b2 = bincode::serialize(&p2).expect("serialize");
        assert_eq!(b1, b2, "proof bytes must not depend on pool/cache state");
    }

    /// Entry values are hashed by the verifier; cap their length so a single
    /// entry cannot make the verifier hash unbounded attacker data.
    #[test]
    fn dos_gate_entry_value_oversized_rejected() {
        let (path, mut entry) = honest_path_with_entry();
        let (proof, root) = prove_and_root(&[path]);
        entry.value = vec![0u8; (1 << 20) + 1];
        let mut chv = FsChallenger::new(b"smp-entries");
        let res = verify_sound_multiproof_with_entries(&proof, &root, &[entry], &mut chv);
        assert!(
            matches!(res, Err(MhotMembershipError::MalformedProof { .. })),
            "oversized entry value must be rejected by the length gate, got {res:?}"
        );
    }
