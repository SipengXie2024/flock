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
        let native = mhot_node_to_sha256_merkle(&input.node).native_root;
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
        let t_root = mhot_node_to_sha256_merkle(&t_node).native_root;
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
        let m_root = mhot_node_to_sha256_merkle(&m_node).native_root;
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
