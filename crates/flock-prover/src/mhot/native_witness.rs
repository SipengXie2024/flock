use crate::r1cs_hashes::sha2::{Compression, SHA256_IV, sha256_compress};

/// Witness data for a single MHOT node: the child digests and which child
/// the membership path descends through.
#[derive(Clone, Debug)]
pub struct MhotNodeWitness {
    pub children: Vec<[u8; 32]>,
    pub selected_child: usize,
}

/// The binary Merkle path extracted from one MHOT node, ready for Flock's
/// SHA-256 `prove_merkle_path`.
#[derive(Clone, Debug)]
pub struct BinaryMerkleWitness {
    pub compressions: Vec<Compression>,
    pub b_bits: Vec<bool>,
    pub leaf: [u32; 8],
    /// Chain root. Legacy order forces b_bits[0]=false so this can differ from
    /// `native_root`; with `native_order = true` the two are equal.
    pub root: [u32; 8],
    /// The real MHOT in-node Merkle root (original tree structure preserved).
    pub native_root: [u32; 8],
}

fn bytes_to_words(h: &[u8; 32]) -> [u32; 8] {
    let mut w = [0u32; 8];
    for i in 0..8 {
        w[i] = u32::from_be_bytes([h[4 * i], h[4 * i + 1], h[4 * i + 2], h[4 * i + 3]]);
    }
    w
}

/// Convert an MHOT node into a binary Merkle path witness for SHA-256.
///
/// The node's children are padded to the next power of two with zero hashes,
/// then a binary Merkle tree is built bottom-up. The path from
/// `selected_child` to the root yields one `Compression` per tree level.
///
/// `native_order = false` (legacy): the depth-0 leaf is forced into
/// X_L = M[0..8] with b_bits[0] = false. When the selected child is at an odd
/// position this produces a chain root (`root`) that differs from
/// `native_root`. Callers such as `mhot_to_sha256` and `merkle_path_common`'s
/// R1CS binding rely on this convention.
///
/// `native_order = true`: every depth including 0 places the current hash per
/// its real side bit, so the chain is the true in-node tree path and
/// `root == native_root`. The merkle shift authenticates real b_bits[0] since
/// the forced-B(0)=0 convention was removed (Route-2 foundation).
pub fn mhot_node_to_sha256_merkle(
    node: &MhotNodeWitness,
    native_order: bool,
) -> BinaryMerkleWitness {
    assert!(!node.children.is_empty(), "node must have at least one child");
    assert!(
        node.selected_child < node.children.len(),
        "selected_child {} out of range (fanout {})",
        node.selected_child,
        node.children.len(),
    );

    let n = node.children.len().next_power_of_two();
    let depth = n.trailing_zeros() as usize;
    assert!(depth >= 1, "need at least 2 children for a binary Merkle tree");

    let mut leaves_w: Vec<[u32; 8]> = node
        .children
        .iter()
        .map(|c| bytes_to_words(c))
        .collect();
    leaves_w.resize(n, [0u32; 8]);

    // Build the complete binary tree bottom-up (for sibling lookups and native root).
    let mut layers: Vec<Vec<[u32; 8]>> = Vec::with_capacity(depth + 1);
    layers.push(leaves_w);

    for d in 0..depth {
        let prev = &layers[d];
        let mut cur = Vec::with_capacity(prev.len() / 2);
        for pair in prev.chunks_exact(2) {
            let mut m = [0u32; 16];
            m[..8].copy_from_slice(&pair[0]);
            m[8..].copy_from_slice(&pair[1]);
            cur.push(sha256_compress(&SHA256_IV, &m));
        }
        layers.push(cur);
    }

    let native_root = layers[depth][0];
    let leaf = layers[0][node.selected_child];

    let mut compressions = Vec::with_capacity(depth);
    let mut b_bits = Vec::with_capacity(depth);
    let mut idx = node.selected_child;
    let mut current = leaf;

    for d in 0..depth {
        let sibling_idx = idx ^ 1;
        let sibling = layers[d][sibling_idx];
        let is_right = (idx & 1) == 1;
        let place_right = is_right && (native_order || d > 0);

        let mut m = [0u32; 16];
        if place_right {
            m[..8].copy_from_slice(&sibling);
            m[8..].copy_from_slice(&current);
        } else {
            m[..8].copy_from_slice(&current);
            m[8..].copy_from_slice(&sibling);
        }

        compressions.push((SHA256_IV, m));
        b_bits.push(place_right);
        current = sha256_compress(&SHA256_IV, &m);
        idx >>= 1;
    }

    let flock_root = current;

    BinaryMerkleWitness {
        compressions,
        b_bits,
        leaf,
        root: flock_root,
        native_root,
    }
}

/// Convert a sequence of MHOT nodes into per-node binary Merkle witnesses.
///
/// Each node is converted independently (not concatenated into a single chain).
/// Cross-node linking (parent's selected child == child's Merkle root) is
/// verified at a higher protocol level, not within the Merkle-path proofs.
pub fn mhot_nodes_to_sha256_merkle(
    nodes: &[MhotNodeWitness],
) -> Vec<BinaryMerkleWitness> {
    nodes.iter().map(|n| mhot_node_to_sha256_merkle(n, false)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_random_children(n: usize, seed: u64) -> Vec<[u8; 32]> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                let mut h = [0u8; 32];
                for b in h.iter_mut() {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    *b = s as u8;
                }
                h
            })
            .collect()
    }

    #[test]
    fn single_node_fanout8_witness() {
        let children = make_random_children(8, 0xDEADBEEF);
        let node = MhotNodeWitness {
            children: children.clone(),
            selected_child: 3,
        };
        let w = mhot_node_to_sha256_merkle(&node, false);

        assert_eq!(w.compressions.len(), 3, "fanout 8 → depth 3");
        assert_eq!(w.b_bits.len(), 3);
        assert_eq!(w.leaf, bytes_to_words(&children[3]));
        assert!(!w.b_bits[0], "b_bits[0] must be false (Flock convention)");

        // Verify the chain: replaying compressions must produce root.
        let mut current = w.leaf;
        for (i, comp) in w.compressions.iter().enumerate() {
            let (iv, m) = comp;
            assert_eq!(*iv, SHA256_IV);
            if !w.b_bits[i] {
                assert_eq!(&m[..8], &current[..], "left child must be current hash");
            } else {
                assert_eq!(&m[8..], &current[..], "right child must be current hash");
            }
            current = sha256_compress(iv, m);
        }
        assert_eq!(current, w.root, "chain must lead to Flock root");
    }

    #[test]
    fn witness_bbits0_always_false() {
        for sel in 0..8 {
            let children = make_random_children(8, 0xF000 + sel as u64);
            let node = MhotNodeWitness {
                children,
                selected_child: sel,
            };
            let w = mhot_node_to_sha256_merkle(&node, false);
            assert!(!w.b_bits[0], "b_bits[0] must be false for selected_child={sel}");
        }
    }

    #[test]
    fn native_root_matches_when_leaf_is_left_child() {
        let children = make_random_children(4, 0xAAAA);
        let node = MhotNodeWitness {
            children,
            selected_child: 0,
        };
        let w = mhot_node_to_sha256_merkle(&node, false);
        assert_eq!(
            w.root, w.native_root,
            "when leaf is a left child, Flock root == native root"
        );
    }

    #[test]
    fn native_order_root_equals_native_root() {
        for fanout in [2usize, 4, 5, 8] {
            for sel in 0..fanout {
                let children = make_random_children(fanout, 0x4E00 + (fanout * 100 + sel) as u64);
                let node = MhotNodeWitness {
                    children,
                    selected_child: sel,
                };
                let w = mhot_node_to_sha256_merkle(&node, true);
                assert_eq!(
                    w.root, w.native_root,
                    "native_order root must equal native_root (fanout={fanout} sel={sel})"
                );
                assert_eq!(
                    w.b_bits[0],
                    sel & 1 == 1,
                    "b_bits[0] must be the real depth-0 side (fanout={fanout} sel={sel})"
                );
                // Replaying the chain with the real side bits must land on native_root.
                let mut current = w.leaf;
                for (i, (iv, m)) in w.compressions.iter().enumerate() {
                    if !w.b_bits[i] {
                        assert_eq!(&m[..8], &current[..], "left placement at depth {i}");
                    } else {
                        assert_eq!(&m[8..], &current[..], "right placement at depth {i}");
                    }
                    current = sha256_compress(iv, m);
                }
                assert_eq!(current, w.native_root, "chain must lead to native_root");
            }
        }
    }

    #[test]
    fn multi_node_witnesses_independent() {
        let nodes: Vec<MhotNodeWitness> = vec![
            MhotNodeWitness {
                children: make_random_children(8, 111),
                selected_child: 2,
            },
            MhotNodeWitness {
                children: make_random_children(4, 222),
                selected_child: 1,
            },
            MhotNodeWitness {
                children: make_random_children(2, 333),
                selected_child: 0,
            },
        ];
        let witnesses = mhot_nodes_to_sha256_merkle(&nodes);
        assert_eq!(witnesses.len(), 3);
        assert_eq!(witnesses[0].compressions.len(), 3);
        assert_eq!(witnesses[1].compressions.len(), 2);
        assert_eq!(witnesses[2].compressions.len(), 1);
        for w in &witnesses {
            assert!(!w.b_bits[0], "b_bits[0] must be false for each node");
        }
    }
}
