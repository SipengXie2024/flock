use crate::r1cs_hashes::sha2::{self, Compression, SHA256_IV, sha256_compress};

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
    pub root: [u32; 8],
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
/// then a binary Merkle tree is built bottom-up using SHA-256 compression.
/// The path from `selected_child` to the root yields one `Compression` per
/// tree level, along with the direction bits.
pub fn mhot_node_to_sha256_merkle(node: &MhotNodeWitness) -> BinaryMerkleWitness {
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
    // Pad with zeros to the next power of two.
    leaves_w.resize(n, [0u32; 8]);

    // Build the complete binary tree bottom-up.
    // Layer 0 = leaves, layer `depth` = root.
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

    let root = layers[depth][0];
    let leaf = layers[0][node.selected_child];

    // Extract path: walk from the leaf upward.
    let mut compressions = Vec::with_capacity(depth);
    let mut b_bits = Vec::with_capacity(depth);
    let mut idx = node.selected_child;

    for d in 0..depth {
        let sibling_idx = idx ^ 1;
        let sibling = layers[d][sibling_idx];
        // b_bit = false means "selected is left child (M[0..8])"
        // b_bit = true  means "selected is right child (M[8..16])"
        let is_right = (idx & 1) == 1;

        let layer_hash = layers[d][idx];
        let m = if !is_right {
            let mut m = [0u32; 16];
            m[..8].copy_from_slice(&layer_hash);
            m[8..].copy_from_slice(&sibling);
            m
        } else {
            let mut m = [0u32; 16];
            m[..8].copy_from_slice(&sibling);
            m[8..].copy_from_slice(&layer_hash);
            m
        };

        compressions.push((SHA256_IV, m));
        b_bits.push(is_right);
        idx >>= 1;
    }

    BinaryMerkleWitness {
        compressions,
        b_bits,
        leaf,
        root,
    }
}

/// Convert a multi-node MHOT path (root-to-leaf order) into a single
/// concatenated binary Merkle witness for a chained `prove_merkle_path` call.
///
/// The per-node witnesses are concatenated in the given order. The overall
/// leaf is the selected child of the last (deepest) MHOT node, and the
/// overall root is the Merkle root of the first (topmost) MHOT node.
pub fn mhot_path_to_sha256_merkle(
    nodes: &[MhotNodeWitness],
) -> BinaryMerkleWitness {
    assert!(!nodes.is_empty(), "path must contain at least one node");

    let mut all_compressions: Vec<Compression> = Vec::new();
    let mut all_b_bits: Vec<bool> = Vec::new();

    let first_w = mhot_node_to_sha256_merkle(&nodes[0]);
    let root = first_w.root;

    for node in nodes {
        let w = mhot_node_to_sha256_merkle(node);
        all_compressions.extend_from_slice(&w.compressions);
        all_b_bits.extend_from_slice(&w.b_bits);
    }

    let last_w = mhot_node_to_sha256_merkle(&nodes[nodes.len() - 1]);
    let leaf = last_w.leaf;

    BinaryMerkleWitness {
        compressions: all_compressions,
        b_bits: all_b_bits,
        leaf,
        root,
    }
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
        let w = mhot_node_to_sha256_merkle(&node);

        assert_eq!(w.compressions.len(), 3, "fanout 8 → depth 3");
        assert_eq!(w.b_bits.len(), 3);
        assert_eq!(w.leaf, bytes_to_words(&children[3]));

        // Verify the path manually: replaying compressions must produce root.
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
        assert_eq!(current, w.root, "path must lead to root");
    }

    #[test]
    fn multi_node_path_witness() {
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
        let w = mhot_path_to_sha256_merkle(&nodes);
        // fanout 8 → depth 3, fanout 4 → depth 2, fanout 2 → depth 1 = total 6
        assert_eq!(w.compressions.len(), 6, "total compressions for [8,4,2] path");
        assert_eq!(w.b_bits.len(), 6);
    }
}
