//! Re-export of the MHOT node → SHA-256 Merkle path conversion from
//! `native_witness`, plus additional cross-verification tests that compare
//! the Flock witness builder against an independent CPU-side Merkle root
//! computation (mirroring native MHOT's `build_merkle_root` algorithm).

pub use super::native_witness::{
    BinaryMerkleWitness as Sha256MerkleWitness,
    MhotNodeWitness as MhotNodeMerkleInput,
    mhot_node_to_sha256_merkle,
    mhot_path_to_sha256_merkle,
};

use crate::r1cs_hashes::sha2::{SHA256_IV, sha256_compress};

/// Independent CPU-side binary Merkle root computation, mirroring the native
/// MHOT `HotInnerNode::build_merkle_root` algorithm:
///   1. Pad leaves to next power of 2 with zero hashes
///   2. Hash pairs bottom-up using SHA-256 compression: H_in = SHA256_IV,
///      M = left_child(8 words) || right_child(8 words)
///
/// This is intentionally written from scratch (not calling `mhot_node_to_sha256_merkle`)
/// so it serves as a genuine cross-check.
pub fn cpu_merkle_root(children_bytes: &[[u8; 32]]) -> [u32; 8] {
    assert!(!children_bytes.is_empty());

    fn bytes_to_words_be(h: &[u8; 32]) -> [u32; 8] {
        let mut w = [0u32; 8];
        for i in 0..8 {
            w[i] = u32::from_be_bytes([h[4 * i], h[4 * i + 1], h[4 * i + 2], h[4 * i + 3]]);
        }
        w
    }

    let n = children_bytes.len().next_power_of_two();
    let mut layer: Vec<[u32; 8]> = children_bytes.iter().map(|c| bytes_to_words_be(c)).collect();
    layer.resize(n, [0u32; 8]);

    while layer.len() > 1 {
        let mut next = Vec::with_capacity(layer.len() / 2);
        for pair in layer.chunks_exact(2) {
            let mut m = [0u32; 16];
            m[..8].copy_from_slice(&pair[0]);
            m[8..].copy_from_slice(&pair[1]);
            next.push(sha256_compress(&SHA256_IV, &m));
        }
        layer = next;
    }

    layer[0]
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
    fn mhot_node_merkle_root_matches_cpu() {
        for &fanout in &[2, 3, 4, 5, 7, 8, 16, 17, 31, 32] {
            let children = make_random_children(fanout, 0xCAFE_0000 + fanout as u64);
            let node = MhotNodeMerkleInput {
                children: children.clone(),
                selected_child: 0,
            };
            let witness = mhot_node_to_sha256_merkle(&node);
            let cpu_root = cpu_merkle_root(&children);

            assert_eq!(
                witness.root, cpu_root,
                "root mismatch for fanout={fanout}: witness builder vs independent CPU"
            );
        }
    }

    #[test]
    fn mhot_node_small_fanout() {
        let children = make_random_children(2, 0xBEEF_DEAD);
        let node = MhotNodeMerkleInput {
            children: children.clone(),
            selected_child: 0,
        };
        let w = mhot_node_to_sha256_merkle(&node);

        assert_eq!(w.compressions.len(), 1, "fanout 2 → depth 1 → 1 compression");
        assert_eq!(w.b_bits.len(), 1);
        assert!(!w.b_bits[0], "child 0 is left → b_bit = false");

        fn bytes_to_words_be(h: &[u8; 32]) -> [u32; 8] {
            let mut w = [0u32; 8];
            for i in 0..8 {
                w[i] = u32::from_be_bytes([h[4 * i], h[4 * i + 1], h[4 * i + 2], h[4 * i + 3]]);
            }
            w
        }

        let leaf = bytes_to_words_be(&children[0]);
        assert_eq!(w.leaf, leaf, "leaf must be child 0");

        let mut m = [0u32; 16];
        m[..8].copy_from_slice(&bytes_to_words_be(&children[0]));
        m[8..].copy_from_slice(&bytes_to_words_be(&children[1]));
        let expected_root = sha256_compress(&SHA256_IV, &m);
        assert_eq!(w.root, expected_root, "root must be compress(IV, child0||child1)");

        // Also verify selected_child=1 (right child)
        let node_r = MhotNodeMerkleInput {
            children: children.clone(),
            selected_child: 1,
        };
        let wr = mhot_node_to_sha256_merkle(&node_r);
        assert_eq!(wr.compressions.len(), 1);
        assert!(wr.b_bits[0], "child 1 is right → b_bit = true");
        assert_eq!(wr.leaf, bytes_to_words_be(&children[1]));
        assert_eq!(wr.root, expected_root, "root is the same regardless of selected child");
    }

    #[test]
    fn mhot_node_non_power_of_two_fanout() {
        let children = make_random_children(5, 0x5555_5555);
        let node = MhotNodeMerkleInput {
            children: children.clone(),
            selected_child: 4,
        };
        let w = mhot_node_to_sha256_merkle(&node);

        // fanout 5 → padded to 8 → depth 3
        assert_eq!(w.compressions.len(), 3, "fanout 5 padded to 8 → depth 3");

        let cpu_root = cpu_merkle_root(&children);
        assert_eq!(w.root, cpu_root, "root must match independent CPU computation");

        // Replay path
        let mut current = w.leaf;
        for (i, (iv, m)) in w.compressions.iter().enumerate() {
            assert_eq!(*iv, SHA256_IV);
            if !w.b_bits[i] {
                assert_eq!(&m[..8], &current[..]);
            } else {
                assert_eq!(&m[8..], &current[..]);
            }
            current = sha256_compress(iv, m);
        }
        assert_eq!(current, w.root);
    }
}
