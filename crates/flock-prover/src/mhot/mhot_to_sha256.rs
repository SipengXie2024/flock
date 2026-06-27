//! Re-export of the MHOT node → SHA-256 Merkle path conversion from
//! `native_witness`, plus additional cross-verification tests that compare
//! the Flock witness builder against an independent CPU-side Merkle root
//! computation (mirroring native MHOT's `build_merkle_root` algorithm).
//!
//! Also provides the Level 2 content-hash chain: serializes the MHOT content
//! preimage `H(masks || sparse_keys || merkle_root || counts)` into SHA-256
//! Merkle-Damgard compressions suitable for `Sha256HybridSetup::prove_chain`.

pub use super::native_witness::{
    BinaryMerkleWitness as Sha256MerkleWitness,
    MhotNodeWitness as MhotNodeMerkleInput,
    mhot_node_to_sha256_merkle,
    mhot_nodes_to_sha256_merkle,
};

use crate::r1cs_hashes::sha2::{Compression, SHA256_IV, sha256_compress};

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

/// Serialize the MHOT content-hash preimage into raw bytes, matching the
/// native `compute_content_hash` layout on little-endian platforms:
///   extraction_masks (4 x u64 LE) || sparse_keys (len x u32 LE) ||
///   merkle_root (32 bytes raw)    || counts (len x u32 LE)
///
/// `len` = number of children (= sparse_keys.len() = counts.len()).
pub fn content_preimage_bytes(
    extraction_masks: &[u64; 4],
    sparse_keys: &[u32],
    merkle_root: &[u8; 32],
    counts: &[u32],
) -> Vec<u8> {
    assert_eq!(
        sparse_keys.len(),
        counts.len(),
        "sparse_keys and counts must have the same length"
    );
    let len = sparse_keys.len();
    let total = 32 + len * 4 + 32 + len * 4;
    let mut buf = Vec::with_capacity(total);

    for &mask in extraction_masks {
        buf.extend_from_slice(&mask.to_le_bytes());
    }
    for &spk in sparse_keys {
        buf.extend_from_slice(&spk.to_le_bytes());
    }
    buf.extend_from_slice(merkle_root);
    for &count in counts {
        buf.extend_from_slice(&count.to_le_bytes());
    }

    debug_assert_eq!(buf.len(), total);
    buf
}

/// Apply NIST SHA-256 padding (FIPS 180-4 section 5.1.1) to an arbitrary
/// byte message, then split into 64-byte blocks, returning a sequence of
/// `Compression = ([u32; 8], [u32; 16])` with proper Merkle-Damgard chaining.
///
/// Block 0 uses `SHA256_IV` as `H_in`; subsequent blocks chain the output of
/// the previous compression as `H_in`.
pub fn bytes_to_sha256_chain(data: &[u8]) -> Vec<Compression> {
    let padded = sha256_pad(data);
    assert_eq!(padded.len() % 64, 0);

    let n_blocks = padded.len() / 64;
    let mut compressions = Vec::with_capacity(n_blocks);
    let mut cv = SHA256_IV;

    for block_idx in 0..n_blocks {
        let block = &padded[block_idx * 64..(block_idx + 1) * 64];
        let m = bytes_to_message_words(block);
        compressions.push((cv, m));
        cv = sha256_compress(&cv, &m);
    }

    compressions
}

/// The final chaining value (the SHA-256 digest as 8 x u32 words) after
/// processing all compressions in the chain.
pub fn sha256_chain_output(compressions: &[Compression]) -> [u32; 8] {
    let mut cv = compressions[0].0;
    for (h_in, m) in compressions {
        debug_assert_eq!(*h_in, cv);
        cv = sha256_compress(h_in, m);
    }
    cv
}

/// Convert the content-hash preimage into a SHA-256 Merkle-Damgard chain of
/// compressions. This is the Level 2 entry point: given the MHOT node's
/// semantic fields, produce `Vec<Compression>` ready for `prove_chain`.
///
/// Returns `(compressions, content_hash_words)` where `content_hash_words`
/// is the 8-word SHA-256 output (the content hash).
pub fn content_hash_to_sha256_chain(
    extraction_masks: &[u64; 4],
    sparse_keys: &[u32],
    merkle_root: &[u8; 32],
    counts: &[u32],
) -> (Vec<Compression>, [u32; 8]) {
    let preimage = content_preimage_bytes(extraction_masks, sparse_keys, merkle_root, counts);
    let compressions = bytes_to_sha256_chain(&preimage);
    let digest = sha256_chain_output(&compressions);
    (compressions, digest)
}

/// Reference SHA-256 of arbitrary bytes via the standard Merkle-Damgard
/// construction. Returns the digest as 32 bytes (big-endian word encoding).
/// Used for cross-checking against native MHOT.
pub fn sha256_bytes(data: &[u8]) -> [u8; 32] {
    let compressions = bytes_to_sha256_chain(data);
    let words = sha256_chain_output(&compressions);
    let mut out = [0u8; 32];
    for (i, &w) in words.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&w.to_be_bytes());
    }
    out
}

// ── internal helpers ─────────────────────────────────────────────────────

/// NIST SHA-256 padding: append 0x80, zero-pad to 56 mod 64, append 64-bit
/// big-endian bit length.
fn sha256_pad(data: &[u8]) -> Vec<u8> {
    let bit_len = (data.len() as u64) * 8;
    let mut padded = Vec::with_capacity(data.len() + 72);
    padded.extend_from_slice(data);
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0x00);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());
    debug_assert_eq!(padded.len() % 64, 0);
    padded
}

/// Parse a 64-byte block into 16 big-endian u32 words (SHA-256 convention).
fn bytes_to_message_words(block: &[u8]) -> [u32; 16] {
    let mut m = [0u32; 16];
    for i in 0..16 {
        m[i] = u32::from_be_bytes([
            block[4 * i],
            block[4 * i + 1],
            block[4 * i + 2],
            block[4 * i + 3],
        ]);
    }
    m
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
    fn mhot_node_native_root_matches_cpu() {
        for &fanout in &[2, 3, 4, 5, 7, 8, 16, 17, 31, 32] {
            let children = make_random_children(fanout, 0xCAFE_0000 + fanout as u64);
            let node = MhotNodeMerkleInput {
                children: children.clone(),
                selected_child: 0,
            };
            let witness = mhot_node_to_sha256_merkle(&node);
            let cpu_root = cpu_merkle_root(&children);

            assert_eq!(
                witness.native_root, cpu_root,
                "native_root mismatch for fanout={fanout}: witness vs CPU"
            );
            // selected_child=0 is always a left child, so Flock root == native root.
            assert_eq!(
                witness.root, cpu_root,
                "when selected=0 (left child), Flock root must match native root"
            );
        }
    }

    #[test]
    fn mhot_node_small_fanout() {
        let children = make_random_children(2, 0xBEEF_DEAD);

        fn bytes_to_words_be(h: &[u8; 32]) -> [u32; 8] {
            let mut w = [0u32; 8];
            for i in 0..8 {
                w[i] = u32::from_be_bytes([h[4 * i], h[4 * i + 1], h[4 * i + 2], h[4 * i + 3]]);
            }
            w
        }

        // selected_child=0 (left child): Flock root == native root
        let node = MhotNodeMerkleInput {
            children: children.clone(),
            selected_child: 0,
        };
        let w = mhot_node_to_sha256_merkle(&node);

        assert_eq!(w.compressions.len(), 1, "fanout 2 → depth 1 → 1 compression");
        assert!(!w.b_bits[0], "b_bits[0] must be false (Flock convention)");
        assert_eq!(w.leaf, bytes_to_words_be(&children[0]));

        let mut m = [0u32; 16];
        m[..8].copy_from_slice(&bytes_to_words_be(&children[0]));
        m[8..].copy_from_slice(&bytes_to_words_be(&children[1]));
        let expected_native_root = sha256_compress(&SHA256_IV, &m);
        assert_eq!(w.root, expected_native_root);
        assert_eq!(w.native_root, expected_native_root);

        // selected_child=1 (right child): b_bits[0] still false, Flock root differs
        let node_r = MhotNodeMerkleInput {
            children: children.clone(),
            selected_child: 1,
        };
        let wr = mhot_node_to_sha256_merkle(&node_r);
        assert_eq!(wr.compressions.len(), 1);
        assert!(!wr.b_bits[0], "b_bits[0] must be false even for right child");
        assert_eq!(wr.leaf, bytes_to_words_be(&children[1]));
        assert_eq!(wr.native_root, expected_native_root,
            "native root is the same regardless of selected child");
        // Flock root: compress(IV, child1||child0) ≠ compress(IV, child0||child1)
        let mut m_swapped = [0u32; 16];
        m_swapped[..8].copy_from_slice(&bytes_to_words_be(&children[1]));
        m_swapped[8..].copy_from_slice(&bytes_to_words_be(&children[0]));
        assert_eq!(wr.root, sha256_compress(&SHA256_IV, &m_swapped));
    }

    #[test]
    fn mhot_node_non_power_of_two_fanout() {
        let children = make_random_children(5, 0x5555_5555);
        let node = MhotNodeMerkleInput {
            children: children.clone(),
            selected_child: 4,
        };
        let w = mhot_node_to_sha256_merkle(&node);

        assert_eq!(w.compressions.len(), 3, "fanout 5 padded to 8 → depth 3");

        let cpu_root = cpu_merkle_root(&children);
        assert_eq!(w.native_root, cpu_root, "native root must match CPU");

        // Replay chain from leaf to Flock root
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
        assert_eq!(current, w.root, "chain must lead to Flock root");
    }

    // ─── Content hash chain tests ────────────────────────────────────────

    #[test]
    fn content_preimage_layout_matches_native() {
        let masks: [u64; 4] = [0x0102030405060708, 0x090A0B0C0D0E0F10, 0x1112131415161718, 0x191A1B1C1D1E1F20];
        let keys: Vec<u32> = vec![0xAABBCCDD, 0x11223344];
        let root = [0x42u8; 32];
        let counts: Vec<u32> = vec![100, 1];

        let buf = content_preimage_bytes(&masks, &keys, &root, &counts);

        // extraction_masks: 32 bytes LE
        assert_eq!(&buf[0..8], &masks[0].to_le_bytes());
        assert_eq!(&buf[8..16], &masks[1].to_le_bytes());
        assert_eq!(&buf[16..24], &masks[2].to_le_bytes());
        assert_eq!(&buf[24..32], &masks[3].to_le_bytes());

        // sparse_keys: 2 * 4 = 8 bytes LE
        assert_eq!(&buf[32..36], &keys[0].to_le_bytes());
        assert_eq!(&buf[36..40], &keys[1].to_le_bytes());

        // merkle_root: 32 bytes raw
        assert_eq!(&buf[40..72], &root);

        // counts: 2 * 4 = 8 bytes LE
        assert_eq!(&buf[72..76], &counts[0].to_le_bytes());
        assert_eq!(&buf[76..80], &counts[1].to_le_bytes());

        assert_eq!(buf.len(), 80);
    }

    #[test]
    fn sha256_pad_produces_correct_blocks() {
        // Empty message: 1 block (0x80 + 55 zeros + 8 length bytes)
        let padded = sha256_pad(&[]);
        assert_eq!(padded.len(), 64);
        assert_eq!(padded[0], 0x80);
        for &b in &padded[1..56] {
            assert_eq!(b, 0);
        }
        assert_eq!(&padded[56..64], &0u64.to_be_bytes());

        // 55 bytes: exactly fills to 1 block
        let data55 = vec![0xAB; 55];
        let padded55 = sha256_pad(&data55);
        assert_eq!(padded55.len(), 64);
        assert_eq!(padded55[55], 0x80);
        let bit_len = (55u64 * 8).to_be_bytes();
        assert_eq!(&padded55[56..64], &bit_len);

        // 56 bytes: spills to 2 blocks
        let data56 = vec![0xCD; 56];
        let padded56 = sha256_pad(&data56);
        assert_eq!(padded56.len(), 128);

        // 64 bytes: spills to 2 blocks
        let data64 = vec![0xEF; 64];
        let padded64 = sha256_pad(&data64);
        assert_eq!(padded64.len(), 128);
    }

    #[test]
    fn content_chain_roundtrip() {
        let masks: [u64; 4] = [0xFF, 0, 0, 0];
        let keys: Vec<u32> = vec![42, 7, 99];
        let root = [0xABu8; 32];
        let counts: Vec<u32> = vec![1000, 500, 1];

        let (compressions, digest) =
            content_hash_to_sha256_chain(&masks, &keys, &root, &counts);

        // The chain must be self-consistent: replaying compressions yields
        // the same digest.
        let replayed = sha256_chain_output(&compressions);
        assert_eq!(replayed, digest);

        // Cross-check: sha256_bytes of the same preimage must match.
        let preimage = content_preimage_bytes(&masks, &keys, &root, &counts);
        let ref_digest_bytes = sha256_bytes(&preimage);
        let mut digest_bytes = [0u8; 32];
        for (i, &w) in digest.iter().enumerate() {
            digest_bytes[4 * i..4 * i + 4].copy_from_slice(&w.to_be_bytes());
        }
        assert_eq!(digest_bytes, ref_digest_bytes, "chain digest must match reference");

        // Chaining values: first compression starts from IV, subsequent
        // compressions chain the previous output.
        assert_eq!(compressions[0].0, SHA256_IV);
        for i in 1..compressions.len() {
            let expected_cv = sha256_compress(&compressions[i - 1].0, &compressions[i - 1].1);
            assert_eq!(
                compressions[i].0, expected_cv,
                "chaining value mismatch at block {i}"
            );
        }
    }

    #[test]
    fn content_chain_various_fanouts() {
        for fanout in [1, 2, 4, 8, 16, 22, 32] {
            let masks: [u64; 4] = [
                (1u64 << fanout.min(64)) - 1,
                0,
                0,
                0,
            ];
            let keys: Vec<u32> = (0..fanout).map(|i| i as u32 * 7 + 3).collect();
            let root = [fanout as u8; 32];
            let counts: Vec<u32> = (0..fanout).map(|i| (fanout - i) as u32).collect();

            let (compressions, digest) =
                content_hash_to_sha256_chain(&masks, &keys, &root, &counts);

            // Preimage size = 32 + fanout*4 + 32 + fanout*4 = 64 + 8*fanout
            let preimage_len = 64 + 8 * fanout;
            // After padding: ceil((preimage_len + 9) / 64) blocks
            let expected_blocks = (preimage_len + 9 + 63) / 64;
            assert_eq!(
                compressions.len(),
                expected_blocks,
                "wrong block count for fanout={fanout} (preimage {preimage_len} bytes)"
            );

            let replayed = sha256_chain_output(&compressions);
            assert_eq!(replayed, digest, "replay mismatch for fanout={fanout}");

            let ref_bytes = sha256_bytes(&content_preimage_bytes(&masks, &keys, &root, &counts));
            let mut digest_bytes = [0u8; 32];
            for (i, &w) in digest.iter().enumerate() {
                digest_bytes[4 * i..4 * i + 4].copy_from_slice(&w.to_be_bytes());
            }
            assert_eq!(
                digest_bytes, ref_bytes,
                "digest mismatch vs reference for fanout={fanout}"
            );
        }
    }

    #[test]
    fn content_chain_wrong_masks_different_digest() {
        let masks_a: [u64; 4] = [0xFFFF, 0, 0, 0];
        let masks_b: [u64; 4] = [0xFFFE, 0, 0, 0];
        let keys = vec![1u32, 2, 3, 4];
        let root = [0u8; 32];
        let counts = vec![10u32, 5, 2, 1];

        let (_, digest_a) = content_hash_to_sha256_chain(&masks_a, &keys, &root, &counts);
        let (_, digest_b) = content_hash_to_sha256_chain(&masks_b, &keys, &root, &counts);

        assert_ne!(
            digest_a, digest_b,
            "different masks must produce different content hashes"
        );
    }

    #[test]
    fn sha256_bytes_matches_known_vector() {
        // SHA-256("abc") = ba7816bf 8f01cfea 414140de 5dae2223
        //                  b00361a3 96177a9c b410ff61 f20015ad
        let digest = sha256_bytes(b"abc");
        assert_eq!(
            digest,
            [
                0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
                0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
                0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
                0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad,
            ],
            "SHA-256('abc') known test vector mismatch"
        );
    }
}
