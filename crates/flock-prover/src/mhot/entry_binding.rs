//! Entry binding: native HOT routing + leaf hashing, mirrored from
//! `mhot-verify` (proof.rs / node.rs). Re-exported by `merkle_membership`.

// ---------------------------------------------------------------------------
// Entry binding: native HOT routing + leaf hashing, mirrored from mhot-verify
// (proof.rs / node.rs). The verifier re-runs the routing on the authenticated
// ContentMetas and pins each path's terminal leaf to a public (key, value).
// ---------------------------------------------------------------------------

/// One public membership claim: the (key, value) pair a path proves.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PathEntry {
    pub key: [u8; 32],
    pub value: Vec<u8>,
}

/// Mirror of `mhot-verify/src/proof.rs::compute_dense_key`: extract the
/// discriminative bits of `key` selected by the four BE u64 mask words,
/// LSB-first within each word.
pub fn compute_dense_key(key: &[u8; 32], extraction_masks: &[u64; 4]) -> u32 {
    let mut dense = 0u32;
    let mut bit_pos = 0u32;
    for (chunk_idx, &mask) in extraction_masks.iter().enumerate() {
        if mask == 0 {
            continue;
        }
        let key_chunk =
            u64::from_be_bytes(key[chunk_idx * 8..(chunk_idx + 1) * 8].try_into().unwrap());
        let mut m = mask;
        while m != 0 {
            let bit = m.trailing_zeros();
            if (key_chunk >> bit) & 1 != 0 {
                dense |= 1 << bit_pos;
            }
            bit_pos += 1;
            m &= m - 1;
        }
    }
    dense
}

/// Mirror of `mhot-verify/src/proof.rs::search_in_sparse_keys`.
pub fn search_in_sparse_keys(dense_key: u32, sparse_keys: &[u32]) -> usize {
    for i in (0..sparse_keys.len()).rev() {
        let sparse = sparse_keys[i];
        if (dense_key & sparse) == sparse {
            return i;
        }
    }
    0
}

/// Leaf content hash: full SHA-256 over the native `LeafData` bincode encoding
/// (fixint little-endian: key ‖ value_len as u64 LE ‖ value). Matches
/// `mhot-verify/src/node.rs::LeafData::compute_node_id`'s hashed bytes. Uses the
/// in-crate SHA-256 (same pad+compress the R1CS witnesses mirror) so the leaf
/// check cannot drift from a future in-circuit leaf proof.
pub fn leaf_content_hash(entry: &PathEntry) -> [u8; 32] {
    let mut buf = Vec::with_capacity(32 + 8 + entry.value.len());
    buf.extend_from_slice(&entry.key);
    buf.extend_from_slice(&(entry.value.len() as u64).to_le_bytes());
    buf.extend_from_slice(&entry.value);
    super::mhot_to_sha256::sha256_bytes(&buf)
}
