//! CPU-side MHOT semantic oracles.
//!
//! C4 M31 fold-layout cross-check:
//! `/home/ubuntu/gkr-probe/ExpanderCompilerCollection/circuit-std-rs/tests/
//! mhot_complete_path.rs` uses Poseidon over M31 lanes, not Keccak bytes. Its
//! `four_to_one{,_cpu}` builds a 16-lane Poseidon state by taking lanes
//! `slot[0..4]` from each of four child digest slots, in slot order. The
//! selected child is placed into `child_index % 4` by `place_arity4_slots` /
//! `place_slots_cpu`, with siblings occupying the other slots.
//!
//! There is no Keccak `pad10*1` padding and no fold-domain byte in the M31
//! fold. Domain separation appears in the separate content/leaf Poseidon
//! preimages (`INTERNAL_TAG_CONST`, `LEAF_TAG_CONST`), not in the 4-ary fold.
//! Flock's current `ref_witness::fold_state` is therefore not byte-layout
//! identical: it packs four 32-byte digests at bytes 0..128 and applies Keccak
//! padding at bytes 128 (`0x01`) and 135 (`0x80`). The aligned semantic is the
//! ordered 4-child slot fold; the concrete hash state layout differs by design.

/// Verify MHOT compact membership content constraints (CPU oracle).
///
/// 1. Mask bits must be contiguous from bit 0.
/// 2. Key bits above the mask width must be zero.
pub fn check_compact_content(
    key: &[bool; KEY_BITS],
    mask: &[bool; KEY_BITS],
) -> Result<(), ContentError> {
    let popcount = mask.iter().filter(|&&bit| bit).count();

    for (bit, &set) in mask.iter().enumerate() {
        if bit < popcount && !set {
            return Err(ContentError::MaskNotCompact { bit, popcount });
        }
        if bit >= popcount && set {
            return Err(ContentError::MaskNotCompact { bit, popcount });
        }
    }

    for (bit, &set) in key.iter().enumerate().skip(popcount) {
        if set {
            return Err(ContentError::KeyBitAboveMask { bit });
        }
    }

    Ok(())
}

/// Verify MHOT subtree counts along a membership path.
///
/// `counts[i]` is the subtree count at depth `i` (`0` is root). This CPU
/// oracle checks only the path-local invariants available in the PoC: the path
/// is non-empty, every count is positive, the leaf count is one, and counts do
/// not increase as we descend the path.
pub fn check_subtree_counts(counts: &[u64]) -> Result<(), CountError> {
    if counts.is_empty() {
        return Err(CountError::EmptyPath);
    }

    let leaf_count = *counts.last().expect("non-empty counts");
    if leaf_count != 1 {
        return Err(CountError::LeafNotOne { leaf_count });
    }

    for (depth, &count) in counts.iter().enumerate() {
        if count == 0 {
            return Err(CountError::ZeroCount { depth });
        }
    }

    for depth in 1..counts.len() {
        let parent = counts[depth - 1];
        let child = counts[depth];
        if child > parent {
            return Err(CountError::ChildExceedsParent {
                depth,
                parent,
                child,
            });
        }
    }

    Ok(())
}

/// Verify MHOT absence: the query key must differ from the leaf key.
///
/// MHOT absence is structural: route to the position, then assert the key at
/// that position is not the query key. No size comparison is needed.
pub fn check_absence(
    query_key: &[bool; KEY_BITS],
    leaf_key: &[bool; KEY_BITS],
) -> Result<(), AbsenceError> {
    if query_key == leaf_key {
        return Err(AbsenceError::KeysEqual);
    }
    Ok(())
}

/// Build the M31 reference fold input state before Poseidon permutation.
///
/// The M31 circuit's 4-ary fold takes the first four lanes of each 16-lane
/// child digest slot and concatenates them into a 16-lane Poseidon state:
/// `slot0[0..4] || slot1[0..4] || slot2[0..4] || slot3[0..4]`.
pub fn fold_state_m31_compatible(slots: &[[u32; M31_WIDTH]; M31_FANOUT]) -> [u32; M31_WIDTH] {
    let mut state = [0u32; M31_WIDTH];
    for (slot_idx, slot) in slots.iter().enumerate() {
        let out = slot_idx * M31_QUARTER;
        state[out..out + M31_QUARTER].copy_from_slice(&slot[..M31_QUARTER]);
    }
    state
}

pub const KEY_BITS: usize = 256;
pub const M31_WIDTH: usize = 16;
pub const M31_FANOUT: usize = 4;
pub const M31_QUARTER: usize = M31_WIDTH / M31_FANOUT;

#[derive(Debug, PartialEq, Eq)]
pub enum ContentError {
    MaskNotCompact { bit: usize, popcount: usize },
    KeyBitAboveMask { bit: usize },
}

#[derive(Debug, PartialEq, Eq)]
pub enum CountError {
    EmptyPath,
    LeafNotOne {
        leaf_count: u64,
    },
    ZeroCount {
        depth: usize,
    },
    ChildExceedsParent {
        depth: usize,
        parent: u64,
        child: u64,
    },
}

#[derive(Debug, PartialEq, Eq)]
pub enum AbsenceError {
    KeysEqual,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_compact_content() {
        let mut key = [false; KEY_BITS];
        let mut mask = [false; KEY_BITS];
        mask[0] = true;
        mask[1] = true;
        key[1] = true;

        assert!(check_compact_content(&key, &mask).is_ok());
    }

    #[test]
    fn non_compact_mask_rejected() {
        let key = [false; KEY_BITS];
        let mut mask = [false; KEY_BITS];
        mask[0] = true;
        mask[2] = true;

        assert_eq!(
            check_compact_content(&key, &mask),
            Err(ContentError::MaskNotCompact {
                bit: 1,
                popcount: 2
            })
        );
    }

    #[test]
    fn key_above_mask_rejected() {
        let mut key = [false; KEY_BITS];
        let mut mask = [false; KEY_BITS];
        mask[0] = true;
        mask[1] = true;
        key[0] = true;
        key[5] = true;

        assert_eq!(
            check_compact_content(&key, &mask),
            Err(ContentError::KeyBitAboveMask { bit: 5 })
        );
    }

    #[test]
    fn valid_counts() {
        assert!(check_subtree_counts(&[1000, 50, 3, 1]).is_ok());
    }

    #[test]
    fn leaf_not_one_rejected() {
        assert_eq!(
            check_subtree_counts(&[100, 50, 2]),
            Err(CountError::LeafNotOne { leaf_count: 2 })
        );
    }

    #[test]
    fn child_exceeds_parent_rejected() {
        assert_eq!(
            check_subtree_counts(&[10, 50, 1]),
            Err(CountError::ChildExceedsParent {
                depth: 1,
                parent: 10,
                child: 50
            })
        );
    }

    #[test]
    fn zero_count_rejected() {
        assert_eq!(
            check_subtree_counts(&[100, 0, 1]),
            Err(CountError::ZeroCount { depth: 1 })
        );
    }

    #[test]
    fn absence_different_keys_accepted() {
        let mut query = [false; KEY_BITS];
        let leaf = [false; KEY_BITS];
        query[0] = true;

        assert!(check_absence(&query, &leaf).is_ok());
    }

    #[test]
    fn absence_same_keys_rejected() {
        let key = [false; KEY_BITS];

        assert_eq!(check_absence(&key, &key), Err(AbsenceError::KeysEqual));
    }

    #[test]
    fn m31_fold_state_uses_slot_quarters() {
        let slots: [[u32; M31_WIDTH]; M31_FANOUT] =
            std::array::from_fn(|slot| std::array::from_fn(|lane| (slot * 100 + lane) as u32));

        let state = fold_state_m31_compatible(&slots);
        assert_eq!(&state[0..4], &[0, 1, 2, 3]);
        assert_eq!(&state[4..8], &[100, 101, 102, 103]);
        assert_eq!(&state[8..12], &[200, 201, 202, 203]);
        assert_eq!(&state[12..16], &[300, 301, 302, 303]);
    }
}
