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

pub const KEY_BITS: usize = 256;

#[derive(Debug, PartialEq, Eq)]
pub enum ContentError {
    MaskNotCompact { bit: usize, popcount: usize },
    KeyBitAboveMask { bit: usize },
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
}
