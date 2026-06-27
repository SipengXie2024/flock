use crate::r1cs_hashes::sha2::{
    Compression, MerklePathProof, MerklePathVerifyError, Sha256HybridSetup,
};
use flock_core::challenger::Challenger;
use flock_core::pcs::Commitment;

use super::native_witness::{MhotNodeWitness, mhot_node_to_sha256_merkle, mhot_path_to_sha256_merkle};

/// Prove a single MHOT membership path (root-to-leaf node sequence) using
/// Flock's SHA-256 binary Merkle proof. Each MHOT node's children form a
/// binary Merkle tree; the in-node paths are concatenated into one big
/// Merkle path proved by `Sha256HybridSetup::prove_merkle_path`.
///
/// Returns `(proof, commitment, leaf, root)` so the caller can hand them
/// to `verify_mhot_sha256_path`.
pub fn prove_mhot_sha256_path<Ch: Challenger>(
    nodes: &[MhotNodeWitness],
    challenger: &mut Ch,
) -> (MerklePathProof, Commitment, [u32; 8], [u32; 8]) {
    let w = mhot_path_to_sha256_merkle(nodes);
    let n = w.compressions.len();

    let setup = Sha256HybridSetup::new(n);
    let needed = setup.n_block_slots();

    // Pad compressions and b_bits to `needed` (power of 2, >= 8).
    let mut compressions = w.compressions;
    let mut b_bits = w.b_bits;
    pad_to_power_of_two(&mut compressions, &mut b_bits, needed);

    let (proof, commitment) = setup.prove_merkle_path(&compressions, &b_bits, challenger);
    (proof, commitment, w.leaf, w.root)
}

/// Verify a previously proved MHOT membership path.
pub fn verify_mhot_sha256_path<Ch: Challenger>(
    n_compressions: usize,
    commitment: &Commitment,
    proof: &MerklePathProof,
    leaf: &[u32; 8],
    root: &[u32; 8],
    b_bits_orig: &[bool],
    challenger: &mut Ch,
) -> Result<(), MerklePathVerifyError> {
    let setup = Sha256HybridSetup::new(n_compressions);
    let needed = setup.n_block_slots();
    let mut b_bits = b_bits_orig.to_vec();
    // Pad b_bits the same way prover did.
    b_bits.resize(needed, false);
    setup.verify_merkle_path(commitment, proof, leaf, root, &b_bits, challenger)
}

/// Pad compressions and b_bits to `needed` slots with dummy identity
/// compressions. The dummy compression hashes `(current_output, zeros)` so the
/// chain remains valid but inert.
fn pad_to_power_of_two(
    compressions: &mut Vec<Compression>,
    b_bits: &mut Vec<bool>,
    needed: usize,
) {
    use crate::r1cs_hashes::sha2::{SHA256_IV, sha256_compress};

    if compressions.len() >= needed {
        return;
    }

    // The last real compression's output becomes the chain continuation value.
    let last_output = if compressions.is_empty() {
        [0u32; 8]
    } else {
        let (iv, m) = &compressions[compressions.len() - 1];
        sha256_compress(iv, m)
    };

    let mut current = last_output;
    while compressions.len() < needed {
        let sibling = [0u32; 8];
        let mut m = [0u32; 16];
        // Put current in left (b_bit = false), zero sibling in right.
        m[..8].copy_from_slice(&current);
        m[8..].copy_from_slice(&sibling);
        compressions.push((SHA256_IV, m));
        b_bits.push(false);
        current = sha256_compress(&SHA256_IV, &m);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mhot::native_witness::MhotNodeWitness;
    use flock_core::challenger::FsChallenger;

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
    fn merkle_membership_single_node_roundtrip() {
        let node = MhotNodeWitness {
            children: make_random_children(8, 0xABCD_1234),
            selected_child: 5,
        };
        let mut ch = FsChallenger::new(b"mhot-sha256-single");
        let (proof, commitment, leaf, root) =
            prove_mhot_sha256_path(&[node.clone()], &mut ch);

        let w = crate::mhot::native_witness::mhot_node_to_sha256_merkle(&node);
        let n = w.compressions.len();

        let mut chv = FsChallenger::new(b"mhot-sha256-single");
        verify_mhot_sha256_path(
            n,
            &commitment,
            &proof,
            &leaf,
            &root,
            &w.b_bits,
            &mut chv,
        )
        .expect("single node roundtrip must verify");
    }

    #[test]
    fn merkle_membership_three_node_path_roundtrip() {
        let nodes = vec![
            MhotNodeWitness {
                children: make_random_children(8, 0x1111),
                selected_child: 2,
            },
            MhotNodeWitness {
                children: make_random_children(4, 0x2222),
                selected_child: 1,
            },
            MhotNodeWitness {
                children: make_random_children(2, 0x3333),
                selected_child: 0,
            },
        ];
        let mut ch = FsChallenger::new(b"mhot-sha256-3node");
        let (proof, commitment, leaf, root) =
            prove_mhot_sha256_path(&nodes, &mut ch);

        let w = crate::mhot::native_witness::mhot_path_to_sha256_merkle(&nodes);
        let n = w.compressions.len();

        let mut chv = FsChallenger::new(b"mhot-sha256-3node");
        verify_mhot_sha256_path(
            n,
            &commitment,
            &proof,
            &leaf,
            &root,
            &w.b_bits,
            &mut chv,
        )
        .expect("3-node path roundtrip must verify");
    }

    #[test]
    fn merkle_membership_rejects_wrong_leaf() {
        let node = MhotNodeWitness {
            children: make_random_children(8, 0xBAD_CAFE),
            selected_child: 0,
        };
        let mut ch = FsChallenger::new(b"mhot-sha256-wrong-leaf");
        let (proof, commitment, leaf, root) =
            prove_mhot_sha256_path(&[node.clone()], &mut ch);

        let w = crate::mhot::native_witness::mhot_node_to_sha256_merkle(&node);
        let n = w.compressions.len();

        let mut bad_leaf = leaf;
        bad_leaf[0] ^= 1;

        let mut chv = FsChallenger::new(b"mhot-sha256-wrong-leaf");
        let res = verify_mhot_sha256_path(
            n,
            &commitment,
            &proof,
            &bad_leaf,
            &root,
            &w.b_bits,
            &mut chv,
        );
        assert!(res.is_err(), "verifier must reject wrong leaf");
    }
}
