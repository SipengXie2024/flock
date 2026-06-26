use super::ref_witness::Digest;
use super::schedule::{MhotHashSchedule, WireEndpoint};

/// Error indicating which wire failed the equality check.
#[derive(Debug)]
pub struct WiringError {
    pub wire_index: usize,
    pub src_digest: Digest,
    pub dst_digest: Digest,
}

/// CPU-side check: verify that every wire's src and dst carry the same 256-bit digest.
///
/// `atom_outputs[atom_id]` = the 256-bit digest output of that atom.
/// `atom_inputs[atom_id]` = the 200-byte input state of that atom.
/// `leaf_digests[leaf_index]` = the external leaf digest.
/// `public_root` = the claimed tree root.
pub fn check_wiring_cpu(
    sched: &MhotHashSchedule,
    atom_outputs: &[Digest],
    atom_inputs: &[[u8; 200]],
    leaf_digests: &[Digest],
    public_root: &Digest,
) -> Result<(), WiringError> {
    for (wire_index, wire) in sched.wires.iter().enumerate() {
        let src_digest = endpoint_digest(
            &wire.src,
            atom_outputs,
            atom_inputs,
            leaf_digests,
            public_root,
        );
        let dst_digest = endpoint_digest(
            &wire.dst,
            atom_outputs,
            atom_inputs,
            leaf_digests,
            public_root,
        );
        if src_digest != dst_digest {
            return Err(WiringError {
                wire_index,
                src_digest,
                dst_digest,
            });
        }
    }

    Ok(())
}

/// Helper: compute atom outputs by running keccak-f on each atom's input state.
/// Uses the same keccak-f as ref_witness.
pub fn compute_atom_outputs(atom_states: &[[u8; 200]]) -> Vec<Digest> {
    atom_states
        .iter()
        .map(crate::mhot::ref_witness::keccak_f_digest)
        .collect()
}

fn endpoint_digest(
    endpoint: &WireEndpoint,
    atom_outputs: &[Digest],
    atom_inputs: &[[u8; 200]],
    leaf_digests: &[Digest],
    public_root: &Digest,
) -> Digest {
    match *endpoint {
        WireEndpoint::AtomOutput { atom_id } => atom_outputs[atom_id],
        WireEndpoint::AtomInput {
            atom_id,
            child_slot,
        } => atom_input_digest(&atom_inputs[atom_id], child_slot),
        WireEndpoint::PublicRoot => *public_root,
        WireEndpoint::LeafDigest { leaf_index } => leaf_digests[leaf_index],
    }
}

fn atom_input_digest(state: &[u8; 200], child_slot: usize) -> Digest {
    assert!(child_slot < 4, "fold child slot must be in 0..4");
    let start = child_slot * 32;
    let mut digest = [0u8; 32];
    digest.copy_from_slice(&state[start..start + 32]);
    digest
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mhot::ref_witness::{build_ref_witness, Digest};
    use crate::mhot::schedule::MhotHashSchedule;

    #[test]
    fn glue_accepts_valid() {
        let sched = MhotHashSchedule::from_fanouts(&[8, 4, 2]);
        let witness = build_ref_witness(&sched, 42);
        let outputs = compute_atom_outputs(&witness.atom_states);
        let leaf_digests = make_leaf_digests(&sched, 42);
        assert!(
            check_wiring_cpu(
                &sched,
                &outputs,
                &witness.atom_states,
                &leaf_digests,
                &witness.expected_root
            )
            .is_ok(),
            "valid wiring must pass"
        );
    }

    #[test]
    fn glue_rejects_tampered_wire() {
        let sched = MhotHashSchedule::from_fanouts(&[8, 4, 2]);
        let witness = build_ref_witness(&sched, 42);
        let mut outputs = compute_atom_outputs(&witness.atom_states);
        outputs[1][3] ^= 0x01;
        let leaf_digests = make_leaf_digests(&sched, 42);
        assert!(
            check_wiring_cpu(
                &sched,
                &outputs,
                &witness.atom_states,
                &leaf_digests,
                &witness.expected_root
            )
            .is_err(),
            "tampered wire must be caught"
        );
    }

    #[test]
    fn glue_rejects_tampered_root() {
        let sched = MhotHashSchedule::from_fanouts(&[8, 4, 2]);
        let witness = build_ref_witness(&sched, 42);
        let outputs = compute_atom_outputs(&witness.atom_states);
        let leaf_digests = make_leaf_digests(&sched, 42);
        let mut bad_root = witness.expected_root;
        bad_root[0] ^= 0x01;
        assert!(
            check_wiring_cpu(
                &sched,
                &outputs,
                &witness.atom_states,
                &leaf_digests,
                &bad_root
            )
            .is_err(),
            "tampered root must be caught"
        );
    }

    /// Recreate the leaf digests used by build_ref_witness.
    fn make_leaf_digests(sched: &MhotHashSchedule, seed: u64) -> Vec<Digest> {
        let total_leaves: usize = sched.fanouts.iter().sum();
        (0..total_leaves)
            .map(|i| crate::mhot::ref_witness::leaf_digest(seed, i))
            .collect()
    }
}
