use super::schedule::{KeccakAtom, MhotHashSchedule, WireEndpoint};
use crate::r1cs_hashes::keccak::{keccak_f, State, LANE_BITS, N_LANES, STATE_BITS};

const DIGEST_BYTES: usize = 32;
const STATE_BYTES: usize = 200;
const RATE_BYTES: usize = 136;
const FOLD_INPUT_BYTES: usize = 4 * DIGEST_BYTES;
const PAD_START_BYTE: usize = FOLD_INPUT_BYTES;
const PAD_FINAL_BYTE: usize = RATE_BYTES - 1;

/// A 256-bit digest represented as 32 bytes.
pub type Digest = [u8; 32];

/// Reference witness: per-atom keccak input states and the expected root.
pub struct RefWitness {
    /// For each atom (indexed by atom_id), the full 200-byte keccak-f input state.
    pub atom_states: Vec<[u8; 200]>,
    /// The expected tree root (output of the root node's last fold atom).
    pub expected_root: Digest,
}

/// Build a reference witness from a schedule, using deterministic leaf digests.
///
/// Leaf digests are derived deterministically from `seed` and the leaf index.
/// The fold is computed fully on CPU, no Flock circuits involved.
pub fn build_ref_witness(sched: &MhotHashSchedule, seed: u64) -> RefWitness {
    validate_schedule_atoms(sched);

    let atoms_by_node = atoms_by_node(sched);
    let leaf_starts = leaf_starts(&sched.fanouts);
    let mut atom_states = vec![[0u8; STATE_BYTES]; sched.hash_atoms.len()];
    let mut lower_root = None;

    for node in (0..sched.fanouts.len()).rev() {
        let fanout = sched.fanouts[node];
        let first_leaf = leaf_starts[node];
        let mut children: Vec<Digest> = (0..fanout)
            .map(|i| leaf_digest(seed, first_leaf + i))
            .collect();

        if let Some(root) = lower_root {
            if let Some(first_child) = children.first_mut() {
                *first_child = root;
            }
        }

        lower_root = if atoms_by_node[node].is_empty() {
            children.first().copied()
        } else {
            Some(fill_node_atom_states(
                &atoms_by_node[node],
                &children,
                &mut atom_states,
            ))
        };
    }

    RefWitness {
        atom_states,
        expected_root: lower_root.unwrap_or([0u8; DIGEST_BYTES]),
    }
}

/// Independently compute the fold root from a schedule and atom input states,
/// using a standard keccak-f[1600] CPU implementation.
///
/// This is the cross-check oracle: it takes the same atom_states as the prover
/// would, runs keccak-f on each, follows the wires to propagate outputs to inputs,
/// and returns the root digest.
pub fn cpu_fold_root(sched: &MhotHashSchedule, atom_states: &[[u8; 200]]) -> Digest {
    assert_eq!(
        atom_states.len(),
        sched.hash_atoms.len(),
        "atom_states must be indexed by atom_id"
    );
    validate_schedule_atoms(sched);

    let atoms_by_node = atoms_by_node(sched);
    let mut states = atom_states.to_vec();
    let mut root = None;

    for node in (0..sched.fanouts.len()).rev() {
        for atom in &atoms_by_node[node] {
            let digest = keccak_f_digest(&states[atom.atom_id]);
            propagate_atom_output(sched, atom.atom_id, digest, &mut states, &mut root);
        }
    }

    root.unwrap_or([0u8; DIGEST_BYTES])
}

fn validate_schedule_atoms(sched: &MhotHashSchedule) {
    for (idx, atom) in sched.hash_atoms.iter().enumerate() {
        assert_eq!(atom.atom_id, idx, "hash_atoms must be indexed by atom_id");
        assert!(
            atom.node < sched.fanouts.len(),
            "atom {} references missing node {}",
            atom.atom_id,
            atom.node
        );
    }
}

fn atoms_by_node(sched: &MhotHashSchedule) -> Vec<Vec<&KeccakAtom>> {
    let mut by_node = vec![Vec::new(); sched.fanouts.len()];
    for atom in &sched.hash_atoms {
        by_node[atom.node].push(atom);
    }
    for atoms in &mut by_node {
        atoms.sort_by_key(|atom| atom.fold_step);
    }
    by_node
}

fn leaf_starts(fanouts: &[usize]) -> Vec<usize> {
    let mut starts = Vec::with_capacity(fanouts.len());
    let mut next = 0usize;
    for &fanout in fanouts {
        starts.push(next);
        next += fanout;
    }
    starts
}

fn fill_node_atom_states(
    atoms: &[&KeccakAtom],
    children: &[Digest],
    atom_states: &mut [[u8; STATE_BYTES]],
) -> Digest {
    let mut next_child = 0usize;
    let mut prev_digest = None;

    for atom in atoms {
        let mut inputs = Vec::with_capacity(4);
        if let Some(prev) = prev_digest {
            inputs.push(prev);
            let n_external = (children.len() - next_child).min(3);
            inputs.extend_from_slice(&children[next_child..next_child + n_external]);
            next_child += n_external;
        } else {
            let n_external = children.len().min(4);
            inputs.extend_from_slice(&children[..n_external]);
            next_child = n_external;
        }

        assert_eq!(
            inputs.len(),
            atom.n_children,
            "atom {} n_children does not match fold inputs",
            atom.atom_id
        );

        let state = fold_state(&inputs);
        atom_states[atom.atom_id] = state;
        prev_digest = Some(keccak_f_digest(&state));
    }

    prev_digest.expect("node with atoms must produce a root digest")
}

pub fn leaf_digest(seed: u64, leaf_index: usize) -> Digest {
    let mut state = seed ^ (leaf_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut digest = [0u8; DIGEST_BYTES];
    for chunk in digest.chunks_exact_mut(8) {
        chunk.copy_from_slice(&splitmix64(&mut state).to_le_bytes());
    }
    digest
}

fn fold_state(inputs: &[Digest]) -> [u8; STATE_BYTES] {
    assert!(
        (1..=4).contains(&inputs.len()),
        "fold atom must consume 1..=4 child digests"
    );

    let mut state = [0u8; STATE_BYTES];
    for (slot, digest) in inputs.iter().enumerate() {
        let start = slot * DIGEST_BYTES;
        state[start..start + DIGEST_BYTES].copy_from_slice(digest);
    }
    state[PAD_START_BYTE] = 0x01;
    state[PAD_FINAL_BYTE] = 0x80;
    state
}

pub(crate) fn keccak_f_digest(state: &[u8; STATE_BYTES]) -> Digest {
    let state = keccak_f_state(state);
    let mut digest = [0u8; DIGEST_BYTES];
    digest.copy_from_slice(&state[..DIGEST_BYTES]);
    digest
}

fn keccak_f_state(state: &[u8; STATE_BYTES]) -> [u8; STATE_BYTES] {
    let mut logical = bytes_to_logical_state(state);
    keccak_f(&mut logical);
    logical_state_to_bytes(&logical)
}

fn propagate_atom_output(
    sched: &MhotHashSchedule,
    src_atom_id: usize,
    digest: Digest,
    states: &mut [[u8; STATE_BYTES]],
    root: &mut Option<Digest>,
) {
    for wire in &sched.wires {
        match wire.src {
            WireEndpoint::AtomOutput { atom_id } if atom_id == src_atom_id => match wire.dst {
                WireEndpoint::AtomInput {
                    atom_id,
                    child_slot,
                } => write_child_digest(&mut states[atom_id], child_slot, &digest),
                WireEndpoint::PublicRoot => *root = Some(digest),
                WireEndpoint::AtomOutput { .. } | WireEndpoint::LeafDigest { .. } => {}
            },
            _ => {}
        }
    }
}

fn write_child_digest(state: &mut [u8; STATE_BYTES], child_slot: usize, digest: &Digest) {
    assert!(child_slot < 4, "fold child slot must be in 0..4");
    let start = child_slot * DIGEST_BYTES;
    state[start..start + DIGEST_BYTES].copy_from_slice(digest);
}

pub fn bytes_to_logical_state(bytes: &[u8; STATE_BYTES]) -> State {
    let mut state = [false; STATE_BITS];
    for lane in 0..N_LANES {
        for z in 0..LANE_BITS {
            let byte = bytes[lane * 8 + z / 8];
            state[lane + N_LANES * z] = ((byte >> (z % 8)) & 1) == 1;
        }
    }
    state
}

pub(crate) fn logical_state_to_bytes(state: &State) -> [u8; STATE_BYTES] {
    let mut bytes = [0u8; STATE_BYTES];
    for lane in 0..N_LANES {
        for z in 0..LANE_BITS {
            if state[lane + N_LANES * z] {
                bytes[lane * 8 + z / 8] |= 1u8 << (z % 8);
            }
        }
    }
    bytes
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mhot::schedule::MhotHashSchedule;

    #[test]
    fn ref_root_matches_cpu_keccak() {
        let sched = MhotHashSchedule::from_fanouts(&[8, 4, 2]);
        let witness = build_ref_witness(&sched, 42);
        let cpu_root = cpu_fold_root(&sched, &witness.atom_states);
        assert_eq!(
            cpu_root, witness.expected_root,
            "schedule+witness semantics must be self-consistent"
        );
    }

    #[test]
    fn different_seeds_different_roots() {
        let sched = MhotHashSchedule::from_fanouts(&[4, 2]);
        let w1 = build_ref_witness(&sched, 1);
        let w2 = build_ref_witness(&sched, 2);
        assert_ne!(
            w1.expected_root, w2.expected_root,
            "different seeds must produce different roots"
        );
    }

    #[test]
    fn single_node_fold() {
        let sched = MhotHashSchedule::from_fanouts(&[4]);
        let w = build_ref_witness(&sched, 7);
        assert_eq!(sched.hash_atoms.len(), 1);
        let cpu_root = cpu_fold_root(&sched, &w.atom_states);
        assert_eq!(cpu_root, w.expected_root);
    }

    #[test]
    fn realistic_fanouts() {
        let sched = MhotHashSchedule::from_fanouts(&[28, 24, 22, 16, 8]);
        let w = build_ref_witness(&sched, 99);
        let cpu_root = cpu_fold_root(&sched, &w.atom_states);
        assert_eq!(cpu_root, w.expected_root);
    }
}
