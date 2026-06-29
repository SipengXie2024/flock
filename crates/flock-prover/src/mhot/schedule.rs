/// A logical keccak-f atom in the MHOT fold tree.
/// Each atom represents one keccak-f permutation call that folds up to 4 child digests into 1.
#[derive(Clone, Debug)]
pub struct KeccakAtom {
    /// Global atom index (0-based, across all nodes).
    pub atom_id: usize,
    /// Which tree node (by depth index, 0 = root node) this atom belongs to.
    pub node: usize,
    /// Which fold step within this node (0-based).
    pub fold_step: usize,
    /// How many child digests this atom actually consumes (1..=4).
    /// Last fold step of a node may consume fewer than 4.
    pub n_children: usize,
}

/// A wire connecting an output of one atom to an input of another atom (or to the public root).
#[derive(Clone, Debug)]
pub enum WireEndpoint {
    /// Output digest (256 bits) of a keccak atom.
    AtomOutput { atom_id: usize },
    /// One of the input child slots of a keccak atom.
    AtomInput { atom_id: usize, child_slot: usize },
    /// The tree's public root (output of the last fold of node 0).
    PublicRoot,
    /// A leaf child digest (external input, identified by a global leaf index).
    LeafDigest { leaf_index: usize },
}

/// A wire connecting a source endpoint to a destination endpoint.
/// Each wire represents a 256-bit digest equality constraint.
#[derive(Clone, Debug)]
pub struct Wire {
    pub src: WireEndpoint,
    pub dst: WireEndpoint,
}

/// The complete hash schedule for an MHOT path: all atoms and wires needed
/// to fold a path from leaves to root.
#[derive(Clone, Debug)]
pub struct MhotHashSchedule {
    /// The fanout sequence from root to leaf (e.g. [28, 24, 22, 16, 8]).
    pub fanouts: Vec<usize>,
    /// All keccak-f atoms, ordered by node (root first) then fold step.
    pub hash_atoms: Vec<KeccakAtom>,
    /// All wires (digest equalities) between atoms and between atoms and leaf/root.
    pub wires: Vec<Wire>,
}

impl MhotHashSchedule {
    pub fn from_fanouts(fanouts: &[usize]) -> Self {
        let mut hash_atoms = Vec::new();
        let mut wires = Vec::new();
        let mut node_root_atoms = Vec::with_capacity(fanouts.len());
        let mut node_first_inputs = Vec::with_capacity(fanouts.len());
        let mut node_first_leaf_indices = Vec::with_capacity(fanouts.len());
        let mut next_leaf_index = 0usize;

        for (node, &fanout) in fanouts.iter().enumerate() {
            assert!(fanout >= 1, "MHOT fanout must be at least 1");

            let first_leaf_index = next_leaf_index;
            next_leaf_index += fanout;
            node_first_leaf_indices.push(first_leaf_index);

            let n_atoms = atom_count_for_fanout(fanout);
            let mut next_child = 0usize;
            let mut prev_atom = None;
            let mut first_input = None;

            for fold_step in 0..n_atoms {
                let atom_id = hash_atoms.len();
                let external_children = if fold_step == 0 {
                    (fanout - next_child).min(4)
                } else {
                    (fanout - next_child).min(3)
                };
                let n_children = external_children + usize::from(fold_step > 0);

                hash_atoms.push(KeccakAtom {
                    atom_id,
                    node,
                    fold_step,
                    n_children,
                });

                if fold_step == 0 {
                    first_input = Some((atom_id, 0usize));
                } else {
                    let prev_atom = prev_atom.expect("previous fold atom must exist");
                    wires.push(Wire {
                        src: WireEndpoint::AtomOutput { atom_id: prev_atom },
                        dst: WireEndpoint::AtomInput {
                            atom_id,
                            child_slot: 0,
                        },
                    });
                }

                let first_external_slot = usize::from(fold_step > 0);
                for slot_offset in 0..external_children {
                    wires.push(Wire {
                        src: WireEndpoint::LeafDigest {
                            leaf_index: first_leaf_index + next_child,
                        },
                        dst: WireEndpoint::AtomInput {
                            atom_id,
                            child_slot: first_external_slot + slot_offset,
                        },
                    });
                    next_child += 1;
                }

                prev_atom = Some(atom_id);
            }

            node_root_atoms.push(prev_atom);
            node_first_inputs.push(first_input);
        }

        for node in 1..fanouts.len() {
            if let Some(src_atom_id) = node_root_atoms[node] {
                if let Some((dst_atom_id, child_slot)) =
                    nearest_parent_input(node, &node_first_inputs)
                {
                    wires.push(Wire {
                        src: WireEndpoint::AtomOutput {
                            atom_id: src_atom_id,
                        },
                        dst: WireEndpoint::AtomInput {
                            atom_id: dst_atom_id,
                            child_slot,
                        },
                    });
                }
            }
        }

        remove_leaf_wires_shadowed_by_atom_outputs(&mut wires);

        if let Some(src) = public_root_source(&node_root_atoms, &node_first_leaf_indices) {
            wires.push(Wire {
                src,
                dst: WireEndpoint::PublicRoot,
            });
        }

        Self {
            fanouts: fanouts.to_vec(),
            hash_atoms,
            wires,
        }
    }

    pub fn atom_output_f128(&self, atom_id: usize) -> (usize, usize) {
        let block = atom_id / 3;
        let sub = atom_id % 3;
        let base = block * 1024 + (2 * sub + 1) * 16;
        (base, base + 1)
    }

    pub fn atom_input_f128(&self, atom_id: usize, child_slot: usize) -> (usize, usize) {
        let block = atom_id / 3;
        let sub = atom_id % 3;
        let base = block * 1024 + 2 * sub * 16 + 2 * child_slot;
        (base, base + 1)
    }

    pub fn endpoint_f128_with_offset(
        &self,
        ep: &WireEndpoint,
        atom_offset: usize,
    ) -> Option<(usize, usize)> {
        match *ep {
            WireEndpoint::AtomOutput { atom_id } => {
                let global = atom_id + atom_offset;
                let block = global / 3;
                let sub = global % 3;
                let base = block * 1024 + (2 * sub + 1) * 16;
                Some((base, base + 1))
            }
            WireEndpoint::AtomInput { atom_id, child_slot } => {
                let global = atom_id + atom_offset;
                let block = global / 3;
                let sub = global % 3;
                let base = block * 1024 + 2 * sub * 16 + 2 * child_slot;
                Some((base, base + 1))
            }
            WireEndpoint::PublicRoot | WireEndpoint::LeafDigest { .. } => None,
        }
    }
}

fn atom_count_for_fanout(fanout: usize) -> usize {
    if fanout <= 1 {
        0
    } else {
        (fanout - 1).div_ceil(3)
    }
}

fn nearest_parent_input(
    node: usize,
    node_first_inputs: &[Option<(usize, usize)>],
) -> Option<(usize, usize)> {
    node_first_inputs[..node]
        .iter()
        .rev()
        .copied()
        .flatten()
        .next()
}

fn remove_leaf_wires_shadowed_by_atom_outputs(wires: &mut Vec<Wire>) {
    let mut internal_inputs = Vec::new();
    for wire in wires.iter() {
        if matches!(wire.src, WireEndpoint::AtomOutput { .. }) {
            if let WireEndpoint::AtomInput {
                atom_id,
                child_slot,
            } = wire.dst
            {
                internal_inputs.push((atom_id, child_slot));
            }
        }
    }

    wires.retain(|wire| match wire {
        Wire {
            src: WireEndpoint::LeafDigest { .. },
            dst:
                WireEndpoint::AtomInput {
                    atom_id,
                    child_slot,
                },
        } => !internal_inputs.contains(&(*atom_id, *child_slot)),
        _ => true,
    });
}

fn public_root_source(
    node_root_atoms: &[Option<usize>],
    node_first_leaf_indices: &[usize],
) -> Option<WireEndpoint> {
    if let Some(atom_id) = node_root_atoms.iter().copied().flatten().next() {
        Some(WireEndpoint::AtomOutput { atom_id })
    } else {
        node_first_leaf_indices
            .first()
            .copied()
            .map(|leaf_index| WireEndpoint::LeafDigest { leaf_index })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_atom_count_no_pad() {
        let sched = MhotHashSchedule::from_fanouts(&[28, 24, 22, 16, 8]);
        let expect: usize = [28, 24, 22, 16, 8]
            .iter()
            .map(|&f| if f <= 1 { 0 } else { (f - 1 + 2) / 3 })
            .sum();
        assert_eq!(
            sched.hash_atoms.len(),
            expect,
            "atom count must equal actual 4-ary demand, no padding to max fanout"
        );
    }

    #[test]
    fn schedule_atom_count_small() {
        let sched = MhotHashSchedule::from_fanouts(&[4, 2]);
        assert_eq!(sched.hash_atoms.len(), 2);
    }

    #[test]
    fn schedule_fanout_1_no_atoms() {
        let sched = MhotHashSchedule::from_fanouts(&[4, 1, 8]);
        assert_eq!(sched.hash_atoms.len(), 4);
    }

    #[test]
    fn schedule_wires_exist() {
        let sched = MhotHashSchedule::from_fanouts(&[8, 4]);
        assert!(!sched.wires.is_empty(), "schedule must produce wires");
        let root_wires: Vec<_> = sched
            .wires
            .iter()
            .filter(|w| matches!(w.dst, WireEndpoint::PublicRoot))
            .collect();
        assert_eq!(root_wires.len(), 1, "exactly one wire to PublicRoot");
    }

    #[test]
    fn schedule_atom_n_children_valid() {
        let sched = MhotHashSchedule::from_fanouts(&[28, 24, 22, 16, 8]);
        for atom in &sched.hash_atoms {
            assert!(
                atom.n_children >= 1 && atom.n_children <= 4,
                "atom {} has invalid n_children={}",
                atom.atom_id,
                atom.n_children
            );
        }
    }

    #[test]
    fn schedule_atom_input_has_single_source_kind() {
        let sched = MhotHashSchedule::from_fanouts(&[8, 4, 2]);
        for leaf_wire in &sched.wires {
            let Wire {
                src: WireEndpoint::LeafDigest { .. },
                dst:
                    WireEndpoint::AtomInput {
                        atom_id,
                        child_slot,
                    },
            } = leaf_wire
            else {
                continue;
            };

            for atom_wire in &sched.wires {
                assert!(
                    !matches!(
                        atom_wire,
                        Wire {
                            src: WireEndpoint::AtomOutput { .. },
                            dst: WireEndpoint::AtomInput {
                                atom_id: other_atom_id,
                                child_slot: other_child_slot,
                            },
                        } if other_atom_id == atom_id && other_child_slot == child_slot
                    ),
                    "atom input ({atom_id}, {child_slot}) cannot be both external leaf and internal root"
                );
            }
        }
    }
}
