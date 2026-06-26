//! ZK privacy layer stubs for the MHOT -> Flock migration.
//!
//! Flock currently provides succinct proofs only, not zero-knowledge proofs.
//! This module contains interface stubs and design documentation for the MHOT
//! ZK layer, to be implemented when Flock adds ZK support via hiding
//! commitments and zero-knowledge sumcheck techniques.

/// Design: Private MUX atom (D1).
///
/// Current F_route already places the MUX
/// `selected_out = take ? child_digest : selected_in` inside the R1CS atom.
/// For ZK mode:
///
/// 1. `mask` and `selected_idx` move from public statement to private witness.
/// 2. The R1CS constraints do not change: MUX is already non-linear and inside
///    the atom.
/// 3. Flock needs to provide:
///    - Hiding polynomial commitments, replacing transparent BaseFold/Ligerito
///      commitments for witness polynomials.
///    - Zero-knowledge sumcheck, masking partial sums.
///    - ZK-compatible PCS opening.
/// 4. The F_route prove/verify wrapper needs a mode flag to select succinct vs
///    ZK proving.
///
/// Estimated additional work: minimal MHOT R1CS changes, major Flock
/// infrastructure changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrivateMuxConfig {
    /// Whether the route mask is private in ZK mode or public in the current PoC.
    pub mask_private: bool,
    /// Whether selected_idx is private.
    pub selected_idx_private: bool,
}

impl PrivateMuxConfig {
    pub const fn public() -> Self {
        Self {
            mask_private: false,
            selected_idx_private: false,
        }
    }

    pub const fn private() -> Self {
        Self {
            mask_private: true,
            selected_idx_private: true,
        }
    }
}

/// Design: Popcount Boolean counter (D2).
///
/// When the route mask is private, `popcount(mask)` cannot be checked offline by
/// the verifier. It must be proven in-circuit.
///
/// Approach: binary full-adder reduction tree over 256 mask bits.
///
/// - Full adder:
///   `sum = a XOR b XOR cin`
///   `cout = (a * b) XOR (cin * (a XOR b))`
/// - Two AND gates per full adder.
/// - 256 input bits require about 256 full adders, or about 512 R1CS
///   multiplication rows.
/// - Output is a 9-bit popcount in the range 0..=256.
/// - Assert the popcount equals a claimed public width in ZK mode.
///
/// This can be embedded in F_route, extending `USEFUL_BITS` by roughly 1200, or
/// moved into a separate F_popcount atom.
pub const POPCOUNT_AND_GATES: usize = 512;
pub const POPCOUNT_OUTPUT_BITS: usize = 9;

/// Stub: CPU-side stand-in for the future in-circuit popcount check.
pub fn check_popcount_cpu(mask: &[bool; 256], expected_popcount: usize) -> bool {
    let actual = mask.iter().filter(|&&bit| bit).count();
    actual == expected_popcount
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn private_mux_config() {
        let public = PrivateMuxConfig::public();
        assert!(!public.mask_private);
        assert!(!public.selected_idx_private);

        let private = PrivateMuxConfig::private();
        assert!(private.mask_private);
        assert!(private.selected_idx_private);
    }

    #[test]
    fn popcount_cpu_correct() {
        let mut mask = [false; 256];
        assert!(check_popcount_cpu(&mask, 0));

        mask[0] = true;
        mask[5] = true;
        mask[100] = true;
        assert!(check_popcount_cpu(&mask, 3));
        assert!(!check_popcount_cpu(&mask, 2));
    }
}
