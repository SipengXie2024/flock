//! Binary Merkle tree with SHA-256, ARM crypto-extension accelerated.
//!
//! Layout for `num_leaves = 2^k` leaves:
//!   tree[0..num_leaves]                              = leaf hashes (level k)
//!   tree[num_leaves..3·num_leaves/2]                 = level k−1
//!   ...
//!   tree[2·num_leaves − 2..2·num_leaves − 1]         = root (level 0)
//!
//! Total nodes: `2·num_leaves − 1`. The flat layout keeps the tree contiguous
//! in memory for cheap Merkle-path extraction later.
//!
//! Hash uses the [`sha2`] crate. On aarch64 with the `sha2` target feature
//! (set implicitly by `target-cpu=native` on M-series), the crate uses
//! `sha256h`/`sha256h2`/`sha256su0`/`sha256su1` ARM crypto extension
//! instructions; this is detected at runtime by [`cpufeatures`].
//!
//! No domain separation between leaf and internal hashes — this is a
//! micro-benchmark module, not production code. A production PCS commit
//! should prepend `0x00`/`0x01` (or equivalent) to distinguish the two
//! pre-images and avoid second-preimage attacks via interpretation collision.

use rayon::prelude::*;
use sha2::{Digest, Sha256};

pub type Hash = [u8; 32];

/// 4-way interleaved SHA-256 using ARM crypto-extension intrinsics.
///
/// The M-series SHA unit is pipelined: a single dependent compress
/// chain runs at ~21 ns/compress, while interleaved independent
/// streams sustain ~16 ns/compress on real (distinct) data — a ~1.35×
/// throughput win, measured on M4 Max at m=30. The `sha2` crate hashes
/// one stream at a time, so bulk Merkle hashing (independent leaves /
/// independent nodes within a level) leaves that on the table.
///
/// Digests are byte-identical to `Sha256::digest`.
/// 4-way interleaved SHA-256 using x86_64 AVX2 intrinsics.
///
/// Each `__m256i` holds 8 × u32 lanes. We pack 4 independent SHA-256
/// states into lanes 0..3 (the upper 4 lanes mirror a second quad when
/// doing 8-way, but here we always process quads of 4). All SHA-256
/// round operations (Ch, Maj, Sigma, message schedule) are pure
/// bitwise + add, so they vectorise trivially across lanes.
///
/// Digests are byte-identical to `sha2::Sha256::digest`.
#[cfg(target_arch = "x86_64")]
mod sha256x4_x86 {
    use super::Hash;

    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    const IV: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];

    /// Right-rotate a 32-bit value packed in each lane of a __m256i.
    macro_rules! rotr32 {
        ($x:expr, $n:literal) => {{
            // SAFETY: caller ensures AVX2 is available.
            _mm256_or_si256(
                _mm256_srli_epi32($x, $n),
                _mm256_slli_epi32($x, 32 - $n),
            )
        }};
    }

    // SAFETY for all helpers below: caller ensures AVX2 is available.
    // Functions are `unsafe fn` (body is an unsafe context), so no inner
    // `unsafe {}` block is needed; intrinsics are called directly.

    // SAFETY for all helpers below: they are `unsafe fn` with
    // `#[target_feature(enable = "avx2")]`, making the body an implicit
    // unsafe context in edition 2024. Callers are responsible for ensuring
    // AVX2 is available (enforced by the same target_feature gate).

    /// SHA-256 Ch(e, f, g) = (e & f) ^ (~e & g)
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn ch(e: __m256i, f: __m256i, g: __m256i) -> __m256i {
        _mm256_xor_si256(
            _mm256_and_si256(e, f),
            _mm256_andnot_si256(e, g),
        )
    }

    /// SHA-256 Maj(a, b, c) = (a & b) ^ (a & c) ^ (b & c)
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn maj(a: __m256i, b: __m256i, c: __m256i) -> __m256i {
        _mm256_xor_si256(
            _mm256_xor_si256(
                _mm256_and_si256(a, b),
                _mm256_and_si256(a, c),
            ),
            _mm256_and_si256(b, c),
        )
    }

    /// SHA-256 big Sigma0(a) = ROTR(2, a) ^ ROTR(13, a) ^ ROTR(22, a)
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn bsig0(a: __m256i) -> __m256i {
        _mm256_xor_si256(
            _mm256_xor_si256(rotr32!(a, 2), rotr32!(a, 13)),
            rotr32!(a, 22),
        )
    }

    /// SHA-256 big Sigma1(e) = ROTR(6, e) ^ ROTR(11, e) ^ ROTR(25, e)
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn bsig1(e: __m256i) -> __m256i {
        _mm256_xor_si256(
            _mm256_xor_si256(rotr32!(e, 6), rotr32!(e, 11)),
            rotr32!(e, 25),
        )
    }

    /// SHA-256 small sigma0(x) = ROTR(7, x) ^ ROTR(18, x) ^ SHR(3, x)
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn ssig0(x: __m256i) -> __m256i {
        _mm256_xor_si256(
            _mm256_xor_si256(rotr32!(x, 7), rotr32!(x, 18)),
            _mm256_srli_epi32(x, 3),
        )
    }

    /// SHA-256 small sigma1(x) = ROTR(17, x) ^ ROTR(19, x) ^ SHR(10, x)
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn ssig1(x: __m256i) -> __m256i {
        _mm256_xor_si256(
            _mm256_xor_si256(rotr32!(x, 17), rotr32!(x, 19)),
            _mm256_srli_epi32(x, 10),
        )
    }

    /// Load 4 big-endian u32 words at the same offset from 4 different
    /// message blocks, returning them packed into lanes [0..3] of a
    /// __m256i (lanes 4..7 are zero).
    ///
    /// `word_idx` is the 0-based u32 index within the 64-byte block.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn load_msg_word(blocks: &[*const u8; 4], word_idx: usize) -> __m256i {
        // SAFETY: caller ensures each pointer is valid for 64 bytes and AVX2.
        unsafe {
            let off = word_idx * 4;
            let w0 = u32::from_be_bytes([
                *blocks[0].add(off),
                *blocks[0].add(off + 1),
                *blocks[0].add(off + 2),
                *blocks[0].add(off + 3),
            ]);
            let w1 = u32::from_be_bytes([
                *blocks[1].add(off),
                *blocks[1].add(off + 1),
                *blocks[1].add(off + 2),
                *blocks[1].add(off + 3),
            ]);
            let w2 = u32::from_be_bytes([
                *blocks[2].add(off),
                *blocks[2].add(off + 1),
                *blocks[2].add(off + 2),
                *blocks[2].add(off + 3),
            ]);
            let w3 = u32::from_be_bytes([
                *blocks[3].add(off),
                *blocks[3].add(off + 1),
                *blocks[3].add(off + 2),
                *blocks[3].add(off + 3),
            ]);
            _mm256_setr_epi32(
                w0 as i32, w1 as i32, w2 as i32, w3 as i32,
                0, 0, 0, 0,
            )
        }
    }

    /// Compress one 64-byte block from each of 4 independent messages.
    ///
    /// `state` holds [a, b, c, d, e, f, g, h] as 8 __m256i, each with 4
    /// lanes (one per message). Modified in place (state += round result).
    #[target_feature(enable = "avx2")]
    unsafe fn compress4(state: &mut [__m256i; 8], blocks: &[*const u8; 4]) {
        // SAFETY: caller ensures AVX2 is available, pointers valid for 64 bytes.
        unsafe {
            let mut w = [_mm256_setzero_si256(); 64];

            for t in 0..16 {
                w[t] = load_msg_word(blocks, t);
            }

            // Message schedule: W[t] = ssig1(W[t-2]) + W[t-7] + ssig0(W[t-15]) + W[t-16]
            for t in 16..64 {
                w[t] = _mm256_add_epi32(
                    _mm256_add_epi32(ssig1(w[t - 2]), w[t - 7]),
                    _mm256_add_epi32(ssig0(w[t - 15]), w[t - 16]),
                );
            }

            let state_save = *state;
            let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;

            for t in 0..64 {
                let ki = _mm256_set1_epi32(K[t] as i32);

                // T1 = h + bsig1(e) + ch(e,f,g) + K[t] + W[t]
                let t1 = _mm256_add_epi32(
                    _mm256_add_epi32(
                        _mm256_add_epi32(h, bsig1(e)),
                        _mm256_add_epi32(ch(e, f, g), ki),
                    ),
                    w[t],
                );
                // T2 = bsig0(a) + maj(a,b,c)
                let t2 = _mm256_add_epi32(bsig0(a), maj(a, b, c));

                h = g;
                g = f;
                f = e;
                e = _mm256_add_epi32(d, t1);
                d = c;
                c = b;
                b = a;
                a = _mm256_add_epi32(t1, t2);
            }

            state[0] = _mm256_add_epi32(a, state_save[0]);
            state[1] = _mm256_add_epi32(b, state_save[1]);
            state[2] = _mm256_add_epi32(c, state_save[2]);
            state[3] = _mm256_add_epi32(d, state_save[3]);
            state[4] = _mm256_add_epi32(e, state_save[4]);
            state[5] = _mm256_add_epi32(f, state_save[5]);
            state[6] = _mm256_add_epi32(g, state_save[6]);
            state[7] = _mm256_add_epi32(h, state_save[7]);
        }
    }

    /// Extract the u32 at the given lane index (0..7) from a __m256i.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn extract_lane(v: __m256i, lane: usize) -> u32 {
        // SAFETY: AVX2 ensured by caller; tmp is 32-byte aligned on stack.
        unsafe {
            let mut tmp = [0u32; 8];
            _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, v);
            tmp[lane]
        }
    }

    /// Hash 4 equal-length inputs, producing 4 standard SHA-256 digests.
    #[target_feature(enable = "avx2")]
    pub unsafe fn hash4_equal_len(inputs: [&[u8]; 4], out: &mut [Hash]) {
        let len = inputs[0].len();
        debug_assert!(inputs.iter().all(|x| x.len() == len));
        debug_assert!(out.len() >= 4);

        // SAFETY: AVX2 availability ensured by caller (runtime detection in hash4).
        unsafe {
            let mut state: [__m256i; 8] = [
                _mm256_set1_epi32(IV[0] as i32),
                _mm256_set1_epi32(IV[1] as i32),
                _mm256_set1_epi32(IV[2] as i32),
                _mm256_set1_epi32(IV[3] as i32),
                _mm256_set1_epi32(IV[4] as i32),
                _mm256_set1_epi32(IV[5] as i32),
                _mm256_set1_epi32(IV[6] as i32),
                _mm256_set1_epi32(IV[7] as i32),
            ];

            let n_full = len / 64;
            for blk in 0..n_full {
                let off = blk * 64;
                compress4(
                    &mut state,
                    &[
                        inputs[0].as_ptr().add(off),
                        inputs[1].as_ptr().add(off),
                        inputs[2].as_ptr().add(off),
                        inputs[3].as_ptr().add(off),
                    ],
                );
            }

            // Tail: remaining bytes + 0x80 + zero pad + 64-bit BE bit length.
            let rem = len % 64;
            let bit_len = (len as u64) * 8;
            let n_tail = if rem < 56 { 1 } else { 2 };
            let mut tails = [[0u8; 128]; 4];
            for i in 0..4 {
                tails[i][..rem].copy_from_slice(&inputs[i][len - rem..]);
                tails[i][rem] = 0x80;
                tails[i][n_tail * 64 - 8..n_tail * 64]
                    .copy_from_slice(&bit_len.to_be_bytes());
            }
            for blk in 0..n_tail {
                let off = blk * 64;
                compress4(
                    &mut state,
                    &[
                        tails[0].as_ptr().add(off),
                        tails[1].as_ptr().add(off),
                        tails[2].as_ptr().add(off),
                        tails[3].as_ptr().add(off),
                    ],
                );
            }

            for msg_idx in 0..4 {
                for word_idx in 0..8 {
                    let w = extract_lane(state[word_idx], msg_idx);
                    out[msg_idx][word_idx * 4..word_idx * 4 + 4]
                        .copy_from_slice(&w.to_be_bytes());
                }
            }
        }
    }

    /// Safe wrapper: checks AVX2 at runtime, then calls the unsafe
    /// `hash4_equal_len`. Falls back to scalar if AVX2 is absent (should
    /// not happen on the target environment).
    pub fn hash4(inputs: [&[u8]; 4], out: &mut [Hash]) {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: feature detection confirmed AVX2 is available.
            unsafe { hash4_equal_len(inputs, out) }
        } else {
            for (i, inp) in inputs.iter().enumerate() {
                use sha2::{Digest, Sha256};
                out[i] = Sha256::digest(inp).into();
            }
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "sha2"))]
mod sha256x4 {
    use super::Hash;
    use core::arch::aarch64::*;

    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    const IV_ABCD: [u32; 4] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a];
    const IV_EFGH: [u32; 4] = [0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];

    /// One interleaved compression round over 4 independent states.
    /// `blocks[i]` must be ≥ 64 bytes; only the first 64 are consumed.
    #[inline]
    unsafe fn compress4(
        abcd: &mut [uint32x4_t; 4],
        efgh: &mut [uint32x4_t; 4],
        blocks: [*const u8; 4],
    ) {
        unsafe {
            let mut msg0 = [vdupq_n_u32(0); 4];
            let mut msg1 = [vdupq_n_u32(0); 4];
            let mut msg2 = [vdupq_n_u32(0); 4];
            let mut msg3 = [vdupq_n_u32(0); 4];
            for i in 0..4 {
                // SHA-256 message words are big-endian.
                msg0[i] = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(blocks[i])));
                msg1[i] = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(blocks[i].add(16))));
                msg2[i] = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(blocks[i].add(32))));
                msg3[i] = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(blocks[i].add(48))));
            }
            let abcd_save = *abcd;
            let efgh_save = *efgh;

            macro_rules! rounds4 {
                ($msg:expr, $ki:expr) => {{
                    let kv = vld1q_u32(K.as_ptr().add($ki));
                    for i in 0..4 {
                        let wk = vaddq_u32($msg[i], kv);
                        let t = abcd[i];
                        abcd[i] = vsha256hq_u32(abcd[i], efgh[i], wk);
                        efgh[i] = vsha256h2q_u32(efgh[i], t, wk);
                    }
                }};
            }
            macro_rules! sched {
                ($m0:expr, $m1:expr, $m2:expr, $m3:expr) => {
                    for i in 0..4 {
                        $m0[i] = vsha256su1q_u32(vsha256su0q_u32($m0[i], $m1[i]), $m2[i], $m3[i]);
                    }
                };
            }

            rounds4!(msg0, 0);
            rounds4!(msg1, 4);
            rounds4!(msg2, 8);
            rounds4!(msg3, 12);
            for r in 1..4 {
                sched!(msg0, msg1, msg2, msg3);
                sched!(msg1, msg2, msg3, msg0);
                sched!(msg2, msg3, msg0, msg1);
                sched!(msg3, msg0, msg1, msg2);
                rounds4!(msg0, 16 * r);
                rounds4!(msg1, 16 * r + 4);
                rounds4!(msg2, 16 * r + 8);
                rounds4!(msg3, 16 * r + 12);
            }
            for i in 0..4 {
                abcd[i] = vaddq_u32(abcd[i], abcd_save[i]);
                efgh[i] = vaddq_u32(efgh[i], efgh_save[i]);
            }
        }
    }

    /// Hash 4 equal-length inputs, producing 4 standard SHA-256 digests.
    #[inline]
    pub fn hash4_equal_len(inputs: [&[u8]; 4], out: &mut [Hash]) {
        let len = inputs[0].len();
        debug_assert!(inputs.iter().all(|x| x.len() == len));
        debug_assert!(out.len() >= 4);

        unsafe {
            let mut abcd = [vld1q_u32(IV_ABCD.as_ptr()); 4];
            let mut efgh = [vld1q_u32(IV_EFGH.as_ptr()); 4];

            // Full 64-byte blocks.
            let n_full = len / 64;
            for blk in 0..n_full {
                compress4(
                    &mut abcd,
                    &mut efgh,
                    [
                        inputs[0].as_ptr().add(blk * 64),
                        inputs[1].as_ptr().add(blk * 64),
                        inputs[2].as_ptr().add(blk * 64),
                        inputs[3].as_ptr().add(blk * 64),
                    ],
                );
            }

            // Tail: remaining bytes + 0x80 + zero pad + 64-bit BE bit length.
            // One extra block when rem ≤ 55, two when 56 ≤ rem ≤ 63.
            let rem = len % 64;
            let bit_len = (len as u64) * 8;
            let n_tail = if rem < 56 { 1 } else { 2 };
            let mut tails = [[0u8; 128]; 4];
            for i in 0..4 {
                tails[i][..rem].copy_from_slice(&inputs[i][len - rem..]);
                tails[i][rem] = 0x80;
                tails[i][n_tail * 64 - 8..n_tail * 64].copy_from_slice(&bit_len.to_be_bytes());
            }
            for blk in 0..n_tail {
                compress4(
                    &mut abcd,
                    &mut efgh,
                    [
                        tails[0].as_ptr().add(blk * 64),
                        tails[1].as_ptr().add(blk * 64),
                        tails[2].as_ptr().add(blk * 64),
                        tails[3].as_ptr().add(blk * 64),
                    ],
                );
            }

            // Digest = big-endian a..h.
            for i in 0..4 {
                let be_lo = vrev32q_u8(vreinterpretq_u8_u32(abcd[i]));
                let be_hi = vrev32q_u8(vreinterpretq_u8_u32(efgh[i]));
                vst1q_u8(out[i].as_mut_ptr(), be_lo);
                vst1q_u8(out[i].as_mut_ptr().add(16), be_hi);
            }
        }
    }
}

/// Global SHA-256 call/compression counters, enabled with
/// `--features hash-count` (e.g. by `benches/verifier_hash_count.rs`).
/// Relaxed atomics — exact totals, no ordering guarantees across threads.
#[cfg(feature = "hash-count")]
pub mod hash_count {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    pub static LEAF_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static LEAF_COMPRESSIONS: AtomicU64 = AtomicU64::new(0);
    pub static PAIR_CALLS: AtomicU64 = AtomicU64::new(0);

    /// SHA-256 compression count for a one-shot hash of `len` bytes:
    /// ceil((len + 9) / 64) — payload + 0x80 pad + 8-byte length.
    #[inline]
    pub fn sha256_blocks(len: usize) -> u64 {
        ((len + 9).div_ceil(64)) as u64
    }

    pub fn reset() {
        LEAF_CALLS.store(0, Relaxed);
        LEAF_COMPRESSIONS.store(0, Relaxed);
        PAIR_CALLS.store(0, Relaxed);
    }

    /// (leaf_calls, leaf_compressions, pair_calls). Each pair hash is
    /// 2 compressions (64 B payload + padding block).
    pub fn snapshot() -> (u64, u64, u64) {
        (
            LEAF_CALLS.load(Relaxed),
            LEAF_COMPRESSIONS.load(Relaxed),
            PAIR_CALLS.load(Relaxed),
        )
    }
}

/// Hash one leaf of arbitrary byte length.
#[inline]
pub fn hash_leaf(data: &[u8]) -> Hash {
    #[cfg(feature = "hash-count")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        hash_count::LEAF_CALLS.fetch_add(1, Relaxed);
        hash_count::LEAF_COMPRESSIONS.fetch_add(hash_count::sha256_blocks(data.len()), Relaxed);
    }
    Sha256::digest(data).into()
}

/// Hash a pair of children into a parent node (64 B → 32 B).
#[inline]
pub fn hash_pair(left: &Hash, right: &Hash) -> Hash {
    #[cfg(feature = "hash-count")]
    hash_count::PAIR_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut h = Sha256::new();
    h.update(left);
    h.update(right);
    h.finalize().into()
}

/// Compute the Merkle root of `data` split into `num_leaves` equal-sized leaves.
///
/// Multi-threaded via rayon. `num_leaves` must be a power of two and divide
/// `data.len()`. Returns the 32-byte root. The intermediate tree is allocated
/// and dropped; if you need it for path opening, use [`merkle_tree`] instead.
pub fn merkle_root(data: &[u8], num_leaves: usize) -> Hash {
    let tree = merkle_tree(data, num_leaves);
    tree[tree.len() - 1]
}

/// Compute the full Merkle tree (flat layout, see module docs) for `data`
/// split into `num_leaves` equal-sized leaves.
pub fn merkle_tree(data: &[u8], num_leaves: usize) -> Vec<Hash> {
    assert!(
        num_leaves.is_power_of_two() && num_leaves > 0,
        "num_leaves must be power of 2"
    );
    assert_eq!(
        data.len() % num_leaves,
        0,
        "data length must be a multiple of num_leaves"
    );

    let leaf_size = data.len() / num_leaves;
    let total_nodes = 2 * num_leaves - 1;
    // Uninit alloc — every node is written exactly once before being read:
    // leaves at step 1, then each internal level reads the level below (which
    // was just written) and writes itself.
    let mut tree: Vec<Hash> = crate::alloc_uninit_vec(total_nodes);

    // 1. Leaves — fully parallel; 4-way interleaved SHA where available.
    #[cfg(all(target_arch = "aarch64", target_feature = "sha2"))]
    {
        tree[..num_leaves]
            .par_chunks_mut(4)
            .zip(data.par_chunks(4 * leaf_size))
            .for_each(|(outs, leaves)| {
                if outs.len() == 4 {
                    #[cfg(feature = "hash-count")]
                    {
                        use std::sync::atomic::Ordering::Relaxed;
                        hash_count::LEAF_CALLS.fetch_add(4, Relaxed);
                        hash_count::LEAF_COMPRESSIONS
                            .fetch_add(4 * hash_count::sha256_blocks(leaf_size), Relaxed);
                    }
                    sha256x4::hash4_equal_len(
                        [
                            &leaves[..leaf_size],
                            &leaves[leaf_size..2 * leaf_size],
                            &leaves[2 * leaf_size..3 * leaf_size],
                            &leaves[3 * leaf_size..],
                        ],
                        outs,
                    );
                } else {
                    for (out, leaf) in outs.iter_mut().zip(leaves.chunks(leaf_size)) {
                        *out = hash_leaf(leaf);
                    }
                }
            });
    }
    #[cfg(all(not(all(target_arch = "aarch64", target_feature = "sha2")), target_arch = "x86_64"))]
    {
        tree[..num_leaves]
            .par_chunks_mut(4)
            .zip(data.par_chunks(4 * leaf_size))
            .for_each(|(outs, leaves)| {
                if outs.len() == 4 {
                    #[cfg(feature = "hash-count")]
                    {
                        use std::sync::atomic::Ordering::Relaxed;
                        hash_count::LEAF_CALLS.fetch_add(4, Relaxed);
                        hash_count::LEAF_COMPRESSIONS
                            .fetch_add(4 * hash_count::sha256_blocks(leaf_size), Relaxed);
                    }
                    sha256x4_x86::hash4(
                        [
                            &leaves[..leaf_size],
                            &leaves[leaf_size..2 * leaf_size],
                            &leaves[2 * leaf_size..3 * leaf_size],
                            &leaves[3 * leaf_size..],
                        ],
                        outs,
                    );
                } else {
                    for (out, leaf) in outs.iter_mut().zip(leaves.chunks(leaf_size)) {
                        *out = hash_leaf(leaf);
                    }
                }
            });
    }
    #[cfg(not(any(all(target_arch = "aarch64", target_feature = "sha2"), target_arch = "x86_64")))]
    {
        tree[..num_leaves]
            .par_iter_mut()
            .zip(data.par_chunks(leaf_size))
            .for_each(|(out, leaf)| *out = hash_leaf(leaf));
    }

    // 2. Internal levels — parallel within a level, sequential across levels.
    let mut read_start = 0usize;
    let mut read_len = num_leaves;
    while read_len > 1 {
        let next_len = read_len >> 1;
        // Split the buffer at the end of the current level so we get two
        // non-overlapping mutable slices: `read` (input) and `write` (output).
        let (read, rest) = tree[read_start..].split_at_mut(read_len);
        let write = &mut rest[..next_len];

        // 4 parents at a time = 8 contiguous children = 256 contiguous bytes;
        // each parent hashes its 64-byte child pair, interleaved 4-way.
        #[cfg(all(target_arch = "aarch64", target_feature = "sha2"))]
        {
            let read_bytes: &[u8] =
                unsafe { core::slice::from_raw_parts(read.as_ptr() as *const u8, read.len() * 32) };
            let hash_quad = |outs: &mut [Hash], children: &[u8]| {
                if outs.len() == 4 {
                    #[cfg(feature = "hash-count")]
                    hash_count::PAIR_CALLS.fetch_add(4, std::sync::atomic::Ordering::Relaxed);
                    sha256x4::hash4_equal_len(
                        [
                            &children[..64],
                            &children[64..128],
                            &children[128..192],
                            &children[192..256],
                        ],
                        outs,
                    );
                } else {
                    for (i, out) in outs.iter_mut().enumerate() {
                        let l: &Hash = children[i * 64..i * 64 + 32].try_into().unwrap();
                        let r: &Hash = children[i * 64 + 32..i * 64 + 64].try_into().unwrap();
                        *out = hash_pair(l, r);
                    }
                }
            };
            // Small upper levels can't fill the cores (≤ SERIAL_LEVEL_NODES / 4
            // SHA-x4 tasks), so a rayon dispatch per level costs more than the
            // hashing itself (~3× at the top of a 2^18 tree). Hash them serially
            // — still 4-way SIMD — and only fan out the wide lower levels.
            const SERIAL_LEVEL_NODES: usize = 1024;
            if write.len() <= SERIAL_LEVEL_NODES {
                for (outs, children) in write.chunks_mut(4).zip(read_bytes.chunks(256)) {
                    hash_quad(outs, children);
                }
            } else {
                write
                    .par_chunks_mut(4)
                    .zip(read_bytes.par_chunks(256))
                    .for_each(|(outs, children)| hash_quad(outs, children));
            }
        }
        #[cfg(all(not(all(target_arch = "aarch64", target_feature = "sha2")), target_arch = "x86_64"))]
        {
            // SAFETY: `read` is a contiguous &[Hash] (each 32 bytes); reinterpreting
            // as &[u8] is valid because Hash = [u8; 32] has alignment 1.
            let read_bytes: &[u8] =
                unsafe { core::slice::from_raw_parts(read.as_ptr() as *const u8, read.len() * 32) };
            let hash_quad = |outs: &mut [Hash], children: &[u8]| {
                if outs.len() == 4 {
                    #[cfg(feature = "hash-count")]
                    hash_count::PAIR_CALLS.fetch_add(4, std::sync::atomic::Ordering::Relaxed);
                    sha256x4_x86::hash4(
                        [
                            &children[..64],
                            &children[64..128],
                            &children[128..192],
                            &children[192..256],
                        ],
                        outs,
                    );
                } else {
                    for (i, out) in outs.iter_mut().enumerate() {
                        let l: &Hash = children[i * 64..i * 64 + 32].try_into().unwrap();
                        let r: &Hash = children[i * 64 + 32..i * 64 + 64].try_into().unwrap();
                        *out = hash_pair(l, r);
                    }
                }
            };
            const SERIAL_LEVEL_NODES: usize = 1024;
            if write.len() <= SERIAL_LEVEL_NODES {
                for (outs, children) in write.chunks_mut(4).zip(read_bytes.chunks(256)) {
                    hash_quad(outs, children);
                }
            } else {
                write
                    .par_chunks_mut(4)
                    .zip(read_bytes.par_chunks(256))
                    .for_each(|(outs, children)| hash_quad(outs, children));
            }
        }
        #[cfg(not(any(all(target_arch = "aarch64", target_feature = "sha2"), target_arch = "x86_64")))]
        {
            write
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, out)| *out = hash_pair(&read[2 * i], &read[2 * i + 1]));
        }

        read_start += read_len;
        read_len = next_len;
    }

    tree
}

/// Sequential (single-threaded) version of [`merkle_tree`]. Used for
/// benchmark comparison and as the test oracle.
pub fn merkle_tree_sequential(data: &[u8], num_leaves: usize) -> Vec<Hash> {
    assert!(num_leaves.is_power_of_two() && num_leaves > 0);
    assert_eq!(data.len() % num_leaves, 0);

    let leaf_size = data.len() / num_leaves;
    let total_nodes = 2 * num_leaves - 1;
    let mut tree: Vec<Hash> = crate::alloc_uninit_vec(total_nodes);

    for (i, leaf) in data.chunks(leaf_size).enumerate() {
        tree[i] = hash_leaf(leaf);
    }
    let mut read_start = 0usize;
    let mut read_len = num_leaves;
    while read_len > 1 {
        let next_len = read_len >> 1;
        for i in 0..next_len {
            let left = tree[read_start + 2 * i];
            let right = tree[read_start + 2 * i + 1];
            tree[read_start + read_len + i] = hash_pair(&left, &right);
        }
        read_start += read_len;
        read_len = next_len;
    }
    tree
}

// ---------------------------------------------------------------------------
// Merkle path opening and verification.
// ---------------------------------------------------------------------------

/// Build an opening proof for leaf `index`: the sibling hashes from the leaf
/// level up to (but not including) the root.
///
/// `tree` must be the flat tree produced by [`merkle_tree`] or
/// [`merkle_tree_sequential`] for `num_leaves` leaves. The returned vector has
/// length `log2(num_leaves)`.
///
/// Verify with [`verify_merkle_proof`].
pub fn merkle_proof(tree: &[Hash], num_leaves: usize, index: usize) -> Vec<Hash> {
    assert!(num_leaves.is_power_of_two() && num_leaves > 0);
    assert!(index < num_leaves);
    assert_eq!(tree.len(), 2 * num_leaves - 1);

    let log_n = num_leaves.trailing_zeros() as usize;
    let mut proof = Vec::with_capacity(log_n);

    let mut level_start = 0usize;
    let mut level_len = num_leaves;
    let mut idx = index;
    while level_len > 1 {
        let sibling_idx = idx ^ 1;
        proof.push(tree[level_start + sibling_idx]);
        level_start += level_len;
        level_len >>= 1;
        idx >>= 1;
    }
    proof
}

/// Verify a Merkle opening: recomputes the root from `leaf_hash`, the path,
/// and the leaf index. Returns true iff the recomputed root matches `root`.
pub fn verify_merkle_proof(root: &Hash, leaf_hash: &Hash, index: usize, proof: &[Hash]) -> bool {
    let mut acc = *leaf_hash;
    let mut idx = index;
    for sibling in proof {
        // If idx is even, our node is the LEFT child; sibling is on the RIGHT.
        let (left, right) = if idx & 1 == 0 {
            (acc, *sibling)
        } else {
            (*sibling, acc)
        };
        acc = hash_pair(&left, &right);
        idx >>= 1;
    }
    &acc == root
}

// ---------------------------------------------------------------------------
// Multi-proof (Octopus / batched opening): one shared proof for multiple leaf
// positions, deduplicating siblings that lie on multiple paths.
// ---------------------------------------------------------------------------

/// Build a Merkle multi-proof for `positions`. Returns the sibling hashes
/// needed to verify ALL positions against the root, in the canonical
/// bottom-up sorted-by-position traversal order.
///
/// `positions` need not be sorted or unique; the function sorts + dedupes
/// internally. For `q` queries in a tree of depth `d`, the output is at
/// most `q · d` hashes (matching `q` independent paths) and typically much
/// smaller (siblings shared across multiple paths are emitted once).
///
/// Verify with [`verify_merkle_multi_proof`].
pub fn merkle_multi_proof(tree: &[Hash], num_leaves: usize, positions: &[usize]) -> Vec<Hash> {
    assert!(num_leaves.is_power_of_two() && num_leaves > 0);
    assert_eq!(tree.len(), 2 * num_leaves - 1);

    if positions.is_empty() || num_leaves == 1 {
        return Vec::new();
    }

    let mut active: Vec<usize> = positions.to_vec();
    active.sort_unstable();
    active.dedup();
    debug_assert!(active.iter().all(|&p| p < num_leaves));

    let mut proof = Vec::new();
    let mut level_start = 0usize;
    let mut level_len = num_leaves;

    while level_len > 1 {
        let mut next = Vec::with_capacity(active.len());
        let mut i = 0;
        while i < active.len() {
            let p = active[i];
            let sib_active = i + 1 < active.len() && active[i + 1] == (p ^ 1);
            if sib_active {
                // Both children active — no sibling hash needed; both fold into
                // the same parent.
                i += 2;
            } else {
                // Sibling not in active set; emit it.
                proof.push(tree[level_start + (p ^ 1)]);
                i += 1;
            }
            next.push(p >> 1);
        }
        // `next` is sorted-unique by construction: the input was sorted-unique;
        // consecutive sibling pairs (handled above) collapse to one; otherwise
        // p >> 1 preserves strict ordering.
        active = next;
        level_start += level_len;
        level_len >>= 1;
    }

    proof
}

/// Verify a Merkle multi-proof produced by [`merkle_multi_proof`].
///
/// `sorted_unique_positions` and `leaf_hashes` must be aligned and sorted:
/// `leaf_hashes[i]` is the hash of the leaf at `sorted_unique_positions[i]`,
/// and the position list is strictly ascending. Returns true iff the
/// reconstructed root equals `root` and the proof is consumed exactly.
pub fn verify_merkle_multi_proof(
    root: &Hash,
    num_leaves: usize,
    sorted_unique_positions: &[usize],
    leaf_hashes: &[Hash],
    proof: &[Hash],
) -> bool {
    if !num_leaves.is_power_of_two() || num_leaves == 0 {
        return false;
    }
    if sorted_unique_positions.len() != leaf_hashes.len() {
        return false;
    }
    if sorted_unique_positions.is_empty() {
        // Vacuous; nothing to verify. Treat as "ok" iff the proof is empty.
        return proof.is_empty();
    }
    // Verify the position list is sorted strictly ascending + in range.
    for (i, &p) in sorted_unique_positions.iter().enumerate() {
        if p >= num_leaves {
            return false;
        }
        if i > 0 && sorted_unique_positions[i - 1] >= p {
            return false;
        }
    }
    // Edge case: 1-leaf tree, no proof needed.
    if num_leaves == 1 {
        return proof.is_empty() && leaf_hashes[0] == *root;
    }

    let mut active: Vec<(usize, Hash)> = sorted_unique_positions
        .iter()
        .copied()
        .zip(leaf_hashes.iter().copied())
        .collect();
    let mut proof_iter = proof.iter().copied();
    let mut level_len = num_leaves;

    while level_len > 1 {
        let mut next = Vec::with_capacity(active.len());
        let mut i = 0;
        while i < active.len() {
            let (p, h) = active[i];
            let sib_active = i + 1 < active.len() && active[i + 1].0 == (p ^ 1);
            let (left, right) = if sib_active {
                let (_, h_sib) = active[i + 1];
                // Sorted strictly ascending → active[i+1].0 = p + 1 (= p ^ 1
                // since p is even when p ^ 1 = p + 1). So p is LEFT child.
                debug_assert_eq!(p & 1, 0);
                i += 2;
                (h, h_sib)
            } else {
                let sib = match proof_iter.next() {
                    Some(s) => s,
                    None => return false,
                };
                i += 1;
                if p & 1 == 0 { (h, sib) } else { (sib, h) }
            };
            next.push((p >> 1, hash_pair(&left, &right)));
        }
        active = next;
        level_len >>= 1;
    }

    // After the loop, `active` has exactly one element (the root). Reject
    // any leftover proof bytes.
    if proof_iter.next().is_some() {
        return false;
    }
    active.len() == 1 && active[0].1 == *root
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_leaves_matches_hand_computation() {
        // Two 8-byte leaves: [0,1,2,3,4,5,6,7] and [8,9,10,11,12,13,14,15].
        let data: Vec<u8> = (0..16).collect();
        let tree = merkle_tree(&data, 2);
        assert_eq!(tree.len(), 3); // 2 leaves + 1 root

        let h0 = hash_leaf(&data[0..8]);
        let h1 = hash_leaf(&data[8..16]);
        let root = hash_pair(&h0, &h1);

        assert_eq!(tree[0], h0);
        assert_eq!(tree[1], h1);
        assert_eq!(tree[2], root);
    }

    #[test]
    fn one_leaf_root_is_the_leaf_hash() {
        let data: Vec<u8> = (0..32).collect();
        let root = merkle_root(&data, 1);
        assert_eq!(root, hash_leaf(&data));
    }

    #[test]
    fn parallel_matches_sequential() {
        // Use a non-trivial size: 1024 leaves × 64 B = 64 KB.
        let n_leaves = 1024;
        let leaf_size = 64;
        let mut data = vec![0u8; n_leaves * leaf_size];
        // Fill with a deterministic pattern.
        for (i, b) in data.iter_mut().enumerate() {
            *b = ((i.wrapping_mul(0x9E3779B9)) & 0xff) as u8;
        }
        let par = merkle_tree(&data, n_leaves);
        let seq = merkle_tree_sequential(&data, n_leaves);
        assert_eq!(par, seq);
    }

    /// Leaf sizes chosen to hit every SHA-256 tail shape in the 4-way
    /// interleaved path: rem = 0 (block-aligned), rem < 56 (one tail block),
    /// and rem ≥ 56 (two tail blocks). Also a non-multiple-of-4 leaf count
    /// for the remainder fallback.
    #[test]
    fn parallel_matches_sequential_tail_shapes() {
        for (n_leaves, leaf_size) in [(64, 1024), (64, 100), (64, 60), (64, 56), (2, 48), (16, 1)] {
            let mut data = vec![0u8; n_leaves * leaf_size];
            for (i, b) in data.iter_mut().enumerate() {
                *b = ((i.wrapping_mul(0x6C8E944D)) & 0xff) as u8;
            }
            let par = merkle_tree(&data, n_leaves);
            let seq = merkle_tree_sequential(&data, n_leaves);
            assert_eq!(par, seq, "n_leaves={n_leaves} leaf_size={leaf_size}");
        }
    }

    #[test]
    fn root_changes_when_any_leaf_changes() {
        let n_leaves = 64;
        let leaf_size = 32;
        let mut data = vec![0u8; n_leaves * leaf_size];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(31);
        }
        let r0 = merkle_root(&data, n_leaves);
        // Flip one bit deep in the buffer.
        data[n_leaves * leaf_size - 1] ^= 0x01;
        let r1 = merkle_root(&data, n_leaves);
        assert_ne!(r0, r1, "single-bit change should change the root");
    }

    #[test]
    fn power_of_two_assertion() {
        let data = vec![0u8; 64];
        // Should not panic for power-of-two leaf counts.
        let _ = merkle_root(&data, 1);
        let _ = merkle_root(&data, 2);
        let _ = merkle_root(&data, 4);
        let _ = merkle_root(&data, 8);
    }

    #[test]
    #[should_panic(expected = "num_leaves must be power of 2")]
    fn rejects_non_power_of_two() {
        let data = vec![0u8; 30];
        let _ = merkle_root(&data, 3);
    }

    #[test]
    fn merkle_proof_roundtrips_at_every_leaf() {
        let n_leaves = 16;
        let leaf_size = 8;
        let mut data = vec![0u8; n_leaves * leaf_size];
        for (i, b) in data.iter_mut().enumerate() {
            *b = ((i.wrapping_mul(0x9E3779B9)) & 0xff) as u8;
        }
        let tree = merkle_tree(&data, n_leaves);
        let root = *tree.last().unwrap();

        for i in 0..n_leaves {
            let leaf_hash = hash_leaf(&data[i * leaf_size..(i + 1) * leaf_size]);
            let proof = merkle_proof(&tree, n_leaves, i);
            assert_eq!(proof.len(), 4); // log2(16) = 4
            assert!(
                verify_merkle_proof(&root, &leaf_hash, i, &proof),
                "verify failed at i={i}"
            );
        }
    }

    #[test]
    fn merkle_proof_rejects_wrong_index() {
        let n_leaves = 8;
        let leaf_size = 16;
        let data: Vec<u8> = (0..(n_leaves * leaf_size) as u8).collect();
        let tree = merkle_tree(&data, n_leaves);
        let root = *tree.last().unwrap();

        let leaf_hash = hash_leaf(&data[0..leaf_size]);
        let proof = merkle_proof(&tree, n_leaves, 0);

        // Same proof, but claim it's for index 1 → should fail (different sibling structure).
        assert!(!verify_merkle_proof(&root, &leaf_hash, 1, &proof));
    }

    #[test]
    fn merkle_proof_rejects_tampered_path() {
        let n_leaves = 8;
        let leaf_size = 16;
        let data: Vec<u8> = (0..(n_leaves * leaf_size) as u8).collect();
        let tree = merkle_tree(&data, n_leaves);
        let root = *tree.last().unwrap();

        let leaf_hash = hash_leaf(&data[0..leaf_size]);
        let mut proof = merkle_proof(&tree, n_leaves, 0);
        // Flip a byte in the first sibling.
        proof[0][0] ^= 1;
        assert!(!verify_merkle_proof(&root, &leaf_hash, 0, &proof));
    }

    fn random_data(n_leaves: usize, leaf_size: usize, seed: u64) -> Vec<u8> {
        let mut data = vec![0u8; n_leaves * leaf_size];
        let mut z = seed;
        for b in data.iter_mut() {
            z = z.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
            *b = ((z >> 33) & 0xff) as u8;
        }
        data
    }

    #[test]
    fn multi_proof_single_position_matches_single_proof() {
        let (n_leaves, leaf_size) = (16, 8);
        let data = random_data(n_leaves, leaf_size, 42);
        let tree = merkle_tree(&data, n_leaves);
        let root = *tree.last().unwrap();

        for i in 0..n_leaves {
            let multi = merkle_multi_proof(&tree, n_leaves, &[i]);
            let single = merkle_proof(&tree, n_leaves, i);
            assert_eq!(
                multi, single,
                "multi-proof of [{i}] must equal single proof"
            );

            let leaf_hash = hash_leaf(&data[i * leaf_size..(i + 1) * leaf_size]);
            assert!(verify_merkle_multi_proof(
                &root,
                n_leaves,
                &[i],
                &[leaf_hash],
                &multi
            ));
        }
    }

    #[test]
    fn multi_proof_sibling_pair_emits_no_hashes_at_leaf_level() {
        // Sibling pair (0,1) at the leaf level shares its parent → no leaf-level
        // sibling is needed; one sibling per remaining level.
        let n_leaves = 8;
        let leaf_size = 4;
        let data = random_data(n_leaves, leaf_size, 7);
        let tree = merkle_tree(&data, n_leaves);
        let root = *tree.last().unwrap();

        let multi = merkle_multi_proof(&tree, n_leaves, &[0, 1]);
        assert_eq!(
            multi.len(),
            2,
            "sibling pair at leaves saves the leaf-level hash"
        );

        let leaves: Vec<Hash> = [0usize, 1]
            .iter()
            .map(|&i| hash_leaf(&data[i * leaf_size..(i + 1) * leaf_size]))
            .collect();
        assert!(verify_merkle_multi_proof(
            &root,
            n_leaves,
            &[0, 1],
            &leaves,
            &multi
        ));
    }

    #[test]
    fn multi_proof_full_query_set_is_root_only() {
        // Every leaf queried → the verifier already knows everything, so the
        // multi-proof should be empty.
        let n_leaves = 16;
        let leaf_size = 8;
        let data = random_data(n_leaves, leaf_size, 99);
        let tree = merkle_tree(&data, n_leaves);
        let root = *tree.last().unwrap();

        let positions: Vec<usize> = (0..n_leaves).collect();
        let multi = merkle_multi_proof(&tree, n_leaves, &positions);
        assert!(
            multi.is_empty(),
            "full-set multi-proof should have zero hashes"
        );

        let leaves: Vec<Hash> = (0..n_leaves)
            .map(|i| hash_leaf(&data[i * leaf_size..(i + 1) * leaf_size]))
            .collect();
        assert!(verify_merkle_multi_proof(
            &root, n_leaves, &positions, &leaves, &multi
        ));
    }

    #[test]
    fn multi_proof_random_subsets_roundtrip() {
        let n_leaves = 64;
        let leaf_size = 16;
        let data = random_data(n_leaves, leaf_size, 2024);
        let tree = merkle_tree(&data, n_leaves);
        let root = *tree.last().unwrap();

        let all_leaves: Vec<Hash> = (0..n_leaves)
            .map(|i| hash_leaf(&data[i * leaf_size..(i + 1) * leaf_size]))
            .collect();

        let subsets: &[&[usize]] = &[
            &[0],
            &[63],
            &[0, 63],
            &[3, 17, 41],
            &[10, 11, 12, 13],
            &[0, 1, 2, 3, 60, 61, 62, 63],
            &[5, 5, 5, 17, 17],
            &[0, 8, 16, 24, 32, 40, 48, 56],
        ];
        for positions in subsets {
            let multi = merkle_multi_proof(&tree, n_leaves, positions);

            let mut sorted: Vec<usize> = positions.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            let leaves: Vec<Hash> = sorted.iter().map(|&p| all_leaves[p]).collect();

            assert!(
                verify_merkle_multi_proof(&root, n_leaves, &sorted, &leaves, &multi),
                "roundtrip failed for positions={positions:?}"
            );

            let log_n = n_leaves.trailing_zeros() as usize;
            assert!(
                multi.len() <= sorted.len() * log_n,
                "multi-proof can't exceed sum of independent paths"
            );
        }
    }

    #[test]
    fn multi_proof_rejects_wrong_leaf() {
        let (n_leaves, leaf_size) = (32, 8);
        let data = random_data(n_leaves, leaf_size, 1);
        let tree = merkle_tree(&data, n_leaves);
        let root = *tree.last().unwrap();

        let positions = vec![3usize, 7, 19, 28];
        let multi = merkle_multi_proof(&tree, n_leaves, &positions);
        let mut leaves: Vec<Hash> = positions
            .iter()
            .map(|&p| hash_leaf(&data[p * leaf_size..(p + 1) * leaf_size]))
            .collect();

        assert!(verify_merkle_multi_proof(
            &root, n_leaves, &positions, &leaves, &multi
        ));
        leaves[1][0] ^= 1;
        assert!(!verify_merkle_multi_proof(
            &root, n_leaves, &positions, &leaves, &multi
        ));
    }

    #[test]
    fn multi_proof_rejects_tampered_proof_hash() {
        let (n_leaves, leaf_size) = (32, 8);
        let data = random_data(n_leaves, leaf_size, 2);
        let tree = merkle_tree(&data, n_leaves);
        let root = *tree.last().unwrap();

        let positions = vec![1usize, 14, 27];
        let mut multi = merkle_multi_proof(&tree, n_leaves, &positions);
        let leaves: Vec<Hash> = positions
            .iter()
            .map(|&p| hash_leaf(&data[p * leaf_size..(p + 1) * leaf_size]))
            .collect();

        assert!(verify_merkle_multi_proof(
            &root, n_leaves, &positions, &leaves, &multi
        ));
        multi[0][0] ^= 1;
        assert!(!verify_merkle_multi_proof(
            &root, n_leaves, &positions, &leaves, &multi
        ));
    }

    #[test]
    fn multi_proof_rejects_extra_or_missing_hashes() {
        let (n_leaves, leaf_size) = (16, 8);
        let data = random_data(n_leaves, leaf_size, 3);
        let tree = merkle_tree(&data, n_leaves);
        let root = *tree.last().unwrap();

        let positions = vec![2usize, 11];
        let multi = merkle_multi_proof(&tree, n_leaves, &positions);
        let leaves: Vec<Hash> = positions
            .iter()
            .map(|&p| hash_leaf(&data[p * leaf_size..(p + 1) * leaf_size]))
            .collect();

        let mut extra = multi.clone();
        extra.push([0xaa; 32]);
        assert!(!verify_merkle_multi_proof(
            &root, n_leaves, &positions, &leaves, &extra
        ));

        let mut short = multi.clone();
        short.pop();
        assert!(!verify_merkle_multi_proof(
            &root, n_leaves, &positions, &leaves, &short
        ));
    }

    #[test]
    fn multi_proof_rejects_unsorted_positions() {
        let (n_leaves, leaf_size) = (16, 8);
        let data = random_data(n_leaves, leaf_size, 5);
        let tree = merkle_tree(&data, n_leaves);
        let root = *tree.last().unwrap();

        let positions = vec![2usize, 11];
        let multi = merkle_multi_proof(&tree, n_leaves, &positions);
        let leaves: Vec<Hash> = positions
            .iter()
            .map(|&p| hash_leaf(&data[p * leaf_size..(p + 1) * leaf_size]))
            .collect();

        let unsorted = vec![11usize, 2];
        let unsorted_leaves = vec![leaves[1], leaves[0]];
        assert!(!verify_merkle_multi_proof(
            &root,
            n_leaves,
            &unsorted,
            &unsorted_leaves,
            &multi
        ));
    }

    #[test]
    fn multi_proof_beats_independent_paths_at_scale() {
        let n_leaves = 1024;
        let leaf_size = 8;
        let data = random_data(n_leaves, leaf_size, 4096);
        let tree = merkle_tree(&data, n_leaves);
        let root = *tree.last().unwrap();
        let log_n = n_leaves.trailing_zeros() as usize;

        let positions_raw: Vec<usize> = (0..100)
            .map(|i| {
                let mut z = (i as u64).wrapping_mul(0xDEAD_BEEF_F0F0_F0F0);
                z ^= z >> 27;
                (z as usize) & (n_leaves - 1)
            })
            .collect();
        let multi = merkle_multi_proof(&tree, n_leaves, &positions_raw);

        let mut positions = positions_raw.clone();
        positions.sort_unstable();
        positions.dedup();
        let leaves: Vec<Hash> = positions
            .iter()
            .map(|&p| hash_leaf(&data[p * leaf_size..(p + 1) * leaf_size]))
            .collect();

        assert!(verify_merkle_multi_proof(
            &root, n_leaves, &positions, &leaves, &multi
        ));
        assert!(
            multi.len() < positions.len() * log_n,
            "multi-proof should beat independent paths: got {} vs {} × {}",
            multi.len(),
            positions.len(),
            log_n
        );
    }

    #[test]
    fn sha256x4_x86_matches_scalar() {
        let inputs: [[u8; 64]; 4] = [
            {
                let mut a = [0u8; 64];
                for (i, b) in a.iter_mut().enumerate() {
                    *b = (i as u8).wrapping_mul(0x37);
                }
                a
            },
            {
                let mut a = [0u8; 64];
                for (i, b) in a.iter_mut().enumerate() {
                    *b = (i as u8).wrapping_add(0xAB);
                }
                a
            },
            {
                let mut a = [0u8; 64];
                for (i, b) in a.iter_mut().enumerate() {
                    *b = (i as u8) ^ 0xCD;
                }
                a
            },
            {
                let mut a = [0u8; 64];
                for (i, b) in a.iter_mut().enumerate() {
                    *b = ((i * 7 + 3) & 0xFF) as u8;
                }
                a
            },
        ];

        let scalar: Vec<Hash> = inputs
            .iter()
            .map(|inp| Sha256::digest(inp).into())
            .collect();

        let mut batch = [[0u8; 32]; 4];
        #[cfg(target_arch = "x86_64")]
        {
            super::sha256x4_x86::hash4(
                [&inputs[0][..], &inputs[1][..], &inputs[2][..], &inputs[3][..]],
                &mut batch,
            );
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            for i in 0..4 {
                batch[i] = Sha256::digest(&inputs[i]).into();
            }
        }

        for i in 0..4 {
            assert_eq!(
                batch[i], scalar[i],
                "sha256x4 lane {i} differs from scalar"
            );
        }
    }

    #[test]
    fn sha256x4_x86_various_lengths() {
        for len in [1, 7, 32, 55, 56, 63, 64, 100, 128, 200] {
            let inputs: Vec<Vec<u8>> = (0..4u8)
                .map(|seed| {
                    (0..len)
                        .map(|i| (i as u8).wrapping_mul(seed.wrapping_add(0x31)))
                        .collect()
                })
                .collect();

            let scalar: Vec<Hash> = inputs
                .iter()
                .map(|inp| Sha256::digest(inp).into())
                .collect();

            let mut batch = [[0u8; 32]; 4];
            #[cfg(target_arch = "x86_64")]
            {
                super::sha256x4_x86::hash4(
                    [&inputs[0], &inputs[1], &inputs[2], &inputs[3]],
                    &mut batch,
                );
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                for i in 0..4 {
                    batch[i] = Sha256::digest(&inputs[i]).into();
                }
            }

            for i in 0..4 {
                assert_eq!(
                    batch[i], scalar[i],
                    "sha256x4 lane {i} differs at len={len}"
                );
            }
        }
    }

    #[test]
    fn merkle_tree_x86_matches_scalar() {
        for (n_leaves, leaf_size) in [
            (4, 32),
            (8, 64),
            (16, 8),
            (64, 100),
            (256, 16),
            (1024, 64),
        ] {
            let data = random_data(n_leaves, leaf_size, 0xDEAD_CAFE);
            let tree = merkle_tree(&data, n_leaves);
            let seq = merkle_tree_sequential(&data, n_leaves);
            assert_eq!(
                tree, seq,
                "merkle_tree (x86 path) differs from sequential at n_leaves={n_leaves} leaf_size={leaf_size}"
            );
        }
    }
}
