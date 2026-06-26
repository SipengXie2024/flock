# MHOT → Flock Benchmark Results

## Environment

- **CPU**: x86_64 (scalar fallback — NOT Flock's optimized target)
- **Flock target**: Apple M NEON (rerun needed for paper numbers)
- **PCS**: Ligerito, minimum embedded config `m=22`
- **All configs pad to m=22 floor**: 2-32 atoms share the same proof cost

## Hash-Only (keccak3 prove/verify)

| Config | Fanouts | Atoms | Prove (ms) | Verify (ms) | Total (ms) | ms/atom |
|--------|---------|------:|----------:|----------:|----------:|--------:|
| small | [4,2] | 2 | 101 | 71 | 173 | 86 |
| medium | [8,4,2] | 5 | 96 | 72 | 167 | 33 |
| wide | [16,8,4] | 9 | 94 | 71 | 165 | 18 |
| realistic | [28,24,22,16,8] | 32 | 99 | 72 | 171 | 5.3 |

## Multi-Base (F_hash + F_route)

| Config | Fanouts | Atoms | Routes | Prove (ms) | Verify (ms) | Total (ms) | ms/atom |
|--------|---------|------:|-------:|----------:|----------:|----------:|--------:|
| small | [4,2] | 2 | 2 | 153 | 101 | 254 | 127 |
| medium | [8,4,2] | 5 | 3 | 151 | 103 | 254 | 51 |
| wide | [16,8,4] | 9 | 3 | 147 | 101 | 248 | 28 |
| realistic | [28,24,22,16,8] | 32 | 5 | 154 | 101 | 255 | 8 |

## Key Observations

1. **Ligerito floor effect**: All configs use the same m=22 Ligerito setup (minimum embedded config). Absolute wall-clock is nearly constant (~165ms hash-only, ~250ms multi-base) regardless of atom count. Per-atom cost drops from ~86ms to ~5ms as atoms increase.

2. **Batch amortization potential**: The flat cost curve means Flock's batch amortization kicks in at scale. The current PoC tests ≤32 atoms (single MHOT path); at 2^14-2^16 paths the per-path cost should drop dramatically.

3. **x86 vs Apple M**: These numbers are x86 scalar fallback. Flock is optimized for Apple M NEON. Expect 3-10x speedup on Apple M (based on Flock paper's reported throughput).

4. **Multi-base overhead**: Adding F_route to F_hash costs ~85ms extra (~50% overhead), dominated by the second prove_fast_core + PCS opening. The route R1CS (K=32768) is smaller than keccak3 (K=131072), so the overhead is sub-linear.

## Comparison with MHOT@Expander-M31

Direct comparison is not meaningful (different field, different proof system, different metric):
- M31 numbers are `total_cost` (constraint count), not wall-clock
- M31 MHOT membership: ~52.66M constraints (Poseidon)
- Flock numbers are wall-clock on x86 scalar (binary field, batch prover)

The meaningful comparison for the paper is Flock MHOT vs Flock keccak-chain baseline, both on Apple M.

## What's Proven

The SNARK proves:
- ✅ Every keccak-f atom is correctly computed (hash correctness)
- ✅ Every PEXT route selects the correct child (route correctness)
- ✅ Both share the same Fiat-Shamir transcript (binding)
- ⚠️ Wire equality between atoms is CPU-oracle verified, not in-circuit (needs custom wiring sumcheck)
- ⚠️ Content sponge constraints are CPU-oracle verified, not in-circuit
- ⚠️ No ZK (Flock is succinct-only currently)

## Test Coverage

46 mhot tests (38 unit + 8 acceptance):
- schedule: 6 (atom counts, wires, fanout edge cases)
- ref_witness: 4 (CPU fold root, self-consistency)
- wide_glue: 3 (wiring oracle, tamper detection)
- hash_only: 3 (prove/verify roundtrip, cross-check, multi-fanout)
- route: 6 (R1CS satisfies, prove/verify, negative controls)
- multi_base: 4 (roundtrip + 3 transcript soundness negatives)
- content: 10 (compact mask, absence, counts)
- zk_stub: 2 (config, popcount CPU)
- acceptance: 8 (end-to-end membership + absence)
