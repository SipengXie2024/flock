//! MHOT prover throughput benchmark (x86 scalar fallback).
//! Run: cargo bench -p flock-prover --bench mhot_bench

use std::hint::black_box;
use std::time::Instant;

use flock_prover::mhot::{
    hash_only::{prove_mhot_hash_only, verify_mhot_hash_only},
    multi_base::{prove_multi, verify_multi},
    ref_witness::build_ref_witness,
    route::{self, RouteWitness},
    schedule::MhotHashSchedule,
};

struct BenchConfig {
    name: &'static str,
    fanouts: &'static [usize],
}

struct Timing {
    prove_ms: f64,
    verify_ms: f64,
}

fn main() {
    let configs = [
        BenchConfig {
            name: "small [4,2]",
            fanouts: &[4, 2],
        },
        BenchConfig {
            name: "medium [8,4,2]",
            fanouts: &[8, 4, 2],
        },
        BenchConfig {
            name: "wide [16,8,4]",
            fanouts: &[16, 8, 4],
        },
        BenchConfig {
            name: "realistic [28,24,22,16,8]",
            fanouts: &[28, 24, 22, 16, 8],
        },
    ];

    println!("=== MHOT Hash-Only Prove/Verify Benchmark (x86 scalar) ===");
    println!(
        "{:<30} {:>8} {:>12} {:>12} {:>12}",
        "Config", "Atoms", "Prove(ms)", "Verify(ms)", "Total(ms)"
    );
    println!("{}", "-".repeat(76));

    for config in &configs {
        let sched = MhotHashSchedule::from_fanouts(config.fanouts);
        let witness = build_ref_witness(&sched, 42);
        let timing = bench_hash_only(&sched, &witness);

        println!(
            "{:<30} {:>8} {:>12.1} {:>12.1} {:>12.1}",
            config.name,
            sched.hash_atoms.len(),
            timing.prove_ms,
            timing.verify_ms,
            timing.prove_ms + timing.verify_ms
        );
    }

    println!();
    println!("=== MHOT Multi-Base Prove/Verify Benchmark (x86 scalar) ===");
    println!(
        "{:<30} {:>8} {:>8} {:>12} {:>12} {:>12}",
        "Config", "Atoms", "Routes", "Prove(ms)", "Verify(ms)", "Total(ms)"
    );
    println!("{}", "-".repeat(86));

    for config in &configs {
        let sched = MhotHashSchedule::from_fanouts(config.fanouts);
        let witness = build_ref_witness(&sched, 42);
        let routes = route_witnesses_for_schedule(&sched);
        let timing = bench_multi_base(&sched, &witness, &routes);

        println!(
            "{:<30} {:>8} {:>8} {:>12.1} {:>12.1} {:>12.1}",
            config.name,
            sched.hash_atoms.len(),
            routes.len(),
            timing.prove_ms,
            timing.verify_ms,
            timing.prove_ms + timing.verify_ms
        );
    }

    println!();
    println!("NOTE: x86 scalar fallback. Flock is optimized for Apple M NEON.");
    println!("      Treat these as x86 baseline trends; rerun on Apple M for paper numbers.");
}

fn bench_hash_only(
    sched: &MhotHashSchedule,
    witness: &flock_prover::mhot::ref_witness::RefWitness,
) -> Timing {
    let warmup = prove_mhot_hash_only(sched, witness);
    verify_mhot_hash_only(sched.hash_atoms.len(), &warmup).expect("warmup hash-only verify");
    black_box(&warmup);

    let t0 = Instant::now();
    let proof = prove_mhot_hash_only(sched, witness);
    let prove_ms = elapsed_ms(t0);

    let t1 = Instant::now();
    verify_mhot_hash_only(sched.hash_atoms.len(), &proof).expect("hash-only verify");
    let verify_ms = elapsed_ms(t1);
    black_box(&proof);

    Timing {
        prove_ms,
        verify_ms,
    }
}

fn bench_multi_base(
    sched: &MhotHashSchedule,
    witness: &flock_prover::mhot::ref_witness::RefWitness,
    routes: &[RouteWitness],
) -> Timing {
    let warmup = prove_multi(sched, witness, routes);
    verify_multi(&warmup).expect("warmup multi-base verify");
    black_box(&warmup);

    let t0 = Instant::now();
    let proof = prove_multi(sched, witness, routes);
    let prove_ms = elapsed_ms(t0);

    let t1 = Instant::now();
    verify_multi(&proof).expect("multi-base verify");
    let verify_ms = elapsed_ms(t1);
    black_box(&proof);

    Timing {
        prove_ms,
        verify_ms,
    }
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn route_witnesses_for_schedule(sched: &MhotHashSchedule) -> Vec<RouteWitness> {
    sched
        .fanouts
        .iter()
        .enumerate()
        .map(|(node, _)| route_witness_for_node(node))
        .collect()
}

fn route_witness_for_node(node: usize) -> RouteWitness {
    let mut key = [false; route::KEY_BITS];
    let mut mask = [false; route::KEY_BITS];
    key[0] = (node & 1) != 0;
    key[1] = true;
    mask[0] = true;
    mask[1] = true;

    let children: Vec<[bool; route::DIGEST_BITS]> = (0..route::FANOUT)
        .map(|child| std::array::from_fn(|bit| ((node * 31 + child * 17 + bit) & 1) != 0))
        .collect();

    RouteWitness::new(key, mask, children)
}
