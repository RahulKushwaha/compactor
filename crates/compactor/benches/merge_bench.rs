//! Head-to-head perf comparison across two independent axes:
//! - **Algorithm**: `HeapMerger` (RocksDB-style binary heap, see
//!   table/merging_iterator.cc's minHeap_) vs `LoserTreeMerger` (tournament
//!   loser tree).
//! - **Representation**: `VecSource` (owned `Vec<u8>` per key/value, one
//!   allocation each) vs `ArenaSource` (one arena allocation per source,
//!   `(offset,len)` index).
//!
//! Three combinations are benchmarked (heap+Vec, loser_tree+Vec,
//! loser_tree+Arena) since heap+Arena isn't a live comparison of the
//! representation's real benefit — the interesting representation
//! comparison is against whichever algorithm is faster.
//!
//! A fourth axis, **comparator**, is benchmarked separately in
//! `bench_comparator_prefix_fastpath`: `InternalKeyComparator` (no
//! `sort_prefix` fast path — its suffix needs numeric reinterpretation, see
//! `cmp.rs`) vs `BytewiseComparator` (`ByteComparable`, `sort_prefix`
//! returns `Some`) over otherwise-identical loser-tree/Arena merges. This
//! isolates the cost/benefit of the offset-value comparison shortcut itself
//! from the algorithm/representation choices above.

use compactor::merge::{
    ArenaSourceBuilder, BytewiseComparator, HeapMerger, InternalKeyComparator, KWayMerge,
    LoserTreeMerger, VecSource,
};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn make_internal_key(user_key: u64, seq: u64) -> Vec<u8> {
    let mut k = format!("key{:012}", user_key).into_bytes();
    let packed = (seq << 8) | 1u64; // kTypeValue
    k.extend_from_slice(&packed.to_le_bytes());
    k
}

/// Builds `num_sources` sorted sources (as plain owned-Vec entries) whose
/// merged output has `total_keys` distinct user keys, each source holding a
/// random subset (mimicking overlapping SST key ranges during compaction).
fn build_entries(num_sources: usize, total_keys: u64, seed: u64) -> Vec<Vec<(Vec<u8>, Vec<u8>)>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut per_source: Vec<Vec<(Vec<u8>, Vec<u8>)>> = vec![Vec::new(); num_sources];

    for user_key in 0..total_keys {
        // Each key lands in 1-3 sources with distinct seq numbers, simulating
        // overlapping versions across input SSTs.
        let hits = rng.gen_range(1..=3);
        let mut chosen: Vec<usize> = (0..num_sources).collect();
        for i in 0..chosen.len() {
            let j = rng.gen_range(0..chosen.len());
            chosen.swap(i, j);
        }
        for (seq, &src) in chosen.iter().take(hits).enumerate() {
            let key = make_internal_key(user_key, (hits - seq) as u64);
            let value = format!("value_for_{}_seq_{}", user_key, hits - seq).into_bytes();
            per_source[src].push((key, value));
        }
    }

    for src in &mut per_source {
        src.sort_by(|a, b| compactor::merge::cmp::compare_internal_keys(&a.0, &b.0));
    }
    per_source
}

/// Keys usable, byte-for-byte, by BOTH comparators under test with the SAME
/// resulting sort order — 8-byte big-endian `user_key` followed by a
/// constant 8-byte suffix (seq=0, type=1, packed LE like a real internal
/// key). `InternalKeyComparator` compares the user-key part (always
/// distinct here, so the constant suffix never needs to break a tie);
/// `BytewiseComparator` memcmps the full 16 bytes, and since the suffix is
/// constant, that agrees with the same ascending order. This is what makes
/// the two benchmarks below a fair A/B on comparator cost alone: identical
/// bytes, identical order, only the comparator (and its fast path) differs.
///
/// (An earlier version of this benchmark used bare 8-byte keys with no
/// suffix, which happened to also be "valid" input to
/// `InternalKeyComparator::compare` — but that function treats an 8-byte
/// key as ALL suffix, comparing it as a descending-order packed seq/type
/// value. That silently sorted the two comparators' outputs in opposite
/// directions, an apples-to-oranges comparison rather than an isolated one.)
///
/// Also: keys are raw big-endian bytes, not a decimal string
/// (`format!("key{:016}", n)` zero-pads so almost every key shares the same
/// leading 8 bytes, defeating `sort_prefix` entirely since the varying
/// digits land past byte 8).
fn build_bytewise_entries(num_sources: usize, total_keys: u64) -> Vec<Vec<(Vec<u8>, Vec<u8>)>> {
    build_bytewise_entries_padded(num_sources, total_keys, 0)
}

/// Same as `build_bytewise_entries`, but with `extra_pad_bytes` of constant
/// filler appended after the 8-byte varying prefix and before the constant
/// suffix — lets the bench vary total key length while keeping the
/// leading, differentiating bytes (and thus `sort_prefix`'s behavior)
/// unchanged, to test whether the fast path's benefit (skipping the padding
/// bytes that a full `compare` would otherwise scan) shows up on longer
/// keys even though it doesn't on short ones.
fn build_bytewise_entries_padded(
    num_sources: usize,
    total_keys: u64,
    extra_pad_bytes: usize,
) -> Vec<Vec<(Vec<u8>, Vec<u8>)>> {
    const CONSTANT_SUFFIX: [u8; 8] = 1u64.to_le_bytes(); // seq=0, type=1
    let per_src = (total_keys / num_sources as u64).max(1);
    (0..num_sources)
        .map(|src| {
            (0..per_src)
                .map(|i| {
                    let user_key = src as u64 + i * num_sources as u64;
                    let mut key = user_key.to_be_bytes().to_vec();
                    key.extend(std::iter::repeat(0xABu8).take(extra_pad_bytes));
                    key.extend_from_slice(&CONSTANT_SUFFIX);
                    let value = format!("value_for_{}", user_key).into_bytes();
                    (key, value)
                })
                .collect()
        })
        .collect()
}

fn to_vec_sources(per_source: &[Vec<(Vec<u8>, Vec<u8>)>]) -> Vec<VecSource> {
    per_source.iter().cloned().map(VecSource::new).collect()
}

fn to_arena_sources(per_source: &[Vec<(Vec<u8>, Vec<u8>)>]) -> Vec<compactor::merge::ArenaSource> {
    per_source
        .iter()
        .map(|entries| {
            let mut builder = ArenaSourceBuilder::with_capacity(
                entries.len(),
                entries.iter().map(|(k, v)| k.len() + v.len()).sum(),
            );
            for (k, v) in entries {
                builder.push(k, v);
            }
            builder.build()
        })
        .collect()
}

fn bench_mergers(c: &mut Criterion) {
    let total_keys = 50_000u64;

    for &num_sources in &[4usize, 8, 16, 32] {
        let per_source = build_entries(num_sources, total_keys, 42);
        let mut group = c.benchmark_group(format!("merge_k{}", num_sources));

        group.bench_with_input(
            BenchmarkId::new("heap_vec", num_sources),
            &per_source,
            |b, per_source| {
                b.iter_batched(
                    || to_vec_sources(per_source),
                    |sources| {
                        let mut merger: HeapMerger<VecSource> = HeapMerger::new(sources);
                        let mut count = 0usize;
                        merger.run(|_k, _v| count += 1);
                        count
                    },
                    criterion::BatchSize::LargeInput,
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("loser_tree_vec", num_sources),
            &per_source,
            |b, per_source| {
                b.iter_batched(
                    || to_vec_sources(per_source),
                    |sources| {
                        let mut merger: LoserTreeMerger<VecSource> = LoserTreeMerger::new(sources);
                        let mut count = 0usize;
                        merger.run(|_k, _v| count += 1);
                        count
                    },
                    criterion::BatchSize::LargeInput,
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("loser_tree_arena", num_sources),
            &per_source,
            |b, per_source| {
                b.iter_batched(
                    || to_arena_sources(per_source),
                    |sources| {
                        let mut merger: LoserTreeMerger<compactor::merge::ArenaSource> =
                            LoserTreeMerger::new(sources);
                        let mut count = 0usize;
                        merger.run(|_k, _v| count += 1);
                        count
                    },
                    criterion::BatchSize::LargeInput,
                )
            },
        );

        group.finish();
    }
}

/// Isolates the offset-value (`sort_prefix`) comparison fast path: same
/// algorithm (loser tree) and representation (Arena) throughout, varying
/// only the comparator. `InternalKeyComparator` always falls back to
/// `compare`; `BytewiseComparator` short-circuits on differing prefixes.
fn bench_comparator_prefix_fastpath(c: &mut Criterion) {
    let total_keys = 50_000u64;

    for &num_sources in &[4usize, 8, 16, 32] {
        let per_source = build_bytewise_entries(num_sources, total_keys);
        let mut group = c.benchmark_group(format!("comparator_k{}", num_sources));

        group.bench_with_input(
            BenchmarkId::new("internal_key_comparator", num_sources),
            &per_source,
            |b, per_source| {
                b.iter_batched(
                    || to_arena_sources(per_source),
                    |sources| {
                        let mut merger: LoserTreeMerger<
                            compactor::merge::ArenaSource,
                            InternalKeyComparator,
                        > = LoserTreeMerger::new(sources);
                        let mut count = 0usize;
                        merger.run(|_k, _v| count += 1);
                        count
                    },
                    criterion::BatchSize::LargeInput,
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("bytewise_comparator", num_sources),
            &per_source,
            |b, per_source| {
                b.iter_batched(
                    || to_arena_sources(per_source),
                    |sources| {
                        let mut merger: LoserTreeMerger<
                            compactor::merge::ArenaSource,
                            BytewiseComparator,
                        > = LoserTreeMerger::new(sources);
                        let mut count = 0usize;
                        merger.run(|_k, _v| count += 1);
                        count
                    },
                    criterion::BatchSize::LargeInput,
                )
            },
        );

        group.finish();
    }
}

/// Same isolation as `bench_comparator_prefix_fastpath`, but with 120 bytes
/// of constant padding inserted before the suffix (128-byte keys total) —
/// tests whether the fast path's win shows up once a full `compare` has
/// more bytes to scan, given it didn't on the 16-byte keys above.
fn bench_comparator_prefix_fastpath_long_keys(c: &mut Criterion) {
    let total_keys = 50_000u64;

    for &num_sources in &[4usize, 16] {
        let per_source = build_bytewise_entries_padded(num_sources, total_keys, 1016);
        let mut group = c.benchmark_group(format!("comparator_long_k{}", num_sources));

        group.bench_with_input(
            BenchmarkId::new("internal_key_comparator", num_sources),
            &per_source,
            |b, per_source| {
                b.iter_batched(
                    || to_arena_sources(per_source),
                    |sources| {
                        let mut merger: LoserTreeMerger<
                            compactor::merge::ArenaSource,
                            InternalKeyComparator,
                        > = LoserTreeMerger::new(sources);
                        let mut count = 0usize;
                        merger.run(|_k, _v| count += 1);
                        count
                    },
                    criterion::BatchSize::LargeInput,
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("bytewise_comparator", num_sources),
            &per_source,
            |b, per_source| {
                b.iter_batched(
                    || to_arena_sources(per_source),
                    |sources| {
                        let mut merger: LoserTreeMerger<
                            compactor::merge::ArenaSource,
                            BytewiseComparator,
                        > = LoserTreeMerger::new(sources);
                        let mut count = 0usize;
                        merger.run(|_k, _v| count += 1);
                        count
                    },
                    criterion::BatchSize::LargeInput,
                )
            },
        );

        group.finish();
    }
}

criterion_group!(
    benches,
    bench_mergers,
    bench_comparator_prefix_fastpath,
    bench_comparator_prefix_fastpath_long_keys
);
criterion_main!(benches);
