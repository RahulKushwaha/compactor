//! Ties reader (compactor-sst) + merge (compactor-merge) + writer
//! (compactor-format) into an actual compaction job: N input SSTs in,
//! 1 output SST out.
//!
//! What to keep vs drop is decided by a pluggable [`CompactionStrategy`] —
//! see that trait's doc comment for the strategies provided
//! (`BottommostStrategy`, `SnapshotAwareStrategy`, `PassthroughStrategy`)
//! and why one hardcoded policy wasn't enough (RocksDB itself branches on
//! exactly this: `bottommost_level_` / `earliest_snapshot_` in
//! db/compaction/compaction_iterator.h).
//!
//! Still-narrow scope, independent of strategy choice:
//! - Single output file (no target_file_size splitting yet).
//! - No range-deletion tombstone handling yet (RangeDeletion entries pass
//!   through as regular entries, which is wrong but out of scope for now;
//!   flagged so it isn't silently assumed correct).

use crate::format::internal_key;
use crate::format::sst_builder::{SstBuilderOptions, build_sst};
use crate::format::{block, footer};
use crate::io::FileReader;
use crate::merge::{ArenaSource, ArenaSourceBuilder, KWayMerge, LoserTreeMerger};

pub mod strategy;
pub use strategy::{
    BottommostStrategy, CompactionStrategy, KeepDecision, PassthroughStrategy,
    SnapshotAwareStrategy,
};

/// Reads every entry out of one SST into an `ArenaSource`, ready to feed a
/// merge. One allocation for the whole file's worth of keys+values (see
/// compactor-merge's ArenaSource doc comment for why this beats collecting
/// into `Vec<(Vec<u8>, Vec<u8>)>` first).
///
/// Issues exactly ONE `read_at` for the whole file, then decodes footer,
/// metaindex, top-level index, and every data block purely in memory (via
/// `crate::format`'s sync decoders) — no `SstReader` incremental walk here.
/// This matters a lot in practice: `SstReader::for_each_entry` issues one
/// `read_at` per data block, and each one round-trips through the blocking
/// I/O backend's `spawn_blocking` (a real thread hop + syscall per call on
/// platforms without io_uring). Measured against real `ldb compact` on a
/// 13-file/41MB workload, that per-block-call pattern made compactor ~5x
/// slower wall-clock than RocksDB's own compaction, almost entirely in
/// `sys` time (confirmed via `/usr/bin/time -l`: user time nearly
/// identical, sys time 5-10x higher) rather than CPU/algorithm cost.
/// Reading once and decoding in memory cut that to ~1.6-1.8x. Compaction
/// always wants the whole file's data eventually, so the whole-file read
/// has no real downside here (unlike `SstReader`, which stays incremental
/// for callers like the CLI dump tool that may want to stop early or avoid
/// holding a multi-GB file in memory at once).
pub async fn load_arena_source<R: FileReader>(reader: R) -> Result<ArenaSource, String> {
    let file_size = reader.file_size();
    let file_bytes = reader
        .read_at(0, file_size as usize)
        .await
        .map_err(|e| format!("failed to read whole file: {}", e))?;

    let tail = &file_bytes[file_bytes.len() - footer::FOOTER_TAIL_LEN..];
    let ft = footer::decode_footer(tail, file_size)?;

    let metaindex_raw = decode_block_in_memory(&file_bytes, &ft, ft.metaindex_handle)?;
    let metaindex = block::parse_block(metaindex_raw)?;

    let index_handle = match ft.index_handle {
        Some(h) => h,
        None => find_meta_handle(&metaindex, "rocksdb.index")?
            .ok_or_else(|| "index block handle missing from metaindex".to_string())?,
    };
    let index_raw = decode_block_in_memory(&file_bytes, &ft, index_handle)?;
    let index = block::parse_block(index_raw)?;

    let mut index_key_scratch = Vec::new();
    let mut data_handles = Vec::new();
    block::for_each_index_entry(&index, &mut index_key_scratch, |_key, handle| {
        data_handles.push(handle);
    })?;

    let mut builder = ArenaSourceBuilder::new();
    let mut data_key_scratch = Vec::new();
    for handle in data_handles {
        let raw = decode_block_in_memory(&file_bytes, &ft, handle)?;
        let parsed = block::parse_block(raw)?;
        block::for_each_entry(&parsed, &mut data_key_scratch, |k, v| {
            builder.push(k, v);
        })?;
    }
    Ok(builder.build())
}

fn decode_block_in_memory(
    file_bytes: &[u8],
    ft: &footer::Footer,
    handle: footer::BlockHandle,
) -> Result<Vec<u8>, String> {
    let start = handle.offset as usize;
    let len = block::block_read_len(handle);
    let raw_with_trailer = file_bytes
        .get(start..start + len)
        .ok_or_else(|| format!("block at offset {} out of file bounds", handle.offset))?
        .to_vec();
    block::decode_block_contents(
        raw_with_trailer,
        handle,
        ft.checksum_type,
        ft.base_context_checksum,
    )
}

fn find_meta_handle(
    metaindex: &block::ParsedBlock,
    name: &str,
) -> Result<Option<footer::BlockHandle>, String> {
    let mut scratch = Vec::new();
    let mut found = None;
    block::for_each_entry(metaindex, &mut scratch, |key, value| {
        if found.is_none() && key == name.as_bytes() {
            let mut pos = 0usize;
            if let (Some(offset), Some(size)) = (
                crate::format::varint::get_varint64(value, &mut pos),
                crate::format::varint::get_varint64(value, &mut pos),
            ) {
                found = Some(footer::BlockHandle { offset, size });
            }
        }
    })?;
    Ok(found)
}

#[derive(Clone)]
pub struct CompactionOptions {
    pub sst_builder: SstBuilderOptions,
}

/// Runs a full compaction using [`BottommostStrategy`] (drop all but the
/// newest version per user key, drop tombstones outright) — the default,
/// backward-compatible entry point for callers that don't need a different
/// policy. See [`compact_with_strategy`] to plug in
/// [`SnapshotAwareStrategy`], [`PassthroughStrategy`], or a custom
/// [`CompactionStrategy`] impl.
pub fn compact(sources: Vec<ArenaSource>, opts: &CompactionOptions) -> Result<Vec<u8>, String> {
    compact_with_strategy(sources, opts, &mut BottommostStrategy)
}

/// Runs a compaction: merges `sources` (each already sorted, per
/// `crate::merge::cmp::compare_internal_keys`) and returns a single output
/// SST's bytes, keeping or dropping each entry per `strategy`'s decision.
pub fn compact_with_strategy(
    sources: Vec<ArenaSource>,
    opts: &CompactionOptions,
    strategy: &mut dyn CompactionStrategy,
) -> Result<Vec<u8>, String> {
    let mut merger: LoserTreeMerger<ArenaSource> = LoserTreeMerger::new(sources);
    let mut output: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    // Reused scratch buffer for the previous entry's user key, instead of
    // `Option<Vec<u8>>` — that pattern allocates a fresh Vec on every single
    // merge-output entry just to remember one key for comparison. Confirmed
    // as the dominant per-entry allocation in this loop; fixing it is the
    // core of making this hot path allocation-free (values are already
    // borrowed straight from the merge sources' backing arenas — see
    // ArenaSource in crate::merge — right up until the final `output.push`
    // clone, which is unavoidable since `build_sst` needs owned entries to
    // outlive the merger).
    let mut last_user_key: Vec<u8> = Vec::new();
    let mut has_last_user_key = false;

    merger.run(|key, value| {
        let ik = match internal_key::split(key) {
            Ok(ik) => ik,
            Err(_) => return, // malformed entry; skip rather than propagate through a closure
        };

        // Merge output is sorted by (user_key asc, seq desc), so the first
        // entry seen for a given user_key is the newest version.
        let is_first_for_key = !has_last_user_key || last_user_key != ik.user_key;
        last_user_key.clear();
        last_user_key.extend_from_slice(ik.user_key);
        has_last_user_key = true;

        if strategy.decide(&ik, is_first_for_key) == KeepDecision::Drop {
            return;
        }

        output.push((key.to_vec(), value.to_vec()));
    });

    Ok(build_sst(&output, &opts.sst_builder))
}

/// Runs `num_shards` subcompactions concurrently, each producing its own
/// output SST. Boundary keys are chosen by evenly slicing the *largest*
/// input source's index range into `num_shards` pieces (a simplified stand-
/// in for RocksDB's approximate-size-based subcompaction split — good
/// enough when inputs are roughly similarly distributed; a real
/// implementation would estimate total output bytes per candidate boundary
/// instead of just index position). Every input source is then sliced at
/// the *same* boundary keys via `ArenaSource::lower_bound`, so no user key
/// straddles two shards regardless of which source(s) contain it.
///
/// Each shard's `compact()` call runs inside `tokio::task::spawn_blocking`
/// (see compactor-io's `io.rs` doc comment on the same bridge pattern for
/// why: this reuses the existing synchronous merge/write path rather than
/// rewriting it as `async fn`, while still getting real thread-pool
/// concurrency across shards — the thing actually asked for here). True
/// non-blocking-thread async merging would need an async-aware merge loop,
/// deferred as a further step past this.
///
/// Returns each shard's output SST bytes, in shard order (shard 0 covers
/// the lowest key range). A caller wanting one logical "compaction job"
/// output writes each as a separate output file, same as RocksDB's own
/// subcompaction model (N inputs -> M outputs when M subcompactions ran).
pub async fn compact_sharded(
    sources: Vec<ArenaSource>,
    num_shards: usize,
    opts: CompactionOptions,
) -> Result<Vec<Vec<u8>>, String> {
    if num_shards <= 1 || sources.is_empty() {
        return Ok(vec![compact(sources, &opts)?]);
    }

    let boundaries = pick_shard_boundaries(&sources, num_shards);

    // Build each shard's sliced sources up front (cheap: Arc clone + index
    // window per source, no bytes copied), then hand each shard's Vec to
    // its own spawn_blocking task.
    let mut shard_inputs: Vec<Vec<ArenaSource>> = (0..boundaries.len() + 1)
        .map(|_| Vec::with_capacity(sources.len()))
        .collect();
    for source in &sources {
        let mut start = 0usize;
        for (shard_idx, boundary) in boundaries.iter().enumerate() {
            let end = source.lower_bound(boundary);
            shard_inputs[shard_idx].push(source.shard(start, end));
            start = end;
        }
        let last_idx = shard_inputs.len() - 1;
        shard_inputs[last_idx].push(source.shard(start, source.total_len()));
    }

    let mut tasks = Vec::with_capacity(shard_inputs.len());
    for shard_sources in shard_inputs {
        let shard_opts = opts.clone();
        tasks.push(tokio::task::spawn_blocking(move || {
            compact(shard_sources, &shard_opts)
        }));
    }

    let mut outputs = Vec::with_capacity(tasks.len());
    for task in tasks {
        let result = task
            .await
            .map_err(|e| format!("subcompaction shard task panicked: {}", e))?;
        outputs.push(result?);
    }
    Ok(outputs)
}

/// Picks `num_shards - 1` boundary keys by evenly slicing the largest input
/// source's index range. See `compact_sharded` doc comment for why this is
/// a simplified stand-in for size-based splitting.
fn pick_shard_boundaries(sources: &[ArenaSource], num_shards: usize) -> Vec<Vec<u8>> {
    let largest = sources
        .iter()
        .max_by_key(|s| s.total_len())
        .expect("sources is non-empty (checked by caller)");
    let n = largest.total_len();
    if n == 0 {
        return Vec::new();
    }
    (1..num_shards)
        .map(|i| {
            let idx = (n * i) / num_shards;
            largest.key_at(idx.min(n - 1)).to_vec()
        })
        .collect()
}
