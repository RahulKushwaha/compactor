# compactor

[![e2e](https://github.com/RahulKushwaha/compactor/actions/workflows/e2e.yml/badge.svg)](https://github.com/RahulKushwaha/compactor/actions/workflows/e2e.yml)

A from-scratch Rust reimplementation of RocksDB's SST file format and offline
compaction. Reads and writes real RocksDB block-based SST files, merges N
sorted input SSTs, drops obsolete versions and tombstones, and writes a
single output SST — without linking RocksDB's C++.

## Why

Compaction is CPU/IO-bound background work that doesn't need to live inside
the storage engine's own process. This crate is a standalone tool that can
read the exact files a real RocksDB instance produces, compact them, and
hand back a file RocksDB will accept — as a step toward running compaction
as an external service. Correctness is checked against RocksDB itself, not
just against this codebase's own assumptions (see [Testing
strategy](#testing-strategy)).

## Layout

```
crates/
  compactor/   the real thing: format codec, merge engine, compaction driver, I/O, 2 binaries
  oracle/      independent naive reference implementation (BTreeMap-based), no dependency on compactor
  fuzzer/      shared random-input generation + differential tests (compactor vs oracle, compactor vs real RocksDB)
```

### `crates/compactor`

- `format/` — pure, allocation-conscious codec for the RocksDB block-based
  table format: footer, metaindex/properties/index blocks, data block
  entry encoding (shared-prefix delta encoding + restart points),
  checksums (CRC32C / XXHash / XXHash64 / XXH3), block compression
  (Snappy / LZ4 / LZ4HC / Zstd on read; uncompressed only on write so far),
  varint codec, internal-key layout.
- `merge/` — k-way merge over sorted sources. Two interchangeable algorithms
  (`HeapMerger`, a RocksDB-style binary heap; `LoserTreeMerger`, a tournament
  loser tree) behind one `KWayMerge` trait, generic over both the source
  representation (`VecSource` vs the single-allocation `ArenaSource`) and the
  key comparator (`KeyComparator`/`ByteComparable`, defaulting to
  `InternalKeyComparator`).
- `compaction/` — ties format + merge + I/O into an actual compaction job.
  Drop/keep policy is pluggable via `CompactionStrategy`
  (`BottommostStrategy`, `SnapshotAwareStrategy`, `PassthroughStrategy`).
  Supports sharded subcompactions (`compact_sharded`) run concurrently via
  `tokio::task::spawn_blocking`.
- `io/` — async positional file reads, one backend (`BlockingFileReader`;
  see its doc comment for why a `tokio-uring` backend was tried and
  abandoned as a structural, not just porting, mismatch).
- `sst/` — glues `io` + `format` together for incremental, on-demand SST
  reading (as opposed to `compaction`'s whole-file-at-once read path, which
  exists for a measured performance reason — see its module doc comment).
- Two binaries:
  - `compactor <path.sst> [--dump-all]` — inspect/dump a single SST file.
  - `compact_files [--for-ingest] <output.sst> <input1.sst> [input2.sst ...]`
    — standalone offline compaction: reads N input SSTs, compacts, writes
    one output SST. `--for-ingest` stamps the SstFileWriter-compatible
    properties needed for RocksDB's `ldb ingest_extern_sst` to accept it.

### `crates/oracle`

A second, deliberately naive implementation of the same compaction
semantics — dedupe by user key, drop tombstones — built on a plain
`BTreeMap`, with its own small duplicate of the internal-key decode logic.
No dependency on `compactor`, on purpose: if both implementations agreed
because they shared a bug, this oracle would prove nothing.

### `crates/fuzzer`

Shared randomized-input generation (`generate_sources`) plus two
differential test suites:
- `tests/differential.rs` — `compactor` vs the `oracle` crate, across random
  source/key/version shapes.
- `tests/rocksdb_oracle.rs` — `compactor` vs a **real RocksDB instance**
  (shells out to `ldb`/`sst_dump`): runs a live workload, compacts the
  resulting real SSTs with `compact_files`, and diffs against RocksDB's own
  ground truth. Also verifies the output can be ingested back into RocksDB
  and survives `ldb repair` + restart.

## Building and testing

```bash
cargo build --workspace
cargo test --workspace
cargo bench --bench merge_bench   # criterion; compares heap vs loser-tree, Vec vs Arena sources, comparator fast paths
```

Tests in `crates/fuzzer/tests/rocksdb_oracle.rs` shell out to real RocksDB
tooling (`ldb`, `sst_dump`) — expect these to be slower than the rest of the
suite. They resolve either binary naming in use (Homebrew's
`rocksdb_ldb`/`rocksdb_sst_dump` on macOS, or plain `ldb`/`sst_dump` from
Debian/Ubuntu's `rocksdb-tools` package) and skip with a message if neither
is on PATH.

## CI

`.github/workflows/e2e.yml` runs the full merge → ingest → verify workflow on
every push/PR: install `rocksdb-tools`, build, then run
`rocksdb_oracle`'s tests — real RocksDB workload, snapshot the
pre-compaction SSTs, merge them with `compact_files`, ingest the result into
a fresh RocksDB instance (and separately, drop it into an existing DB's
directory and run `ldb repair` to rebuild the manifest — the closest thing to
"reboot RocksDB" without a live server process), then diff what RocksDB
reads back against ground truth captured before compaction ran. The job
fails loudly (not a silent pass) if the RocksDB tools aren't actually
present, since that test skips gracefully by design for local dev machines
without them.

## Testing strategy

Three independent layers of correctness check, deliberately not sharing
implementation:
1. **Unit tests** per module (format codec round-trips, merge algorithm
   correctness including a 200-trial randomized differential test against a
   naive sort, comparator behavior).
2. **`compactor` vs `oracle`** — same semantics, structurally unrelated
   implementations, checked over randomized inputs.
3. **`compactor` vs real RocksDB** — the actual system this crate reads
   files for and is meant to eventually offload compaction from.

## Status / scope

Read side: block compression (Snappy/LZ4/LZ4HC/Zstd), all four checksum
types, format_version >= 6 metaindex-based index handle. Write side: single
output file only (no `target_file_size` splitting beyond explicit
subcompaction sharding), no compression yet (always writes uncompressed
blocks), no range-deletion tombstone handling yet (range deletions pass
through as regular entries — not yet correct).
