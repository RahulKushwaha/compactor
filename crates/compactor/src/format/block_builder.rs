//! Write-side mirror of the block entry/restart-point encoding read by
//! `block::for_each_entry` / `block::for_each_index_entry`. See
//! table/block_based/block_builder.cc for the reference implementation.

use crate::format::footer::BlockHandle;
use crate::format::varint::{put_varint32, put_varint64};

/// RocksDB's default `block_restart_interval` (BlockBasedTableOptions).
pub const DEFAULT_RESTART_INTERVAL: usize = 16;

/// Builds a single data block: shared-prefix delta-encoded entries plus the
/// trailing restart-point array. Produces output readable by
/// `block::for_each_entry`.
///
/// Only the plain `kDataBlockBinarySearch` layout is supported (no data
/// block hash index, no separated KV storage) — matches what our reader
/// assumes today (see block.rs: the restart-count trailer is read as a raw
/// u32, not the packed hash-index/separated-KV bit layout DataBlockFooter
/// uses in general; that packed layout collapses to a plain count when
/// neither feature is enabled, which is always true for blocks this builder
/// produces).
pub struct BlockBuilder {
    restart_interval: usize,
    buffer: Vec<u8>,
    restarts: Vec<u32>,
    last_key: Vec<u8>,
    counter: usize,
    finished: bool,
}

impl BlockBuilder {
    pub fn new(restart_interval: usize) -> Self {
        assert!(restart_interval >= 1);
        BlockBuilder {
            restart_interval,
            buffer: Vec::new(),
            restarts: vec![0],
            last_key: Vec::new(),
            counter: 0,
            finished: false,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Approximate encoded size so far, including the eventual restart-array
    /// trailer. Used by the caller to decide when to cut a new block.
    pub fn current_size_estimate(&self) -> usize {
        self.buffer.len() + self.restarts.len() * 4 + 4
    }

    /// REQUIRES: `key` is strictly greater than any previously added key
    /// (same requirement as BlockBuilder::Add in block_builder.h).
    pub fn add(&mut self, key: &[u8], value: &[u8]) {
        assert!(!self.finished);
        assert!(self.counter <= self.restart_interval);

        let shared = if self.counter >= self.restart_interval {
            0
        } else {
            shared_prefix_len(&self.last_key, key)
        };

        if self.counter >= self.restart_interval {
            self.restarts.push(self.buffer.len() as u32);
            self.counter = 0;
        }

        let non_shared = key.len() - shared;
        put_varint32(&mut self.buffer, shared as u32);
        put_varint32(&mut self.buffer, non_shared as u32);
        put_varint32(&mut self.buffer, value.len() as u32);
        self.buffer.extend_from_slice(&key[shared..]);
        self.buffer.extend_from_slice(value);

        self.last_key.clear();
        self.last_key.extend_from_slice(key);
        self.counter += 1;
    }

    /// Finalizes the block: appends the restart-point array and count.
    /// Returns the raw (uncompressed) block contents ready for
    /// compression + trailer + checksum by the caller.
    pub fn finish(mut self) -> Vec<u8> {
        for &r in &self.restarts {
            self.buffer.extend_from_slice(&r.to_le_bytes());
        }
        self.buffer
            .extend_from_slice(&(self.restarts.len() as u32).to_le_bytes());
        self.finished = true;
        self.buffer
    }
}

fn shared_prefix_len(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Builds a top-level (or partitioned-leaf) index block: 2-varint
/// (shared, non_shared) key header with no value_length field, followed by a
/// self-describing `BlockHandle` value — a full (offset, size) pair for the
/// first entry after each restart point, or a signed varint size-delta
/// against the previous handle otherwise. Mirrors
/// `block::for_each_index_entry`'s decode and table/format.cc
/// `IndexValue::EncodeTo` (the `previous_handle` non-null path).
pub struct IndexBlockBuilder {
    restart_interval: usize,
    buffer: Vec<u8>,
    restarts: Vec<u32>,
    last_key: Vec<u8>,
    last_handle: Option<BlockHandle>,
    counter: usize,
}

impl IndexBlockBuilder {
    pub fn new(restart_interval: usize) -> Self {
        assert!(restart_interval >= 1);
        IndexBlockBuilder {
            restart_interval,
            buffer: Vec::new(),
            restarts: vec![0],
            last_key: Vec::new(),
            last_handle: None,
            counter: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn current_size_estimate(&self) -> usize {
        self.buffer.len() + self.restarts.len() * 4 + 4
    }

    /// REQUIRES: `key` is strictly greater than any previously added key.
    pub fn add(&mut self, key: &[u8], handle: BlockHandle) {
        let at_restart = self.counter >= self.restart_interval;
        let shared = if at_restart {
            0
        } else {
            shared_prefix_len(&self.last_key, key)
        };

        if at_restart {
            self.restarts.push(self.buffer.len() as u32);
            self.counter = 0;
        }

        let non_shared = key.len() - shared;
        put_varint32(&mut self.buffer, shared as u32);
        put_varint32(&mut self.buffer, non_shared as u32);
        self.buffer.extend_from_slice(&key[shared..]);

        // First entry after a restart (shared == 0 there by construction)
        // always gets a full handle; later entries in the interval get a
        // delta against the previous handle's size.
        if at_restart || shared == 0 {
            put_varint64(&mut self.buffer, handle.offset);
            put_varint64(&mut self.buffer, handle.size);
        } else {
            let prev = self
                .last_handle
                .expect("delta encode with no previous handle");
            let delta = handle.size as i64 - prev.size as i64;
            put_varsigned64(&mut self.buffer, delta);
        }

        self.last_key.clear();
        self.last_key.extend_from_slice(key);
        self.last_handle = Some(handle);
        self.counter += 1;
    }

    pub fn finish(mut self) -> Vec<u8> {
        for &r in &self.restarts {
            self.buffer.extend_from_slice(&r.to_le_bytes());
        }
        self.buffer
            .extend_from_slice(&(self.restarts.len() as u32).to_le_bytes());
        self.buffer
    }
}

/// Zigzag varint64 encode, per util/coding.h PutVarsignedint64.
fn put_varsigned64(out: &mut Vec<u8>, v: i64) {
    let u = ((v << 1) ^ (v >> 63)) as u64;
    put_varint64(out, u);
}
