/// A single sorted input to a k-way merge: internal keys in strictly
/// increasing order per `cmp::compare_internal_keys`. Implementations own
/// their own iteration state (e.g. an index into an in-memory Vec, or an SST
/// data-block cursor); the merger only calls `peek`/`advance`.
///
/// Kept minimal and allocation-free on the hot path: `peek` borrows, it
/// doesn't clone.
pub trait MergeSource {
    /// Current front element, or `None` if exhausted.
    fn peek(&self) -> Option<(&[u8], &[u8])>;

    /// Discards the current front element, advancing to the next.
    /// REQUIRES: `peek()` was `Some` before this call.
    fn advance(&mut self);
}

/// Simple in-memory `MergeSource` backed by a pre-sorted `Vec`. Used for
/// tests and for the perf-comparison harness (isolates merge-algorithm cost
/// from block-decode cost).
#[derive(Clone)]
pub struct VecSource {
    entries: Vec<(Vec<u8>, Vec<u8>)>,
    pos: usize,
}

impl VecSource {
    /// REQUIRES: `entries` sorted per `cmp::compare_internal_keys`.
    pub fn new(entries: Vec<(Vec<u8>, Vec<u8>)>) -> Self {
        VecSource { entries, pos: 0 }
    }
}

impl MergeSource for VecSource {
    fn peek(&self) -> Option<(&[u8], &[u8])> {
        self.entries
            .get(self.pos)
            .map(|(k, v)| (k.as_slice(), v.as_slice()))
    }

    fn advance(&mut self) {
        self.pos += 1;
    }
}

/// One entry's location within an `ArenaSource`'s backing buffer.
#[derive(Clone, Copy)]
pub struct EntrySpan {
    key_off: u32,
    key_len: u32,
    val_off: u32,
    val_len: u32,
}

/// `MergeSource` backed by one contiguous byte arena plus a compact index
/// of `(offset, len)` spans, instead of a `Vec<(Vec<u8>, Vec<u8>)>`.
///
/// Motivation (the "offset-encoded" side of the perf comparison): a
/// `VecSource` with N entries holds 2*N separate heap allocations (one Vec
/// per key, one per value) scattered across the allocator's memory: bad
/// cache locality when scanning, and real allocation overhead when building
/// sources from decoded blocks. `ArenaSource` holds exactly one allocation
/// for all keys+values combined (built once via `ArenaSourceBuilder`), so
/// scanning through entries in order is a sequential memory scan, and the
/// index array (12 bytes/entry: 3x u32) is small enough to stay cache-hot
/// even for large sources.
///
/// u32 offsets/lengths cap a single source's arena at 4 GiB, which is fine
/// for a single SST's data (SST sizes are bounded well under that) — see
/// the >4GiB SST index issue already tracked elsewhere in this project for
/// why we don't casually assume "small" without checking.
pub struct ArenaSource {
    arena: std::sync::Arc<Vec<u8>>,
    spans: std::sync::Arc<Vec<EntrySpan>>,
    pos: usize,
    end: usize,
}

impl ArenaSource {
    fn key(&self, span: &EntrySpan) -> &[u8] {
        &self.arena[span.key_off as usize..(span.key_off + span.key_len) as usize]
    }

    fn val(&self, span: &EntrySpan) -> &[u8] {
        &self.arena[span.val_off as usize..(span.val_off + span.val_len) as usize]
    }

    /// Number of entries in this source (or shard).
    pub fn len(&self) -> usize {
        self.end - self.pos.min(self.end)
    }

    /// Splits off a zero-copy shard covering entry index range
    /// `[start, end)` (indices relative to the *original* unsharded
    /// source). Shares the same `Arc<arena>`/`Arc<spans>` — no bytes are
    /// copied; only a new `(pos, end)` window is created. Used to split one
    /// input SST's worth of decoded entries into concurrent subcompaction
    /// shards at a chosen key boundary.
    ///
    /// REQUIRES: `start <= end_idx <= self.spans.len()`. Boundaries must be
    /// chosen at user-key boundaries by the caller (never splitting the same
    /// user key across two shards), or a subcompaction could see only part
    /// of a key's versions — this type has no way to enforce that itself,
    /// since it doesn't know about user-key grouping.
    pub fn shard(&self, start: usize, end_idx: usize) -> ArenaSource {
        assert!(start <= end_idx && end_idx <= self.spans.len());
        ArenaSource {
            arena: self.arena.clone(),
            spans: self.spans.clone(),
            pos: start,
            end: end_idx,
        }
    }

    /// Total entry count in the original (unsharded) source.
    pub fn total_len(&self) -> usize {
        self.spans.len()
    }

    /// Borrow of entry `i`'s key (for shard-boundary key comparisons by
    /// callers, without going through `peek`/`advance`).
    pub fn key_at(&self, i: usize) -> &[u8] {
        self.key(&self.spans[i])
    }

    /// Returns the index of the first entry whose USER KEY is `>=` the user
    /// key of `boundary_key`, via binary search over the (already sorted)
    /// spans. Used by shard-boundary pickers to split this source at the
    /// same logical point as every other input source.
    ///
    /// Compares only the user-key portion (via `cmp::compare_user_keys`),
    /// not the full internal key: entries are sorted by (user_key asc, seq
    /// desc), so all versions of one user key are contiguous, but they are
    /// NOT identical internal keys. Comparing full internal keys here would
    /// let a boundary fall *between* two versions of the same user key
    /// (their seq/type suffixes differ), splitting that key's versions
    /// across two shards — each shard's independent compaction would then
    /// see its own slice as "the newest version" and keep it, duplicating
    /// the key across shard outputs. This was caught by
    /// compactor-compaction's sharded-vs-unsharded differential test.
    pub fn lower_bound(&self, boundary_key: &[u8]) -> usize {
        let spans = &self.spans[..];
        spans.partition_point(|span| {
            let k = self.key(span);
            crate::merge::cmp::compare_user_keys(k, boundary_key) == std::cmp::Ordering::Less
        })
    }
}

impl MergeSource for ArenaSource {
    fn peek(&self) -> Option<(&[u8], &[u8])> {
        if self.pos >= self.end {
            return None;
        }
        let span = self.spans.get(self.pos)?;
        Some((self.key(span), self.val(span)))
    }

    fn advance(&mut self) {
        self.pos += 1;
    }
}

/// Builds an `ArenaSource` by appending entries into a single growing
/// buffer, recording each entry's span. REQUIRES: entries appended in
/// increasing order per `cmp::compare_internal_keys` (not checked here,
/// same contract as `VecSource::new`).
pub struct ArenaSourceBuilder {
    arena: Vec<u8>,
    spans: Vec<EntrySpan>,
}

impl ArenaSourceBuilder {
    pub fn new() -> Self {
        ArenaSourceBuilder {
            arena: Vec::new(),
            spans: Vec::new(),
        }
    }

    pub fn with_capacity(entries_hint: usize, bytes_hint: usize) -> Self {
        ArenaSourceBuilder {
            arena: Vec::with_capacity(bytes_hint),
            spans: Vec::with_capacity(entries_hint),
        }
    }

    pub fn push(&mut self, key: &[u8], value: &[u8]) {
        let key_off = self.arena.len() as u32;
        self.arena.extend_from_slice(key);
        let key_len = key.len() as u32;

        let val_off = self.arena.len() as u32;
        self.arena.extend_from_slice(value);
        let val_len = value.len() as u32;

        self.spans.push(EntrySpan {
            key_off,
            key_len,
            val_off,
            val_len,
        });
    }

    pub fn build(self) -> ArenaSource {
        let end = self.spans.len();
        ArenaSource {
            arena: std::sync::Arc::new(self.arena),
            spans: std::sync::Arc::new(self.spans),
            pos: 0,
            end,
        }
    }
}

impl Default for ArenaSourceBuilder {
    fn default() -> Self {
        Self::new()
    }
}
