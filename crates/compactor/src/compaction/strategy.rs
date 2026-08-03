//! Pluggable compaction strategies: what to do with each *version* of a
//! user key as the merge walks entries in `(user_key asc, seq desc)` order.
//!
//! RocksDB itself does not have one hardcoded drop policy either —
//! `CompactionIterator` (db/compaction/compaction_iterator.h/.cc) branches
//! on `bottommost_level_` and `earliest_snapshot_` to decide whether an
//! obsolete version or a tombstone can be dropped. Those two axes are
//! exactly what the strategies below capture:
//!
//! - [`BottommostStrategy`]: no live snapshot below this level can see an
//!   older value, so drop every version but the newest, and drop the
//!   newest too if it's a tombstone. Correct only when this compaction's
//!   output sits at the true bottom of the LSM (or, equivalently, no
//!   snapshot predates any input file) — see `compactor::compaction`'s
//!   original narrow-scope doc comment for the assumption this used to
//!   make silently before the trait existed.
//! - [`SnapshotAwareStrategy`]: holds a live snapshot sequence number
//!   below which any version might still be read; the newest version
//!   *visible to that snapshot* must be kept even if newer versions above
//!   it get dropped, and a tombstone is only droppable if no snapshot
//!   needs to see past it. Mirrors RocksDB's real (non-bottommost)
//!   behavior.
//! - [`PassthroughStrategy`]: keeps every entry unchanged (no dedup, no
//!   tombstone drop) — useful for tests that want to inspect the raw
//!   merge output, or as a deliberately-conservative choice when the
//!   caller doesn't have enough information to make a drop decision
//!   safely (better to over-keep than to silently lose a version a
//!   snapshot needed).

use crate::format::internal_key::{InternalKey, ValueType};

/// What to do with one entry (one version of one user key) as the merge
/// walk visits it, in `(user_key asc, seq desc)` order — so for a given
/// user key, `is_first_for_key` entries arrive newest-seq-first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeepDecision {
    /// Write this entry to the output SST.
    Keep,
    /// Drop this entry; it will never be visible to any reader the output
    /// needs to serve.
    Drop,
}

/// Decides, per merge-output entry, whether to keep or drop it.
///
/// Called once per entry in merge order (which is also internal-key
/// order): all versions of one user key arrive consecutively, newest
/// sequence number first. Implementations that need to remember state
/// across a user key's versions (e.g. "have I already kept one visible
/// version?") do so via `&mut self` and must reset that state themselves
/// when `is_first_for_key` is true.
pub trait CompactionStrategy {
    /// `ik` is the current entry's decoded internal key; `is_first_for_key`
    /// is true exactly when this is the newest (first-seen) version of its
    /// user key in this merge.
    fn decide(&mut self, ik: &InternalKey<'_>, is_first_for_key: bool) -> KeepDecision;
}

/// Bottommost/full-compaction semantics: keep only the newest version of
/// each user key, and drop it too if that newest version is a tombstone.
/// See module doc comment for the correctness precondition (no live
/// snapshot needs an older value).
#[derive(Debug, Default, Clone, Copy)]
pub struct BottommostStrategy;

impl CompactionStrategy for BottommostStrategy {
    fn decide(&mut self, ik: &InternalKey<'_>, is_first_for_key: bool) -> KeepDecision {
        if !is_first_for_key {
            return KeepDecision::Drop;
        }
        if matches!(
            ik.value_type,
            ValueType::Deletion | ValueType::SingleDeletion
        ) {
            return KeepDecision::Drop;
        }
        KeepDecision::Keep
    }
}

/// Non-bottommost semantics: a live snapshot at `snapshot_seqno` might
/// still need to read a version visible as of that snapshot, so:
/// - The newest version *visible to the snapshot* (i.e. the first entry
///   for this user key with `sequence <= snapshot_seqno`, since versions
///   arrive newest-first) must be kept even though newer versions above
///   it are dropped.
/// - Versions newer than the snapshot are always droppable (nothing below
///   this compaction's output needs them — a snapshot can't see the
///   future) UNLESS they're the single newest version overall, mirroring
///   RocksDB's rule that the current/newest version is always kept
///   regardless of snapshots (a compaction never removes the very latest
///   write).
/// - A tombstone is kept if it might still be hiding an older value the
///   snapshot could otherwise see; it's only safely droppable once we've
///   established nothing below the snapshot boundary remains reachable
///   (approximated here as: drop a tombstone only if it's newer than the
///   snapshot AND not the newest version, matching "nothing can observe
///   it": readers newer than the snapshot see the newest live version
///   instead, and the snapshot reader will hit the *next* older version,
///   which this strategy keeps).
///
/// This mirrors the two RocksDB `CompactionIterator` fields that drive the
/// same decision (`earliest_snapshot_`, `bottommost_level_` — see
/// db/compaction/compaction_iterator.cc `NextFromInput`), simplified to a
/// single snapshot boundary rather than a full snapshot list (RocksDB
/// itself only needs the earliest one for this decision — later versions
/// above it are safe to collapse into whichever of {newest, at-or-below
/// snapshot} a reader would actually observe).
pub struct SnapshotAwareStrategy {
    snapshot_seqno: u64,
    /// Reset to false at the start of each user key (`is_first_for_key`);
    /// tracks whether we've already kept the "visible to snapshot" version
    /// for the current key, so later (older) versions of the same key are
    /// unconditionally dropped once that's satisfied.
    kept_visible_version_for_current_key: bool,
}

impl SnapshotAwareStrategy {
    pub fn new(snapshot_seqno: u64) -> Self {
        SnapshotAwareStrategy {
            snapshot_seqno,
            kept_visible_version_for_current_key: false,
        }
    }
}

impl CompactionStrategy for SnapshotAwareStrategy {
    fn decide(&mut self, ik: &InternalKey<'_>, is_first_for_key: bool) -> KeepDecision {
        if is_first_for_key {
            self.kept_visible_version_for_current_key = false;
        }

        let is_tombstone = matches!(
            ik.value_type,
            ValueType::Deletion | ValueType::SingleDeletion
        );

        if is_first_for_key {
            // The newest version is always kept, even if it's a tombstone
            // newer than the snapshot: a reader newer than the snapshot
            // must see it (a compaction never hides the current value from
            // current readers), and if it's at-or-below the snapshot
            // seqno, the snapshot reader needs it directly.
            if ik.sequence <= self.snapshot_seqno {
                self.kept_visible_version_for_current_key = true;
            }
            return KeepDecision::Keep;
        }

        if self.kept_visible_version_for_current_key {
            // Already have a version satisfying the snapshot; every
            // older version is unreachable by any reader this output
            // needs to serve.
            return KeepDecision::Drop;
        }

        if ik.sequence <= self.snapshot_seqno {
            // First version at-or-below the snapshot boundary: this is
            // what the snapshot reader observes. Keep it even if it's a
            // tombstone (the snapshot reader needs to see the delete, not
            // silently fall through to an older value).
            self.kept_visible_version_for_current_key = true;
            return KeepDecision::Keep;
        }

        // Newer than the snapshot but not the newest version for this key:
        // no reader this output serves can observe it (current readers see
        // the newest version already kept above; the snapshot reader
        // hasn't reached its boundary yet).
        let _ = is_tombstone; // already accounted for via the newest-version branch above
        KeepDecision::Drop
    }
}

/// Keeps every entry unchanged: no dedup, no tombstone drop. See module
/// doc comment for when this is the right (conservative) choice.
#[derive(Debug, Default, Clone, Copy)]
pub struct PassthroughStrategy;

impl CompactionStrategy for PassthroughStrategy {
    fn decide(&mut self, _ik: &InternalKey<'_>, _is_first_for_key: bool) -> KeepDecision {
        KeepDecision::Keep
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::internal_key;

    fn key(user_key: &str, seq: u64, ty: u8) -> Vec<u8> {
        let mut k = user_key.as_bytes().to_vec();
        let packed = (seq << 8) | ty as u64;
        k.extend_from_slice(&packed.to_le_bytes());
        k
    }

    const TYPE_VALUE: u8 = 1;
    const TYPE_DELETION: u8 = 0;

    #[test]
    fn bottommost_keeps_only_newest_and_drops_tombstone_winner() {
        let mut strat = BottommostStrategy;
        let k1 = key("a", 5, TYPE_VALUE);
        let k2 = key("a", 3, TYPE_VALUE);
        let ik1 = internal_key::split(&k1).unwrap();
        let ik2 = internal_key::split(&k2).unwrap();

        assert_eq!(strat.decide(&ik1, true), KeepDecision::Keep);
        assert_eq!(strat.decide(&ik2, false), KeepDecision::Drop);

        let mut strat2 = BottommostStrategy;
        let kd = key("b", 5, TYPE_DELETION);
        let ikd = internal_key::split(&kd).unwrap();
        assert_eq!(strat2.decide(&ikd, true), KeepDecision::Drop);
    }

    #[test]
    fn passthrough_keeps_everything() {
        let mut strat = PassthroughStrategy;
        let kd = key("a", 5, TYPE_DELETION);
        let ikd = internal_key::split(&kd).unwrap();
        assert_eq!(strat.decide(&ikd, true), KeepDecision::Keep);
        assert_eq!(strat.decide(&ikd, false), KeepDecision::Keep);
    }

    #[test]
    fn snapshot_aware_keeps_newest_and_the_version_visible_to_snapshot() {
        // Versions arrive newest-first: seq 10 (newest), 7, 4, 2 for key "a".
        // Snapshot at seqno 5 should see seq 4 (first version <= 5).
        let mut strat = SnapshotAwareStrategy::new(5);
        let versions = [
            (10u64, TYPE_VALUE),
            (7, TYPE_VALUE),
            (4, TYPE_VALUE),
            (2, TYPE_VALUE),
        ];
        let mut decisions = Vec::new();
        for (i, (seq, ty)) in versions.iter().enumerate() {
            let k = key("a", *seq, *ty);
            let ik = internal_key::split(&k).unwrap();
            decisions.push(strat.decide(&ik, i == 0));
        }
        assert_eq!(
            decisions,
            vec![
                KeepDecision::Keep, // seq 10: newest, always kept
                KeepDecision::Drop, // seq 7: newer than snapshot, not newest
                KeepDecision::Keep, // seq 4: first at-or-below snapshot boundary
                KeepDecision::Drop, // seq 2: snapshot already satisfied
            ]
        );
    }

    #[test]
    fn snapshot_aware_keeps_tombstone_visible_to_snapshot() {
        // Newest version is a live value (seq 10), but the snapshot at seqno
        // 5 should observe a delete at seq 4 — that delete must be kept so
        // the snapshot reader sees "not found", not the pre-delete value.
        let mut strat = SnapshotAwareStrategy::new(5);
        let k_new = key("a", 10, TYPE_VALUE);
        let k_del = key("a", 4, TYPE_DELETION);
        let k_old = key("a", 2, TYPE_VALUE);

        let ik_new = internal_key::split(&k_new).unwrap();
        let ik_del = internal_key::split(&k_del).unwrap();
        let ik_old = internal_key::split(&k_old).unwrap();

        assert_eq!(strat.decide(&ik_new, true), KeepDecision::Keep);
        assert_eq!(strat.decide(&ik_del, false), KeepDecision::Keep);
        assert_eq!(strat.decide(&ik_old, false), KeepDecision::Drop);
    }

    #[test]
    fn snapshot_aware_newest_version_below_snapshot_only_keeps_once() {
        // If the newest version is already at-or-below the snapshot, it
        // satisfies the snapshot itself; older versions are all dropped.
        let mut strat = SnapshotAwareStrategy::new(100);
        let k1 = key("a", 5, TYPE_VALUE);
        let k2 = key("a", 3, TYPE_VALUE);
        let ik1 = internal_key::split(&k1).unwrap();
        let ik2 = internal_key::split(&k2).unwrap();
        assert_eq!(strat.decide(&ik1, true), KeepDecision::Keep);
        assert_eq!(strat.decide(&ik2, false), KeepDecision::Drop);
    }
}
