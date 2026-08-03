//! Comparators, abstracted as a trait so merge algorithms (see
//! `crate::merge::merger`) can be generic over key ordering instead of
//! hardcoding internal-key comparison.
//!
//! `InternalKeyComparator` mirrors db/dbformat.h InternalKeyComparator::Compare:
//! order by (1) increasing user key (bytewise), (2) decreasing sequence
//! number, (3) decreasing type. The trailing 8 bytes are a little-endian
//! packed `(seq << 8) | type` (see internal_key.rs in compactor-format) —
//! comparing them numerically (not as raw memcmp bytes) is required, since
//! LE byte order does not match numeric order. That numeric reinterpretation
//! is exactly why `InternalKeyComparator` is NOT byte-comparable (see
//! `KeyComparator::sort_prefix`).

use std::cmp::Ordering;

const SUFFIX_LEN: usize = 8;

/// A key comparator, decoupled from any specific key encoding so merge
/// algorithms (`HeapMerger`, `LoserTreeMerger`) can be generic over it.
pub trait KeyComparator {
    /// Total order over full key bytes (whatever "key" means for this
    /// comparator: a full internal key, a bare user key, etc.).
    fn compare(a: &[u8], b: &[u8]) -> Ordering;

    /// Whether `sort_prefix` is a safe fast path for this comparator. A
    /// `const`, not a runtime flag: callers branch on it
    /// (`if C::HAS_SORT_PREFIX { ... }`) inside a function generic over
    /// `C`, so after monomorphization the branch is decided at compile
    /// time — for a comparator that leaves this `false`, the whole
    /// prefix-computation path is dead code, not a runtime no-op check.
    /// (An earlier version used `sort_prefix() -> Option<u64>` as the
    /// opt-in signal instead of this const; benchmarking showed that
    /// version's `Option` construction/matching on every comparison added
    /// measurable overhead even for comparators that always returned
    /// `Some`, and added a real branch for comparators that always
    /// returned `None` — const dispatch avoids both.)
    const HAS_SORT_PREFIX: bool = false;

    /// Cheap fast-path prefix for accelerating hot comparison loops (heap
    /// push/pop, loser-tree replay): a numeric value computed from the
    /// leading bytes of `key`, cheap to hold in a register and compare with
    /// plain integer `<`/`>` instead of a byte-slice compare.
    ///
    /// Only called when `HAS_SORT_PREFIX` is `true`. Contract: for any two
    /// keys `a`, `b`, if `sort_prefix(a) != sort_prefix(b)`, then
    /// `sort_prefix(a).cmp(&sort_prefix(b)) == compare(a, b)`. Equal
    /// prefixes still require the full `compare` call (the prefix only
    /// covers the leading bytes, so it can't distinguish keys that agree on
    /// those bytes but differ later).
    ///
    /// Only safe to implement (with `HAS_SORT_PREFIX = true`) when
    /// `compare`'s order agrees with plain byte-wise memcmp over full key
    /// bytes — i.e. for `ByteComparable` comparators.
    fn sort_prefix(_key: &[u8]) -> u64 {
        0
    }
}

/// Marker for comparators whose `compare` is equivalent to plain byte-wise
/// memcmp over the full key. Implementing this (and overriding
/// `sort_prefix` to return `Some`) is what "unlocks" the offset-value
/// comparison fast path in the mergers: a numeric prefix taken straight from
/// the leading key bytes sorts identically to the bytes themselves, with no
/// reinterpretation, so comparing it as a plain integer is safe.
///
/// `InternalKeyComparator` deliberately does not implement this: its suffix
/// is a little-endian packed (seq, type) compared numerically, which does
/// not agree with memcmp over the raw suffix bytes.
pub trait ByteComparable: KeyComparator {}

/// Loads up to the first 8 bytes of `key` into a big-endian `u64`, zero
/// padded if shorter. Big-endian is required so integer comparison order
/// matches byte order: `u64::from_be_bytes` puts the first byte in the most
/// significant position, exactly where memcmp gives it the most weight.
pub fn load_be_prefix(key: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    let n = key.len().min(8);
    buf[..n].copy_from_slice(&key[..n]);
    u64::from_be_bytes(buf)
}

/// Internal-key comparator: see module doc comment. NOT `ByteComparable`.
pub struct InternalKeyComparator;

impl KeyComparator for InternalKeyComparator {
    fn compare(a: &[u8], b: &[u8]) -> Ordering {
        let (a_user, a_suffix) = split(a);
        let (b_user, b_suffix) = split(b);
        match a_user.cmp(b_user) {
            Ordering::Equal => {
                let a_packed = u64::from_le_bytes(a_suffix.try_into().unwrap());
                let b_packed = u64::from_le_bytes(b_suffix.try_into().unwrap());
                // Higher packed value (newer seq, or same seq/higher type) sorts first.
                b_packed.cmp(&a_packed)
            }
            other => other,
        }
    }
    // No sort_prefix override: the user-key prefix alone would be a safe
    // fast path if we only ever needed to disambiguate distinct user keys,
    // but two keys sharing a user-key prefix within 8 bytes still need the
    // full compare to break the tie on seq/type, and a plain prefix load
    // can't tell "genuinely equal prefix, need full compare" apart from
    // "these are the same user key" without extra bookkeeping. Left as the
    // safe default (None) rather than a subtly-wrong fast path.
}

/// Plain byte-wise comparator (RocksDB's `leveldb.BytewiseComparator`).
/// `ByteComparable` by construction: `compare` *is* memcmp.
pub struct BytewiseComparator;

impl KeyComparator for BytewiseComparator {
    fn compare(a: &[u8], b: &[u8]) -> Ordering {
        a.cmp(b)
    }

    // HAS_SORT_PREFIX stays at the trait default (false) here: measured on
    // the dev desktop (aarch64, 3 key lengths — 16B, 128B, 1KB — via
    // crates/compactor/benches/merge_bench.rs's
    // bench_comparator_prefix_fastpath{,_long_keys}), enabling sort_prefix
    // for this comparator was consistently ~17-25% SLOWER than plain
    // `compare`, not faster, at every length tried. Root cause: `a.cmp(b)`
    // on `&[u8]` already lowers to a vectorized memcmp-equivalent that
    // early-exits at the first differing byte; extracting an 8-byte prefix
    // by hand (zero-init a stack buffer, copy, reinterpret as u64) costs
    // more than the comparison it's meant to short-circuit, for a
    // comparator whose `compare` is *already* just that memcmp. The
    // fast path (`sort_prefix`/`load_be_prefix` below) is left implemented
    // and tested as the documented extension point for a future
    // `ByteComparable` comparator whose `compare` does real per-byte work
    // beyond memcmp (e.g. decode-then-compare) — there, skipping most of
    // that work via a cheap prefix could plausibly win. That hasn't been
    // measured yet; don't assume it without benchmarking that case too.
    fn sort_prefix(key: &[u8]) -> u64 {
        load_be_prefix(key)
    }
}

impl ByteComparable for BytewiseComparator {}

fn split(key: &[u8]) -> (&[u8], &[u8]) {
    let at = key.len() - SUFFIX_LEN;
    (&key[..at], &key[at..])
}

/// Extracts just the user-key portion (strips the trailing 8-byte packed
/// seq/type suffix).
pub fn user_key(internal_key: &[u8]) -> &[u8] {
    split(internal_key).0
}

/// Compares two full internal keys. Thin wrapper over
/// `InternalKeyComparator::compare` kept for callers that don't need to be
/// generic over comparator choice.
pub fn compare_internal_keys(a: &[u8], b: &[u8]) -> Ordering {
    InternalKeyComparator::compare(a, b)
}

/// Compares only the user-key portion of two internal keys, ignoring
/// seq/type entirely. Used for shard-boundary decisions, where a boundary
/// must never fall between two versions of the same user key (see
/// compactor-merge::source::ArenaSource::lower_bound and
/// compactor-compaction's pick_shard_boundaries).
pub fn compare_user_keys(a: &[u8], b: &[u8]) -> Ordering {
    user_key(a).cmp(user_key(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_key(user_key: &[u8], seq: u64, ty: u8) -> Vec<u8> {
        let mut k = user_key.to_vec();
        let packed = (seq << 8) | ty as u64;
        k.extend_from_slice(&packed.to_le_bytes());
        k
    }

    #[test]
    fn orders_by_user_key_first() {
        let a = make_key(b"aaa", 5, 1);
        let b = make_key(b"bbb", 1, 1);
        assert_eq!(compare_internal_keys(&a, &b), Ordering::Less);
    }

    #[test]
    fn same_user_key_higher_seq_first() {
        let newer = make_key(b"key", 10, 1);
        let older = make_key(b"key", 5, 1);
        assert_eq!(compare_internal_keys(&newer, &older), Ordering::Less);
        assert_eq!(compare_internal_keys(&older, &newer), Ordering::Greater);
    }

    #[test]
    fn same_user_key_same_seq_higher_type_first() {
        let a = make_key(b"key", 5, 7);
        let b = make_key(b"key", 5, 1);
        assert_eq!(compare_internal_keys(&a, &b), Ordering::Less);
    }

    #[test]
    fn bytewise_comparator_matches_slice_cmp() {
        let a = b"abc";
        let b = b"abd";
        assert_eq!(BytewiseComparator::compare(a, b), a.cmp(b));
    }

    #[test]
    fn bytewise_sort_prefix_agrees_with_compare_on_differing_prefixes() {
        let a = b"aaaaaaaaX";
        let b = b"bbbbbbbbX";
        let pa = BytewiseComparator::sort_prefix(a);
        let pb = BytewiseComparator::sort_prefix(b);
        assert_ne!(pa, pb);
        assert_eq!(pa.cmp(&pb), BytewiseComparator::compare(a, b));
    }

    #[test]
    fn bytewise_sort_prefix_ties_on_shared_prefix_need_full_compare() {
        // Both keys share the same leading 8 bytes; sort_prefix alone can't
        // distinguish them, as documented on the trait.
        let a = b"aaaaaaaaX";
        let b = b"aaaaaaaaY";
        assert_eq!(
            BytewiseComparator::sort_prefix(a),
            BytewiseComparator::sort_prefix(b)
        );
        assert_eq!(BytewiseComparator::compare(a, b), Ordering::Less);
    }

    #[test]
    fn internal_key_comparator_has_no_fast_path() {
        assert!(!InternalKeyComparator::HAS_SORT_PREFIX);
    }
}
