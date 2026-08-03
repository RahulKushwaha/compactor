//! Independent reference implementation of "compact N sorted sources of
//! internal-key/value pairs into one deduplicated, tombstone-dropped
//! dataset" — the same semantics `compactor::compaction::compact` aims for,
//! implemented via a completely different, deliberately naive path (no
//! merge iterator, no loser tree, just a `BTreeMap` keyed by user key).
//!
//! Deliberately has NO dependency on the `compactor` crate: if both
//! implementations agreed because they shared a bug in common code, this
//! oracle would be worthless as a correctness check. Its internal-key
//! decode is its own small duplicate of the format, not a reuse of
//! `compactor::format::internal_key`.

use std::collections::BTreeMap;

const SUFFIX_LEN: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    Deletion,
    Value,
    SingleDeletion,
    Other(u8),
}

fn decode_suffix(internal_key: &[u8]) -> (&[u8], u64, ValueType) {
    let at = internal_key.len() - SUFFIX_LEN;
    let user_key = &internal_key[..at];
    let packed = u64::from_le_bytes(internal_key[at..].try_into().unwrap());
    let seq = packed >> 8;
    let ty = match (packed & 0xff) as u8 {
        0x0 => ValueType::Deletion,
        0x1 => ValueType::Value,
        0x7 => ValueType::SingleDeletion,
        other => ValueType::Other(other),
    };
    (user_key, seq, ty)
}

/// Naive compaction: for each user key across all sources, keep only the
/// version with the highest sequence number; drop it too if that version is
/// a Deletion/SingleDeletion (bottommost-compaction semantics — same scope
/// restriction as `compactor::compaction::compact`, see that module's doc
/// comment for why: no snapshot pinning, valid only when nothing below
/// bottommost needs an older version).
///
/// Returns entries sorted by user key ascending, ready to compare directly
/// against a decoded compactor output (after stripping the internal-key
/// suffix on both sides, since the oracle doesn't reconstruct one).
pub fn naive_compact(sources: &[Vec<(Vec<u8>, Vec<u8>)>]) -> Vec<(Vec<u8>, Vec<u8>)> {
    // user_key -> (best_seq, value, is_tombstone)
    let mut best: BTreeMap<Vec<u8>, (u64, Vec<u8>, bool)> = BTreeMap::new();

    for source in sources {
        for (key, value) in source {
            let (user_key, seq, ty) = decode_suffix(key);
            let is_tombstone = matches!(ty, ValueType::Deletion | ValueType::SingleDeletion);
            match best.get(user_key) {
                Some((existing_seq, _, _)) if *existing_seq >= seq => {
                    // Existing entry is newer or equal; keep it. Equal
                    // sequence numbers shouldn't occur across independent
                    // versions in well-formed input, but ">=" makes the
                    // tie-break deterministic rather than input-order
                    // dependent either way.
                }
                _ => {
                    best.insert(user_key.to_vec(), (seq, value.clone(), is_tombstone));
                }
            }
        }
    }

    best.into_iter()
        .filter(|(_, (_, _, is_tombstone))| !is_tombstone)
        .map(|(user_key, (_, value, _))| (user_key, value))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(user_key: &str, seq: u64, ty: u8) -> Vec<u8> {
        let mut k = user_key.as_bytes().to_vec();
        let packed = (seq << 8) | ty as u64;
        k.extend_from_slice(&packed.to_le_bytes());
        k
    }

    #[test]
    fn keeps_newest_drops_tombstone_winner() {
        let sources = vec![
            vec![
                (key("a", 1, 1), b"old_a".to_vec()),
                (key("b", 1, 1), b"only_b".to_vec()),
            ],
            vec![
                (key("a", 5, 1), b"new_a".to_vec()),
                (key("b", 5, 0), b"".to_vec()),
            ],
        ];
        let out = naive_compact(&sources);
        assert_eq!(out, vec![(b"a".to_vec(), b"new_a".to_vec())]);
    }
}
