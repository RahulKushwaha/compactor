//! K-way merge over `MergeSource`s, in two swappable algorithm flavors so
//! they can be benchmarked head-to-head:
//! - `HeapMerger`: binary min-heap over source indices, same structure as
//!   RocksDB's real `MergingIterator` (table/merging_iterator.cc: a
//!   `BinaryHeap` keyed by the internal-key comparator, pop+push per step —
//!   see minHeap_ usage there).
//! - `LoserTreeMerger`: classic tournament loser tree (Knuth, TAOCP Vol 3,
//!   §5.4.1). Advancing only replays comparisons along one fixed
//!   leaf-to-root path (O(log k) comparisons, no data movement beyond
//!   updating nodes on that path in place — no heap-style swaps, no
//!   re-copying the winner's key into a heap entry every step).
//!
//! Both are generic over `MergeSource`, so the same benchmark can vary the
//! key/value *representation* (plain `VecSource` vs an offset-encoded arena
//! source) independently of the *algorithm*.
//!
//! Both are also generic over `KeyComparator` (`C`), defaulted to
//! `InternalKeyComparator` so existing callers (`HeapMerger<S>`,
//! `LoserTreeMerger<S>`) keep compiling unchanged. When `C` additionally
//! implements `ByteComparable` (`HAS_SORT_PREFIX = true`), every hot
//! comparison first checks a cheap integer prefix computed straight off the
//! key bytes — no allocation, no stored/encoded copy, just a value derived
//! on the fly for that one comparison — before falling back to `C::compare`
//! on a tie. This is the offset-value fast path: the prefix isn't persisted
//! anywhere, only computed transiently inside the priority-queue/loser-tree
//! comparisons that already have to look at the key.

use crate::merge::cmp::{InternalKeyComparator, KeyComparator};
use crate::merge::source::MergeSource;
use std::cmp::Ordering;
use std::marker::PhantomData;

/// Drives a k-way merge to completion, calling `f` once per output entry in
/// sorted order. Callback-based (no output Vec) to keep the merge itself
/// allocation-free; callers that want a Vec can push inside the closure.
pub trait KWayMerge<S: MergeSource> {
    fn new(sources: Vec<S>) -> Self;
    fn run(&mut self, f: impl FnMut(&[u8], &[u8]));
}

/// Compares two keys under comparator `C`, using its `sort_prefix` fast path
/// when available: distinct prefixes settle the comparison as a plain `u64`
/// compare; equal prefixes (or `HAS_SORT_PREFIX == false`) fall back to
/// `C::compare`. `C::HAS_SORT_PREFIX` is a `const`, so for a comparator that
/// leaves it `false` this monomorphizes down to a direct call to
/// `C::compare` with no prefix computation at all.
#[inline]
fn compare_with_prefix<C: KeyComparator>(a: &[u8], b: &[u8]) -> Ordering {
    if C::HAS_SORT_PREFIX {
        let by_prefix = C::sort_prefix(a).cmp(&C::sort_prefix(b));
        if by_prefix != Ordering::Equal {
            return by_prefix;
        }
    }
    C::compare(a, b)
}

// ---------------------------------------------------------------------
// Heap-based merger (RocksDB-style baseline)
// ---------------------------------------------------------------------

pub struct HeapMerger<S: MergeSource, C: KeyComparator = InternalKeyComparator> {
    sources: Vec<S>,
    /// Min-heap of source indices, ordered by each source's current front
    /// key. `std::collections::BinaryHeap` is a max-heap, so `HeapEntry`'s
    /// `Ord` impl reverses the comparison (standard "wrap for min-heap"
    /// trick) rather than wrapping in `std::cmp::Reverse`, since we also
    /// need `source_idx` carried alongside the key.
    heap: std::collections::BinaryHeap<HeapEntry<C>>,
}

struct HeapEntry<C: KeyComparator> {
    source_idx: usize,
    // Key bytes copied out for the duration this entry sits in the heap.
    // Unavoidable with a plain BinaryHeap: it needs Ord on data it owns
    // independent of `sources`, and we can't borrow `sources[i]` while a
    // `&mut sources` reference is also live for advancing other entries.
    // RocksDB's C++ heap sidesteps this by comparing through iterator
    // pointers it owns outside the heap; this per-step key copy is exactly
    // the cost the loser tree below is designed to avoid.
    key: Vec<u8>,
    _comparator: PhantomData<C>,
}

impl<C: KeyComparator> PartialEq for HeapEntry<C> {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}
impl<C: KeyComparator> Eq for HeapEntry<C> {}
impl<C: KeyComparator> PartialOrd for HeapEntry<C> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<C: KeyComparator> Ord for HeapEntry<C> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed: BinaryHeap is a max-heap, we want the smallest key on top.
        compare_with_prefix::<C>(&other.key, &self.key)
    }
}

impl<S: MergeSource, C: KeyComparator> KWayMerge<S> for HeapMerger<S, C> {
    fn new(sources: Vec<S>) -> Self {
        let mut heap = std::collections::BinaryHeap::with_capacity(sources.len());
        for (idx, s) in sources.iter().enumerate() {
            if let Some((k, _)) = s.peek() {
                heap.push(HeapEntry {
                    source_idx: idx,
                    key: k.to_vec(),
                    _comparator: PhantomData,
                });
            }
        }
        HeapMerger { sources, heap }
    }

    fn run(&mut self, mut f: impl FnMut(&[u8], &[u8])) {
        while let Some(HeapEntry { source_idx, .. }) = self.heap.pop() {
            let (k, v) = self.sources[source_idx].peek().expect("heap entry stale");
            f(k, v);
            self.sources[source_idx].advance();
            if let Some((k2, _)) = self.sources[source_idx].peek() {
                self.heap.push(HeapEntry {
                    source_idx,
                    key: k2.to_vec(),
                    _comparator: PhantomData,
                });
            }
        }
    }
}

// ---------------------------------------------------------------------
// Loser-tree merger
// ---------------------------------------------------------------------

/// Tournament loser tree over `k_padded` leaves (real sources plus phantom
/// "always empty" leaves padding up to a power of two, which keeps the
/// tree a perfect complete binary tree and makes `parent(x) = x/2` exact —
/// avoids the well-known fiddliness of building a loser tree over a
/// non-power-of-two leaf count).
///
/// Layout: leaves occupy positions `k_padded..2*k_padded-1`; leaf position
/// `k_padded + i` corresponds to source `i` for `i < sources.len()`, and to
/// a phantom (always-losing) leaf otherwise. Internal nodes occupy
/// `1..k_padded-1`; each stores the *loser* index of its subtree's final
/// match. The overall winner is tracked separately in `winner`.
pub struct LoserTreeMerger<S: MergeSource, C: KeyComparator = InternalKeyComparator> {
    sources: Vec<S>,
    tree: Vec<usize>,
    winner: usize,
    k_padded: usize,
    _comparator: PhantomData<C>,
}

const NONE_YET: usize = usize::MAX;

impl<S: MergeSource, C: KeyComparator> LoserTreeMerger<S, C> {
    /// A leaf index is "real" if it names an actual source; indices at or
    /// beyond `sources.len()` are phantom padding and always lose (treated
    /// as permanently exhausted).
    fn peek_at(&self, leaf: usize) -> Option<(&[u8], &[u8])> {
        if leaf >= self.sources.len() {
            None
        } else {
            self.sources[leaf].peek()
        }
    }

    fn less(&self, a: usize, b: usize) -> bool {
        match (self.peek_at(a), self.peek_at(b)) {
            (Some((ka, _)), Some((kb, _))) => compare_with_prefix::<C>(ka, kb) == Ordering::Less,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => false,
        }
    }

    /// Recursively computes the winner of the subtree rooted at `node`,
    /// recording the loser of the final match at `tree[node]` for every
    /// internal node visited. Standard bottom-up tournament-tree build
    /// (equivalent to building a min-segment-tree): O(k_padded) total work,
    /// each internal node visited exactly once.
    fn build_subtree(&mut self, node: usize) -> usize {
        if node >= self.k_padded {
            return node - self.k_padded; // leaf: source index (may be phantom)
        }
        let left = self.build_subtree(2 * node);
        let right = self.build_subtree(2 * node + 1);
        let (winner, loser) = if self.less(left, right) {
            (left, right)
        } else {
            (right, left)
        };
        self.tree[node] = loser;
        winner
    }

    fn build(&mut self) {
        self.winner = self.build_subtree(1);
    }

    /// After `leaf`'s source has been advanced, replays comparisons along
    /// `leaf`'s path to the root against the recorded losers, updating the
    /// path and the overall winner. Correct only when called with `leaf`
    /// equal to the *previous* winner (the only leaf whose value just
    /// changed) — see `run()`.
    fn replay_from_leaf(&mut self, leaf: usize) {
        let mut node = (leaf + self.k_padded) / 2;
        let mut contender = leaf;
        while node >= 1 {
            let loser_here = self.tree[node];
            if !self.less(contender, loser_here) {
                self.tree[node] = contender;
                contender = loser_here;
            }
            // else: contender still wins; tree[node] (the other subtree's
            // result) is unaffected and stays as-is.
            if node == 1 {
                self.winner = contender;
                return;
            }
            node /= 2;
        }
    }
}

impl<S: MergeSource, C: KeyComparator> KWayMerge<S> for LoserTreeMerger<S, C> {
    fn new(sources: Vec<S>) -> Self {
        let k_padded = sources.len().max(1).next_power_of_two();
        let mut merger = LoserTreeMerger {
            sources,
            tree: vec![NONE_YET; k_padded],
            winner: 0,
            k_padded,
            _comparator: PhantomData,
        };
        merger.build();
        merger
    }

    fn run(&mut self, mut f: impl FnMut(&[u8], &[u8])) {
        loop {
            if self.winner >= self.sources.len() {
                break; // winner is a phantom leaf: all real sources exhausted
            }
            let (k, v) = match self.sources[self.winner].peek() {
                Some(kv) => kv,
                None => break,
            };
            f(k, v);
            let winner = self.winner;
            self.sources[winner].advance();
            self.replay_from_leaf(winner);
        }
    }
}
