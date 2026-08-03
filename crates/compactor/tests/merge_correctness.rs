use compactor::merge::cmp::compare_internal_keys;
use compactor::merge::{ArenaSourceBuilder, HeapMerger, KWayMerge, LoserTreeMerger, VecSource};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn to_arena(entries: &[(Vec<u8>, Vec<u8>)]) -> compactor::merge::ArenaSource {
    let mut builder = ArenaSourceBuilder::new();
    for (k, v) in entries {
        builder.push(k, v);
    }
    builder.build()
}

fn make_key(user_key: &str, seq: u64, ty: u8) -> Vec<u8> {
    let mut k = user_key.as_bytes().to_vec();
    let packed = (seq << 8) | ty as u64;
    k.extend_from_slice(&packed.to_le_bytes());
    k
}

fn collect<S: compactor::merge::MergeSource, M: KWayMerge<S>>(
    sources: Vec<S>,
) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut merger = M::new(sources);
    let mut out = Vec::new();
    merger.run(|k, v| out.push((k.to_vec(), v.to_vec())));
    out
}

#[test]
fn empty_sources() {
    let sources: Vec<VecSource> = vec![VecSource::new(vec![]), VecSource::new(vec![])];
    let out = collect::<_, HeapMerger<VecSource>>(sources.clone());
    assert!(out.is_empty());
    let out2 = collect::<_, LoserTreeMerger<VecSource>>(sources);
    assert!(out2.is_empty());
}

#[test]
fn single_source() {
    let entries = vec![
        (make_key("a", 1, 1), b"va".to_vec()),
        (make_key("b", 1, 1), b"vb".to_vec()),
        (make_key("c", 1, 1), b"vc".to_vec()),
    ];
    let sources = vec![VecSource::new(entries.clone())];
    let out = collect::<_, HeapMerger<VecSource>>(sources.clone());
    assert_eq!(out, entries);
    let out2 = collect::<_, LoserTreeMerger<VecSource>>(sources);
    assert_eq!(out2, entries);
}

#[test]
fn interleaved_disjoint_sources() {
    let s1 = vec![
        (make_key("a", 1, 1), b"1".to_vec()),
        (make_key("c", 1, 1), b"1".to_vec()),
        (make_key("e", 1, 1), b"1".to_vec()),
    ];
    let s2 = vec![
        (make_key("b", 1, 1), b"2".to_vec()),
        (make_key("d", 1, 1), b"2".to_vec()),
        (make_key("f", 1, 1), b"2".to_vec()),
    ];
    let expected_keys = vec!["a", "b", "c", "d", "e", "f"];

    for (name, out) in [
        (
            "heap",
            collect::<_, HeapMerger<VecSource>>(vec![
                VecSource::new(s1.clone()),
                VecSource::new(s2.clone()),
            ]),
        ),
        (
            "loser_tree",
            collect::<_, LoserTreeMerger<VecSource>>(vec![
                VecSource::new(s1.clone()),
                VecSource::new(s2.clone()),
            ]),
        ),
    ] {
        let got_keys: Vec<String> = out
            .iter()
            .map(|(k, _)| String::from_utf8(k[..k.len() - 8].to_vec()).unwrap())
            .collect();
        assert_eq!(got_keys, expected_keys, "mismatch for {}", name);
    }
}

#[test]
fn overlapping_versions_newest_seq_first_in_output() {
    // Same user key from two sources with different sequence numbers: the
    // merge output must yield BOTH versions (compaction dedup is a separate
    // concern, not the merger's job), newest seq first.
    let s1 = vec![(make_key("k", 5, 1), b"new".to_vec())];
    let s2 = vec![(make_key("k", 2, 1), b"old".to_vec())];

    let out = collect::<_, HeapMerger<VecSource>>(vec![
        VecSource::new(s1.clone()),
        VecSource::new(s2.clone()),
    ]);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].1, b"new");
    assert_eq!(out[1].1, b"old");

    let out2 =
        collect::<_, LoserTreeMerger<VecSource>>(vec![VecSource::new(s1), VecSource::new(s2)]);
    assert_eq!(out2.len(), 2);
    assert_eq!(out2[0].1, b"new");
    assert_eq!(out2[1].1, b"old");
}

/// Randomized differential test: both mergers must agree with each other
/// AND with a naive sort-everything oracle, across many random shapes
/// (source count, per-source length, key overlap). This is the real
/// correctness gate for the loser tree, which is easy to get subtly wrong
/// (e.g. the classic "phantom leaf" / non-power-of-two edge cases).
#[test]
fn randomized_differential_against_naive_sort() {
    let mut rng = StdRng::seed_from_u64(1234);

    for trial in 0..200 {
        let num_sources = rng.gen_range(1..=17); // include awkward non-power-of-two counts
        let mut all_entries: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        let mut per_source: Vec<Vec<(Vec<u8>, Vec<u8>)>> = vec![Vec::new(); num_sources];

        let num_keys = rng.gen_range(0..=60);
        for i in 0..num_keys {
            let user_key = format!("k{:04}", rng.gen_range(0..30)); // force overlap
            let seq = (num_keys - i) as u64 + 1;
            let key = make_key(&user_key, seq, 1);
            let value = format!("v{}_{}", trial, i).into_bytes();
            let src = rng.gen_range(0..num_sources);
            per_source[src].push((key.clone(), value.clone()));
            all_entries.push((key, value));
        }

        for src in &mut per_source {
            src.sort_by(|a, b| compare_internal_keys(&a.0, &b.0));
        }
        all_entries.sort_by(|a, b| compare_internal_keys(&a.0, &b.0));

        let heap_sources: Vec<VecSource> = per_source.iter().cloned().map(VecSource::new).collect();
        let loser_sources: Vec<VecSource> =
            per_source.iter().cloned().map(VecSource::new).collect();
        let arena_loser_sources: Vec<_> = per_source.iter().map(|e| to_arena(e)).collect();

        let heap_out = collect::<_, HeapMerger<VecSource>>(heap_sources);
        let loser_out = collect::<_, LoserTreeMerger<VecSource>>(loser_sources);
        let arena_out =
            collect::<_, LoserTreeMerger<compactor::merge::ArenaSource>>(arena_loser_sources);

        assert_eq!(
            heap_out, all_entries,
            "trial {}: heap merger diverged from naive sort oracle",
            trial
        );
        assert_eq!(
            loser_out, all_entries,
            "trial {}: loser tree merger (VecSource) diverged from naive sort oracle",
            trial
        );
        assert_eq!(
            arena_out, all_entries,
            "trial {}: loser tree merger (ArenaSource) diverged from naive sort oracle",
            trial
        );
    }
}

#[test]
fn bytewise_and_internal_key_comparator_agree_on_shared_suffix_dataset() {
    use compactor::merge::{ArenaSource, BytewiseComparator, InternalKeyComparator};

    const CONSTANT_SUFFIX: [u8; 8] = 1u64.to_le_bytes();
    let num_sources = 4usize;
    let total_keys = 400u64;
    let per_src = total_keys / num_sources as u64;
    let per_source: Vec<Vec<(Vec<u8>, Vec<u8>)>> = (0..num_sources)
        .map(|src| {
            (0..per_src)
                .map(|i| {
                    let user_key = src as u64 + i * num_sources as u64;
                    let mut key = user_key.to_be_bytes().to_vec();
                    key.extend_from_slice(&CONSTANT_SUFFIX);
                    let value = format!("v{}", user_key).into_bytes();
                    (key, value)
                })
                .collect()
        })
        .collect();

    let to_arena = |entries: &[(Vec<u8>, Vec<u8>)]| -> ArenaSource {
        let mut b = ArenaSourceBuilder::new();
        for (k, v) in entries {
            b.push(k, v);
        }
        b.build()
    };

    let sources_a: Vec<ArenaSource> = per_source.iter().map(|e| to_arena(e)).collect();
    let sources_b: Vec<ArenaSource> = per_source.iter().map(|e| to_arena(e)).collect();

    let out_internal = collect::<_, LoserTreeMerger<ArenaSource, InternalKeyComparator>>(sources_a);
    let out_bytewise = collect::<_, LoserTreeMerger<ArenaSource, BytewiseComparator>>(sources_b);

    assert_eq!(
        out_internal, out_bytewise,
        "InternalKeyComparator and BytewiseComparator must agree on this shared-suffix dataset"
    );
}
