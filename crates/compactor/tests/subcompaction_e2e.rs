use compactor::compaction::{CompactionOptions, compact, compact_sharded};
use compactor::format::sst_builder::SstBuilderOptions;
use compactor::merge::ArenaSourceBuilder;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn internal_key(user_key: &str, seq: u64, value_type: u8) -> Vec<u8> {
    let mut k = user_key.as_bytes().to_vec();
    let packed = (seq << 8) | value_type as u64;
    k.extend_from_slice(&packed.to_le_bytes());
    k
}

const TYPE_VALUE: u8 = 1;
const TYPE_DELETION: u8 = 0;

/// Generates the per-source entry lists once per seed; each test iteration
/// rebuilds fresh `ArenaSource`s from these plain entries (cheap, and
/// avoids needing a Clone impl on ArenaSource that no real caller needs).
fn build_test_entries(
    seed: u64,
    num_sources: usize,
    num_keys: usize,
) -> Vec<Vec<(Vec<u8>, Vec<u8>)>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut per_source: Vec<Vec<(Vec<u8>, Vec<u8>)>> = vec![Vec::new(); num_sources];

    for i in 0..num_keys {
        let user_key = format!("key{:06}", i);
        let versions = rng.gen_range(1..=3);
        for v in 0..versions {
            let seq = (versions - v) as u64;
            let ty = if v == 0 && rng.gen_bool(0.1) {
                TYPE_DELETION
            } else {
                TYPE_VALUE
            };
            let src = rng.gen_range(0..num_sources);
            per_source[src].push((
                internal_key(&user_key, seq, ty),
                format!("value_{}_{}", i, seq).into_bytes(),
            ));
        }
    }

    for entries in &mut per_source {
        entries.sort_by(|a, b| compactor::merge::cmp::compare_internal_keys(&a.0, &b.0));
    }
    per_source
}

fn to_arena_sources(per_source: &[Vec<(Vec<u8>, Vec<u8>)>]) -> Vec<compactor::merge::ArenaSource> {
    per_source
        .iter()
        .map(|entries| {
            let mut builder = ArenaSourceBuilder::new();
            for (k, v) in entries {
                builder.push(k, v);
            }
            builder.build()
        })
        .collect()
}

/// The core subcompaction correctness invariant: splitting one compaction
/// job into N independent shards must produce EXACTLY the same logical
/// dataset (same keys, same values, same drops) as running it unsharded.
/// This is the test flagged as most important when subcompactions were
/// first scoped — get shard boundary placement wrong (e.g. split a user
/// key across two shards) and this is exactly what would catch it.
#[test]
fn sharded_compaction_matches_unsharded_for_various_shard_counts() {
    let rt = tokio::runtime::Runtime::new().unwrap();

    for seed in 0..10u64 {
        let per_source = build_test_entries(seed, 5, 400);
        let opts = CompactionOptions {
            sst_builder: SstBuilderOptions {
                base_context_checksum: 0xABCDEF,
                ..Default::default()
            },
        };

        let unsharded = compact(to_arena_sources(&per_source), &opts)
            .expect("unsharded compaction should succeed");
        let unsharded_entries = decode_all_entries(&[unsharded]);

        for &num_shards in &[1usize, 2, 3, 4, 7] {
            let shard_outputs = rt
                .block_on(compact_sharded(
                    to_arena_sources(&per_source),
                    num_shards,
                    opts.clone(),
                ))
                .expect("sharded compaction should succeed");
            let sharded_entries = decode_all_entries(&shard_outputs);

            assert_eq!(
                sharded_entries, unsharded_entries,
                "seed {}: num_shards={} produced different logical dataset than unsharded",
                seed, num_shards
            );
        }
    }
}

fn decode_all_entries(ssts: &[Vec<u8>]) -> Vec<(String, String)> {
    use compactor::format::block;
    use compactor::format::footer;

    let mut out = Vec::new();
    for bytes in ssts {
        let footer = footer::decode_footer(
            &bytes[bytes.len() - footer::FOOTER_TAIL_LEN..],
            bytes.len() as u64,
        )
        .expect("footer should decode");

        let metaindex_raw = block::decode_block_contents(
            bytes[footer.metaindex_handle.offset as usize
                ..footer.metaindex_handle.offset as usize
                    + block::block_read_len(footer.metaindex_handle)]
                .to_vec(),
            footer.metaindex_handle,
            footer.checksum_type,
            footer.base_context_checksum,
        )
        .expect("metaindex should decode");
        let metaindex = block::parse_block(metaindex_raw).expect("metaindex should parse");

        let mut scratch = Vec::new();
        let mut index_handle = None;
        block::for_each_entry(&metaindex, &mut scratch, |k, v| {
            if k == b"rocksdb.index" {
                let mut pos = 0;
                let offset = compactor::format::varint::get_varint64(v, &mut pos).unwrap();
                let size = compactor::format::varint::get_varint64(v, &mut pos).unwrap();
                index_handle = Some(footer::BlockHandle { offset, size });
            }
        })
        .expect("metaindex iteration should succeed");
        let index_handle = index_handle.expect("index handle must be present");

        let index_raw = block::decode_block_contents(
            bytes[index_handle.offset as usize
                ..index_handle.offset as usize + block::block_read_len(index_handle)]
                .to_vec(),
            index_handle,
            footer.checksum_type,
            footer.base_context_checksum,
        )
        .expect("index block should decode");
        let index = block::parse_block(index_raw).expect("index block should parse");

        let mut data_handles = Vec::new();
        let mut index_scratch = Vec::new();
        block::for_each_index_entry(&index, &mut index_scratch, |_k, h| data_handles.push(h))
            .expect("index iteration should succeed");

        let mut data_scratch = Vec::new();
        for handle in data_handles {
            let raw = block::decode_block_contents(
                bytes[handle.offset as usize
                    ..handle.offset as usize + block::block_read_len(handle)]
                    .to_vec(),
                handle,
                footer.checksum_type,
                footer.base_context_checksum,
            )
            .expect("data block should decode");
            let parsed = block::parse_block(raw).expect("data block should parse");
            block::for_each_entry(&parsed, &mut data_scratch, |k, v| {
                let ik = compactor::format::internal_key::split(k).expect("valid internal key");
                out.push((
                    String::from_utf8_lossy(ik.user_key).to_string(),
                    String::from_utf8_lossy(v).to_string(),
                ));
            })
            .expect("data block iteration should succeed");
        }
    }
    // Different shard counts produce different numbers of output files with
    // different key ranges each, but the union across all of them must be
    // identical; sort before comparing so file boundaries don't matter.
    out.sort();
    out
}
