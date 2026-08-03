//! Randomized input generation + differential checking between
//! `compactor`'s real compaction path and `oracle`'s independent naive
//! reference. Library form (callable from tests, from a `cargo fuzz`
//! target, or from a standalone binary) so the generation logic isn't
//! duplicated across harnesses.

use rand::Rng;

pub const TYPE_VALUE: u8 = 1;
pub const TYPE_DELETION: u8 = 0;

pub fn make_internal_key(user_key: &str, seq: u64, ty: u8) -> Vec<u8> {
    let mut k = user_key.as_bytes().to_vec();
    let packed = (seq << 8) | ty as u64;
    k.extend_from_slice(&packed.to_le_bytes());
    k
}

/// Generates `num_sources` sorted sources (per compactor's internal-key
/// order) covering `num_keys` distinct user keys, each with 1-3 randomly
/// distributed versions and an occasional tombstone as the newest version.
/// Mirrors the shape used in compactor's own subcompaction differential
/// tests, factored out here so the fuzzer and any future test can share it.
pub fn generate_sources(
    rng: &mut impl Rng,
    num_sources: usize,
    num_keys: usize,
) -> Vec<Vec<(Vec<u8>, Vec<u8>)>> {
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
                make_internal_key(&user_key, seq, ty),
                format!("value_{}_{}", i, seq).into_bytes(),
            ));
        }
    }

    for entries in &mut per_source {
        entries.sort_by(|a, b| compactor::merge::cmp::compare_internal_keys(&a.0, &b.0));
    }
    per_source
}

/// Runs compactor's real compaction over `per_source` and decodes the
/// resulting SST back into plain `(user_key, value)` pairs, sorted, ready
/// to diff against `oracle::naive_compact`'s output.
pub fn run_compactor(per_source: &[Vec<(Vec<u8>, Vec<u8>)>]) -> Vec<(Vec<u8>, Vec<u8>)> {
    use compactor::compaction::{CompactionOptions, compact};
    use compactor::format::sst_builder::SstBuilderOptions;
    use compactor::merge::ArenaSourceBuilder;

    let sources = per_source
        .iter()
        .map(|entries| {
            let mut builder = ArenaSourceBuilder::new();
            for (k, v) in entries {
                builder.push(k, v);
            }
            builder.build()
        })
        .collect();

    let opts = CompactionOptions {
        sst_builder: SstBuilderOptions {
            base_context_checksum: 0x1357_9BDF,
            ..Default::default()
        },
    };
    let sst_bytes = compact(sources, &opts).expect("compaction should succeed");
    decode_sst_entries(&sst_bytes)
}

/// Decodes a compactor-produced SST's data blocks into `(user_key, value)`
/// pairs, sorted. Uses compactor's own pure decoders (not the oracle's) —
/// this is fine because decoding is not the thing under differential test;
/// the compaction *logic* (dedup, tombstone drop) is. If the decoder itself
/// were wrong, compactor's own format round-trip tests would already have
/// caught it via real `sst_dump` cross-checks.
pub fn decode_sst_entries(bytes: &[u8]) -> Vec<(Vec<u8>, Vec<u8>)> {
    use compactor::format::{block, footer};

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

    let mut out = Vec::new();
    let mut data_scratch = Vec::new();
    for handle in data_handles {
        let raw = block::decode_block_contents(
            bytes[handle.offset as usize..handle.offset as usize + block::block_read_len(handle)]
                .to_vec(),
            handle,
            footer.checksum_type,
            footer.base_context_checksum,
        )
        .expect("data block should decode");
        let parsed = block::parse_block(raw).expect("data block should parse");
        block::for_each_entry(&parsed, &mut data_scratch, |k, v| {
            let ik = compactor::format::internal_key::split(k).expect("valid internal key");
            out.push((ik.user_key.to_vec(), v.to_vec()));
        })
        .expect("data block iteration should succeed");
    }
    out.sort();
    out
}
