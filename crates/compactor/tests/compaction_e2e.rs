use compactor::compaction::{CompactionOptions, compact};
use compactor::format::sst_builder::SstBuilderOptions;
use compactor::merge::ArenaSourceBuilder;

fn internal_key(user_key: &str, seq: u64, value_type: u8) -> Vec<u8> {
    let mut k = user_key.as_bytes().to_vec();
    let packed = (seq << 8) | value_type as u64;
    k.extend_from_slice(&packed.to_le_bytes());
    k
}

const TYPE_VALUE: u8 = 1;
const TYPE_DELETION: u8 = 0;

/// Two overlapping input "SSTs" (as ArenaSources): older values in source
/// A, newer values (and one delete) in source B. Compacting must keep only
/// the newest version per key, and drop keys whose newest version is a
/// tombstone (bottommost semantics, see compactor-compaction's doc comment
/// on why that's the scope here).
#[test]
fn compact_drops_obsolete_versions_and_tombstones() {
    let mut a = ArenaSourceBuilder::new();
    a.push(&internal_key("apple", 1, TYPE_VALUE), b"old_apple");
    a.push(&internal_key("banana", 1, TYPE_VALUE), b"old_banana");
    a.push(&internal_key("cherry", 1, TYPE_VALUE), b"only_cherry");
    let source_a = a.build();

    let mut b = ArenaSourceBuilder::new();
    b.push(&internal_key("apple", 5, TYPE_VALUE), b"new_apple");
    b.push(&internal_key("banana", 5, TYPE_DELETION), b"");
    let source_b = b.build();

    let opts = CompactionOptions {
        sst_builder: SstBuilderOptions {
            base_context_checksum: 0xC0FFEE,
            ..Default::default()
        },
    };

    let output_bytes = compact(vec![source_a, source_b], &opts).expect("compaction should succeed");
    assert!(!output_bytes.is_empty());

    let out_path = std::env::temp_dir().join("compactor_compaction_e2e_test.sst");
    std::fs::write(&out_path, &output_bytes).expect("failed to write output SST");
    eprintln!("wrote compacted SST to {}", out_path.display());

    // "apple" -> newest value only, "banana" -> dropped (tombstone winner),
    // "cherry" -> only version, unaffected.
    let entries = parse_scan_output(&output_bytes);
    assert_eq!(
        entries,
        vec![
            ("apple".to_string(), "new_apple".to_string()),
            ("cherry".to_string(), "only_cherry".to_string()),
        ]
    );
}

/// Minimal synchronous re-parse of the output bytes using compactor-format's
/// pure decoders directly (no async I/O needed since we already have the
/// full byte buffer) — avoids pulling compactor-sst/compactor-io into this
/// test just to re-validate our own output.
fn parse_scan_output(bytes: &[u8]) -> Vec<(String, String)> {
    use compactor::format::block;
    use compactor::format::footer;

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
            out.push((
                String::from_utf8_lossy(ik.user_key).to_string(),
                String::from_utf8_lossy(v).to_string(),
            ));
        })
        .expect("data block iteration should succeed");
    }
    out
}
