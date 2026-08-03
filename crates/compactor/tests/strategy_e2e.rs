use compactor::compaction::{
    BottommostStrategy, CompactionOptions, PassthroughStrategy, SnapshotAwareStrategy,
    compact_with_strategy,
};
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

fn opts() -> CompactionOptions {
    CompactionOptions {
        sst_builder: SstBuilderOptions {
            base_context_checksum: 0xABCD1234,
            ..Default::default()
        },
    }
}

/// Real end-to-end proof (full pipeline: merge -> strategy -> SST write ->
/// re-parse), not just the strategy's own unit-level `decide()` tests:
/// with a live snapshot between two versions of a key, SnapshotAwareStrategy
/// must keep BOTH the newest version (for current readers) and the version
/// visible to the snapshot (for a reader pinned to that snapshot) — a plain
/// BottommostStrategy compaction of the same input would incorrectly drop
/// the older version.
#[test]
fn snapshot_aware_keeps_both_current_and_snapshot_visible_versions() {
    let mut a = ArenaSourceBuilder::new();
    a.push(&internal_key("apple", 2, TYPE_VALUE), b"old_apple");
    let source_a = a.build();

    let mut b = ArenaSourceBuilder::new();
    b.push(&internal_key("apple", 10, TYPE_VALUE), b"new_apple");
    let source_b = b.build();

    // Snapshot at seqno 5 sits between the two versions (2 <= 5 < 10): a
    // reader pinned to it must still see "old_apple", so compaction can't
    // drop seq=2 even though it's not the newest.
    let mut strategy = SnapshotAwareStrategy::new(5);
    let output = compact_with_strategy(vec![source_a, source_b], &opts(), &mut strategy)
        .expect("compaction should succeed");
    let entries = parse_scan_output(&output);
    assert_eq!(
        entries,
        vec![
            ("apple".to_string(), "new_apple".to_string()),
            ("apple".to_string(), "old_apple".to_string()),
        ],
        "snapshot-aware compaction must keep both the newest and the snapshot-visible version"
    );

    // Sanity: the SAME input under BottommostStrategy drops the older
    // version, proving the two strategies genuinely diverge rather than
    // one being a no-op wrapper around the other.
    let mut a2 = ArenaSourceBuilder::new();
    a2.push(&internal_key("apple", 2, TYPE_VALUE), b"old_apple");
    let mut b2 = ArenaSourceBuilder::new();
    b2.push(&internal_key("apple", 10, TYPE_VALUE), b"new_apple");
    let mut bottommost = BottommostStrategy;
    let output2 = compact_with_strategy(vec![a2.build(), b2.build()], &opts(), &mut bottommost)
        .expect("compaction should succeed");
    let entries2 = parse_scan_output(&output2);
    assert_eq!(
        entries2,
        vec![("apple".to_string(), "new_apple".to_string())],
        "bottommost compaction of the identical input must drop the older version"
    );
}

/// A tombstone that a live snapshot might still need to observe must be
/// kept, not dropped, even though it isn't the absolute newest version.
#[test]
fn snapshot_aware_keeps_tombstone_visible_to_snapshot_end_to_end() {
    let mut a = ArenaSourceBuilder::new();
    a.push(&internal_key("key", 10, TYPE_VALUE), b"current_value");
    a.push(&internal_key("key", 4, TYPE_DELETION), b"");
    a.push(&internal_key("key", 2, TYPE_VALUE), b"ancient_value");
    let source = a.build();

    let mut strategy = SnapshotAwareStrategy::new(5);
    let output = compact_with_strategy(vec![source], &opts(), &mut strategy)
        .expect("compaction should succeed");
    let entries = parse_scan_output(&output);
    // Only "current_value" survives as a real entry; the delete at seq=4 is
    // kept internally (needed for a snapshot@5 reader) but our test parser
    // only reports Value-typed entries as visible key/value pairs — confirm
    // via raw entry count instead that the tombstone entry is physically
    // present in the output, not silently dropped.
    assert_eq!(
        entries,
        vec![("key".to_string(), "current_value".to_string())]
    );

    let raw_entries = parse_all_entries_including_tombstones(&output);
    assert_eq!(
        raw_entries.len(),
        2,
        "expected current value + the snapshot-visible tombstone, got {:?}",
        raw_entries
    );
}

/// PassthroughStrategy must keep every physical entry unchanged: no dedup,
/// no tombstone drop, even at the full-pipeline level.
#[test]
fn passthrough_keeps_all_versions_end_to_end() {
    let mut a = ArenaSourceBuilder::new();
    a.push(&internal_key("k", 3, TYPE_VALUE), b"v3");
    a.push(&internal_key("k", 2, TYPE_VALUE), b"v2");
    a.push(&internal_key("k", 1, TYPE_DELETION), b"");
    let source = a.build();

    let mut strategy = PassthroughStrategy;
    let output = compact_with_strategy(vec![source], &opts(), &mut strategy)
        .expect("compaction should succeed");
    let raw_entries = parse_all_entries_including_tombstones(&output);
    assert_eq!(
        raw_entries.len(),
        3,
        "passthrough must keep every entry, got {:?}",
        raw_entries
    );
}

/// Same decode path as compaction_e2e.rs's parse_scan_output, but returns
/// EVERY entry (including tombstones/deletions) rather than filtering to
/// look like a plain key/value scan — needed here because a
/// snapshot-visible tombstone is a real, load-bearing output entry, not
/// noise to filter out.
fn parse_all_entries_including_tombstones(bytes: &[u8]) -> Vec<(String, u64, u8)> {
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
            let _ = v;
            let ty_byte = match ik.value_type {
                compactor::format::internal_key::ValueType::Deletion => 0u8,
                compactor::format::internal_key::ValueType::Value => 1u8,
                _ => 255u8,
            };
            out.push((
                String::from_utf8_lossy(ik.user_key).to_string(),
                ik.sequence,
                ty_byte,
            ));
        })
        .expect("data block iteration should succeed");
    }
    out
}

/// Same decode as compaction_e2e.rs's parse_scan_output (Value-typed
/// entries only, as plain key/value pairs) — duplicated locally so this
/// file doesn't depend on the other test file's private helper.
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
            if matches!(
                ik.value_type,
                compactor::format::internal_key::ValueType::Value
            ) {
                out.push((
                    String::from_utf8_lossy(ik.user_key).to_string(),
                    String::from_utf8_lossy(v).to_string(),
                ));
            }
        })
        .expect("data block iteration should succeed");
    }
    out
}
