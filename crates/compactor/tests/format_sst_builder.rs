use compactor::format::sst_builder::{SstBuilderOptions, build_sst};

/// Builds a small SST via our writer and writes it to a fixed path so an
/// external test script can point real RocksDB tools (`ldb`/`sst_dump`) at
/// it. This test only checks our own writer produces *something*; end-to-end
/// validation against real RocksDB tooling happens out-of-band (see
/// scripts/verify_with_rocksdb.sh) since that requires shelling out to
/// binaries not available in a plain `cargo test` environment.
#[test]
fn build_sst_smoke_test() {
    let mut entries = Vec::new();
    for i in 0..500u32 {
        let user_key = format!("key{:05}", i).into_bytes();
        let mut internal_key = user_key;
        internal_key.extend_from_slice(&pack_seq_and_type(0, 1)); // kTypeValue
        let value = format!("value_{:05}_padding_to_make_it_longer", i).into_bytes();
        entries.push((internal_key, value));
    }

    let opts = SstBuilderOptions {
        base_context_checksum: 0xdead_beef,
        ..Default::default()
    };
    let sst_bytes = build_sst(&entries, &opts);

    assert!(!sst_bytes.is_empty());

    let out_path = std::env::temp_dir().join("compactor_sst_builder_test.sst");
    std::fs::write(&out_path, &sst_bytes).expect("failed to write test SST");
    eprintln!("wrote test SST to {}", out_path.display());
}

/// Every index separator must be >= the last key of the block it points at.
///
/// RocksDB's `kBinarySearch` index seek descends into the FIRST block whose
/// separator is >= the target. A separator smaller than keys inside its own block
/// therefore sends a point lookup one block too far, and `Get` returns NotFound
/// for a key that is present. A sequential scan cannot catch it, since it walks
/// blocks in order without consulting separators: an SST with this defect passes
/// `sst_dump --command=scan`, `ldb scan`, and `sst_dump --command=check` while
/// failing every `Get` for a key not first in its block. That is exactly how it
/// escaped the existing tests.
#[test]
fn index_separators_cover_their_blocks() {
    use compactor::format::block;
    use compactor::format::footer::{self, BlockHandle};
    use compactor::format::varint;

    // Enough entries, with values big enough, to span many data blocks. A
    // single-block file cannot exhibit the bug at all.
    let mut entries = Vec::new();
    for i in 0..2000u32 {
        let mut internal_key = format!("key{:08}", i).into_bytes();
        internal_key.extend_from_slice(&pack_seq_and_type(u64::from(i) + 1, 1));
        entries.push((internal_key, vec![b'v'; 512]));
    }

    let opts = SstBuilderOptions {
        base_context_checksum: 0xdead_beef,
        ..Default::default()
    };
    let sst = build_sst(&entries, &opts);

    // Walk the footer -> metaindex -> index chain to get the index block.
    let tail = &sst[sst.len() - footer::FOOTER_TAIL_LEN..];
    let ft = footer::decode_footer(tail, sst.len() as u64).expect("decode footer");

    let read_block = |h: BlockHandle| -> block::ParsedBlock {
        let start = h.offset as usize;
        let raw = sst[start..start + block::block_read_len(h)].to_vec();
        let contents =
            block::decode_block_contents(raw, h, ft.checksum_type, ft.base_context_checksum)
                .expect("decode block");
        block::parse_block(contents).expect("parse block")
    };

    let metaindex = read_block(ft.metaindex_handle);
    let index_handle = match ft.index_handle {
        Some(h) => h,
        None => {
            let mut found = None;
            let mut scratch = Vec::new();
            block::for_each_entry(&metaindex, &mut scratch, |key, value| {
                if key == b"rocksdb.index" {
                    // Metaindex values are a bare 2-varint BlockHandle.
                    let mut pos = 0usize;
                    let offset = varint::get_varint64(value, &mut pos).expect("handle offset");
                    let size = varint::get_varint64(value, &mut pos).expect("handle size");
                    found = Some(BlockHandle { offset, size });
                }
            })
            .expect("iterate metaindex");
            found.expect("index handle in metaindex")
        }
    };

    let index = read_block(index_handle);
    let mut separators: Vec<(Vec<u8>, BlockHandle)> = Vec::new();
    let mut scratch = Vec::new();
    block::for_each_index_entry(&index, &mut scratch, |key, handle| {
        separators.push((key.to_vec(), handle));
    })
    .expect("iterate index");

    assert!(
        separators.len() > 1,
        "test needs a multi-block file to be meaningful, got {} block(s)",
        separators.len()
    );

    for (separator, handle) in &separators {
        let data = read_block(*handle);
        let mut keys = Vec::new();
        let mut s = Vec::new();
        block::for_each_entry(&data, &mut s, |key, _| keys.push(key.to_vec()))
            .expect("iterate data block");
        let last = keys.last().expect("non-empty data block");
        assert!(
            separator >= last,
            "index separator {:?} is smaller than the last key {:?} in the block it points at; \
             point lookups for keys after the separator will miss",
            String::from_utf8_lossy(&separator[..separator.len() - 8]),
            String::from_utf8_lossy(&last[..last.len() - 8]),
        );
    }
}

fn pack_seq_and_type(seq: u64, value_type: u8) -> [u8; 8] {
    let packed = (seq << 8) | value_type as u64;
    packed.to_le_bytes()
}
