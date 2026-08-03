use compactor::format::block::{self, decode_block_contents, encode_block_with_trailer};
use compactor::format::block_builder::BlockBuilder;
use compactor::format::footer::{self, BlockHandle, ChecksumType};

#[test]
fn data_block_roundtrip_no_compression() {
    let mut builder = BlockBuilder::new(4); // small restart interval to force multiple restarts
    let entries: Vec<(Vec<u8>, Vec<u8>)> = (0..50)
        .map(|i| {
            (
                format!("key{:04}", i).into_bytes(),
                format!("value_{:04}_payload", i).into_bytes(),
            )
        })
        .collect();
    for (k, v) in &entries {
        builder.add(k, v);
    }
    let raw = builder.finish();

    let checksum_type = ChecksumType::XxH3;
    let offset = 12345u64;
    let base_context_checksum = 0xdead_beefu32;
    let with_trailer = encode_block_with_trailer(raw, offset, checksum_type, base_context_checksum);

    let handle = BlockHandle {
        offset,
        size: (with_trailer.len() - block::BLOCK_TRAILER_SIZE) as u64,
    };
    let decoded = decode_block_contents(with_trailer, handle, checksum_type, base_context_checksum)
        .expect("decode should succeed");
    let parsed = block::parse_block(decoded).expect("parse should succeed");

    let mut scratch = Vec::new();
    let mut got = Vec::new();
    block::for_each_entry(&parsed, &mut scratch, |k, v| {
        got.push((k.to_vec(), v.to_vec()));
    })
    .expect("iteration should succeed");

    assert_eq!(got, entries);
}

#[test]
fn data_block_roundtrip_checksum_tamper_detected() {
    let mut builder = BlockBuilder::new(16);
    builder.add(b"a", b"1");
    builder.add(b"b", b"2");
    let raw = builder.finish();

    let checksum_type = ChecksumType::Crc32c;
    let offset = 0u64;
    let mut with_trailer = encode_block_with_trailer(raw, offset, checksum_type, 0);
    // Flip a bit in the raw contents after the trailer was computed.
    with_trailer[0] ^= 0x01;

    let handle = BlockHandle {
        offset,
        size: (with_trailer.len() - block::BLOCK_TRAILER_SIZE) as u64,
    };
    let result = decode_block_contents(with_trailer, handle, checksum_type, 0);
    assert!(
        result.is_err(),
        "tampered block should fail checksum verification"
    );
}

#[test]
fn footer_roundtrip_format_version_7() {
    let checksum_type = ChecksumType::XxH3;
    let base_context_checksum = 0x1234_5678u32;
    let metaindex_handle = BlockHandle {
        offset: 1000,
        size: 42,
    };
    // metaindex must be immediately followed by its trailer, then the footer.
    let footer_offset =
        metaindex_handle.offset + metaindex_handle.size + block::BLOCK_TRAILER_SIZE as u64;

    let encoded = footer::encode_footer_v6_plus(
        7,
        checksum_type,
        metaindex_handle,
        base_context_checksum,
        footer_offset,
    );
    let file_size = footer_offset + encoded.len() as u64;

    let decoded = footer::decode_footer(&encoded, file_size).expect("footer should decode");
    assert_eq!(decoded.format_version, 7);
    assert_eq!(decoded.checksum_type, checksum_type);
    assert_eq!(decoded.base_context_checksum, base_context_checksum);
    assert_eq!(decoded.metaindex_handle.offset, metaindex_handle.offset);
    assert_eq!(decoded.metaindex_handle.size, metaindex_handle.size);
    assert_eq!(decoded.table_magic_number, footer::BLOCK_BASED_TABLE_MAGIC);
}

#[test]
fn index_block_roundtrip_delta_encoded_handles() {
    use compactor::format::block_builder::IndexBlockBuilder;

    let mut builder = IndexBlockBuilder::new(3);
    let entries = vec![
        (
            b"aaa".to_vec(),
            BlockHandle {
                offset: 0,
                size: 100,
            },
        ),
        (
            b"bbb".to_vec(),
            BlockHandle {
                offset: 105,
                size: 110,
            },
        ),
        (
            b"ccc".to_vec(),
            BlockHandle {
                offset: 220,
                size: 90,
            },
        ),
        (
            b"ddd".to_vec(),
            BlockHandle {
                offset: 315,
                size: 130,
            },
        ), // new restart point
        (
            b"eee".to_vec(),
            BlockHandle {
                offset: 450,
                size: 50,
            },
        ),
    ];
    for (k, h) in &entries {
        builder.add(k, *h);
    }
    let raw = builder.finish();
    let parsed = block::parse_block(raw).expect("parse should succeed");

    let mut scratch = Vec::new();
    let mut got = Vec::new();
    block::for_each_index_entry(&parsed, &mut scratch, |k, h| {
        got.push((k.to_vec(), h));
    })
    .expect("iteration should succeed");

    assert_eq!(got.len(), entries.len());
    for ((got_key, got_handle), (want_key, want_handle)) in got.iter().zip(entries.iter()) {
        assert_eq!(got_key, want_key);
        assert_eq!(got_handle.offset, want_handle.offset);
        assert_eq!(got_handle.size, want_handle.size);
    }
}
