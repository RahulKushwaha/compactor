//! Top-level SST assembly: takes sorted internal-key/value pairs and
//! produces a complete, valid block-based table file in memory. Pure and
//! synchronous — no I/O. `compactor-sst` wraps this with the actual async
//! file write.
//!
//! Fixed choices for this first cut (see block_builder.rs / properties.rs
//! doc comments for the RocksDB source citations backing each default):
//! - format_version 7, block-based table, XXH3 checksums with per-block
//!   context-checksum modifier (matches what modern RocksDB writes today).
//! - No compression (`CompressionType::NoCompression`).
//! - Plain `kDataBlockBinarySearch` data blocks (no hash index, no
//!   separated KV storage).
//! - No filter block, no partitioned index, no range deletions.
//! - block_restart_interval = 16 (data), 1 (index) — RocksDB's own defaults
//!   (include/rocksdb/table.h).
//! - target data block size ~4096 bytes before cutting a new one
//!   (BlockBasedTableOptions::block_size default).

use crate::format::block::encode_block_with_trailer;
use crate::format::block_builder::{BlockBuilder, IndexBlockBuilder};
use crate::format::footer::{self, BlockHandle, ChecksumType};
use crate::format::properties::{self, PropertiesInput};

const DATA_BLOCK_RESTART_INTERVAL: usize = 16;
const INDEX_BLOCK_RESTART_INTERVAL: usize = 1;
const TARGET_BLOCK_SIZE: usize = 4096;
const FORMAT_VERSION: u32 = 7;

#[derive(Clone)]
pub struct SstBuilderOptions {
    pub checksum_type: ChecksumType,
    pub comparator_name: String,
    /// Column family id/name this SST belongs to. Must match the target
    /// DB's actual CF or `ldb repair` / normal recovery rejects the file
    /// ("inconsistent column family name") — see properties.rs
    /// `PropertiesInput::column_family_name` doc comment for how this was
    /// confirmed. Defaults to the default CF (id 0, name "default").
    pub column_family_id: u64,
    pub column_family_name: String,
    /// Nonzero random value; format_version >= 6 requires this to be
    /// nonzero when block checksums are in use (table/format.cc
    /// FooterBuilder::Build assert). Caller supplies it since this crate
    /// has no RNG dependency by design (see Date.now()/random restrictions
    /// upstream in workflow contexts — keep this crate deterministic).
    pub base_context_checksum: u32,
    /// When `Some`, produces an `ldb ingest_extern_sst` / `IngestExternalFile`
    /// -compatible output: every entry's internal key has its sequence
    /// number forced to 0 (required by SstFileWriter's own contract — see
    /// table/sst_file_writer.cc `AddImpl`, `constexpr SequenceNumber
    /// sequence_number = 0`), and the properties block gets the
    /// external_sst_file.version/global_seqno stamps that let RocksDB's
    /// ingest path accept the file. See properties.rs
    /// `PropertiesInput::external_sst_global_seqno` for the full contract
    /// (entries must already be deduped to one version per user key).
    pub external_sst_global_seqno: Option<u64>,
}

impl Default for SstBuilderOptions {
    fn default() -> Self {
        SstBuilderOptions {
            checksum_type: ChecksumType::XxH3,
            comparator_name: properties::BYTEWISE_COMPARATOR_NAME.to_string(),
            column_family_id: 0,
            column_family_name: "default".to_string(),
            base_context_checksum: 0x1, // caller should override with a real random value
            external_sst_global_seqno: None,
        }
    }
}

/// Builds a complete SST file in memory from internal keys already in
/// sorted order (caller's responsibility — this does not sort or dedup).
/// `entries` are `(internal_key, value)` pairs; internal_key already
/// includes the 8-byte seq/type suffix (see internal_key.rs).
pub fn build_sst(entries: &[(Vec<u8>, Vec<u8>)], opts: &SstBuilderOptions) -> Vec<u8> {
    // For external_sst mode, rewrite every internal key's seq/type suffix to
    // (seq=0, same type) — SstFileWriter's own contract (see
    // SstBuilderOptions::external_sst_global_seqno doc comment). Cloned once
    // up front rather than threading a per-key rewrite through every place
    // below that reads `entries`.
    let rewritten: Option<Vec<(Vec<u8>, Vec<u8>)>> = opts.external_sst_global_seqno.map(|_| {
        entries
            .iter()
            .map(|(k, v)| (zero_out_seqno(k), v.clone()))
            .collect()
    });
    let entries: &[(Vec<u8>, Vec<u8>)] = rewritten.as_deref().unwrap_or(entries);

    let mut file = Vec::new();
    let mut index_builder = IndexBlockBuilder::new(INDEX_BLOCK_RESTART_INTERVAL);

    let mut raw_key_size = 0u64;
    let mut raw_value_size = 0u64;

    let mut block_builder = BlockBuilder::new(DATA_BLOCK_RESTART_INTERVAL);
    // The index separator for a data block must be >= every key IN that block,
    // because RocksDB's `kBinarySearch` index seek does `index_iter->Seek(target)`
    // and then descends into the FIRST block whose separator is >= target
    // (table/block_based/block_based_table_reader.cc, `NewIndexIterator` +
    // `BlockBasedTableIterator::SeekImpl`). Handing it the block's FIRST key
    // instead makes the separator smaller than most of the block's contents, so a
    // point lookup for a key in the middle of block N finds N's separator < target,
    // skips to N+1, and reports the key missing.
    //
    // This is invisible to a sequential scan: an iterator that starts at the
    // beginning walks blocks in order and never consults the separator to choose
    // one, which is why `sst_dump --command=scan` and `ldb scan` accept a file with
    // first-key separators while `Get` on it silently returns NotFound.
    //
    // RocksDB itself uses `FindShortestSeparator(last_key_in_block,
    // first_key_in_next_block)` to get a shorter-but-still-valid separator
    // (`ShortenedIndexBuilder::AddIndexEntry`). The exact last key is the
    // conservative choice: always valid, just a few bytes larger per block.
    let mut pending_last_key: Option<Vec<u8>> = None;

    let flush_data_block = |file: &mut Vec<u8>,
                            index_builder: &mut IndexBlockBuilder,
                            block_builder: BlockBuilder,
                            last_key: &[u8]| {
        if block_builder.is_empty() {
            return;
        }
        let raw = block_builder.finish();
        let offset = file.len() as u64;
        let uncompressed_size = raw.len() as u64;
        let with_trailer =
            encode_block_with_trailer(raw, offset, opts.checksum_type, opts.base_context_checksum);
        let handle = BlockHandle {
            offset,
            size: uncompressed_size,
        };
        file.extend_from_slice(&with_trailer);
        index_builder.add(last_key, handle);
    };

    for (key, value) in entries {
        raw_key_size += key.len() as u64;
        raw_value_size += value.len() as u64;

        block_builder.add(key, value);
        pending_last_key = Some(key.clone());

        if block_builder.current_size_estimate() >= TARGET_BLOCK_SIZE {
            let last_key = pending_last_key.take().unwrap();
            let finished = std::mem::replace(
                &mut block_builder,
                BlockBuilder::new(DATA_BLOCK_RESTART_INTERVAL),
            );
            flush_data_block(&mut file, &mut index_builder, finished, &last_key);
        }
    }
    if let Some(last_key) = pending_last_key.take() {
        flush_data_block(&mut file, &mut index_builder, block_builder, &last_key);
    }
    let data_size = file.len() as u64;

    // Index block.
    let index_raw = index_builder.finish();
    let index_offset = file.len() as u64;
    let index_uncompressed_size = index_raw.len() as u64;
    let index_with_trailer = encode_block_with_trailer(
        index_raw,
        index_offset,
        opts.checksum_type,
        opts.base_context_checksum,
    );
    file.extend_from_slice(&index_with_trailer);
    let index_handle = BlockHandle {
        offset: index_offset,
        size: index_uncompressed_size,
    };

    // Properties block.
    let props_input = PropertiesInput {
        num_entries: entries.len() as u64,
        raw_key_size,
        raw_value_size,
        data_size,
        index_size: index_uncompressed_size,
        comparator_name: opts.comparator_name.clone(),
        column_family_id: opts.column_family_id,
        column_family_name: opts.column_family_name.clone(),
        external_sst_global_seqno: opts.external_sst_global_seqno,
    };
    let props_raw = properties::build_properties_block(&props_input);
    let props_offset = file.len() as u64;
    let props_uncompressed_size = props_raw.len() as u64;
    let props_with_trailer = encode_block_with_trailer(
        props_raw,
        props_offset,
        opts.checksum_type,
        opts.base_context_checksum,
    );
    file.extend_from_slice(&props_with_trailer);
    let props_handle = BlockHandle {
        offset: props_offset,
        size: props_uncompressed_size,
    };

    // Metaindex block: maps "rocksdb.properties" -> props_handle and
    // "rocksdb.index" -> index_handle (format_version >= 6 requires the
    // index handle here, not in the footer).
    let metaindex_raw = properties::build_metaindex_block(&[
        (properties::PROPERTIES_BLOCK_NAME, props_handle),
        (properties::INDEX_BLOCK_NAME, index_handle),
    ]);
    let metaindex_offset = file.len() as u64;
    let metaindex_uncompressed_size = metaindex_raw.len() as u64;
    let metaindex_with_trailer = encode_block_with_trailer(
        metaindex_raw,
        metaindex_offset,
        opts.checksum_type,
        opts.base_context_checksum,
    );
    file.extend_from_slice(&metaindex_with_trailer);
    let metaindex_handle = BlockHandle {
        offset: metaindex_offset,
        size: metaindex_uncompressed_size,
    };

    // Footer.
    let footer_offset = file.len() as u64;
    let footer_bytes = footer::encode_footer_v6_plus(
        FORMAT_VERSION,
        opts.checksum_type,
        metaindex_handle,
        opts.base_context_checksum,
        footer_offset,
    );
    file.extend_from_slice(&footer_bytes);

    file
}

/// Rewrites an internal key's trailing 8-byte packed `(seq << 8) | type`
/// suffix to `seq = 0`, preserving the type byte. See
/// `SstBuilderOptions::external_sst_global_seqno` doc comment for why.
fn zero_out_seqno(internal_key: &[u8]) -> Vec<u8> {
    let mut k = internal_key.to_vec();
    let n = k.len();
    let packed = u64::from_le_bytes(k[n - 8..].try_into().unwrap());
    let value_type = packed & 0xff;
    k[n - 8..].copy_from_slice(&value_type.to_le_bytes());
    k
}
