//! Properties block + metaindex block construction. See table/meta_blocks.h
//! / table/meta_blocks.cc and table/table_properties.cc in RocksDB.
//!
//! Only the minimum property set the reader actually requires or branches
//! on is written (see research citations in module comments below); most of
//! TableProperties' dozens of fields are advisory/stats-only and are safely
//! omitted. Confirmed by direct read of RocksDB source (table/meta_blocks.cc,
//! table/block_based/block_based_table_reader.cc):
//! - Properties block itself is mandatory (reader.cc: "Cannot find
//!   Properties block from file").
//! - `rocksdb.block.based.table.index.type` is mandatory, encoded as a
//!   fixed32 LE (not varint!) — reader.cc: "Missing index type property".
//! - Property/metaindex keys must be added in strictly increasing order
//!   (both are backed by a sorted map in the real implementation; we sort
//!   explicitly since we don't have that map).
//! - Numeric TableProperties fields (num_entries, raw_key_size, etc.) are
//!   varint64-encoded, NOT fixed64 and NOT decimal strings.
//! - format_version >= 6 requires the index block's handle to live in the
//!   metaindex block under key "rocksdb.index" (not the footer).

use crate::format::block_builder::BlockBuilder;
use crate::format::footer::BlockHandle;
use crate::format::varint::put_varint64;

pub const PROPERTIES_BLOCK_NAME: &str = "rocksdb.properties";
pub const INDEX_BLOCK_NAME: &str = "rocksdb.index";

const KEY_INDEX_TYPE: &str = "rocksdb.block.based.table.index.type";
const KEY_NUM_ENTRIES: &str = "rocksdb.num.entries";
const KEY_RAW_KEY_SIZE: &str = "rocksdb.raw.key.size";
const KEY_RAW_VALUE_SIZE: &str = "rocksdb.raw.value.size";
const KEY_DATA_SIZE: &str = "rocksdb.data.size";
const KEY_INDEX_SIZE: &str = "rocksdb.index.size";
const KEY_INDEX_KEY_IS_USER_KEY: &str = "rocksdb.index.key.is.user.key";
const KEY_INDEX_VALUE_IS_DELTA_ENCODED: &str = "rocksdb.index.value.is.delta.encoded";
const KEY_COMPARATOR: &str = "rocksdb.comparator";
const KEY_COLUMN_FAMILY_ID: &str = "rocksdb.column.family.id";
const KEY_COLUMN_FAMILY_NAME: &str = "rocksdb.column.family.name";

/// SstFileWriter-specific properties (table/sst_file_writer_collectors.h
/// ExternalSstFilePropertyNames). Their presence is what lets `ldb
/// ingest_extern_sst` / IngestExternalFile accept a file — plain internal
/// SSTs (even ones RocksDB itself wrote as part of normal flush/compaction)
/// are rejected with "External file version not found", confirmed by
/// testing `ldb ingest_extern_sst` against a genuine RocksDB-internal SST.
/// Unlike every other property here, these two are FIXED-width (fixed32,
/// fixed64), not varint — matches PutFixed32/PutFixed64 in
/// SstFileWriterPropertiesCollector::Finish.
const KEY_EXTERNAL_VERSION: &str = "rocksdb.external_sst_file.version";
const KEY_EXTERNAL_GLOBAL_SEQNO: &str = "rocksdb.external_sst_file.global_seqno";

/// version=2 is what current SstFileWriter stamps (table/sst_file_writer.cc:
/// `new SstFileWriterPropertiesCollectorFactory(2 /* version */, ...)`);
/// version 2 implies "this file carries a global seqno property that the
/// ingesting DB may patch/override at read time".
const EXTERNAL_SST_VERSION: u32 = 2;

/// RocksDB's `BlockBasedTableOptions::IndexType::kBinarySearch` enum value —
/// the only index type this writer produces (matches `IndexBlockBuilder`).
const INDEX_TYPE_BINARY_SEARCH: u32 = 0;

/// Bytewise comparator name RocksDB's default comparator reports via
/// `Comparator::Name()` (util/comparator.cc `kClassName()`).
pub const BYTEWISE_COMPARATOR_NAME: &str = "leveldb.BytewiseComparator";

#[derive(Debug, Clone, Default)]
pub struct PropertiesInput {
    pub num_entries: u64,
    pub raw_key_size: u64,
    pub raw_value_size: u64,
    pub data_size: u64,
    pub index_size: u64,
    pub comparator_name: String,
    /// Column family this SST belongs to. RocksDB's `ldb repair` (and
    /// normal DB open) rejects files whose `column_family_name` property
    /// doesn't match the CF it's being placed into (db/repair.cc: "Table
    /// #N: inconsistent column family name") — confirmed directly by
    /// swapping a real DB's SSTs for a compactor-written file and running
    /// `ldb repair` without this set: it dropped the file entirely
    /// ("recovered 0 files"). Default DB / default CF is id=0, name=
    /// "default" (db/column_family.h kDefaultColumnFamilyName); override
    /// for non-default column families.
    pub column_family_id: u64,
    pub column_family_name: String,
    /// When `Some`, stamps the SstFileWriter-compatible properties
    /// (external_sst_file.version=2, external_sst_file.global_seqno=<this
    /// value>) so the output is accepted by `ldb ingest_extern_sst` /
    /// `DB::IngestExternalFile`. The seqno value itself is a placeholder —
    /// RocksDB's ingest path assigns and applies the real sequence number
    /// at ingest time (see db/external_sst_file_ingestion_job.cc). REQUIRES
    /// (enforced by SstFileWriter, not re-checked here): every entry's
    /// internal key must use seq=0, and user keys must be unique and
    /// strictly ascending (no MVCC versions) — matches compactor::compaction
    /// output shape naturally, since it's already deduped to one version
    /// per key. When `None`, produces a plain internal SST (not
    /// ingest-compatible), which is what compactor normally writes.
    pub external_sst_global_seqno: Option<u64>,
}

/// Builds the properties block. Not restart-encoded in practice (RocksDB
/// uses `block_restart_interval = INT32_MAX`, i.e. no restarts beyond the
/// first); we use `BlockBuilder::new(usize::MAX)` to get the same effect —
/// every key shares nothing with the previous one anyway once sorted
/// lexicographically by unrelated property names, so restarts buy nothing.
pub fn build_properties_block(props: &PropertiesInput) -> Vec<u8> {
    let mut entries: Vec<(String, Vec<u8>)> = Vec::new();

    entries.push((
        KEY_INDEX_TYPE.to_string(),
        INDEX_TYPE_BINARY_SEARCH.to_le_bytes().to_vec(),
    ));
    entries.push((
        KEY_NUM_ENTRIES.to_string(),
        varint64_bytes(props.num_entries),
    ));
    entries.push((
        KEY_RAW_KEY_SIZE.to_string(),
        varint64_bytes(props.raw_key_size),
    ));
    entries.push((
        KEY_RAW_VALUE_SIZE.to_string(),
        varint64_bytes(props.raw_value_size),
    ));
    entries.push((KEY_DATA_SIZE.to_string(), varint64_bytes(props.data_size)));
    entries.push((KEY_INDEX_SIZE.to_string(), varint64_bytes(props.index_size)));
    // Our index keys are full internal keys (user_key + seq/type suffix),
    // never trimmed to user-key-only, so index_key_is_user_key = 0 (false).
    entries.push((KEY_INDEX_KEY_IS_USER_KEY.to_string(), varint64_bytes(0)));
    // IndexBlockBuilder always delta-encodes non-restart entries against the
    // previous handle's size, matching table/format.cc IndexValue's
    // delta-encoded path — so this must be 1 (true), or the real reader
    // parses our index block with the wrong (3-varint, full-value) layout.
    entries.push((
        KEY_INDEX_VALUE_IS_DELTA_ENCODED.to_string(),
        varint64_bytes(1),
    ));
    if !props.comparator_name.is_empty() {
        entries.push((
            KEY_COMPARATOR.to_string(),
            props.comparator_name.as_bytes().to_vec(),
        ));
    }
    entries.push((
        KEY_COLUMN_FAMILY_ID.to_string(),
        varint64_bytes(props.column_family_id),
    ));
    if !props.column_family_name.is_empty() {
        entries.push((
            KEY_COLUMN_FAMILY_NAME.to_string(),
            props.column_family_name.as_bytes().to_vec(),
        ));
    }
    if let Some(global_seqno) = props.external_sst_global_seqno {
        entries.push((
            KEY_EXTERNAL_VERSION.to_string(),
            EXTERNAL_SST_VERSION.to_le_bytes().to_vec(),
        ));
        entries.push((
            KEY_EXTERNAL_GLOBAL_SEQNO.to_string(),
            global_seqno.to_le_bytes().to_vec(),
        ));
    }

    entries.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));

    let mut builder = BlockBuilder::new(usize::MAX);
    for (k, v) in &entries {
        builder.add(k.as_bytes(), v);
    }
    builder.finish()
}

fn varint64_bytes(v: u64) -> Vec<u8> {
    let mut out = Vec::new();
    put_varint64(&mut out, v);
    out
}

/// Builds the metaindex block: maps meta-block names ("rocksdb.properties",
/// "rocksdb.index", ...) to their `BlockHandle`s. RocksDB uses restart
/// interval 1 for this block (meta_blocks.cc); we match that, though for a
/// handful of entries it makes no measurable difference.
pub fn build_metaindex_block(entries: &[(&str, BlockHandle)]) -> Vec<u8> {
    let mut sorted: Vec<&(&str, BlockHandle)> = entries.iter().collect();
    sorted.sort_by_key(|(name, _)| name.as_bytes());

    let mut builder = BlockBuilder::new(1);
    for (name, handle) in sorted {
        let mut value = Vec::new();
        put_varint64(&mut value, handle.offset);
        put_varint64(&mut value, handle.size);
        builder.add(name.as_bytes(), &value);
    }
    builder.finish()
}
