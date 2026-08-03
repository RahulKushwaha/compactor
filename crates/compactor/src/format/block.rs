use crate::format::footer::{BlockHandle, ChecksumType};
use crate::format::varint::{get_varint32, get_varint64};
use std::borrow::Cow;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionType {
    NoCompression,
    Snappy,
    Zlib,
    BZip2,
    Lz4,
    Lz4Hc,
    Xpress,
    Zstd,
}

impl CompressionType {
    fn from_byte(b: u8) -> Result<Self, String> {
        match b {
            0x00 => Ok(CompressionType::NoCompression),
            0x01 => Ok(CompressionType::Snappy),
            0x02 => Ok(CompressionType::Zlib),
            0x03 => Ok(CompressionType::BZip2),
            0x04 => Ok(CompressionType::Lz4),
            0x05 => Ok(CompressionType::Lz4Hc),
            0x06 => Ok(CompressionType::Xpress),
            0x07 => Ok(CompressionType::Zstd),
            other => Err(format!(
                "unsupported/unknown compression type byte {}",
                other
            )),
        }
    }

    pub fn to_byte(self) -> u8 {
        match self {
            CompressionType::NoCompression => 0x00,
            CompressionType::Snappy => 0x01,
            CompressionType::Zlib => 0x02,
            CompressionType::BZip2 => 0x03,
            CompressionType::Lz4 => 0x04,
            CompressionType::Lz4Hc => 0x05,
            CompressionType::Xpress => 0x06,
            CompressionType::Zstd => 0x07,
        }
    }
}

pub const BLOCK_TRAILER_SIZE: usize = 5;

/// Number of bytes to read from the file for this block, including the
/// trailer (1-byte compression type + 4-byte checksum). Callers with I/O
/// access (compactor-sst) use this to size their `read_at` call before
/// handing the bytes to `decode_block_contents`.
pub fn block_read_len(handle: BlockHandle) -> usize {
    handle.size as usize + BLOCK_TRAILER_SIZE
}

/// Assembles the on-disk bytes for a block: raw contents (uncompressed only,
/// for now — see `BlockBuilder`/`IndexBlockBuilder` output) plus a trailer
/// (1-byte compression type + 4-byte checksum, matching what
/// `decode_block_contents` expects to read back). `offset` is the block's
/// eventual position in the file, needed for the format_version >= 6
/// context-checksum modifier (see table/format.h ChecksumModifierForContext).
///
/// Compression is intentionally not supported yet: this always writes
/// `CompressionType::NoCompression`. Wiring up snappy/lz4/zstd on the write
/// path is a follow-up once the uncompressed path round-trips correctly.
pub fn encode_block_with_trailer(
    mut raw_contents: Vec<u8>,
    offset: u64,
    checksum_type: ChecksumType,
    base_context_checksum: u32,
) -> Vec<u8> {
    let comp_byte = CompressionType::NoCompression.to_byte();
    let mut checksum = compute_checksum_with_last_byte(checksum_type, &raw_contents, comp_byte);
    checksum = checksum.wrapping_add(crate::format::footer::checksum_modifier_for_context(
        base_context_checksum,
        offset,
    ));
    raw_contents.push(comp_byte);
    raw_contents.extend_from_slice(&checksum.to_le_bytes());
    raw_contents
}

/// Decodes and decompresses a block already read from disk. `raw_with_trailer`
/// must be exactly `block_read_len(handle)` bytes: the raw (possibly
/// compressed) block contents followed by the 5-byte trailer. Verifies the
/// trailer checksum (see table/block_based/block_based_table_builder.cc
/// WriteMaybeCompressedBlock).
///
/// Returns `Cow::Borrowed` when the block is stored uncompressed (no extra
/// copy); `Cow::Owned` when decompression allocated a new buffer.
pub fn decode_block_contents(
    raw_with_trailer: Vec<u8>,
    handle: BlockHandle,
    checksum_type: ChecksumType,
    base_context_checksum: u32,
) -> Result<Vec<u8>, String> {
    let raw_len = handle.size as usize;
    if raw_with_trailer.len() != raw_len + BLOCK_TRAILER_SIZE {
        return Err(format!(
            "block buffer wrong size: expected {} got {}",
            raw_len + BLOCK_TRAILER_SIZE,
            raw_with_trailer.len()
        ));
    }
    let raw = &raw_with_trailer[..raw_len];
    let trailer = &raw_with_trailer[raw_len..raw_len + BLOCK_TRAILER_SIZE];
    let comp_byte = trailer[0];
    let stored_checksum = u32::from_le_bytes(trailer[1..5].try_into().unwrap());

    let mut computed = compute_checksum_with_last_byte(checksum_type, raw, comp_byte);
    computed = computed.wrapping_add(crate::format::footer::checksum_modifier_for_context(
        base_context_checksum,
        handle.offset,
    ));
    if checksum_type != ChecksumType::NoChecksum && computed != stored_checksum {
        return Err(format!(
            "block checksum mismatch at offset {}: stored=0x{:08x} computed=0x{:08x}",
            handle.offset, stored_checksum, computed
        ));
    }

    let compression = CompressionType::from_byte(comp_byte)?;
    match decompress(compression, raw)? {
        Cow::Borrowed(_) => {
            // Raw bytes are uncompressed: trim the trailer off the buffer we
            // already own instead of copying, by truncating in place.
            let mut owned = raw_with_trailer;
            owned.truncate(raw_len);
            Ok(owned)
        }
        Cow::Owned(v) => Ok(v),
    }
}

pub(crate) fn compute_checksum_with_last_byte(ty: ChecksumType, data: &[u8], last_byte: u8) -> u32 {
    match ty {
        ChecksumType::NoChecksum => 0,
        ChecksumType::Crc32c => {
            let crc = crc32c::crc32c(data);
            let crc = crc32c::crc32c_append(crc, &[last_byte]);
            mask_crc32c(crc)
        }
        ChecksumType::XxHash => {
            let mut state = xxhash_rust::xxh32::Xxh32::new(0);
            state.update(data);
            state.update(&[last_byte]);
            state.digest()
        }
        ChecksumType::XxHash64 => {
            let mut state = xxhash_rust::xxh64::Xxh64::new(0);
            state.update(data);
            state.update(&[last_byte]);
            (state.digest() & 0xffff_ffff) as u32
        }
        ChecksumType::XxH3 => {
            let v = (xxhash_rust::xxh3::xxh3_64(data) & 0xffff_ffff) as u32;
            modify_checksum_for_last_byte(v, last_byte)
        }
    }
}

// See table/format.cc ModifyChecksumForLastByte: used by XXH3's custom
// last-byte handling (avoids re-hashing the whole block for a 1-byte change).
pub(crate) fn modify_checksum_for_last_byte(checksum: u32, last_byte: u8) -> u32 {
    const RANDOM_PRIME: u32 = 0x6b90_83d9;
    checksum ^ (last_byte as u32).wrapping_mul(RANDOM_PRIME)
}

// RocksDB masks the raw CRC32C to avoid stripe-related peculiarities.
// See util/crc32c.h Mask(): kMaskDelta = 0xa282ead8.
fn mask_crc32c(crc: u32) -> u32 {
    const MASK_DELTA: u32 = 0xa282_ead8;
    crc.rotate_right(15).wrapping_add(MASK_DELTA)
}

fn decompress(ty: CompressionType, data: &[u8]) -> Result<Cow<'_, [u8]>, String> {
    match ty {
        CompressionType::NoCompression => Ok(Cow::Borrowed(data)),
        CompressionType::Snappy => {
            // Snappy is the one built-in codec with no leading uncompressed-size
            // varint (its own format self-describes length) — see
            // util/compression.cc Decompressor::ExtractUncompressedSize doc
            // comment: "Standard format... except Snappy".
            let mut decoder = snap::raw::Decoder::new();
            decoder
                .decompress_vec(data)
                .map(Cow::Owned)
                .map_err(|e| format!("snappy decompress failed: {}", e))
        }
        CompressionType::Lz4 | CompressionType::Lz4Hc => {
            // Both LZ4 variants share one decompressor (HC only affects the
            // encoder's compression effort, not the wire format). Payload is
            // varint64(uncompressed_size) ++ lz4-compressed bytes (see
            // util/compression.cc Decompressor::ExtractUncompressedSize +
            // LZ4_DecompressBlock: LZ4_decompress_safe_continue against a
            // fresh stream, no dictionary in compactor's simple case).
            let (uncompressed_size, payload) = split_uncompressed_size_prefix(data)?;
            lz4_flex::block::decompress(payload, uncompressed_size)
                .map(Cow::Owned)
                .map_err(|e| format!("lz4 decompress failed: {}", e))
        }
        CompressionType::Zstd => {
            // Same varint64-prefixed framing as LZ4 (compress_format_version=2).
            // zstd::bulk::decompress needs a capacity hint; we already know
            // the exact size from the prefix, so allocate exactly that.
            let (uncompressed_size, payload) = split_uncompressed_size_prefix(data)?;
            zstd::bulk::decompress(payload, uncompressed_size)
                .map(Cow::Owned)
                .map_err(|e| format!("zstd decompress failed: {}", e))
        }
        other => Err(format!("compression type {:?} not yet supported", other)),
    }
}

/// Strips the leading varint64 uncompressed-size prefix RocksDB writes
/// before LZ4/ZSTD (and other non-Snappy) compressed block payloads
/// (compress_format_version=2; see util/compression.cc
/// Decompressor::ExtractUncompressedSize). Returns (size, remaining bytes).
fn split_uncompressed_size_prefix(data: &[u8]) -> Result<(usize, &[u8]), String> {
    let mut pos = 0usize;
    let size = get_varint64(data, &mut pos)
        .ok_or_else(|| "failed to read uncompressed-size prefix".to_string())?;
    Ok((size as usize, &data[pos..]))
}

/// A parsed block (data or index): the raw uncompressed contents plus the
/// restart point offsets from the trailer, per table/block_based/block_builder.cc:
///   entries...
///   restarts: uint32[num_restarts]
///   num_restarts: uint32
pub struct ParsedBlock {
    pub contents: Vec<u8>,
    pub restarts: Vec<u32>,
}

pub fn parse_block(contents: Vec<u8>) -> Result<ParsedBlock, String> {
    if contents.len() < 4 {
        return Err("block too small to contain restart trailer".to_string());
    }
    let n = contents.len();
    let num_restarts = u32::from_le_bytes(contents[n - 4..n].try_into().unwrap()) as usize;
    let restarts_len = num_restarts * 4;
    if n < 4 + restarts_len {
        return Err("block restart array out of bounds".to_string());
    }
    let restarts_start = n - 4 - restarts_len;
    let mut restarts = Vec::with_capacity(num_restarts);
    for i in 0..num_restarts {
        let off = restarts_start + i * 4;
        restarts.push(u32::from_le_bytes(
            contents[off..off + 4].try_into().unwrap(),
        ));
    }
    Ok(ParsedBlock { contents, restarts })
}

/// Visits every (key, value) entry in a data block in order, applying
/// prefix-shared delta decoding between restart points. See entry format
/// comment in table/block_based/block_builder.cc:
///   shared_bytes: varint32
///   unshared_bytes: varint32
///   value_length: varint32
///   key_delta: char[unshared_bytes]
///   value: char[value_length]
///
/// Zero-alloc: `key_scratch` is a caller-owned reusable buffer for
/// reconstructing delta-encoded keys (mirrors RocksDB's IterKey). Only the
/// shared+delta bytes get copied into it; `value` is always a borrow
/// straight into `block.contents` (values are never delta-encoded in data
/// blocks). `f` receives `(full_key, value)` per entry; no per-entry Vec is
/// allocated.
pub fn for_each_entry(
    block: &ParsedBlock,
    key_scratch: &mut Vec<u8>,
    mut f: impl FnMut(&[u8], &[u8]),
) -> Result<(), String> {
    let restarts_len = block.restarts.len() * 4;
    let entries_end = block.contents.len() - 4 - restarts_len;
    let data = &block.contents[..entries_end];

    key_scratch.clear();
    let mut pos = 0usize;

    while pos < data.len() {
        let shared = get_varint32(data, &mut pos).ok_or("truncated entry: shared_bytes")?;
        let unshared = get_varint32(data, &mut pos).ok_or("truncated entry: unshared_bytes")?;
        let value_len = get_varint32(data, &mut pos).ok_or("truncated entry: value_length")?;

        let shared = shared as usize;
        let unshared = unshared as usize;
        let value_len = value_len as usize;

        if shared > key_scratch.len() {
            return Err("corrupt entry: shared_bytes exceeds last key length".to_string());
        }
        let key_delta = data
            .get(pos..pos + unshared)
            .ok_or("truncated entry: key_delta")?;
        pos += unshared;

        key_scratch.truncate(shared);
        key_scratch.extend_from_slice(key_delta);

        let value = data
            .get(pos..pos + value_len)
            .ok_or("truncated entry: value")?;
        pos += value_len;

        f(key_scratch, value);
    }

    Ok(())
}

/// Visits entries in an index block built with `index_value_is_delta_encoded`
/// (the default for format_version >= 4). Key entries use the 2-varint
/// (shared, non_shared) header with no value_length field (DecodeEntryV4);
/// the value is a self-describing BlockHandle, optionally delta-encoded
/// against the previous entry's handle within the same restart interval
/// (see table/block_based/block.cc IndexBlockIter, table/format.cc
/// IndexValue::DecodeFrom).
///
/// Zero-alloc in the same sense as `for_each_entry`: `key_scratch` is reused
/// across entries, no per-entry Vec.
pub fn for_each_index_entry(
    block: &ParsedBlock,
    key_scratch: &mut Vec<u8>,
    mut f: impl FnMut(&[u8], BlockHandle),
) -> Result<(), String> {
    let restarts_len = block.restarts.len() * 4;
    let entries_end = block.contents.len() - 4 - restarts_len;
    let data = &block.contents[..entries_end];

    let mut restart_offsets: Vec<usize> = block.restarts.iter().map(|&r| r as usize).collect();
    restart_offsets.sort_unstable();

    key_scratch.clear();
    let mut pos = 0usize;
    let mut last_handle: Option<BlockHandle> = None;

    while pos < data.len() {
        let at_restart = restart_offsets.binary_search(&pos).is_ok();

        let shared = get_varint32(data, &mut pos).ok_or("truncated index entry: shared_bytes")?;
        let unshared =
            get_varint32(data, &mut pos).ok_or("truncated index entry: non_shared_bytes")?;

        let shared = shared as usize;
        let unshared = unshared as usize;
        if shared > key_scratch.len() {
            return Err("corrupt index entry: shared_bytes exceeds last key length".to_string());
        }
        let key_delta = data
            .get(pos..pos + unshared)
            .ok_or("truncated index entry: key_delta")?;
        pos += unshared;

        key_scratch.truncate(shared);
        key_scratch.extend_from_slice(key_delta);

        // is_shared in the C++ reader means "this key was delta-encoded",
        // i.e. shared != 0. First entry after a restart always has shared == 0.
        let is_shared = shared != 0;
        let handle = if at_restart || !is_shared {
            let offset = get_varint64(data, &mut pos).ok_or("truncated index value: offset")?;
            let size = get_varint64(data, &mut pos).ok_or("truncated index value: size")?;
            BlockHandle { offset, size }
        } else {
            let prev = last_handle.ok_or("delta-encoded index value with no previous handle")?;
            let delta =
                get_varsigned64(data, &mut pos).ok_or("truncated index value: size delta")?;
            let new_size = (prev.size as i64 + delta) as u64;
            BlockHandle {
                offset: prev.offset + prev.size + BLOCK_TRAILER_SIZE as u64,
                size: new_size,
            }
        };

        last_handle = Some(handle);
        f(key_scratch, handle);
    }

    Ok(())
}

/// Zigzag varint64 decode, per util/coding.h GetVarsignedint64.
fn get_varsigned64(buf: &[u8], pos: &mut usize) -> Option<i64> {
    let u = get_varint64(buf, pos)?;
    Some(((u >> 1) as i64) ^ -((u & 1) as i64))
}
