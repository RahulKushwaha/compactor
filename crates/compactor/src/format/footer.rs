// Mirrors table/format.h / table/format.cc (Footer::DecodeFrom) in RocksDB.

pub const BLOCK_BASED_TABLE_MAGIC: u64 = 0x88e2_41b7_85f4_cff7;

const MAGIC_NUMBER_LEN: usize = 8;
const BLOCK_HANDLE_MAX_ENCODED_LEN: usize = 20; // 2 * kMaxVarint64Length
const NEW_VERSIONS_ENCODED_LEN: usize = 1 + 2 * BLOCK_HANDLE_MAX_ENCODED_LEN + 4 + MAGIC_NUMBER_LEN;
pub const MAX_FOOTER_ENCODED_LEN: usize = NEW_VERSIONS_ENCODED_LEN;
const EXTENDED_MAGIC: [u8; 4] = [0x3e, 0x00, 0x7a, 0x00];

#[derive(Debug, Clone, Copy)]
pub struct BlockHandle {
    pub offset: u64,
    pub size: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChecksumType {
    NoChecksum,
    Crc32c,
    XxHash,
    XxHash64,
    XxH3,
}

impl ChecksumType {
    fn from_byte(b: u8) -> Option<Self> {
        match b {
            0x0 => Some(ChecksumType::NoChecksum),
            0x1 => Some(ChecksumType::Crc32c),
            0x2 => Some(ChecksumType::XxHash),
            0x3 => Some(ChecksumType::XxHash64),
            0x4 => Some(ChecksumType::XxH3),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct Footer {
    pub table_magic_number: u64,
    pub format_version: u32,
    pub checksum_type: ChecksumType,
    pub metaindex_handle: BlockHandle,
    /// Only populated for format_version < 6; for >= 6 the index handle
    /// lives in the metaindex block instead.
    pub index_handle: Option<BlockHandle>,
    pub block_trailer_size: usize,
    /// Nonzero for format_version >= 6 ("context checksum"); see
    /// ChecksumModifierForContext in table/format.h. Zero means the feature
    /// is disabled (format_version < 6, or explicitly 0).
    pub base_context_checksum: u32,
}

/// Mirrors table/format.h ChecksumModifierForContext: derives a per-block
/// checksum modifier from the file's base_context_checksum and the block's
/// offset, so that colocated/moved blocks with identical bytes still get
/// distinct checksums. Returns 0 (no-op) when base_context_checksum == 0.
pub fn checksum_modifier_for_context(base_context_checksum: u32, offset: u64) -> u32 {
    if base_context_checksum == 0 {
        return 0;
    }
    let lower = offset as u32;
    let upper = (offset >> 32) as u32;
    base_context_checksum ^ lower.wrapping_add(upper)
}

fn decode_varint64(buf: &[u8], pos: &mut usize) -> Option<u64> {
    crate::format::varint::get_varint64(buf, pos)
}

fn decode_block_handle(buf: &[u8], pos: &mut usize) -> Option<BlockHandle> {
    let offset = decode_varint64(buf, pos)?;
    let size = decode_varint64(buf, pos)?;
    Some(BlockHandle { offset, size })
}

/// Encodes a format_version >= 6 footer (the only version this writer
/// produces). Mirrors table/format.cc FooterBuilder::Build's format_version
/// >= 6 branch: part1 (checksum type byte) + part2 (extended magic, footer
/// checksum, base_context_checksum, metaindex size, 24 bytes reserved
/// padding) + part3 (format_version, magic number). The footer's own
/// checksum covers all of part1..part3 with the checksum field itself
/// zeroed, plus the context-checksum modifier for the footer's own offset.
///
/// `footer_offset` is where this footer will land in the file (i.e. the
/// file size before appending the footer) — needed for the context-checksum
/// modifier, same as any other block's offset.
pub fn encode_footer_v6_plus(
    format_version: u32,
    checksum_type: ChecksumType,
    metaindex_handle: BlockHandle,
    base_context_checksum: u32,
    footer_offset: u64,
) -> Vec<u8> {
    assert!(format_version >= 6);
    let checksum_byte = match checksum_type {
        ChecksumType::NoChecksum => 0x0,
        ChecksumType::Crc32c => 0x1,
        ChecksumType::XxHash => 0x2,
        ChecksumType::XxHash64 => 0x3,
        ChecksumType::XxH3 => 0x4,
    };

    let mut buf = Vec::with_capacity(NEW_VERSIONS_ENCODED_LEN);
    buf.push(checksum_byte);
    buf.extend_from_slice(&EXTENDED_MAGIC);
    let checksum_field_pos = buf.len();
    buf.extend_from_slice(&0u32.to_le_bytes()); // checksum placeholder
    buf.extend_from_slice(&base_context_checksum.to_le_bytes());
    let metaindex_size: u32 = metaindex_handle
        .size
        .try_into()
        .expect("metaindex block size > 4GB");
    buf.extend_from_slice(&metaindex_size.to_le_bytes());
    buf.extend_from_slice(&[0u8; 24]); // reserved padding (16 unchecked + 8 checked)
    buf.extend_from_slice(&format_version.to_le_bytes());
    buf.extend_from_slice(&BLOCK_BASED_TABLE_MAGIC.to_le_bytes());
    debug_assert_eq!(buf.len(), NEW_VERSIONS_ENCODED_LEN);

    // Checksum covers the whole footer buffer (with the checksum field
    // itself zeroed), per FooterBuilder::Build. This uses the plain
    // ComputeBuiltinChecksum, not the WithLastByte variant used for blocks —
    // the footer has no separate trailing "compression type" byte to fold in.
    let mut checksum = compute_checksum_plain(checksum_type, &buf);
    checksum = checksum.wrapping_add(checksum_modifier_for_context(
        base_context_checksum,
        footer_offset,
    ));
    buf[checksum_field_pos..checksum_field_pos + 4].copy_from_slice(&checksum.to_le_bytes());

    buf
}

/// Plain (non-"WithLastByte") checksum, per table/format.cc
/// ComputeBuiltinChecksum — used only for the footer, which checksums its
/// own fixed-size buffer directly rather than a block+trailer-byte pair.
fn compute_checksum_plain(ty: ChecksumType, data: &[u8]) -> u32 {
    match ty {
        ChecksumType::NoChecksum => 0,
        ChecksumType::Crc32c => crc32c::crc32c(data)
            .rotate_right(15)
            .wrapping_add(0xa282_ead8),
        ChecksumType::XxHash => xxhash_rust::xxh32::xxh32(data, 0),
        ChecksumType::XxHash64 => (xxhash_rust::xxh64::xxh64(data, 0) & 0xffff_ffff) as u32,
        ChecksumType::XxH3 => {
            // See table/format.cc ComputeBuiltinChecksum's kXXH3 case: even
            // the "plain" (non-WithLastByte) variant special-cases the
            // buffer's own last byte, hashing only data[..len-1] and folding
            // in data[len-1] via ModifyChecksumForLastByte. data_size == 0
            // is defined as 0 but never occurs here (footer is fixed-size).
            if data.is_empty() {
                0
            } else {
                let (head, last) = data.split_at(data.len() - 1);
                let v = (xxhash_rust::xxh3::xxh3_64(head) & 0xffff_ffff) as u32;
                crate::format::block::modify_checksum_for_last_byte(v, last[0])
            }
        }
    }
}

/// Minimum number of tail bytes a caller must read from the file before
/// calling `decode_footer` (exactly `NEW_VERSIONS_ENCODED_LEN`, exposed so
/// I/O-capable crates know how much to fetch without duplicating the const).
pub const FOOTER_TAIL_LEN: usize = NEW_VERSIONS_ENCODED_LEN;

/// Decode a footer from exactly the last `NEW_VERSIONS_ENCODED_LEN` bytes of
/// an SST file (the "tail"). `file_size` is the total file size, used to
/// compute the absolute footer offset for format_version >= 6 checksum
/// verification.
pub fn decode_footer(tail: &[u8], file_size: u64) -> Result<Footer, String> {
    if tail.len() < NEW_VERSIONS_ENCODED_LEN {
        return Err("footer tail shorter than minimum encoded length".to_string());
    }
    let magic_pos = tail.len() - MAGIC_NUMBER_LEN;
    let magic = u64::from_le_bytes(tail[magic_pos..magic_pos + 8].try_into().unwrap());

    if magic != BLOCK_BASED_TABLE_MAGIC {
        return Err(format!(
            "unsupported/unknown table magic number: 0x{:016x} (only block-based tables supported)",
            magic
        ));
    }
    let block_trailer_size = 5usize; // kBlockTrailerSize for block-based tables

    let format_version = u32::from_le_bytes(tail[magic_pos - 4..magic_pos].try_into().unwrap());
    if !(2..=7).contains(&format_version) {
        return Err(format!("unsupported format_version {}", format_version));
    }

    let footer_offset = file_size - NEW_VERSIONS_ENCODED_LEN as u64;

    let checksum_byte = tail[0];
    let checksum_type = ChecksumType::from_byte(checksum_byte)
        .ok_or_else(|| format!("unsupported checksum type byte {}", checksum_byte))?;

    let mut pos = 1usize; // consumed checksum type byte

    if format_version >= 6 {
        let ext_magic = &tail[pos..pos + 4];
        if ext_magic != EXTENDED_MAGIC {
            return Err(format!("bad extended magic number: {:02x?}", ext_magic));
        }
        pos += 4;

        let stored_checksum = u32::from_le_bytes(tail[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let base_context_checksum = u32::from_le_bytes(tail[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let metaindex_size = u32::from_le_bytes(tail[pos..pos + 4].try_into().unwrap()) as u64;
        pos += 4;

        {
            let mut zeroed = tail.to_vec();
            zeroed[5..9].fill(0); // checksum field is at part1(1) + ext_magic(4) = offset 5
            let mut computed = compute_checksum_plain(checksum_type, &zeroed);
            computed = computed.wrapping_add(checksum_modifier_for_context(
                base_context_checksum,
                footer_offset,
            ));
            if checksum_type != ChecksumType::NoChecksum && computed != stored_checksum {
                return Err(format!(
                    "footer checksum mismatch at offset {}: stored=0x{:08x} computed=0x{:08x}",
                    footer_offset, stored_checksum, computed
                ));
            }
        }

        let metaindex_end = footer_offset - block_trailer_size as u64;
        let metaindex_handle = BlockHandle {
            offset: metaindex_end - metaindex_size,
            size: metaindex_size,
        };

        // 16 bytes unchecked reserved padding + 8 bytes checked reserved padding
        pos += 16;
        let reserved = u64::from_le_bytes(tail[pos..pos + 8].try_into().unwrap());
        pos += 8;
        if reserved != 0 {
            return Err("file uses a future feature not supported by this reader".to_string());
        }
        debug_assert_eq!(pos, magic_pos - 4);

        Ok(Footer {
            table_magic_number: magic,
            format_version,
            checksum_type,
            metaindex_handle,
            index_handle: None,
            block_trailer_size,
            base_context_checksum,
        })
    } else {
        let part2 = &tail[pos..magic_pos - 4];
        let mut p2pos = 0usize;
        let metaindex_handle =
            decode_block_handle(part2, &mut p2pos).ok_or("failed to decode metaindex handle")?;
        let index_handle =
            decode_block_handle(part2, &mut p2pos).ok_or("failed to decode index handle")?;

        Ok(Footer {
            table_magic_number: magic,
            format_version,
            checksum_type,
            metaindex_handle,
            index_handle: Some(index_handle),
            block_trailer_size,
            base_context_checksum: 0,
        })
    }
}
