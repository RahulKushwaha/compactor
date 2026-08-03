//! Async glue: fetches SST bytes off disk via `crate::io::FileReader` and
//! hands them to `crate::format`'s pure decoders. This is the module that
//! combines I/O with format parsing.

use crate::format::block::{self, ParsedBlock};
use crate::format::footer::{self, BlockHandle, Footer};
use crate::format::varint;
use crate::io::FileReader;

pub struct SstReader<R: FileReader> {
    reader: R,
    pub footer: Footer,
    metaindex: ParsedBlock,
    index: ParsedBlock,
}

const INDEX_BLOCK_META_KEY: &str = "rocksdb.index";

impl<R: FileReader> SstReader<R> {
    /// Opens an SST file: one async read for the footer tail, one for the
    /// metaindex block, one for the top-level index block. No whole-file
    /// slurp at any point.
    pub async fn open(reader: R) -> Result<Self, String> {
        let ft = read_footer(&reader).await?;

        let metaindex_raw = read_block_contents(
            &reader,
            ft.metaindex_handle,
            ft.checksum_type,
            ft.base_context_checksum,
        )
        .await?;
        let metaindex = block::parse_block(metaindex_raw)?;

        let index_handle = match ft.index_handle {
            Some(h) => h,
            None => find_meta_handle(&metaindex, INDEX_BLOCK_META_KEY)?
                .ok_or_else(|| "index block handle missing from metaindex".to_string())?,
        };

        let index_raw = read_block_contents(
            &reader,
            index_handle,
            ft.checksum_type,
            ft.base_context_checksum,
        )
        .await?;
        let index = block::parse_block(index_raw)?;

        Ok(SstReader {
            reader,
            footer: ft,
            metaindex,
            index,
        })
    }

    /// Find the handle of a named meta block (e.g. "rocksdb.properties").
    pub fn find_meta_block_handle(&self, name: &str) -> Result<Option<BlockHandle>, String> {
        find_meta_handle(&self.metaindex, name)
    }

    /// Visits every (key, value) entry across all data blocks, in file order
    /// (which for a valid SST is also key order). Keys are still full
    /// internal keys (user_key + 8-byte seq/type suffix); see
    /// `crate::format::internal_key` to split.
    ///
    /// Data blocks are fetched one at a time (sequential async reads); each
    /// block's entries are visited via `block::for_each_entry` with a single
    /// reused key-reconstruction buffer, so no per-entry or per-block Vec
    /// allocation happens beyond the block read itself.
    pub async fn for_each_entry(&self, mut f: impl FnMut(&[u8], &[u8])) -> Result<(), String> {
        let mut index_key_scratch = Vec::new();
        let mut data_key_scratch = Vec::new();
        let mut handles = Vec::new();

        // Collect data block handles first (index block is small and already
        // in memory); this keeps the borrow of `index_key_scratch` from
        // overlapping with the `.await` below.
        block::for_each_index_entry(&self.index, &mut index_key_scratch, |_key, handle| {
            handles.push(handle);
        })?;

        for handle in handles {
            let raw = read_block_contents(
                &self.reader,
                handle,
                self.footer.checksum_type,
                self.footer.base_context_checksum,
            )
            .await?;
            let parsed = block::parse_block(raw)?;
            block::for_each_entry(&parsed, &mut data_key_scratch, &mut f)?;
        }

        Ok(())
    }
}

/// Async-read the footer of an open SST file: one targeted `read_at` for the
/// tail, no whole-file slurp. Thin wrapper around
/// `crate::format::footer::decode_footer`.
async fn read_footer(reader: &impl FileReader) -> Result<Footer, String> {
    let file_size = reader.file_size();
    if (file_size as usize) < footer::FOOTER_TAIL_LEN {
        return Err("file too short to contain a footer".to_string());
    }
    let tail_offset = file_size - footer::FOOTER_TAIL_LEN as u64;
    let tail = reader
        .read_at(tail_offset, footer::FOOTER_TAIL_LEN)
        .await
        .map_err(|e| format!("failed to read footer tail: {}", e))?;
    footer::decode_footer(&tail, file_size)
}

/// Async-reads the block pointed to by `handle` (raw bytes + trailer, one
/// `read_at` call) and hands it to `crate::format::block::decode_block_contents`
/// for checksum verification + decompression.
async fn read_block_contents(
    reader: &impl FileReader,
    handle: BlockHandle,
    checksum_type: crate::format::footer::ChecksumType,
    base_context_checksum: u32,
) -> Result<Vec<u8>, String> {
    let raw_with_trailer = reader
        .read_at(handle.offset, block::block_read_len(handle))
        .await
        .map_err(|e| format!("failed to read block at offset {}: {}", handle.offset, e))?;
    block::decode_block_contents(
        raw_with_trailer,
        handle,
        checksum_type,
        base_context_checksum,
    )
}

/// Metaindex entries use the standard data-block entry encoding (3-varint
/// header incl. value_length; see MetaBlockIter in table/block_based/block.h),
/// and values are themselves plain BlockHandle encodings (2 varints, no
/// first_key suffix; see meta_blocks.cc).
fn decode_meta_value_handle(value: &[u8]) -> Option<BlockHandle> {
    let mut pos = 0usize;
    let offset = varint::get_varint64(value, &mut pos)?;
    let size = varint::get_varint64(value, &mut pos)?;
    Some(BlockHandle { offset, size })
}

fn find_meta_handle(metaindex: &ParsedBlock, name: &str) -> Result<Option<BlockHandle>, String> {
    let mut scratch = Vec::new();
    let mut found = None;
    block::for_each_entry(metaindex, &mut scratch, |key, value| {
        if found.is_none() && key == name.as_bytes() {
            found = decode_meta_value_handle(value);
        }
    })?;
    Ok(found)
}
