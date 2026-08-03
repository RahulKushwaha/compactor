//! Pure, synchronous parsing of the RocksDB block-based SST file format.
//! No I/O, no async: every function here takes bytes already in memory and
//! returns parsed structures or borrows into those bytes. See `crate::sst`
//! for the async glue that fetches those bytes from disk.

pub mod block;
pub mod block_builder;
pub mod footer;
pub mod internal_key;
pub mod properties;
pub mod sst_builder;
pub mod varint;
