// Mirrors db/dbformat.h: an internal key is user_key ++ (seq:56 | type:8),
// packed little-endian as a u64 (PackSequenceAndType).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    Deletion,
    Value,
    Merge,
    SingleDeletion,
    RangeDeletion,
    BlobIndex,
    Other(u8),
}

impl ValueType {
    fn from_byte(b: u8) -> Self {
        match b {
            0x0 => ValueType::Deletion,
            0x1 => ValueType::Value,
            0x2 => ValueType::Merge,
            0x7 => ValueType::SingleDeletion,
            0xF => ValueType::RangeDeletion,
            0x11 => ValueType::BlobIndex,
            other => ValueType::Other(other),
        }
    }
}

pub struct InternalKey<'a> {
    pub user_key: &'a [u8],
    pub sequence: u64,
    pub value_type: ValueType,
}

const NUM_INTERNAL_BYTES: usize = 8;

pub fn split(internal_key: &[u8]) -> Result<InternalKey<'_>, String> {
    if internal_key.len() < NUM_INTERNAL_BYTES {
        return Err("internal key shorter than 8-byte seq/type suffix".to_string());
    }
    let split_at = internal_key.len() - NUM_INTERNAL_BYTES;
    let user_key = &internal_key[..split_at];
    let packed = u64::from_le_bytes(internal_key[split_at..].try_into().unwrap());
    let sequence = packed >> 8;
    let value_type = ValueType::from_byte((packed & 0xff) as u8);
    Ok(InternalKey {
        user_key,
        sequence,
        value_type,
    })
}
