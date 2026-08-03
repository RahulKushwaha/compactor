/// Standard 7-bit-per-byte, little-endian, MSB-continuation varint decoding
/// (same scheme as util/coding.h GetVarint32Ptr / GetVarint64Ptr in RocksDB).
pub fn get_varint32(buf: &[u8], pos: &mut usize) -> Option<u32> {
    let v = get_varint64(buf, pos)?;
    if v > u32::MAX as u64 {
        return None;
    }
    Some(v as u32)
}

pub fn get_varint64(buf: &[u8], pos: &mut usize) -> Option<u64> {
    let mut result: u64 = 0;
    let mut shift = 0;
    let mut p = *pos;
    loop {
        let byte = *buf.get(p)?;
        p += 1;
        result |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            *pos = p;
            return Some(result);
        }
        shift += 7;
        if shift >= 64 {
            return None;
        }
    }
}

/// Encodes `v` as a standard varint32 (same scheme as util/coding.h
/// EncodeVarint32), appending to `out`.
pub fn put_varint32(out: &mut Vec<u8>, v: u32) {
    put_varint64(out, v as u64);
}

/// Encodes `v` as a standard varint64, appending to `out`.
pub fn put_varint64(out: &mut Vec<u8>, mut v: u64) {
    loop {
        if v < 0x80 {
            out.push(v as u8);
            break;
        } else {
            out.push((v as u8 & 0x7f) | 0x80);
            v >>= 7;
        }
    }
}
