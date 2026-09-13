//! Embedding blob codec.
//!
//! Two on-disk formats, both decodable; only one is written.
//!
//! ```text
//!   legacy  raw little-endian f32              4 * dim = 3072 B
//!   int8    magic + per-vector scale + int8[]  6 + dim =  774 B   (3.97x smaller)
//! ```
//!
//! int8 is the "int8 embeddings in DB" line of the optimization ledger. The
//! saving is real on three axes: bytes read per retrieval scan, bytes stored,
//! and bytes pushed over the Turso sync leg.
//!
//! Quantization is per-vector (`scale = max_abs / 127`), not global, so a vector
//! whose components cluster near zero — which every 768-dim unit vector does,
//! mean |component| ~ 1/sqrt(768) — still uses the full int8 range. A single
//! global scale would collapse such a vector to a handful of distinct levels.
//!
//! Cosine is invariant to the per-vector scale, so the only error introduced is
//! rounding each component to 1/127 of that vector's own maximum. Measured
//! against the f32 original this moves a cosine score by <0.002, and
//! `RETRIEVE_MIN_SCORE` is 0.45 with neighbours separated by far more.
//!
//! The magic byte is `0xE0` and the length is `6 + dim` = 774, which is not a
//! multiple of 4 for dim 768 — so an int8 blob can never be mistaken for a
//! legacy f32 one even if the magic check were skipped.

const MAGIC: u8 = 0xe0;
const FORMAT_INT8: u8 = 0x01;
const INT8_HEADER: usize = 6; // magic(1) + version(1) + scale f32(4)

pub fn encode_f32(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

pub fn encode_int8(v: &[f32]) -> Vec<u8> {
    let max_abs = v.iter().fold(0f32, |m, x| m.max(x.abs()));
    // An all-zero vector has no direction; scale 1 keeps it all-zero on decode,
    // and cosine already returns 0 for a zero-norm vector.
    let scale: f32 = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };

    let mut out = Vec::with_capacity(INT8_HEADER + v.len());
    out.push(MAGIC);
    out.push(FORMAT_INT8);
    out.extend_from_slice(&scale.to_le_bytes());

    let inv = 1.0f32 / scale;
    for x in v {
        // Round half UP, not half away from zero, to match the JS encoder
        // byte for byte: `Math.round(-0.5)` is -0 where Rust's `f32::round`
        // gives -1. Both codecs read each other's rows while the Electron
        // build and this one run side by side, so the formats have to agree.
        let q = (x * inv + 0.5).floor().clamp(-127.0, 127.0);
        out.push(q as i8 as u8);
    }
    out
}

pub fn encode(v: &[f32]) -> Vec<u8> {
    if crate::config::embed_quantize() {
        encode_int8(v)
    } else {
        encode_f32(v)
    }
}

/// Decode either format. `None` for a truncated or garbage blob, which the
/// caller treats as "this row has no usable embedding" rather than an error —
/// one bad row must not stop a retrieval scan.
pub fn decode(blob: &[u8]) -> Option<Vec<f32>> {
    if blob.len() > INT8_HEADER && blob[0] == MAGIC && blob[1] == FORMAT_INT8 {
        let scale = f32::from_le_bytes([blob[2], blob[3], blob[4], blob[5]]);
        if !scale.is_finite() {
            return None;
        }
        return Some(
            blob[INT8_HEADER..]
                .iter()
                .map(|&b| (b as i8) as f32 * scale)
                .collect(),
        );
    }

    if blob.is_empty() || blob.len() % 4 != 0 {
        return None;
    }
    Some(
        blob.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

/// Cosine similarity. Returns 0 for a zero-norm vector or a length mismatch,
/// so a malformed row scores below any threshold instead of poisoning a scan.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0f32, 0f32, 0f32);
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_vector(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        let mut v: Vec<f32> = (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / 8_388_608.0) - 1.0
            })
            .collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v {
            *x /= norm;
        }
        v
    }

    #[test]
    fn int8_blob_is_the_documented_size() {
        let v = unit_vector(768, 7);
        assert_eq!(encode_int8(&v).len(), 774);
        assert_eq!(encode_f32(&v).len(), 3072);
    }

    #[test]
    fn int8_roundtrip_keeps_cosine_within_tolerance() {
        let v = unit_vector(768, 42);
        let back = decode(&encode_int8(&v)).expect("decodes");
        // The documented bound is <0.002 against the f32 original.
        assert!(cosine(&v, &back) > 0.998, "cosine {}", cosine(&v, &back));
    }

    #[test]
    fn legacy_f32_blobs_still_decode() {
        let v = unit_vector(768, 3);
        let back = decode(&encode_f32(&v)).expect("decodes");
        assert_eq!(v, back, "f32 is lossless");
    }

    #[test]
    fn an_int8_blob_is_never_mistaken_for_f32() {
        // 774 is not a multiple of 4, which is the second line of defence after
        // the magic byte.
        assert_ne!(encode_int8(&unit_vector(768, 1)).len() % 4, 0);
    }

    #[test]
    fn all_zero_vector_survives_the_roundtrip() {
        let v = vec![0f32; 768];
        let back = decode(&encode_int8(&v)).expect("decodes");
        assert!(back.iter().all(|x| *x == 0.0));
        assert_eq!(cosine(&v, &back), 0.0, "zero norm scores 0, never NaN");
    }

    #[test]
    fn garbage_blobs_decode_to_none_rather_than_panicking() {
        assert!(decode(&[]).is_none());
        assert!(decode(&[1, 2, 3]).is_none(), "not a multiple of 4");
        // int8 magic with a non-finite scale.
        let mut bad = vec![MAGIC, FORMAT_INT8];
        bad.extend_from_slice(&f32::NAN.to_le_bytes());
        bad.push(1);
        assert!(decode(&bad).is_none());
    }

    #[test]
    fn cosine_rejects_mismatched_lengths() {
        assert_eq!(cosine(&[1.0, 0.0], &[1.0, 0.0, 0.0]), 0.0);
    }
}
