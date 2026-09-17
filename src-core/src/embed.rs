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

    if blob.is_empty() || !blob.len().is_multiple_of(4) {
        return None;
    }
    Some(
        blob.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
    )
}

/// Decode a stored blob into `dim` int8 components appended to `out`, WITHOUT
/// dequantizing. Returns false and appends nothing if the row is unusable.
///
/// Cosine is invariant to positive scaling, so the per-vector scale carries no
/// direction and the retrieval scan never needs it: scoring the raw int8
/// components against their own norm gives exactly the cosine that dequantizing
/// to f32 and re-normalizing would, because the dequantized vector IS
/// `scale * q8` and its unit form is therefore `q8 / |q8|`. Skipping the scale
/// pass removes a multiply per component on every cold cache fill and lets the
/// cache hold each row at a quarter of the width.
///
/// The rejections are the whole reason this returns a bool. An i8 cannot hold
/// an infinity or a NaN, so poison can only arrive in a legacy f32 blob, and
/// that is where the finiteness check lives. A non-finite component would
/// otherwise make the row's norm non-finite and every score against it NaN -
/// and NaN loses no comparison, so the row would rank above real matches and,
/// as the k-th best, let every remaining row through. Quantizing on the way in
/// confines that failure to this one function.
pub fn decode_int8(blob: &[u8], dim: usize, out: &mut Vec<i8>) -> bool {
    let start = out.len();

    // Current format: the components are already what the scan wants.
    if blob.len() > INT8_HEADER && blob[0] == MAGIC && blob[1] == FORMAT_INT8 {
        let body = &blob[INT8_HEADER..];
        if body.len() != dim {
            return false;
        }
        // Unused for direction, but a non-finite scale means the blob is
        // corrupt somewhere, and a corrupt header is not a row to trust.
        if !f32::from_le_bytes([blob[2], blob[3], blob[4], blob[5]]).is_finite() {
            return false;
        }
        out.extend(body.iter().map(|&b| b as i8));
        return true;
    }

    // Legacy f32: quantize on the way in, so the cache holds one format only.
    if blob.len() != dim * 4 {
        return false;
    }
    let comps = blob.as_chunks::<4>().0;
    let mut max_abs = 0f32;
    for c in comps {
        let x = f32::from_le_bytes(*c);
        if !x.is_finite() {
            return false;
        }
        max_abs = max_abs.max(x.abs());
    }
    if max_abs == 0.0 {
        return false; // no direction
    }
    let inv = 127.0f32 / max_abs;
    for c in comps {
        // Round half UP, matching `encode_int8` and the JS codec.
        let q = (f32::from_le_bytes(*c) * inv + 0.5)
            .floor()
            .clamp(-127.0, 127.0);
        out.push(q as i8);
    }
    debug_assert_eq!(out.len() - start, dim);
    true
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

    // ---- decode_int8: the retrieval cache's decoder ----------------------
    // It skips the dequantize pass and appends int8 straight to a shared
    // matrix. That makes its rejections load-bearing for ranking: anything it
    // lets through gets scored, and a non-finite component would make every
    // score against that row NaN.

    #[test]
    fn the_int8_decoder_appends_a_row_and_leaves_the_matrix_before_it_alone() {
        let v = unit_vector(768, 4242);
        let blob = encode_int8(&v);
        let mut mat: Vec<i8> = vec![7; 768];
        assert!(decode_int8(&blob, 768, &mut mat));
        assert_eq!(mat.len(), 1536);
        assert!(
            mat[..768].iter().all(|&x| x == 7),
            "clobbered the row before"
        );
        for i in 0..768 {
            assert_eq!(mat[768 + i], blob[INT8_HEADER + i] as i8, "component {i}");
        }
    }

    #[test]
    fn a_legacy_f32_row_is_quantized_on_the_way_in_and_keeps_its_direction() {
        let v = unit_vector(768, 777);
        let mut mat: Vec<i8> = Vec::new();
        assert!(decode_int8(&encode_f32(&v), 768, &mut mat));
        let as_f32: Vec<f32> = mat.iter().map(|&x| x as f32).collect();
        assert!(
            cosine(&as_f32, &v) > 0.999,
            "direction lost in quantization"
        );
    }

    #[test]
    fn the_int8_decoder_rejects_a_legacy_row_with_a_non_finite_component() {
        // The whole reason it returns a bool. An i8 cannot hold an infinity, so
        // this is the only door poison can come through.
        for bad in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            let mut v = unit_vector(768, 1);
            v[13] = bad;
            let mut mat: Vec<i8> = Vec::new();
            assert!(!decode_int8(&encode_f32(&v), 768, &mut mat), "{bad}");
            assert!(mat.is_empty(), "{bad}: a rejected row left debris behind");
        }
    }

    #[test]
    fn a_rejected_row_appends_nothing_so_the_matrix_stays_aligned() {
        // The rows are addressed by index times EMBED_DIM. A rejection that
        // left a partial row behind would shift every row after it, and every
        // id would then point at someone else's vector.
        let mut mat: Vec<i8> = Vec::new();
        assert!(decode_int8(
            &encode_int8(&unit_vector(768, 3)),
            768,
            &mut mat
        ));
        for bad in [
            encode_int8(&unit_vector(64, 3)),   // wrong width, int8
            encode_f32(&unit_vector(64, 3)),    // wrong width, legacy
            Vec::new(),                         // empty
            vec![0xe0, 0x01, 0, 0, 0xc0, 0x7f], // int8 header, NaN scale
        ] {
            assert!(!decode_int8(&bad, 768, &mut mat));
            assert_eq!(mat.len(), 768, "a rejected row changed the matrix");
        }
        assert!(decode_int8(
            &encode_int8(&unit_vector(768, 5)),
            768,
            &mut mat
        ));
        assert_eq!(mat.len(), 1536, "the good row after the bad ones was lost");
    }

    #[test]
    fn a_zero_vector_has_no_direction_to_quantize() {
        let mut mat: Vec<i8> = Vec::new();
        // Legacy: caught here, because 127/0 is not a scale.
        assert!(!decode_int8(&encode_f32(&vec![0.0f32; 768]), 768, &mut mat));
        // int8: it decodes as all zeros, and the caller drops it on the norm.
        assert!(decode_int8(&encode_int8(&vec![0.0f32; 768]), 768, &mut mat));
        assert!(mat.iter().all(|&x| x == 0));
    }

    #[test]
    fn scoring_raw_int8_against_its_own_norm_is_the_cosine() {
        // The cache skips dequantizing because cosine ignores the per-vector
        // scale. This pins the equivalence: the fast score must match the score
        // computed the long way from the same components.
        let mut q = unit_vector(768, 31);
        let qn = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut q {
            *x /= qn;
        }
        for seed in [1u32, 99, 12345] {
            let mut mat: Vec<i8> = Vec::new();
            assert!(decode_int8(
                &encode_int8(&unit_vector(768, seed)),
                768,
                &mut mat
            ));
            let sq: i32 = mat.iter().map(|&x| x as i32 * x as i32).sum();
            let fast: f32 =
                q.iter().zip(&mat).map(|(a, &b)| a * b as f32).sum::<f32>() / (sq as f32).sqrt();

            // the long way, from the same components, through any positive scale
            let deq: Vec<f32> = mat.iter().map(|&x| x as f32 * 0.0037).collect();
            let slow = cosine(&q, &deq);

            assert!((fast - slow).abs() < 1e-5, "seed {seed}: {fast} vs {slow}");
        }
    }
}
