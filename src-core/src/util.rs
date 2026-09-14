//! Small helpers with no home of their own.

use std::time::{SystemTime, UNIX_EPOCH};

/// The timestamp format every row in the database already uses.
///
/// Rows written by the Electron build carry JavaScript `toISOString()` output -
/// `YYYY-MM-DDTHH:MM:SS.sssZ`, always UTC, always three fractional digits - and
/// the schema stores them as TEXT, so `ORDER BY updated_at DESC` is a string
/// sort. A different shape here (an offset instead of Z, or a variable number of
/// fractional digits) would sort wrong against existing rows, so this reproduces
/// that format exactly.
///
/// Done by hand rather than by pulling in a date crate: this is the only date
/// formatting the core needs, and the conversion below is the whole of it.
pub fn now_iso() -> String {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format_iso(d.as_secs() as i64, d.subsec_millis())
}

fn format_iso(secs: i64, millis: u32) -> String {
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        y,
        m,
        d,
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60,
        millis
    )
}

/// Days since the Unix epoch to a civil (year, month, day).
///
/// Hinnant civil_from_days: it shifts the era so that March is month 1, which
/// puts the leap day at the end of the year and removes every special case for
/// it. Valid across the whole i64 range, so no clamping is needed.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_javascript_iso_shape() {
        // Reference values from `new Date(ms).toISOString()`.
        assert_eq!(format_iso(0, 0), "1970-01-01T00:00:00.000Z");
        assert_eq!(format_iso(1_000_000_000, 0), "2001-09-09T01:46:40.000Z");
        assert_eq!(format_iso(1_709_164_800, 7), "2024-02-29T00:00:00.007Z");
        assert_eq!(format_iso(1_767_225_599, 999), "2025-12-31T23:59:59.999Z");
    }

    #[test]
    fn sorts_as_a_string_in_time_order() {
        let mut v = [format_iso(1_709_164_800, 0), format_iso(0, 0), format_iso(1_000_000_000, 0)];
        v.sort();
        assert_eq!(v[0], "1970-01-01T00:00:00.000Z");
        assert_eq!(v[2], "2024-02-29T00:00:00.000Z");
    }

    #[test]
    fn now_is_well_formed_and_current() {
        let s = now_iso();
        assert_eq!(s.len(), 24, "{s}");
        assert!(s.ends_with('Z'));
        assert!(s.as_str() > "2025-01-01T00:00:00.000Z", "{s}");
    }
}
