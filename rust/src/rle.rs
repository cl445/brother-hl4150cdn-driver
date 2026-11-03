//! Plane encoders for XL2HB: bit packing and sliding-window RLE.
//!
//! `group_bits` reads a byte stream as N-bit groups (MSB first) and
//! `pack_groups` is its inverse. `encode_line` is the native counterpart
//! of rle.sw_rle_encode plus the plane glue, like `_rle_fast.pyx`.

/// Same contract as rle.check_group_size.
pub fn check_group_size(group_size: i32) -> Result<u32, String> {
    if (1..=32).contains(&group_size) {
        Ok(group_size as u32)
    } else {
        Err(format!("group size {group_size} outside 1..32"))
    }
}

/// Group raw bytes into MSB-first values of `group_size` (1..=32) bits each.
///
/// The last group is zero-padded if the bit count is not a multiple of
/// `group_size`; the result has ceil(len * 8 / group_size) values.
pub fn group_bits(data: &[u8], group_size: u32) -> Vec<u32> {
    let total_bits = data.len() * 8;
    let gs = group_size as usize;
    let n_full = total_bits / gs;
    let remaining = (total_bits - n_full * gs) as u32;
    let mask = (1u64 << group_size) - 1;
    let mut out = Vec::with_capacity(total_bits.div_ceil(gs));
    let mut bytes = data.iter();
    let mut buf = 0u64;
    let mut buf_bits = 0u32;

    for _ in 0..n_full {
        while buf_bits < group_size {
            buf = (buf << 8) | u64::from(*bytes.next().unwrap());
            buf_bits += 8;
        }
        out.push(((buf >> (buf_bits - group_size)) & mask) as u32);
        buf_bits -= group_size;
    }
    if remaining > 0 {
        for &b in bytes {
            buf = (buf << 8) | u64::from(b);
            buf_bits += 8;
        }
        // The next `remaining` bits, MSB first, at the high end of a
        // `group_size`-bit value (zero-padded at the bottom).
        let tail = (buf >> (buf_bits - remaining)) & ((1u64 << remaining) - 1);
        out.push((tail << (group_size - remaining)) as u32);
    }
    out
}

/// Pack N-bit values (each masked to `group_size` bits) into bytes, MSB first.
///
/// The result has ceil(len * group_size / 8) bytes.
pub fn pack_groups(groups: &[u64], group_size: u32) -> Vec<u8> {
    let mask = (1u64 << group_size) - 1;
    let mut out = Vec::with_capacity((groups.len() * group_size as usize).div_ceil(8));
    let mut buf = 0u64;
    let mut buf_bits = 0u32;
    for &value in groups {
        buf = (buf << group_size) | (value & mask);
        buf_bits += group_size;
        while buf_bits >= 8 {
            out.push((buf >> (buf_bits - 8)) as u8);
            buf_bits -= 8;
        }
    }
    if buf_bits > 0 {
        out.push((buf << (8 - buf_bits)) as u8);
    }
    out
}

// ---------------------------------------------------------------------------
// Sliding-window RLE encoder (port of rle.sw_rle_encode + plane glue)
// ---------------------------------------------------------------------------

fn count_ext(o: &mut Vec<u8>, mut remaining: isize) {
    while remaining > 254 {
        o.push(0xFF);
        remaining -= 255;
    }
    o.push(remaining as u8);
}

fn emit_run(o: &mut Vec<u8>, value: u32, count: isize, bits: u32) {
    if bits == 10 {
        let v_hi = ((value >> 8) & 0x3) as u8;
        if count <= 31 {
            o.push(0x80 | ((count - 1) << 2) as u8 | v_hi);
            o.push(value as u8);
        } else {
            o.push(0xFC | v_hi);
            o.push(value as u8);
            count_ext(o, count - 32);
        }
        return;
    }
    let v_hi = ((value >> (bits - 4)) & 0xF) as u8;
    if count < 8 {
        o.push(0x80 | ((count - 1) << 4) as u8 | v_hi);
    } else {
        o.push(0xF0 | v_hi);
    }
    if bits > 12 {
        o.push((value >> 8) as u8);
    }
    o.push(value as u8);
    if count >= 8 {
        count_ext(o, count - 8);
    }
}

/// The literal header: a count byte, or 0x7F plus a count extension.
fn literal_header(o: &mut Vec<u8>, count: isize) {
    if count < 0x41 {
        o.push(((count - 2) | 0x40) as u8);
    } else {
        o.push(0x7F);
        count_ext(o, count - 0x41);
    }
}

fn emit_literal(o: &mut Vec<u8>, vals: &[u32], bits: u32) {
    literal_header(o, vals.len() as isize);
    let mut buf = 0u64;
    let mut buf_bits = 0u32;
    for &v in vals {
        buf = (buf << bits) | u64::from(v);
        buf_bits += bits;
        while buf_bits >= 8 {
            o.push((buf >> (buf_bits - 8)) as u8);
            buf_bits -= 8;
        }
    }
    if buf_bits > 0 {
        o.push((buf << (8 - buf_bits)) as u8);
    }
}

/// A literal of `vals`, or a run of one if there is only one word.
fn emit_literal_or_single(o: &mut Vec<u8>, vals: &[u32], bits: u32) {
    if vals.len() == 1 {
        emit_run(o, vals[0], 1, bits);
    } else {
        emit_literal(o, vals, bits);
    }
}

fn emit_context_skip(o: &mut Vec<u8>, count: isize) {
    if count < 64 {
        o.push((count - 1) as u8);
    } else {
        o.push(0x3F);
        count_ext(o, count - 64);
    }
}

/// Split the scanline into `bits`-bit words, MSB first, like rle.group_bits.
/// The last word is zero-padded.
fn extract_words(p: &[u8], bits: u32, out: &mut Vec<u32>) {
    let mask = (1u64 << bits) - 1;
    out.clear();
    let mut buf = 0u64;
    let mut buf_bits = 0u32;
    for &b in p {
        buf = (buf << 8) | u64::from(b);
        buf_bits += 8;
        while buf_bits >= bits {
            buf_bits -= bits;
            out.push(((buf >> buf_bits) & mask) as u32);
        }
    }
    if buf_bits > 0 {
        out.push(((buf << (bits - buf_bits)) & mask) as u32);
    }
}

#[derive(Clone, Copy)]
enum State {
    Main,
    Run,
    Literal,
    ContextSkip,
    FinalizeLit,
}

/// Port of rle.sw_rle_encode; state names follow the Python version.
///
/// `bits` selects the settings of rle.CONFIG_10BIT / CONFIG_12BIT /
/// CONFIG_20BIT (context size, literal overflow); keep the two in sync.
fn sw_rle(words: &[u32], bits: u32, o: &mut Vec<u8>, wbuf: &mut Vec<u32>) {
    if bits == 10 {
        sw_rle_ctx::<5>(words, bits, o, wbuf);
    } else {
        sw_rle_ctx::<3>(words, bits, o, wbuf);
    }
}

/// `sw_rle` with a context window of `CTX` words, fixed at compile time so
/// the window shift is a few register moves.
fn sw_rle_ctx<const CTX: usize>(words: &[u32], bits: u32, o: &mut Vec<u8>, wbuf: &mut Vec<u32>) {
    let n = words.len();
    if n == 0 {
        return;
    }
    let lit_overflow: isize = if bits == 20 { 0x7FF } else { 0xFFF };
    let mut ctx = [0u32; CTX];
    wbuf.clear();
    wbuf.resize(n + 2, 0);
    let mut wlen = 0usize;
    let mut match_count: isize = 0;
    let mut run_len: isize = 0;

    let mut cur = words[0];
    let mut wi = 1;
    ctx[CTX - 1] = cur;
    if cur == 0 {
        match_count = 1;
    }
    let mut state = State::Main;

    // Every state first shifts the context window; the reference word is
    // the one that leaves position 1.
    let shift = |ctx: &mut [u32; CTX]| {
        let reference = ctx[1];
        for k in 0..CTX - 1 {
            ctx[k] = ctx[k + 1];
        }
        reference
    };

    loop {
        match state {
            State::Main => {
                let reference = shift(&mut ctx);
                wbuf[0] = cur;
                wlen = 1;
                run_len = 1;
                if wi >= n {
                    state = State::FinalizeLit;
                    continue;
                }
                let w = words[wi];
                wi += 1;
                match_count = if reference == w { match_count + 1 } else { 0 };
                ctx[CTX - 1] = w;
                if cur == w {
                    run_len = 2;
                    state = State::Run;
                } else if match_count == 2 {
                    state = State::ContextSkip;
                } else {
                    wbuf[1] = w;
                    wlen = 2;
                    run_len = 2;
                    state = State::Literal;
                }
            }
            State::Run => {
                let reference = shift(&mut ctx);
                if wi >= n {
                    emit_run(o, cur, run_len, bits);
                    return;
                }
                let w = words[wi];
                wi += 1;
                match_count = if reference == w { match_count + 1 } else { 0 };
                ctx[CTX - 1] = w;
                if cur == w {
                    run_len += 1;
                    continue;
                }
                emit_run(o, cur, run_len, bits);
                cur = w;
                if match_count != 0 {
                    match_count = 1;
                }
                state = State::Main;
            }
            State::Literal => {
                let reference = shift(&mut ctx);
                if wi >= n {
                    state = State::FinalizeLit;
                    continue;
                }
                let w = words[wi];
                wi += 1;
                match_count = if reference == w { match_count + 1 } else { 0 };
                ctx[CTX - 1] = w;

                if match_count > 1 {
                    if match_count - 1 < run_len {
                        let lit_n = (run_len - match_count + 1) as usize;
                        emit_literal_or_single(o, &wbuf[..lit_n], bits);
                    }
                    state = State::ContextSkip;
                    continue;
                }
                if wbuf[wlen - 1] == w {
                    if run_len > 2 {
                        emit_literal(o, &wbuf[..wlen - 1], bits);
                    } else if run_len == 2 {
                        emit_run(o, wbuf[0], 1, bits);
                    }
                    cur = w;
                    wbuf[0] = w;
                    wlen = 1;
                    run_len = 2;
                    state = State::Run;
                    continue;
                }
                if run_len + 1 > lit_overflow {
                    let lit_n = (run_len - 1) as usize;
                    emit_literal_or_single(o, &wbuf[..lit_n], bits);
                    let last = wbuf[wlen - 1];
                    cur = w;
                    wbuf[0] = last;
                    wbuf[1] = w;
                    wlen = 2;
                    run_len = 2;
                    continue;
                }
                wbuf[wlen] = w;
                wlen += 1;
                run_len += 1;
            }
            State::ContextSkip => {
                let reference = shift(&mut ctx);
                if wi >= n {
                    emit_context_skip(o, match_count);
                    return;
                }
                let w = words[wi];
                wi += 1;
                ctx[CTX - 1] = w;
                if reference != w {
                    emit_context_skip(o, match_count);
                    match_count = 0;
                    cur = w;
                    state = State::Main;
                    continue;
                }
                match_count += 1;
            }
            State::FinalizeLit => {
                if match_count < run_len {
                    let lit_n = (run_len - match_count) as usize;
                    emit_literal_or_single(o, &wbuf[..lit_n], bits);
                }
                if match_count > 0 {
                    emit_context_skip(o, match_count);
                }
                return;
            }
        }
    }
}

/// Whether `bits` is a word size of the plane encoders (10, 12 or 20).
pub fn is_word_size(bits: i32) -> bool {
    matches!(bits, 10 | 12 | 20)
}

/// Append one plane scanline, sliding-window RLE encoded, to `o`.
///
/// Native equivalent of plane_encoders._encode_via_sw_rle over the words
/// from rle.group_bits(data, bits). `bits` is the word size and picks the
/// config: 12 (CONFIG_12BIT), 20 (CONFIG_20BIT) or 10 (CONFIG_10BIT).
/// Appends nothing if every word is zero. Returns the number of bytes appended.
pub fn encode_line(p: &[u8], bits: u32, o: &mut Vec<u8>) -> usize {
    LineEncoder::default().encode(p, bits, o)
}

/// `encode_line` with scratch space kept between lines.
#[derive(Default)]
pub struct LineEncoder {
    words: Vec<u32>,
    wbuf: Vec<u32>,
}

impl LineEncoder {
    /// See `encode_line`.
    pub fn encode(&mut self, p: &[u8], bits: u32, o: &mut Vec<u8>) -> usize {
        let start = o.len();
        // Every word is zero exactly when every byte is (the padding is zero).
        if p.iter().all(|&b| b == 0) {
            return 0;
        }
        extract_words(p, bits, &mut self.words);
        let n_words = self.words.len();
        o.reserve(n_words * 3 + 64);
        sw_rle(&self.words, bits, o, &mut self.wbuf);

        if o.len() - start > p.len() + 0x14 {
            // Raw fallback, see rle.finalize_compressed.
            o.truncate(start);
            literal_header(o, n_words as isize);
            let padded_bytes = (n_words * bits as usize).div_ceil(8);
            o.extend_from_slice(p);
            o.resize(o.len() + padded_bytes - p.len(), 0);
        }
        o.len() - start
    }
}

#[cfg(feature = "python")]
pub use bindings::register;

#[cfg(feature = "python")]
mod bindings {
    use pyo3::buffer::PyBuffer;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::{PyBytes, PyList};

    use crate::buffer::contiguous;

    fn group_size(group_size: i32) -> PyResult<u32> {
        super::check_group_size(group_size).map_err(PyValueError::new_err)
    }

    /// Group raw bytes into MSB-first values of `group_size` bits each.
    ///
    /// Last group is zero-padded if the bit count isn't a multiple of
    /// `group_size`.
    ///
    /// Returns:
    ///     list[int] of length ceil(len(data)*8 / group_size).
    ///
    /// Raises:
    ///     ValueError: If `group_size` is outside 1..32.
    #[pyfunction]
    fn group_bits<'py>(
        py: Python<'py>,
        data: PyBuffer<u8>,
        group_size: i32,
    ) -> PyResult<Bound<'py, PyList>> {
        let group_size = self::group_size(group_size)?;
        PyList::new(py, super::group_bits(contiguous(&data)?, group_size))
    }

    /// Pack a list of N-bit values into bytes, MSB-first.
    ///
    /// Returns:
    ///     bytes of length ceil(len(groups) * group_size / 8).
    ///
    /// Raises:
    ///     ValueError: If `group_size` is outside 1..32.
    #[pyfunction]
    fn pack_groups<'py>(
        py: Python<'py>,
        groups: &Bound<'py, PyList>,
        group_size: i32,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let group_size = self::group_size(group_size)?;
        let values = groups
            .iter()
            .map(|v| v.extract::<u64>())
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyBytes::new(py, &super::pack_groups(&values, group_size)))
    }

    /// Encode one plane scanline with the sliding-window RLE, incl. raw fallback.
    ///
    /// `data` is any contiguous byte buffer; `bits` is 10, 12 or 20.
    ///
    /// Returns:
    ///     Compressed bytes; empty if every word is zero.
    #[pyfunction]
    fn encode_sw_rle<'py>(
        py: Python<'py>,
        data: PyBuffer<u8>,
        bits: i32,
    ) -> PyResult<Bound<'py, PyBytes>> {
        if !super::is_word_size(bits) {
            return Err(PyValueError::new_err(format!(
                "unsupported word size {bits}"
            )));
        }
        let mut out = Vec::new();
        super::encode_line(contiguous(&data)?, bits as u32, &mut out);
        Ok(PyBytes::new(py, &out))
    }

    pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(group_bits, m)?)?;
        m.add_function(wrap_pyfunction!(pack_groups, m)?)?;
        m.add_function(wrap_pyfunction!(encode_sw_rle, m)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_bits_pads_the_last_group() {
        assert_eq!(group_bits(&[0xAB, 0xCD], 12), [0xABC, 0xD00]);
        assert_eq!(group_bits(&[], 12), Vec::<u32>::new());
        assert_eq!(group_bits(&[0xFF; 4], 32), [0xFFFF_FFFF]);
    }

    #[test]
    fn pack_groups_inverts_group_bits() {
        let data: Vec<u8> = (0..=255).collect();
        for size in [1, 10, 12, 20, 31, 32] {
            let groups: Vec<u64> = group_bits(&data, size).into_iter().map(u64::from).collect();
            assert_eq!(pack_groups(&groups, size)[..data.len()], data[..]);
        }
    }

    #[test]
    fn blank_line_encodes_to_nothing() {
        let mut out = Vec::new();
        assert_eq!(encode_line(&[0; 596], 12, &mut out), 0);
        assert!(out.is_empty());
    }

    #[test]
    fn solid_line_is_one_run() {
        // 12-bit words of 0xFFF: 6 bytes are 4 words, a run of 4.
        let mut out = Vec::new();
        encode_line(&[0xFF; 6], 12, &mut out);
        assert_eq!(out, [0x80 | (3 << 4) | 0xF, 0xFF]);
    }
}
