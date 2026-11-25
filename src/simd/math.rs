//! Vectorised elementary functions (`ln`, `exp`) built only from portable
//! pulp operations.
//!
//! pulp has no transcendental functions, so these are implemented with the
//! classic range-reduction + polynomial approach:
//!
//! * `ln`: split `x = 2^k · m` with `m ∈ [√½, √2)` using integer bit
//!   manipulation, then `ln(m)` via the fdlibm `s = f/(2+f)` minimax series.
//!   Max error ≈ 1 ulp (`f64`) over the whole positive range, including
//!   subnormals.
//! * `exp`: `x = k·ln2 + r`, `|r| ≤ ln2/2`, Taylor/Horner polynomial for
//!   `e^r` and `2^k` assembled directly in the exponent bits. Max error ≈ 1–2
//!   ulp. Results below ~1e-307 (`f64`) / ~2e-38 (`f32`) flush to zero.
//!
//! The exponent field is moved with 32-bit lane shifts (pulp offers no
//! 64-bit shifts): on the little-endian targets pulp supports, the high
//! 32-bit word of each `f64` lane holds the sign, exponent and top 20
//! mantissa bits.

// Constants are kept digit-for-digit as published in fdlibm.
#![allow(clippy::excessive_precision)]

use bytemuck::cast;
use pulp::Simd;


/// Lane select on the integer view (pulp's float `select` is a non-inlined
/// default method built from transmute helpers).
#[inline(always)]
fn sel64<S: Simd>(s: S, m: S::m64s, t: S::f64s, f: S::f64s) -> S::f64s {
    cast(s.select_u64s(m, cast(t), cast(f)))
}

#[inline(always)]
fn sel32<S: Simd>(s: S, m: S::m32s, t: S::f32s, f: S::f32s) -> S::f32s {
    cast(s.select_u32s(m, cast(t), cast(f)))
}

// fdlibm e_log.c minimax coefficients for log(1+f) in terms of s = f/(2+f)
const LG1: f64 = 6.666666666666735130e-01;
const LG2: f64 = 3.999999999940941908e-01;
const LG3: f64 = 2.857142874366239149e-01;
const LG4: f64 = 2.222219843214978396e-01;
const LG5: f64 = 1.818357216161805012e-01;
const LG6: f64 = 1.531383769920937332e-01;
const LG7: f64 = 1.479819860511658591e-01;
const LN2_HI: f64 = 6.93147180369123816490e-01;
const LN2_LO: f64 = 1.90821492927058770002e-10;

// ---------------------------------------------------------------------------
// f64
// ---------------------------------------------------------------------------

/// `true` if an `f64` register can be viewed as `u32` lanes (all SIMD ISAs).
/// pulp's 1-lane scalar backend has 8-byte `f64s` but 4-byte `u32s`.
#[inline(always)]
fn wide_words<S: Simd>() -> bool {
    core::mem::size_of::<S::f64s>() == core::mem::size_of::<S::u32s>()
}

/// Lane-wise fallback through `std` for the scalar backend.
#[inline(always)]
fn scalar_f64<S: Simd>(x: S::f64s, f: fn(f64) -> f64) -> S::f64s {
    let mut lanes: S::f64s = x;
    for v in bytemuck::cast_slice_mut::<S::f64s, f64>(core::slice::from_mut(&mut lanes)) {
        *v = f(*v);
    }
    lanes
}

#[inline(always)]
pub(crate) fn ln_f64s<S: Simd>(s: S, x: S::f64s) -> S::f64s {
    if !wide_words::<S>() {
        return scalar_f64::<S>(x, f64::ln);
    }
    let zero = s.splat_f64s(0.0);
    // Scale subnormals into the normal range.
    let tiny = s.less_than_f64s(x, s.splat_f64s(f64::MIN_POSITIVE));
    let xs = sel64(s, tiny, s.mul_f64s(x, s.splat_f64s(18014398509481984.0)), x); // 2^54
    let k_adj = sel64(s, tiny, s.splat_f64s(-54.0), zero);

    let bits: S::u64s = cast(xs);

    // Biased exponent E as a float: shift the high word right by 20, keep the
    // 11 exponent bits in the high word, then read 0x43300000_E00000000 as a
    // double = 2^52 + E·2^32 (exact).
    let words: S::u32s = cast(bits);
    let shifted: S::u64s = cast(s.wrapping_dyn_shr_u32s(words, s.splat_u32s(20)));
    let e_bits = s.or_u64s(
        s.and_u64s(shifted, s.splat_u64s(0x0000_07FF_0000_0000)),
        s.splat_u64s(0x4330_0000_0000_0000),
    );
    let e = s.sub_f64s(cast(e_bits), s.splat_f64s(4503599627370496.0)); // E·2^32
    let mut k = s.mul_add_f64s(e, s.splat_f64s(2.3283064365386963e-10), s.splat_f64s(-1023.0));
    k = s.add_f64s(k, k_adj);

    // Mantissa m ∈ [1, 2), then fold into [√½, √2).
    let m_bits = s.or_u64s(
        s.and_u64s(bits, s.splat_u64s(0x000F_FFFF_FFFF_FFFF)),
        s.splat_u64s(0x3FF0_0000_0000_0000),
    );
    let mut m: S::f64s = cast(m_bits);
    let big = s.less_than_f64s(s.splat_f64s(core::f64::consts::SQRT_2), m);
    m = sel64(s, big, s.mul_f64s(m, s.splat_f64s(0.5)), m);
    k = sel64(s, big, s.add_f64s(k, s.splat_f64s(1.0)), k);

    let one = s.splat_f64s(1.0);
    let f = s.sub_f64s(m, one);
    let sv = s.div_f64s(f, s.add_f64s(s.splat_f64s(2.0), f));
    let z = s.mul_f64s(sv, sv);
    let mut r = s.splat_f64s(LG7);
    for c in [LG6, LG5, LG4, LG3, LG2, LG1] {
        r = s.mul_add_f64s(r, z, s.splat_f64s(c));
    }
    r = s.mul_f64s(r, z);
    let hfsq = s.mul_f64s(s.splat_f64s(0.5), s.mul_f64s(f, f));
    // k·ln2_hi − ((hfsq − (s·(hfsq + R) + k·ln2_lo)) − f)
    let inner = s.mul_add_f64s(sv, s.add_f64s(hfsq, r), s.mul_f64s(k, s.splat_f64s(LN2_LO)));
    let mut out = s.sub_f64s(
        s.mul_f64s(k, s.splat_f64s(LN2_HI)),
        s.sub_f64s(s.sub_f64s(hfsq, inner), f),
    );

    // Special values.
    let inf = s.splat_f64s(f64::INFINITY);
    out = sel64(s, s.equal_f64s(x, inf), inf, out);
    out = sel64(s, s.equal_f64s(x, zero), s.splat_f64s(f64::NEG_INFINITY), out);
    out = sel64(s, s.less_than_f64s(x, zero), s.splat_f64s(f64::NAN), out);
    sel64(s, s.equal_f64s(x, x), out, s.splat_f64s(f64::NAN))
}

const EXP_HI_F64: f64 = 709.782712893384;
const EXP_LO_F64: f64 = -708.0;

#[inline(always)]
pub(crate) fn exp_f64s<S: Simd>(s: S, x: S::f64s) -> S::f64s {
    if !wide_words::<S>() {
        return scalar_f64::<S>(x, f64::exp);
    }
    let xc = s.min_f64s(s.max_f64s(x, s.splat_f64s(EXP_LO_F64)), s.splat_f64s(EXP_HI_F64));
    // k = round(x / ln2) via the 1.5·2^52 trick
    let magic = s.splat_f64s(6755399441055744.0);
    let t = s.mul_f64s(xc, s.splat_f64s(core::f64::consts::LOG2_E));
    let k = s.sub_f64s(s.add_f64s(t, magic), magic);
    let mut r = s.mul_add_f64s(s.sub_f64s(s.splat_f64s(0.0), k), s.splat_f64s(LN2_HI), xc);
    r = s.mul_add_f64s(s.sub_f64s(s.splat_f64s(0.0), k), s.splat_f64s(LN2_LO), r);

    // e^r, |r| ≤ 0.347: Taylor to degree 13 (truncation < 5e-18)
    let mut p = s.splat_f64s(1.0 / 6227020800.0);
    for c in [
        1.0 / 479001600.0,
        1.0 / 39916800.0,
        1.0 / 3628800.0,
        1.0 / 362880.0,
        1.0 / 40320.0,
        1.0 / 5040.0,
        1.0 / 720.0,
        1.0 / 120.0,
        1.0 / 24.0,
        1.0 / 6.0,
        0.5,
        1.0,
        1.0,
    ] {
        p = s.mul_add_f64s(p, r, s.splat_f64s(c));
    }

    // 2^(k−1): (k − 1) + 1023 + 2^20 puts the biased exponent in the low 11
    // bits of the high word; shift it into place. k−1 ∈ [−1022, 1023].
    let y = s.add_f64s(k, s.splat_f64s(1048576.0 + 1022.0));
    let hi = s.and_u64s(cast(y), s.splat_u64s(0x0000_07FF_0000_0000));
    let words: S::u32s = cast(hi);
    let scale: S::f64s = cast(s.wrapping_dyn_shl_u32s(words, s.splat_u32s(20)));
    let mut out = s.mul_f64s(s.mul_f64s(p, scale), s.splat_f64s(2.0));

    out = sel64(s, s.less_than_f64s(s.splat_f64s(EXP_HI_F64), x), s.splat_f64s(f64::INFINITY), out);
    out = sel64(s, s.less_than_f64s(x, s.splat_f64s(EXP_LO_F64)), s.splat_f64s(0.0), out);
    sel64(s, s.equal_f64s(x, x), out, x)
}

// ---------------------------------------------------------------------------
// f32
// ---------------------------------------------------------------------------

const LN2_HI_F32: f32 = 6.9313812256e-01;
const LN2_LO_F32: f32 = 9.0580006145e-06;

#[inline(always)]
pub(crate) fn ln_f32s<S: Simd>(s: S, x: S::f32s) -> S::f32s {
    let zero = s.splat_f32s(0.0);
    let tiny = s.less_than_f32s(x, s.splat_f32s(f32::MIN_POSITIVE));
    let xs = sel32(s, tiny, s.mul_f32s(x, s.splat_f32s(16777216.0)), x); // 2^24
    let k_adj = sel32(s, tiny, s.splat_f32s(-24.0), zero);

    let bits: S::u32s = cast(xs);
    let e_raw = s.and_u32s(s.wrapping_dyn_shr_u32s(bits, s.splat_u32s(23)), s.splat_u32s(0xFF));
    let e: S::f32s = cast(s.or_u32s(e_raw, s.splat_u32s(0x4B00_0000)));
    let mut k = s.sub_f32s(e, s.splat_f32s(8388608.0 + 127.0));
    k = s.add_f32s(k, k_adj);

    let m_bits = s.or_u32s(s.and_u32s(bits, s.splat_u32s(0x007F_FFFF)), s.splat_u32s(0x3F80_0000));
    let mut m: S::f32s = cast(m_bits);
    let big = s.less_than_f32s(s.splat_f32s(core::f32::consts::SQRT_2), m);
    m = sel32(s, big, s.mul_f32s(m, s.splat_f32s(0.5)), m);
    k = sel32(s, big, s.add_f32s(k, s.splat_f32s(1.0)), k);

    let f = s.sub_f32s(m, s.splat_f32s(1.0));
    let sv = s.div_f32s(f, s.add_f32s(s.splat_f32s(2.0), f));
    let z = s.mul_f32s(sv, sv);
    let mut r = s.splat_f32s(LG4 as f32);
    for c in [LG3, LG2, LG1] {
        r = s.mul_add_f32s(r, z, s.splat_f32s(c as f32));
    }
    r = s.mul_f32s(r, z);
    let hfsq = s.mul_f32s(s.splat_f32s(0.5), s.mul_f32s(f, f));
    let inner = s.mul_add_f32s(sv, s.add_f32s(hfsq, r), s.mul_f32s(k, s.splat_f32s(LN2_LO_F32)));
    let mut out = s.sub_f32s(
        s.mul_f32s(k, s.splat_f32s(LN2_HI_F32)),
        s.sub_f32s(s.sub_f32s(hfsq, inner), f),
    );

    let inf = s.splat_f32s(f32::INFINITY);
    out = sel32(s, s.equal_f32s(x, inf), inf, out);
    out = sel32(s, s.equal_f32s(x, zero), s.splat_f32s(f32::NEG_INFINITY), out);
    out = sel32(s, s.less_than_f32s(x, zero), s.splat_f32s(f32::NAN), out);
    sel32(s, s.equal_f32s(x, x), out, s.splat_f32s(f32::NAN))
}

const EXP_HI_F32: f32 = 88.722_83;
const EXP_LO_F32: f32 = -86.9;

#[inline(always)]
pub(crate) fn exp_f32s<S: Simd>(s: S, x: S::f32s) -> S::f32s {
    let xc = s.min_f32s(s.max_f32s(x, s.splat_f32s(EXP_LO_F32)), s.splat_f32s(EXP_HI_F32));
    let magic = s.splat_f32s(12582912.0); // 1.5·2^23
    let t = s.mul_f32s(xc, s.splat_f32s(core::f32::consts::LOG2_E));
    let k = s.sub_f32s(s.add_f32s(t, magic), magic);
    let mut r = s.mul_add_f32s(s.sub_f32s(s.splat_f32s(0.0), k), s.splat_f32s(0.693_359_4), xc);
    r = s.mul_add_f32s(s.sub_f32s(s.splat_f32s(0.0), k), s.splat_f32s(-2.121_944_4e-4), r);

    let mut p = s.splat_f32s(1.0 / 40320.0);
    for c in [1.0 / 5040.0, 1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0] {
        p = s.mul_add_f32s(p, r, s.splat_f32s(c));
    }

    // 2^(k−1): low 8 bits of (k − 1) + 127 + 2^23, shifted into the exponent.
    let y = s.add_f32s(k, s.splat_f32s(8388608.0 + 126.0));
    let e = s.and_u32s(cast(y), s.splat_u32s(0xFF));
    let scale: S::f32s = cast(s.wrapping_dyn_shl_u32s(e, s.splat_u32s(23)));
    let mut out = s.mul_f32s(s.mul_f32s(p, scale), s.splat_f32s(2.0));

    out = sel32(s, s.less_than_f32s(s.splat_f32s(EXP_HI_F32), x), s.splat_f32s(f32::INFINITY), out);
    out = sel32(s, s.less_than_f32s(x, s.splat_f32s(EXP_LO_F32)), s.splat_f32s(0.0), out);
    sel32(s, s.equal_f32s(x, x), out, x)
}
