//! Dense `A · Bᵀ` through Apple's Accelerate framework (macOS only).
//!
//! Accelerate's BLAS runs matrix products on the matrix coprocessor of
//! Apple silicon, which is several times faster than NEON code. It is used
//! for the dot-product based batch metrics (dot, cosine); squared Euclidean
//! distances keep using the SIMD kernel, which sums `(a − b)²` directly
//! instead of the less accurate `|a|² + |b|² − 2a·b`.

use crate::simd::SimdFloat;
use core::any::TypeId;

const ROW_MAJOR: i32 = 101;
const NO_TRANS: i32 = 111;
const TRANS: i32 = 112;

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn cblas_sgemm(
        order: i32, trans_a: i32, trans_b: i32, m: i32, n: i32, k: i32,
        alpha: f32, a: *const f32, lda: i32, b: *const f32, ldb: i32,
        beta: f32, c: *mut f32, ldc: i32,
    );
    #[allow(clippy::too_many_arguments)]
    fn cblas_dgemm(
        order: i32, trans_a: i32, trans_b: i32, m: i32, n: i32, k: i32,
        alpha: f64, a: *const f64, lda: i32, b: *const f64, ldb: i32,
        beta: f64, c: *mut f64, ldc: i32,
    );
}

/// Pointer to the rows as one row-major matrix: the rows themselves if they
/// are laid out back to back (e.g. from [`crate::batch::rows_from_flat`]),
/// otherwise a copy stored in `buf`.
fn matrix<'a, E: Copy>(rows: &'a [&'a [E]], d: usize, buf: &'a mut Vec<E>) -> *const E {
    let base = rows[0].as_ptr();
    let contiguous = rows.iter().enumerate().all(|(i, r)| core::ptr::eq(r.as_ptr(), base.wrapping_add(i * d)));
    if contiguous {
        base
    } else {
        *buf = rows.concat();
        buf.as_ptr()
    }
}

/// Writes `out[i * n + j] = a[i] · b[j]`. Returns `false` (and leaves `out`
/// untouched) if the shapes do not fit the BLAS interface.
pub(crate) fn gemm_abt<E: SimdFloat>(a: &[&[E]], b: &[&[E]], out: &mut [E]) -> bool {
    let (m, n) = (a.len(), b.len());
    if m == 0 || n == 0 {
        return false;
    }
    let d = a[0].len();
    let fits = |x: usize| i32::try_from(x).is_ok();
    if d == 0 || !fits(m) || !fits(n) || !fits(d) || out.len() != m * n {
        return false;
    }
    let (mut abuf, mut bbuf) = (Vec::new(), Vec::new());
    let pa = matrix(a, d, &mut abuf);
    let pb = matrix(b, d, &mut bbuf);
    let (mi, ni, di) = (m as i32, n as i32, d as i32);
    // SAFETY: `pa` points to m × d and `pb` to n × d valid elements (either
    // the caller's contiguous rows or our copies), `out` holds m × n, and the
    // element type is checked before casting.
    unsafe {
        if TypeId::of::<E>() == TypeId::of::<f32>() {
            cblas_sgemm(
                ROW_MAJOR, NO_TRANS, TRANS, mi, ni, di, 1.0, pa.cast(), di, pb.cast(), di, 0.0,
                out.as_mut_ptr().cast(), ni,
            );
        } else if TypeId::of::<E>() == TypeId::of::<f64>() {
            cblas_dgemm(
                ROW_MAJOR, NO_TRANS, TRANS, mi, ni, di, 1.0, pa.cast(), di, pb.cast(), di, 0.0,
                out.as_mut_ptr().cast(), ni,
            );
        } else {
            return false;
        }
    }
    true
}
