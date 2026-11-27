//! Dense `A · Bᵀ` through Apple's Accelerate framework (macOS only).
//!
//! Accelerate's BLAS runs matrix products on the matrix coprocessor of
//! Apple silicon, which is several times faster than NEON code. It is used
//! for the dot-product based batch metrics (dot, cosine); squared Euclidean
//! distances keep using the SIMD kernel, which sums `(a − b)²` directly
//! instead of the less accurate `|a|² + |b|² − 2a·b`. Symmetric matrices
//! (`pdist`, `pairwise`) use SYRK, which computes only the upper triangle.

use crate::simd::SimdFloat;
use core::any::TypeId;

const ROW_MAJOR: i32 = 101;
const NO_TRANS: i32 = 111;
const TRANS: i32 = 112;
const UPPER: i32 = 121;

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
    #[allow(clippy::too_many_arguments)]
    fn cblas_ssyrk(
        order: i32, uplo: i32, trans: i32, n: i32, k: i32, alpha: f32, a: *const f32, lda: i32,
        beta: f32, c: *mut f32, ldc: i32,
    );
    #[allow(clippy::too_many_arguments)]
    fn cblas_dsyrk(
        order: i32, uplo: i32, trans: i32, n: i32, k: i32, alpha: f64, a: *const f64, lda: i32,
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

/// Rows of the output computed per BLAS call. One call over a large output
/// (thousands of rows, short vectors) runs up to 10× slower in Accelerate
/// than the same work split into row blocks of this size.
const ROW_BLOCK: usize = 2048;

/// Rows, depth and the i32 shapes BLAS takes, or `None` if they do not fit.
fn shape<E>(a: &[&[E]], n: usize, out: &[E]) -> Option<(i32, i32, i32)> {
    let m = a.len();
    let d = a.first()?.len();
    let fits = |x: usize| i32::try_from(x).ok();
    if m == 0 || n == 0 || d == 0 || out.len() != m * n {
        return None;
    }
    Some((fits(m)?, fits(n)?, fits(d)?))
}

/// `C (m × n, leading dimension ldc) = A (m × d) · B (n × d)ᵀ`, row major.
///
/// SAFETY: `a`, `b` and `c` must point to that many valid elements and `E`
/// must be `f32` or `f64` (checked by the callers).
unsafe fn gemm<E: 'static>(m: usize, n: usize, d: usize, a: *const E, b: *const E, c: *mut E, ldc: usize) {
    let (m, n, d, ldc) = (m as i32, n as i32, d as i32, ldc as i32);
    if TypeId::of::<E>() == TypeId::of::<f32>() {
        cblas_sgemm(ROW_MAJOR, NO_TRANS, TRANS, m, n, d, 1.0, a.cast(), d, b.cast(), d, 0.0, c.cast(), ldc);
    } else {
        cblas_dgemm(ROW_MAJOR, NO_TRANS, TRANS, m, n, d, 1.0, a.cast(), d, b.cast(), d, 0.0, c.cast(), ldc);
    }
}

/// Upper triangle of `C (n × n, leading dimension ldc) = A (n × d) · Aᵀ`.
///
/// SAFETY: as for [`gemm`].
unsafe fn syrk<E: 'static>(n: usize, d: usize, a: *const E, c: *mut E, ldc: usize) {
    let (n, d, ldc) = (n as i32, d as i32, ldc as i32);
    if TypeId::of::<E>() == TypeId::of::<f32>() {
        cblas_ssyrk(ROW_MAJOR, UPPER, NO_TRANS, n, d, 1.0, a.cast(), d, 0.0, c.cast(), ldc);
    } else {
        cblas_dsyrk(ROW_MAJOR, UPPER, NO_TRANS, n, d, 1.0, a.cast(), d, 0.0, c.cast(), ldc);
    }
}

fn is_blas_float<E: 'static>() -> bool {
    TypeId::of::<E>() == TypeId::of::<f32>() || TypeId::of::<E>() == TypeId::of::<f64>()
}

/// Writes `out[i * n + j] = a[i] · b[j]`. Returns `false` (and leaves `out`
/// untouched) if the shapes do not fit the BLAS interface.
pub(crate) fn gemm_abt<E: SimdFloat>(a: &[&[E]], b: &[&[E]], out: &mut [E]) -> bool {
    if !is_blas_float::<E>() || shape(a, b.len(), out).is_none() {
        return false;
    }
    let (m, n, d) = (a.len(), b.len(), a[0].len());
    let (mut abuf, mut bbuf) = (Vec::new(), Vec::new());
    let pa = matrix(a, d, &mut abuf);
    let pb = matrix(b, d, &mut bbuf);
    for i0 in (0..m).step_by(ROW_BLOCK) {
        let rows = ROW_BLOCK.min(m - i0);
        // SAFETY: `pa` points to m × d and `pb` to n × d valid elements
        // (the caller's contiguous rows or our copies), `out` holds m × n,
        // and the element type was checked above.
        unsafe { gemm(rows, n, d, pa.add(i0 * d), pb, out.as_mut_ptr().add(i0 * n), n) };
    }
    true
}

/// Symmetric rank-k update (SYRK): writes the upper triangle of `A · Aᵀ`
/// (`out[i * n + j]` for `j >= i`), doing about half the work of
/// [`gemm_abt`]. Each block of rows gets SYRK on its diagonal block and GEMM
/// for the columns right of it. Entries below the diagonal are left as they
/// were. Returns `false` (and leaves `out` untouched) if the shapes do not
/// fit the BLAS interface.
pub(crate) fn syrk_aat<E: SimdFloat>(a: &[&[E]], out: &mut [E]) -> bool {
    if !is_blas_float::<E>() || shape(a, a.len(), out).is_none() {
        return false;
    }
    let (n, d) = (a.len(), a[0].len());
    let mut buf = Vec::new();
    let pa = matrix(a, d, &mut buf);
    for i0 in (0..n).step_by(ROW_BLOCK) {
        let rows = ROW_BLOCK.min(n - i0);
        let rest = n - i0 - rows;
        // SAFETY: `pa` points to n × d valid elements, `out` holds n × n;
        // the diagonal block starts at (i0, i0) and the block right of it at
        // (i0, i0 + rows), both inside `out` with leading dimension n. The
        // element type was checked above.
        unsafe {
            let c = out.as_mut_ptr().add(i0 * n + i0);
            syrk(rows, d, pa.add(i0 * d), c, n);
            if rest > 0 {
                gemm(rows, rest, d, pa.add(i0 * d), pa.add((i0 + rows) * d), c.add(rows), n);
            }
        }
    }
    true
}
