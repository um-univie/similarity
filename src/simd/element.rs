//! Element abstraction that lets every SIMD kernel be written once and
//! instantiated for both `f32` and `f64`.

use core::fmt::Debug;
use pulp::Simd;

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Floating-point element types supported by the portable SIMD kernels.
///
/// This trait is sealed and implemented for [`f32`] and [`f64`]. It maps the
/// lane-generic operations of [`pulp::Simd`] onto a single element type so
/// that each kernel in [`crate::simd`] is written once and works for both
/// precisions. `f32` processes twice as many lanes per instruction as `f64`.
pub trait SimdFloat:
    sealed::Sealed
    + num_traits::Float
    + num_traits::FromPrimitive
    + Copy
    + Default
    + Debug
    + Send
    + Sync
    + core::iter::Sum
    + 'static
{
    /// SIMD register holding lanes of `Self` for instruction set `S`.
    type V<S: Simd>: Copy;
    /// Lane mask produced by comparisons for instruction set `S`.
    type M<S: Simd>: Copy;

    /// Number of lanes of `Self` in one register of `S`.
    fn lanes<S: Simd>() -> usize;

    fn as_simd<S: Simd>(x: &[Self]) -> (&[Self::V<S>], &[Self]);
    fn as_mut_simd<S: Simd>(x: &mut [Self]) -> (&mut [Self::V<S>], &mut [Self]);
    fn v_partial_load<S: Simd>(s: S, x: &[Self]) -> Self::V<S>;
    fn v_partial_store<S: Simd>(s: S, x: &mut [Self], v: Self::V<S>);

    fn v_splat<S: Simd>(s: S, v: Self) -> Self::V<S>;
    fn v_add<S: Simd>(s: S, a: Self::V<S>, b: Self::V<S>) -> Self::V<S>;
    fn v_sub<S: Simd>(s: S, a: Self::V<S>, b: Self::V<S>) -> Self::V<S>;
    fn v_mul<S: Simd>(s: S, a: Self::V<S>, b: Self::V<S>) -> Self::V<S>;
    fn v_div<S: Simd>(s: S, a: Self::V<S>, b: Self::V<S>) -> Self::V<S>;
    /// `a * b + c`, fused when the hardware supports it.
    fn v_mul_add<S: Simd>(s: S, a: Self::V<S>, b: Self::V<S>, c: Self::V<S>) -> Self::V<S>;
    fn v_abs<S: Simd>(s: S, a: Self::V<S>) -> Self::V<S>;
    fn v_min<S: Simd>(s: S, a: Self::V<S>, b: Self::V<S>) -> Self::V<S>;
    fn v_max<S: Simd>(s: S, a: Self::V<S>, b: Self::V<S>) -> Self::V<S>;
    fn v_eq<S: Simd>(s: S, a: Self::V<S>, b: Self::V<S>) -> Self::M<S>;
    fn v_gt<S: Simd>(s: S, a: Self::V<S>, b: Self::V<S>) -> Self::M<S>;
    fn v_select<S: Simd>(s: S, m: Self::M<S>, t: Self::V<S>, f: Self::V<S>) -> Self::V<S>;

    /// Natural logarithm per lane (≈1 ulp; `ln 0 = −∞`, negative → NaN).
    fn v_ln<S: Simd>(s: S, a: Self::V<S>) -> Self::V<S>;
    /// `e^x` per lane (≈1–2 ulp; tiny results flush to zero).
    fn v_exp<S: Simd>(s: S, a: Self::V<S>) -> Self::V<S>;
    /// Square root per lane (hardware instruction).
    fn v_sqrt<S: Simd>(s: S, a: Self::V<S>) -> Self::V<S>;

    fn v_reduce_sum<S: Simd>(s: S, a: Self::V<S>) -> Self;
    fn v_reduce_min<S: Simd>(s: S, a: Self::V<S>) -> Self;
    fn v_reduce_max<S: Simd>(s: S, a: Self::V<S>) -> Self;

    /// Lossless widening to `f64` (used for accumulation in mixed APIs).
    fn to_f64_(self) -> f64;
    /// Narrowing from `f64` (may round for `f32`).
    fn from_f64_(v: f64) -> Self;
}

macro_rules! impl_simd_float {
    ($t:ty, $v:ident, $m:ident, $lanes:ident,
     $as_simd:ident, $as_mut_simd:ident, $pload:ident, $pstore:ident,
     $splat:ident, $add:ident, $sub:ident, $mul:ident, $div:ident, $fma:ident,
     $abs:ident, $min:ident, $max:ident, $eq:ident, $gt:ident, $select:ident, $uselect:ident,
     $rsum:ident, $rmin:ident, $rmax:ident, $ln:path, $exp:path, $sqrt:ident) => {
        impl SimdFloat for $t {
            type V<S: Simd> = S::$v;
            type M<S: Simd> = S::$m;

            #[inline(always)]
            fn lanes<S: Simd>() -> usize {
                S::$lanes
            }
            #[inline(always)]
            fn as_simd<S: Simd>(x: &[Self]) -> (&[Self::V<S>], &[Self]) {
                S::$as_simd(x)
            }
            #[inline(always)]
            fn as_mut_simd<S: Simd>(x: &mut [Self]) -> (&mut [Self::V<S>], &mut [Self]) {
                S::$as_mut_simd(x)
            }
            #[inline(always)]
            fn v_partial_load<S: Simd>(s: S, x: &[Self]) -> Self::V<S> {
                s.$pload(x)
            }
            #[inline(always)]
            fn v_partial_store<S: Simd>(s: S, x: &mut [Self], v: Self::V<S>) {
                s.$pstore(x, v)
            }
            #[inline(always)]
            fn v_splat<S: Simd>(s: S, v: Self) -> Self::V<S> {
                s.$splat(v)
            }
            #[inline(always)]
            fn v_add<S: Simd>(s: S, a: Self::V<S>, b: Self::V<S>) -> Self::V<S> {
                s.$add(a, b)
            }
            #[inline(always)]
            fn v_sub<S: Simd>(s: S, a: Self::V<S>, b: Self::V<S>) -> Self::V<S> {
                s.$sub(a, b)
            }
            #[inline(always)]
            fn v_mul<S: Simd>(s: S, a: Self::V<S>, b: Self::V<S>) -> Self::V<S> {
                s.$mul(a, b)
            }
            #[inline(always)]
            fn v_div<S: Simd>(s: S, a: Self::V<S>, b: Self::V<S>) -> Self::V<S> {
                s.$div(a, b)
            }
            #[inline(always)]
            fn v_mul_add<S: Simd>(s: S, a: Self::V<S>, b: Self::V<S>, c: Self::V<S>) -> Self::V<S> {
                s.$fma(a, b, c)
            }
            #[inline(always)]
            fn v_abs<S: Simd>(s: S, a: Self::V<S>) -> Self::V<S> {
                // `max(a, -a)`: two single-cycle instructions on every target.
                // pulp's default `abs` is assembled from non-`inline(always)`
                // bit-mask helpers that do not inline into SIMD code paths.
                let _ = <S as Simd>::$abs;
                s.$max(a, s.$sub(s.$splat(0.0), a))
            }
            #[inline(always)]
            fn v_min<S: Simd>(s: S, a: Self::V<S>, b: Self::V<S>) -> Self::V<S> {
                s.$min(a, b)
            }
            #[inline(always)]
            fn v_max<S: Simd>(s: S, a: Self::V<S>, b: Self::V<S>) -> Self::V<S> {
                s.$max(a, b)
            }
            #[inline(always)]
            fn v_eq<S: Simd>(s: S, a: Self::V<S>, b: Self::V<S>) -> Self::M<S> {
                s.$eq(a, b)
            }
            #[inline(always)]
            fn v_gt<S: Simd>(s: S, a: Self::V<S>, b: Self::V<S>) -> Self::M<S> {
                // `$gt` is `less_than` here; pulp's `greater_than` is a non-inlined default.
                s.$gt(b, a)
            }
            #[inline(always)]
            fn v_select<S: Simd>(s: S, m: Self::M<S>, t: Self::V<S>, f: Self::V<S>) -> Self::V<S> {
                // Bit-select on the integer view; pulp's float `select` goes through
                // `#[inline]` transmute helpers that do not inline into SIMD code.
                let _ = <S as Simd>::$select;
                bytemuck::cast(s.$uselect(m, bytemuck::cast(t), bytemuck::cast(f)))
            }
            #[inline(always)]
            fn v_ln<S: Simd>(s: S, a: Self::V<S>) -> Self::V<S> {
                $ln(s, a)
            }
            #[inline(always)]
            fn v_exp<S: Simd>(s: S, a: Self::V<S>) -> Self::V<S> {
                $exp(s, a)
            }
            #[inline(always)]
            fn v_sqrt<S: Simd>(s: S, a: Self::V<S>) -> Self::V<S> {
                s.$sqrt(a)
            }
            #[inline(always)]
            fn v_reduce_sum<S: Simd>(s: S, a: Self::V<S>) -> Self {
                s.$rsum(a)
            }
            #[inline(always)]
            fn v_reduce_min<S: Simd>(s: S, a: Self::V<S>) -> Self {
                s.$rmin(a)
            }
            #[inline(always)]
            fn v_reduce_max<S: Simd>(s: S, a: Self::V<S>) -> Self {
                s.$rmax(a)
            }
            #[inline(always)]
            fn to_f64_(self) -> f64 {
                self as f64
            }
            #[inline(always)]
            fn from_f64_(v: f64) -> Self {
                v as $t
            }
        }
    };
}

impl_simd_float!(
    f64, f64s, m64s, F64_LANES,
    as_simd_f64s, as_mut_simd_f64s, partial_load_f64s, partial_store_f64s,
    splat_f64s, add_f64s, sub_f64s, mul_f64s, div_f64s, mul_add_f64s,
    abs_f64s, min_f64s, max_f64s, equal_f64s, less_than_f64s, select_f64s, select_u64s,
    reduce_sum_f64s, reduce_min_f64s, reduce_max_f64s,
    super::math::ln_f64s, super::math::exp_f64s, sqrt_f64s
);

impl_simd_float!(
    f32, f32s, m32s, F32_LANES,
    as_simd_f32s, as_mut_simd_f32s, partial_load_f32s, partial_store_f32s,
    splat_f32s, add_f32s, sub_f32s, mul_f32s, div_f32s, mul_add_f32s,
    abs_f32s, min_f32s, max_f32s, equal_f32s, less_than_f32s, select_f32s, select_u32s,
    reduce_sum_f32s, reduce_min_f32s, reduce_max_f32s,
    super::math::ln_f32s, super::math::exp_f32s, sqrt_f32s
);
