//! Runtime instruction-set dispatch.
//!
//! All SIMD entry points go through [`dispatch`], which normally forwards
//! to [`pulp::Arch`] (best ISA detected once per process). In test builds a
//! thread-local override can force a specific ISA so the whole test-suite
//! exercises every code path the host CPU supports.

use pulp::{Arch, Simd, WithSimd};

/// Instruction sets this crate can dispatch to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Isa {
    /// Portable scalar fallback.
    Scalar,
    /// x86-64 AVX2 + FMA ("x86-64-v3").
    Avx2,
    /// x86-64 AVX-512 ("x86-64-v4"); requires the `avx512` crate feature.
    Avx512,
    /// AArch64 NEON.
    Neon,
}

impl Isa {
    /// Instruction sets usable on this CPU (always includes `Scalar`).
    pub fn available() -> Vec<Isa> {
        let mut v = vec![Isa::Scalar];
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if pulp::x86::V3::try_new().is_some() {
                v.push(Isa::Avx2);
            }
            #[cfg(feature = "avx512")]
            if pulp::x86::V4::try_new().is_some() {
                v.push(Isa::Avx512);
            }
        }
        #[cfg(target_arch = "aarch64")]
        if pulp::aarch64::Neon::try_new().is_some() {
            v.push(Isa::Neon);
        }
        v
    }

    /// The instruction set automatic dispatch selects on this CPU.
    pub fn detected() -> Isa {
        struct Probe;
        impl WithSimd for Probe {
            type Output = (usize, usize);
            #[inline(always)]
            fn with_simd<S: Simd>(self, _s: S) -> (usize, usize) {
                (S::F64_LANES, S::REGISTER_COUNT)
            }
        }
        let (lanes, _regs) = Arch::new().dispatch(Probe);
        match lanes {
            1 => Isa::Scalar,
            8 => Isa::Avx512,
            4 => Isa::Avx2,
            _ => {
                if cfg!(target_arch = "aarch64") {
                    Isa::Neon
                } else {
                    Isa::Scalar
                }
            }
        }
    }
}

#[cfg(test)]
thread_local! {
    static FORCED: core::cell::Cell<Option<Isa>> = const { core::cell::Cell::new(None) };
}

/// Runs `f` with every SIMD call on this thread forced to `isa`.
#[cfg(test)]
pub(crate) fn with_isa<R>(isa: Isa, f: impl FnOnce() -> R) -> R {
    let prev = FORCED.with(|c| c.replace(Some(isa)));
    let r = f();
    FORCED.with(|c| c.set(prev));
    r
}

/// Runs `f` once for every ISA available on this CPU.
#[cfg(test)]
pub(crate) fn for_each_isa(mut f: impl FnMut(Isa)) {
    for isa in Isa::available() {
        with_isa(isa, || f(isa));
    }
}

#[cfg(test)]
fn forced<Op: WithSimd>(isa: Isa, op: Op) -> Op::Output {
    match isa {
        Isa::Scalar => Simd::vectorize(pulp::Scalar::new(), op),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Isa::Avx2 => Simd::vectorize(pulp::x86::V3::try_new().expect("AVX2 unavailable"), op),
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "avx512"))]
        Isa::Avx512 => Simd::vectorize(pulp::x86::V4::try_new().expect("AVX-512 unavailable"), op),
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => Simd::vectorize(pulp::aarch64::Neon::try_new().expect("NEON unavailable"), op),
        #[allow(unreachable_patterns)]
        _ => panic!("ISA {isa:?} not supported on this target"),
    }
}

/// Dispatches a SIMD operation to the best available instruction set.
#[inline]
pub(crate) fn dispatch<Op: WithSimd>(op: Op) -> Op::Output {
    #[cfg(test)]
    if let Some(isa) = FORCED.with(|c| c.get()) {
        return forced(isa, op);
    }
    Arch::new().dispatch(op)
}
