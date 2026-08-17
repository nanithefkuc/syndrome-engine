//! Errors-and-erasures: the erasure locator and the Forney (modified)
//! syndrome transform.
//!
//! Erasures are positions the channel already flagged. The engine forms
//! the erasure locator `Γ(x) = Π_f (1 + X_f x)` over those positions, the
//! modified syndromes `T(x) = Γ(x)·S(x) (mod x^{n-k})`, and solves the key
//! equation on the suffix `T_ρ, …, T_{n-k-1}`: for `j ≥ ρ` the erasure
//! contributions have cancelled exactly, so the suffix is a clean
//! errors-only syndrome sequence and Berlekamp–Massey / Euclid recover the
//! *error* locator `Λ` from it. The product locator `Λ·Γ` then drives the
//! Chien search and Forney for both errors and erasures at once — the
//! evaluator identity `Ω = Λ·Γ·S (mod x^{n-k})` yields
//! `e_p = X_p^{1-b}·Ω(X_p^{-1}) / (ΛΓ)'(X_p^{-1})` at every corrupted
//! position, flagged or not. The correction guarantee is the standard
//! `2ν + ρ ≤ d - 1`.
//!
//! This crate does *not* implement pure-erasure fast paths; that is
//! `systematic-rs`'s object (see the charter's scope boundary).

use alloc::vec::Vec;

use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use univariate::Polynomial;

use crate::error::DecodeError;
use crate::params::RsParams;

/// Build the erasure locator `Γ(x) = Π (1 + X_f x)` over the erased
/// positions into `locator`, using `swap` as the product scratch. Both
/// buffers are caller-owned and reused; nothing allocates.
///
/// # Errors
///
/// Returns [`DecodeError::AllocationFailed`] when a buffer cannot be
/// reserved.
pub fn locator_into<F: FieldKernels>(
    _params: &RsParams<F>,
    positions: &[usize],
    locator: &mut Polynomial<F>,
    swap: &mut Polynomial<F>,
) -> Result<(), DecodeError> {
    locator.assign_coefficients(&[F::Elem::ONE])?;
    for &position in positions {
        // Γ ← Γ + X_f·x·Γ = Γ·(1 + X_f·x): one scaled-shift fold per
        // erased position, composed from `univariate`'s ring primitives.
        let erasure_locator = <F as Field>::GENERATOR.pow(position as u64);
        swap.assign_from(locator);
        swap.add_scaled_shifted_assign(erasure_locator, locator, 1)?;
        locator.assign_from(swap);
    }
    Ok(())
}

/// Write the modified-syndrome suffix `T_ρ, …, T_{count-1}` into `out`:
/// the errors-only sequence the key equation solves. Coefficients beyond
/// the stored product degree are zero, so the output length is exactly
/// `count - ρ`.
pub fn modified_into<F: FieldKernels>(
    count: usize,
    erasure_count: usize,
    product: &Polynomial<F>,
    out: &mut Vec<F::Elem>,
) {
    out.clear();
    for degree in erasure_count..count {
        out.push(product.coefficient(degree));
    }
}
