//! Syndrome computation: the decoder's entry point.
//!
//! The syndromes of a received word `R(x) = Σ_p r_p x^p` (coefficient `r_p`
//! at degree `p`) are the `n - k` point evaluations
//! `S_j = R(α^{b+j})` at the code's consecutive roots. The engine owns the
//! index convention (offset `b`, primitive element `α = F::GENERATOR`,
//! low-degree-first coefficient order); the arithmetic is `poly-ring`'s
//! multipoint evaluation — never a private Horner loop.
//!
//! The defining property: an uncorrupted codeword vanishes at every
//! consecutive root, so its syndromes are identically zero and the syndrome
//! vector depends only on the error pattern.

use alloc::vec::Vec;

use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use poly_ring::{MultipointScratch, Polynomial, evaluate_multipoint_into};

use crate::error::DecodeError;
use crate::params::RsParams;

/// Write the syndrome evaluation points `α^{b}, α^{b+1}, …, α^{b+n-k-1}`
/// into `points`, stepping one field multiplication per index.
pub(crate) fn points_into<F: FieldKernels>(params: &RsParams<F>, points: &mut Vec<F::Elem>) {
    points.clear();
    let alpha = <F as Field>::GENERATOR;
    let mut point = alpha.pow(params.b() as u64);
    for _ in 0..params.syndrome_count() {
        points.push(point);
        point = point.mul(alpha);
    }
}

/// Compute the syndromes of a packed received word into `values`.
///
/// `word` holds the received word as a scratch polynomial, `points` the
/// evaluation points (from `points_into`), and `eval` the reusable
/// multipoint machinery; all three are caller-owned so the steady state
/// allocates nothing.
///
/// # Errors
///
/// Returns [`DecodeError::WordGeometry`] when `received` does not hold
/// exactly `n` packed field elements, and [`DecodeError::AllocationFailed`]
/// when an upstream buffer cannot be reserved.
pub fn compute_into<F: FieldKernels>(
    params: &RsParams<F>,
    received: &[u8],
    points: &[F::Elem],
    word: &mut Polynomial<F>,
    eval: &mut MultipointScratch<F>,
    values: &mut Vec<F::Elem>,
) -> Result<(), DecodeError> {
    let expected_bytes = params.n() * F::BYTES;
    if received.len() != expected_bytes {
        return Err(DecodeError::WordGeometry {
            got: received.len() / F::BYTES,
            expected: params.n(),
        });
    }
    if params.syndrome_count() == 0 {
        values.clear();
        return Ok(());
    }
    word.assign_packed(received)?;
    evaluate_multipoint_into(word, points, eval, values)?;
    Ok(())
}

/// Compute the `n - k` syndromes of a packed received word.
///
/// Allocating convenience form of the internal scratch path, for diagnostics
/// and for consumers seeding the syndromes-only decode entry
/// ([`crate::Decoder::decode_syndromes_into`]).
///
/// # Errors
///
/// Returns [`DecodeError::WordGeometry`] when `received` does not hold
/// exactly `n` packed field elements, and [`DecodeError::AllocationFailed`]
/// when an evaluation buffer cannot be reserved.
pub fn syndromes<F: FieldKernels>(
    params: &RsParams<F>,
    received: &[u8],
) -> Result<Vec<F::Elem>, DecodeError> {
    let mut points = Vec::new();
    points_into(params, &mut points);
    let mut word = Polynomial::zero();
    let mut eval = MultipointScratch::new();
    let mut values = Vec::new();
    compute_into(params, received, &points, &mut word, &mut eval, &mut values)?;
    Ok(values)
}
