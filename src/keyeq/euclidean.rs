//! The Sugiyama/Euclidean key-equation backend, composing `univariate`'s
//! truncated EEA.
//!
//! The extended Euclidean algorithm on `(x^{N}, S(x))`, stopped at the first
//! remainder of degree below `ceil(N/2)`, yields the Bézout cofactor of
//! `S` — the error locator up to its constant factor — and the stopped
//! remainder — the error evaluator under the same factor. Normalizing the
//! cofactor to `Λ(0) = 1` gives the canonical pair; on decodable inputs the
//! result is Dornstetter-equivalent to Berlekamp–Massey on the same
//! sequence.
//!
//! No polynomial-division or cofactor loop lives here (S1): the engine
//! supplies the operands and the stopping rule, `univariate` supplies the
//! arithmetic.

use fgf::field::Elem;
use fgf::kernel::FieldKernels;
use univariate::truncated_eea;

use crate::error::DecodeError;
use crate::keyeq::{KeyEqScratch, KeyEquation, KeyEquationSolver};

/// The Euclidean/Sugiyama key-equation backend.
///
/// Composes [`univariate::truncated_eea`] and therefore allocates through
/// its internal buffers; it is the permanent cross-check against
/// Berlekamp–Massey (allocation-free in its steady state) rather
/// than the hot path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Euclidean;

impl<F: FieldKernels> KeyEquationSolver<F> for Euclidean {
    fn solve(
        &self,
        sequences: &[&[F::Elem]],
        out: &mut KeyEquation<F>,
        scratch: &mut KeyEqScratch<F>,
    ) -> Result<(), DecodeError> {
        if sequences.len() != 1 {
            return Err(DecodeError::SyndromeGeometry {
                got: sequences.len(),
                expected: 1,
            });
        }
        let sequence = sequences[0];
        let count = sequence.len();
        let stop = count.div_ceil(2);

        // x^N from the pre-reserved operand buffer: zero it, set the one
        // coefficient, and normalization keeps the canonical form.
        scratch.x_pow.set_zero();
        scratch
            .x_pow
            .set_coefficient(count, F::Elem::ONE)
            .map_err(map_error)?;
        scratch
            .series
            .assign_coefficients(sequence)
            .map_err(map_error)?;

        let stopped = truncated_eea(&scratch.x_pow, &scratch.series, stop).map_err(map_error)?;

        // The stopped cofactor is the locator up to its constant factor; a
        // zero constant term means the recurrence admits no normalized
        // synthesis (explicit zero test, never inferred from a division).
        let constant = stopped.b_cofactor.coefficient(0);
        if constant.is_zero() {
            return Err(DecodeError::Inconsistent);
        }
        out.locator.assign_from(&stopped.b_cofactor);
        out.locator.scale_assign(constant.inv());
        out.evaluator.assign_from(&stopped.remainder);
        out.evaluator.scale_assign(constant.inv());

        validate(out, count)
    }
}

/// Enforce the key-equation contract on a solved pair: `deg Λ ≤ N/2` and
/// `deg Ω < deg Λ`.
pub(crate) fn validate<F: FieldKernels>(
    out: &KeyEquation<F>,
    count: usize,
) -> Result<(), DecodeError> {
    let budget = count / 2;
    let degree = out.locator_degree();
    if degree > budget {
        return Err(DecodeError::TooManyErrors {
            errors: degree,
            erasures: 0,
            limit: budget,
        });
    }
    if !out.evaluator.is_zero()
        && out.evaluator.coefficient_count() >= out.locator.coefficient_count()
    {
        return Err(DecodeError::Inconsistent);
    }
    Ok(())
}

/// Map an upstream polynomial failure to the decode error domain. These
/// paths can only fail on buffer reservation; the context distinguishes the
/// operand stage from the arithmetic itself.
pub(crate) fn map_error(error: univariate::PolynomialError) -> DecodeError {
    let context = match error {
        univariate::PolynomialError::Config(_) => "key-equation polynomial buffer",
        _ => "key-equation arithmetic",
    };
    DecodeError::AllocationFailed { context }
}
