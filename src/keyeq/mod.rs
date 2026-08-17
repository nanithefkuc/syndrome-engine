//! Key-equation solvers: one contract, two Dornstetter-equivalent backends.
//!
//! The key equation is `Λ(x)·S(x) ≡ Ω(x) (mod x^{N})` for a length-`N`
//! syndrome sequence `S_0, …, S_{N-1}`, solved for the error locator `Λ`
//! (normalized to `Λ(0) = 1`) and the error evaluator `Ω` with
//! `deg Ω < deg Λ ≤ N/2`. Berlekamp–Massey (engine-native LFSR synthesis)
//! and Euclidean/Sugiyama (`univariate`'s truncated EEA) compute the same
//! pair on every decodable input; mutual cross-checking is the crate's
//! strongest oracle.
//!
//! The trait takes one slice of syndrome slices so the deferred
//! multi-sequence generalization (a common locator synthesized from several
//! interleaved syndrome rows) is another implementation behind the same
//! contract, not a new interface.
mod berlekamp;
mod equation;
mod euclidean;

pub use berlekamp::BerlekampMassey;
pub use equation::{KeyEqScratch, KeyEquation};
pub use euclidean::Euclidean;

use fgf::kernel::FieldKernels;

use crate::error::DecodeError;

/// Solve the key equation for the error-locator `Λ` and error-evaluator `Ω`.
///
/// Backends: [`Euclidean`] (truncated EEA over the polynomial ring) and
/// [`crate::BerlekampMassey`] (LFSR synthesis over the scalar sequence).
/// They are Dornstetter-equivalent; both are cross-checked on every fixture.
///
/// # Errors
///
/// Returns [`DecodeError::TooManyErrors`] when the synthesized locator
/// degree exceeds `N/2`, and [`DecodeError::Inconsistent`] when the pair
/// admits no normalized solution with `deg Ω < deg Λ`.
pub trait KeyEquationSolver<F: FieldKernels> {
    /// Solve for `(Λ, Ω)` given the syndrome sequences.
    ///
    /// `sequences[0]` holds `S_0..S_{N-1}` (after any Forney-syndrome
    /// modification for erasures). `out` receives the normalized pair;
    /// `scratch` is caller-owned and sized for the decoder's geometry.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError::SyndromeGeometry`] for anything but a single
    /// sequence (until the multi-sequence backend ships), plus the errors
    /// named on the trait.
    fn solve(
        &self,
        sequences: &[&[F::Elem]],
        out: &mut KeyEquation<F>,
        scratch: &mut KeyEqScratch<F>,
    ) -> Result<(), DecodeError>;
}
