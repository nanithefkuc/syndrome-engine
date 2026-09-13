//! Measured crossover selectors.
//!
//! A selector is a pure function of a small cost key; it performs no CPU
//! detection and reads no environment variable. The crossover constants
//! and the exact measurement commands and hardware that set them are
//! recorded in `BENCHMARKS.md`; source carries only a one-line pointer (a
//! crossover changed without a re-measurement is a regression with a green
//! test suite).

use fgf::kernel::FieldKernels;

use crate::error::DecodeError;
use crate::keyeq::{BerlekampMassey, Euclidean, KeyEqScratch, KeyEquation, KeyEquationSolver};

/// Syndrome-sequence length through which Berlekamp–Massey's scalar
/// recurrence beats the Euclidean backend's `poly-ring` truncated EEA.
/// Measured on the record in `BENCHMARKS.md`: Berlekamp–Massey leads
/// through `N ≈ 48` syndromes (1.9× at `N = 10`, parity at `N = 48`) and
/// the packed-kernel EEA wins beyond (2.2× at `N = 123`, 10× at
/// `N = 1095`); the threshold sits at the parity band, erring toward the
/// allocation-free backend. It exists so the choice stays a measured,
/// reversible decision rather than a hardcoded backend.
pub const BM_EUCLIDEAN_CROSSOVER: usize = 48;

/// Key-equation backend choice for one solve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverBackend {
    /// Berlekamp–Massey LFSR synthesis over the sequence.
    BerlekampMassey,
    /// Euclidean/Sugiyama through `poly-ring`'s truncated EEA.
    Euclidean,
}

/// Cost key for choosing a key-equation backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SolverCostKey {
    /// Syndrome-sequence length `N`.
    pub syndromes: usize,
    /// Number of elements in the field.
    pub field_order: u128,
}

/// Choose the key-equation backend. Pure. See `BENCHMARKS.md`.
#[must_use]
pub fn select_solver(key: SolverCostKey) -> SolverBackend {
    if key.syndromes > BM_EUCLIDEAN_CROSSOVER {
        SolverBackend::Euclidean
    } else {
        SolverBackend::BerlekampMassey
    }
}

/// The cost-selected key-equation solver: Berlekamp–Massey on the hot
/// path, the Euclidean backend past the measured crossover. The default
/// decoder solver.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Adaptive;

impl<F: FieldKernels> KeyEquationSolver<F> for Adaptive {
    fn solve(
        &self,
        sequences: &[&[F::Elem]],
        out: &mut KeyEquation<F>,
        scratch: &mut KeyEqScratch<F>,
    ) -> Result<(), DecodeError> {
        let count = sequences
            .first()
            .map_or(0, |sequence: &&[F::Elem]| sequence.len());
        match select_solver(SolverCostKey {
            syndromes: count,
            field_order: F::ORDER,
        }) {
            SolverBackend::BerlekampMassey => BerlekampMassey.solve(sequences, out, scratch),
            SolverBackend::Euclidean => Euclidean.solve(sequences, out, scratch),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selector_picks_bm_on_the_measured_range() {
        for syndromes in [0_usize, 2, 10, 32, 40, 48] {
            assert_eq!(
                select_solver(SolverCostKey {
                    syndromes,
                    field_order: 256,
                }),
                SolverBackend::BerlekampMassey
            );
        }
        assert_eq!(
            select_solver(SolverCostKey {
                syndromes: BM_EUCLIDEAN_CROSSOVER + 1,
                field_order: 256,
            }),
            SolverBackend::Euclidean
        );
    }

    #[test]
    fn adaptive_dispatches_by_sequence_length() {
        use crate::keyeq::{KeyEqScratch, KeyEquation, KeyEquationSolver};
        use fgf::Gf8B;

        // A degree-2 LFSR sequence: 1, 0, 1, 0, ... has locator 1 + x^2
        // over GF(2^8) only if the field contains the roots; instead use a
        // synthetic sequence and check totality + identity under Adaptive.
        let sequence = [
            fgf::gf8b::Elem::from_raw(1),
            fgf::gf8b::Elem::from_raw(2),
            fgf::gf8b::Elem::from_raw(4),
            fgf::gf8b::Elem::from_raw(8),
            fgf::gf8b::Elem::from_raw(16),
            fgf::gf8b::Elem::from_raw(32),
        ];
        let mut out = KeyEquation::<Gf8B>::with_capacity(8).expect("out");
        let mut scratch = KeyEqScratch::with_capacity(sequence.len()).expect("scratch");
        Adaptive
            .solve(&[&sequence], &mut out, &mut scratch)
            .expect("adaptive solve");
        let series = poly_ring::Polynomial::from_coefficients(&sequence).expect("series");
        let product = out
            .locator()
            .multiply_truncated(&series, sequence.len())
            .expect("product");
        assert_eq!(&product, out.evaluator());
        assert_eq!(out.locator().coefficient(0), fgf::gf8b::Elem::from_raw(1));
    }
}
