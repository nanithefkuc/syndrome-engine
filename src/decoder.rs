//! The scratch-owning decoder: the assembled bounded-distance pipeline.

use alloc::vec::Vec;

use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use poly_ring::{MultipointScratch, Polynomial};

use crate::erasure;
use crate::error::{ConfigError, DecodeError};
use crate::forney;
use crate::keyeq::{KeyEqScratch, KeyEquation, KeyEquationSolver};
use crate::locate;
use crate::params::RsParams;
use crate::syndrome;

/// A correction within the bounded-distance radius, borrowing the decode
/// scratch. `positions[i]` is a code position and `magnitudes[i]` the field
/// element that was added there; the corrected word is already written back
/// when one was supplied.
#[derive(Debug)]
pub struct DecodeOutcome<'a, F: FieldKernels> {
    positions: &'a [usize],
    magnitudes: &'a [F::Elem],
}

impl<F: FieldKernels> DecodeOutcome<'_, F> {
    /// The corrected positions, ascending.
    #[must_use]
    pub fn positions(&self) -> &[usize] {
        self.positions
    }

    /// The error magnitude at each position in [`Self::positions`].
    #[must_use]
    pub fn magnitudes(&self) -> &[F::Elem] {
        self.magnitudes
    }

    /// The number of corrected errors.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.positions.len()
    }
}

/// Caller-owned workspace, sized once from [`RsParams`] and reused across a
/// stream of words. Every steady-state decode path is `*_into` and
/// allocates nothing (S5) — proven by the counting allocator in
/// `tests/zero_alloc.rs` for the default solver. Geometry is checked
/// against the decoder's parameters at every entry.
#[derive(Debug)]
pub struct DecodeScratch<F: FieldKernels> {
    geometry: (usize, usize, usize),
    syndrome_points: Vec<F::Elem>,
    position_points: Vec<F::Elem>,
    word: Polynomial<F>,
    syndrome_eval: MultipointScratch<F>,
    position_eval: MultipointScratch<F>,
    syndromes: Vec<F::Elem>,
    keyeq: KeyEquation<F>,
    keyeq_scratch: KeyEqScratch<F>,
    syndrome_poly: Polynomial<F>,
    total_locator: Polynomial<F>,
    total_evaluator: Polynomial<F>,
    erasure_locator: Polynomial<F>,
    erasure_swap: Polynomial<F>,
    erasure_product: Polynomial<F>,
    modified: Vec<F::Elem>,
    erased_sorted: Vec<usize>,
    locate_values: Vec<F::Elem>,
    positions: Vec<usize>,
    forney: crate::forney::ForneyScratch<F>,
    magnitudes: Vec<F::Elem>,
}

impl<F: FieldKernels> DecodeScratch<F> {
    /// Build and pre-size the workspace for one geometry. This is the only
    /// allocating step in the decode path.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::AllocationFailed`] when a buffer cannot be
    /// reserved.
    pub fn new(params: &RsParams<F>) -> Result<Self, ConfigError> {
        let redundancy = params.redundancy();
        let mut syndrome_points = Vec::new();
        syndrome::points_into(params, &mut syndrome_points);
        let mut position_points = Vec::new();
        locate::position_points_into(params, &mut position_points);

        Ok(Self {
            geometry: params.geometry(),
            syndrome_points,
            position_points,
            word: reserved_polynomial(params.n(), "decode word")?,
            syndrome_eval: MultipointScratch::new(),
            position_eval: MultipointScratch::new(),
            syndromes: reserved_elements::<F>(redundancy, "decode syndromes")?,
            keyeq: KeyEquation::with_capacity(redundancy + 1)?,
            keyeq_scratch: KeyEqScratch::with_capacity(redundancy)?,
            syndrome_poly: reserved_polynomial(redundancy, "decode syndrome series")?,
            total_locator: reserved_polynomial(redundancy + 1, "decode total locator")?,
            total_evaluator: reserved_polynomial(redundancy, "decode total evaluator")?,
            erasure_locator: reserved_polynomial(redundancy + 1, "erasure locator")?,
            erasure_swap: reserved_polynomial(redundancy + 1, "erasure locator swap")?,
            erasure_product: reserved_polynomial(redundancy, "modified syndromes")?,
            modified: reserved_elements::<F>(redundancy, "modified syndrome sequence")?,
            erased_sorted: reserved_positions(redundancy, "sorted erasure positions")?,
            locate_values: reserved_elements::<F>(params.n(), "decode locate values")?,
            positions: reserved_positions(params.n(), "decode positions")?,
            forney: crate::forney::ForneyScratch::with_capacity(redundancy + 1)?,
            magnitudes: reserved_elements::<F>(redundancy + 1, "decode magnitudes")?,
        })
    }
}

fn reserved_polynomial<F: FieldKernels>(
    capacity: usize,
    context: &'static str,
) -> Result<Polynomial<F>, ConfigError> {
    let mut polynomial = Polynomial::zero();
    polynomial
        .resize_coefficients(capacity)
        .map_err(|_| ConfigError::AllocationFailed {
            context,
            elements: capacity,
            element_size: F::BYTES,
        })?;
    polynomial.set_zero();
    Ok(polynomial)
}

pub(crate) fn reserved_elements<F: FieldKernels>(
    capacity: usize,
    context: &'static str,
) -> Result<Vec<F::Elem>, ConfigError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| ConfigError::AllocationFailed {
            context,
            elements: capacity,
            element_size: F::BYTES,
        })?;
    Ok(values)
}

fn reserved_positions(capacity: usize, context: &'static str) -> Result<Vec<usize>, ConfigError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| ConfigError::AllocationFailed {
            context,
            elements: capacity,
            element_size: core::mem::size_of::<usize>(),
        })?;
    Ok(values)
}

/// The bounded-distance decoder: syndromes, key equation, Chien search,
/// Forney, and the miscorrection guards, over one code geometry and one
/// key-equation backend.
///
/// The default solver is [`crate::Adaptive`]: Berlekamp–Massey (the
/// allocation-free LFSR synthesis) on the hot path, with the Euclidean
/// backend past the measured crossover; [`crate::Euclidean`] composes
/// `poly-ring`'s allocating truncated EEA and is the cross-check backend.
pub struct Decoder<F: FieldKernels, S: KeyEquationSolver<F> = crate::cost::Adaptive> {
    params: RsParams<F>,
    solver: S,
}

impl<F: FieldKernels, S: KeyEquationSolver<F>> Decoder<F, S> {
    /// Build a decoder from validated parameters and a key-equation solver.
    pub fn new(params: RsParams<F>, solver: S) -> Self {
        Self { params, solver }
    }

    /// The code geometry this decoder is built for.
    #[must_use]
    pub fn params(&self) -> &RsParams<F> {
        &self.params
    }

    /// Allocate a decode workspace sized for this decoder's geometry.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::AllocationFailed`] when a buffer cannot be
    /// reserved.
    pub fn scratch(&self) -> Result<DecodeScratch<F>, ConfigError> {
        DecodeScratch::new(&self.params)
    }

    /// Compute the `n - k` syndromes of a packed received word into the
    /// scratch and return them. The steady-state form of the public
    /// [`crate::syndromes`] entry, for consumers that decode from
    /// syndromes.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError::WordGeometry`] when `received` does not hold
    /// exactly `n` packed field elements, and
    /// [`DecodeError::AllocationFailed`] when an evaluation buffer cannot
    /// be reserved.
    pub fn syndromes_into<'a>(
        &self,
        received: &[u8],
        scratch: &'a mut DecodeScratch<F>,
    ) -> Result<&'a [F::Elem], DecodeError> {
        self.check_scratch(scratch)?;
        syndrome::compute_into(
            &self.params,
            received,
            &scratch.syndrome_points,
            &mut scratch.word,
            &mut scratch.syndrome_eval,
            &mut scratch.syndromes,
        )?;
        Ok(&scratch.syndromes)
    }

    /// Decode a received word in place: the correction is applied to
    /// `received` and the pattern is returned. On any error the word is
    /// left untouched.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError::ScratchMismatch`] when the scratch was sized
    /// for another geometry, [`DecodeError::WordGeometry`] for a wrong word
    /// length, [`DecodeError::TooManyErrors`] beyond the
    /// `2ν + ρ ≤ d - 1` budget, and [`DecodeError::DegreeMismatch`] /
    /// [`DecodeError::Inconsistent`] for the two detected-miscorrection
    /// guards (root count, re-encode to zero syndromes).
    pub fn decode_into<'a>(
        &self,
        received: &mut [u8],
        scratch: &'a mut DecodeScratch<F>,
    ) -> Result<DecodeOutcome<'a, F>, DecodeError> {
        self.check_scratch(scratch)?;
        self.compute_syndromes(received, scratch)?;
        self.solve_pure(scratch)?;
        self.finish_into(received, scratch)
    }

    /// Apply the located correction and run the re-encode guard: the
    /// corrected word's syndromes must all vanish, or the correction was
    /// spurious and is undone.
    fn finish_into<'a>(
        &self,
        received: &mut [u8],
        scratch: &'a mut DecodeScratch<F>,
    ) -> Result<DecodeOutcome<'a, F>, DecodeError> {
        apply_pattern::<F>(received, &scratch.positions, &scratch.magnitudes);
        self.compute_syndromes(received, scratch)?;
        if scratch.syndromes.iter().any(|value| !value.is_zero()) {
            // Spurious correction: undo it and report the detection.
            apply_pattern::<F>(received, &scratch.positions, &scratch.magnitudes);
            return Err(DecodeError::Inconsistent);
        }
        Ok(Self::outcome(scratch))
    }

    /// Errors-and-erasures decode: `erased` are known-erased positions,
    /// folded into the key equation as Forney (modified) syndromes. The
    /// budget widens to `2ν + ρ ≤ n - k`; flagged and unflagged corruption
    /// is corrected in one pass, and the word is left untouched on any
    /// error.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::decode_into`], plus
    /// [`DecodeError::TooManyErasures`] when more than `n - k` positions
    /// are erased, [`DecodeError::ErasurePosition`] for a position at or
    /// beyond `n`, and [`DecodeError::DuplicateErasure`] for a repeated
    /// position.
    pub fn decode_with_erasures_into<'a>(
        &self,
        received: &mut [u8],
        erased: &[usize],
        scratch: &'a mut DecodeScratch<F>,
    ) -> Result<DecodeOutcome<'a, F>, DecodeError> {
        self.check_scratch(scratch)?;
        if erased.len() > self.params.redundancy() {
            return Err(DecodeError::TooManyErasures {
                count: erased.len(),
                limit: self.params.redundancy(),
            });
        }
        scratch.erased_sorted.clear();
        scratch.erased_sorted.extend_from_slice(erased);
        scratch.erased_sorted.sort_unstable();
        for window in scratch.erased_sorted.windows(2) {
            if window[0] == window[1] {
                return Err(DecodeError::DuplicateErasure {
                    position: window[0],
                });
            }
        }
        if let Some(&position) = scratch.erased_sorted.last()
            && position >= self.params.n()
        {
            return Err(DecodeError::ErasurePosition {
                got: position,
                limit: self.params.n(),
            });
        }
        self.compute_syndromes(received, scratch)?;
        self.solve_with_erasures(scratch)?;
        self.finish_into(received, scratch)
    }

    fn compute_syndromes(
        &self,
        received: &[u8],
        scratch: &mut DecodeScratch<F>,
    ) -> Result<(), DecodeError> {
        syndrome::compute_into(
            &self.params,
            received,
            &scratch.syndrome_points,
            &mut scratch.word,
            &mut scratch.syndrome_eval,
            &mut scratch.syndromes,
        )
    }

    fn outcome(scratch: &DecodeScratch<F>) -> DecodeOutcome<'_, F> {
        DecodeOutcome {
            positions: &scratch.positions,
            magnitudes: &scratch.magnitudes,
        }
    }

    /// Decode from supplied syndromes, without a received word. The same
    /// guarantees as [`Self::decode_into`]; the certificate here is that
    /// the returned pattern reproduces the supplied syndromes exactly.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError::ScratchMismatch`] when the scratch was sized
    /// for another geometry, [`DecodeError::SyndromeGeometry`] when the
    /// syndrome count differs from `n - k`, and the decode errors of
    /// [`Self::decode_into`].
    pub fn decode_syndromes_into<'a>(
        &self,
        syndromes: &[F::Elem],
        scratch: &'a mut DecodeScratch<F>,
    ) -> Result<DecodeOutcome<'a, F>, DecodeError> {
        self.check_scratch(scratch)?;
        if syndromes.len() != self.params.syndrome_count() {
            return Err(DecodeError::SyndromeGeometry {
                got: syndromes.len(),
                expected: self.params.syndrome_count(),
            });
        }
        // Copy the caller's syndromes into the scratch buffer the core
        // solves from (reusing capacity; nothing allocates).
        scratch.syndromes.clear();
        scratch.syndromes.extend_from_slice(syndromes);
        self.solve_pure(scratch)?;
        // Syndromes-only certificate: Σ e_p X_p^{b+j} must equal S_j.
        if !self.pattern_matches_syndromes(syndromes, scratch) {
            return Err(DecodeError::Inconsistent);
        }
        Ok(Self::outcome(scratch))
    }

    fn check_scratch(&self, scratch: &DecodeScratch<F>) -> Result<(), DecodeError> {
        if scratch.geometry != self.params.geometry() {
            return Err(DecodeError::ScratchMismatch {
                expected: self.params.n(),
                got: scratch.geometry.0,
            });
        }
        Ok(())
    }

    /// The pure-error core: key equation from `scratch.syndromes` (already
    /// computed), then the shared locate/Forney tail.
    fn solve_pure(&self, scratch: &mut DecodeScratch<F>) -> Result<(), DecodeError> {
        if self.params.syndrome_count() == 0 {
            scratch.positions.clear();
            scratch.magnitudes.clear();
            return Ok(());
        }
        self.solver.solve(
            &[&scratch.syndromes],
            &mut scratch.keyeq,
            &mut scratch.keyeq_scratch,
        )?;
        scratch.total_locator.assign_from(scratch.keyeq.locator());
        self.locate_and_forney(scratch)
    }

    /// The errors-and-erasures core: erasure locator, Forney (modified)
    /// syndromes, key equation on the errors-only suffix, product locator,
    /// then the shared locate/Forney tail.
    fn solve_with_erasures(&self, scratch: &mut DecodeScratch<F>) -> Result<(), DecodeError> {
        let count = self.params.syndrome_count();
        let erasures = scratch.erased_sorted.len();
        if count == 0 || erasures == 0 {
            return self.solve_pure(scratch);
        }
        erasure::locator_into(
            &self.params,
            &scratch.erased_sorted,
            &mut scratch.erasure_locator,
            &mut scratch.erasure_swap,
        )?;
        scratch
            .syndrome_poly
            .assign_coefficients(&scratch.syndromes)?;
        scratch.erasure_locator.multiply_truncated_into(
            &scratch.syndrome_poly,
            count,
            &mut scratch.erasure_product,
        )?;
        erasure::modified_into(
            count,
            erasures,
            &scratch.erasure_product,
            &mut scratch.modified,
        );
        // The suffix is a clean errors-only syndrome sequence; the
        // solver's budget on it is exactly `2ν ≤ n - k - ρ`. Report the
        // budget in the decoder's terms when it rejects.
        self.solver
            .solve(
                &[&scratch.modified],
                &mut scratch.keyeq,
                &mut scratch.keyeq_scratch,
            )
            .map_err(|error| match error {
                DecodeError::TooManyErrors { errors, .. } => DecodeError::TooManyErrors {
                    errors,
                    erasures,
                    limit: count,
                },
                other => other,
            })?;
        scratch
            .keyeq
            .locator()
            .multiply_into(&scratch.erasure_locator, &mut scratch.total_locator)?;
        self.locate_and_forney(scratch)
    }

    /// The shared tail: Chien over the total locator, the root-count
    /// guard, the total evaluator, and Forney.
    fn locate_and_forney(&self, scratch: &mut DecodeScratch<F>) -> Result<(), DecodeError> {
        let count = self.params.syndrome_count();
        if scratch.total_locator.is_zero() {
            return Err(DecodeError::Inconsistent);
        }
        locate::locate_into(
            &scratch.total_locator,
            &scratch.position_points,
            &mut scratch.position_eval,
            &mut scratch.locate_values,
            &mut scratch.positions,
        )?;
        // Miscorrection guard 1: the total locator must split into
        // distinct roots over the position set.
        let degree = scratch.total_locator.coefficient_count() - 1;
        if scratch.positions.len() != degree {
            return Err(DecodeError::DegreeMismatch {
                expected: degree,
                got: scratch.positions.len(),
            });
        }
        scratch
            .syndrome_poly
            .assign_coefficients(&scratch.syndromes)?;
        scratch.total_locator.multiply_truncated_into(
            &scratch.syndrome_poly,
            count,
            &mut scratch.total_evaluator,
        )?;
        forney::magnitudes_into(
            &self.params,
            &scratch.total_locator,
            &scratch.total_evaluator,
            &scratch.positions,
            &mut scratch.forney,
            &mut scratch.magnitudes,
        )?;
        Ok(())
    }

    /// Whether `Σ magnitudes[i]·X_{positions[i]}^{b+j} == S_j` for all `j`.
    /// One running power per position, an engine-native scalar
    /// accumulation over `fgf` elements.
    fn pattern_matches_syndromes(
        &self,
        syndromes: &[F::Elem],
        scratch: &mut DecodeScratch<F>,
    ) -> bool {
        scratch.locate_values.clear();
        scratch.locate_values.extend_from_slice(syndromes);
        let alpha = <F as Field>::GENERATOR;
        // Index walk: `usize` and `F::Elem` are `Copy`, so no pattern
        // buffer is duplicated and nothing allocates.
        for index in 0..scratch.positions.len() {
            let position = scratch.positions[index];
            let magnitude = scratch.magnitudes[index];
            // X_p^b = (α^p)^b — the offset composes into the exponent by
            // multiplication, taken as a power of a power so no u64
            // product can overflow on wide fields.
            let step = alpha.pow(position as u64);
            let locator = step.pow(self.params.b() as u64);
            let mut coefficient = magnitude.mul(locator);
            for value in &mut scratch.locate_values {
                *value = value.add(coefficient);
                coefficient = coefficient.mul(step);
            }
        }
        scratch.locate_values.iter().all(|value| value.is_zero())
    }
}

/// XOR the packed pattern into a received word (addition in the field).
fn apply_pattern<F: FieldKernels>(
    received: &mut [u8],
    positions: &[usize],
    magnitudes: &[F::Elem],
) {
    for (position, magnitude) in positions.iter().zip(magnitudes) {
        let start = position * F::BYTES;
        let current = F::read(&received[start..start + F::BYTES]);
        F::write(
            &mut received[start..start + F::BYTES],
            current.add(*magnitude),
        );
    }
}
