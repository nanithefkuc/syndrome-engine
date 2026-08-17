//! Berlekamp–Massey: LFSR synthesis over the scalar syndrome sequence.
//!
//! The one algorithm this crate owns outright rather than composing
//! `univariate` (settled decision #2): its input is a *sequence* — a
//! decoder concept — not a polynomial. The algorithm runs `N` iterations,
//! each computing a discrepancy `δ = S_r + Σ_{i≥1} Λ_i S_{r-i}` with an
//! explicit `is_zero()` test (S7), and on a nonzero discrepancy updates
//! the connection polynomial by a scalar-times-shifted copy of the
//! previous one, extending the register length only when forced. The
//! final connection polynomial *is* the error locator, normalized to
//! `Λ(0) = 1` by construction; the evaluator is the truncated product
//! `Λ·S mod x^{N}`, the same normalization the Euclidean backend produces,
//! so the two agree exactly on every decodable input (Dornstetter 1987;
//! invariant S3).
//!
//! O(N²) field operations with a tiny constant, entirely over caller-owned
//! scratch: the steady-state solve allocates nothing.

use alloc::vec::Vec;

use fgf::field::Elem;
use fgf::kernel::FieldKernels;

use crate::error::DecodeError;
use crate::keyeq::euclidean::validate;
use crate::keyeq::{KeyEqScratch, KeyEquation, KeyEquationSolver};

/// The Berlekamp–Massey key-equation backend: LFSR synthesis over the
/// syndrome sequence, allocation-free in its steady state. This is the hot
/// path; [`crate::Euclidean`] is its Dornstetter cross-check.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BerlekampMassey;

impl<F: FieldKernels> KeyEquationSolver<F> for BerlekampMassey {
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

        // Registers over pre-reserved buffers, managed by explicit lengths
        // so nothing allocates or reallocates: the connection polynomial
        // `connection` (index = degree, constant term always one), the
        // snapshot `previous` from before the last register-length change,
        // and the syndrome series operand.
        let capacity = scratch.bm.capacity;
        scratch.bm.reset();
        let BmScratch {
            connection,
            previous,
            swap,
            ..
        } = &mut scratch.bm;

        let mut connection_len = 1_usize;
        let mut previous_len = 1_usize;
        let mut length = 0_usize; // LFSR register length L
        let mut shift = 1_usize; // steps since `previous` was captured
        let mut last_discrepancy = F::Elem::ONE;

        for (r, symptom) in sequence.iter().enumerate() {
            // Discrepancy: δ = S_r + Σ_{i=1..min(L, r)} Λ_i·S_{r-i}.
            let mut discrepancy = *symptom;
            let terms = length.min(r);
            for i in 1..=terms {
                discrepancy = discrepancy.add(connection[i].mul(sequence[r - i]));
            }
            if discrepancy.is_zero() {
                shift += 1;
                continue;
            }
            // Massey's update, in three ordered steps:
            //   T = C;   C = C + (d/b)·x^shift·B;   B = T, b = d, L grows.
            // The current update always uses the *previous* snapshot B and
            // the *previous* discrepancy b; the snapshot for future
            // updates is the connection polynomial from before this
            // update, captured in the swap register.
            let factor = discrepancy.mul(last_discrepancy.inv());
            let length_change = 2 * length <= r;
            let mut swap_len = 0_usize;
            if length_change {
                swap[..connection_len].copy_from_slice(&connection[..connection_len]);
                swap_len = connection_len;
            }
            // Λ ← Λ + factor·x^shift·B: zero the widened gap, fold the
            // update in, then trim trailing zeros back.
            let target = (shift + previous_len).min(capacity);
            if target > connection_len {
                for coefficient in &mut connection[connection_len..target] {
                    *coefficient = F::Elem::ZERO;
                }
                connection_len = target;
            }
            for i in 0..previous_len {
                let update = factor.mul(previous[i]);
                let coefficient = &mut connection[i + shift];
                *coefficient = coefficient.add(update);
            }
            while connection_len > 1 && connection[connection_len - 1].is_zero() {
                connection_len -= 1;
            }
            if length_change {
                previous[..swap_len].copy_from_slice(&swap[..swap_len]);
                previous_len = swap_len;
                last_discrepancy = discrepancy;
                length = r + 1 - length;
                shift = 1;
            } else {
                shift += 1;
            }
            // The discrepancy loop reads coefficients up to the register
            // length L, and `deg Λ ≤ L` holds once L is updated — but a
            // cancellation can trim the stored length below L, leaving
            // stale coefficients in the gap. Zero them: past the degree
            // they are zero by definition.
            if connection_len <= length {
                for coefficient in &mut connection[connection_len..=length] {
                    *coefficient = F::Elem::ZERO;
                }
            }
        }

        out.locator
            .assign_coefficients(&connection[..connection_len])?;
        scratch.series.assign_coefficients(sequence)?;
        out.locator
            .multiply_truncated_into(&scratch.series, count, &mut out.evaluator)?;
        validate(out, count)
    }
}

/// Caller-owned Berlekamp–Massey registers, pre-reserved once and reused
/// by explicit lengths.
#[derive(Debug)]
pub struct BmScratch<F: FieldKernels> {
    connection: Vec<F::Elem>,
    previous: Vec<F::Elem>,
    swap: Vec<F::Elem>,
    capacity: usize,
}

impl<F: FieldKernels> BmScratch<F> {
    /// Registers reserved for sequences up to `capacity` syndromes.
    ///
    /// # Errors
    ///
    /// Returns [`crate::ConfigError::AllocationFailed`] when a register
    /// cannot be reserved.
    pub(crate) fn with_capacity(capacity: usize) -> Result<Self, crate::error::ConfigError> {
        let reserve = |context: &'static str| -> Result<Vec<F::Elem>, crate::error::ConfigError> {
            let mut values = Vec::new();
            values.try_reserve_exact(capacity + 1).map_err(|_| {
                crate::error::ConfigError::AllocationFailed {
                    context,
                    elements: capacity + 1,
                    element_size: F::BYTES,
                }
            })?;
            values.resize(capacity + 1, F::Elem::ZERO);
            Ok(values)
        };
        Ok(Self {
            connection: reserve("berlekamp-massey connection register")?,
            previous: reserve("berlekamp-massey snapshot register")?,
            swap: reserve("berlekamp-massey swap register")?,
            capacity: capacity + 1,
        })
    }

    fn reset(&mut self) {
        for register in [&mut self.connection, &mut self.previous, &mut self.swap] {
            for value in register.iter_mut() {
                *value = F::Elem::ZERO;
            }
        }
        self.connection[0] = F::Elem::ONE;
        self.previous[0] = F::Elem::ONE;
    }
}
