//! Forney's algorithm: the error magnitudes.
//!
//! With the locators `X_p = α^{p}` known from the Chien search, each
//! magnitude is one evaluation-and-division:
//!
//! `e_p = X_p^{1-b} · Ω(X_p^{-1}) / Λ'(X_p^{-1})`
//!
//! in the characteristic-two form, computed here as
//! `e_p = X_p · Ω(y_p) / (X_p^{b} · Λ'(y_p))` with `y_p = X_p^{-1}`, so the
//! exponent `b` folds into the batch-inverted denominators and no modular
//! exponent arithmetic is left. The formal derivative is evaluated
//! pointwise through `univariate`'s order-1 Hasse derivative — in
//! characteristic two that keeps exactly the odd-degree terms, the
//! half-work simplification, without materializing the derivative
//! polynomial.
//!
//! Denominators are inverted together by Montgomery's batch trick: one
//! field inversion plus `3(ν-1)` multiplications. There is no upstream
//! batch-inversion helper to compose (neither `fgf` nor `univariate` ships
//! one), and the loop is scalar sequence logic over `fgf::field::Elem` —
//! the same sanction Berlekamp–Massey has. `inv(0) == 0` is inherited from
//! `fgf`; a zero denominator is an explicit `is_zero()` rejection before
//! the inversion (S7), never a value inferred from a division.

use alloc::vec::Vec;

use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use univariate::Polynomial;

use crate::error::DecodeError;
use crate::params::RsParams;

/// Compute the error magnitudes for the located positions against the total
/// locator `Λ·Γ` and total evaluator `Ω = Λ·Γ·S mod x^{n-k}`.
///
/// `numerators`, `denominators`, and `prefix` are scratch; `out` receives
/// one magnitude per position, in the given position order.
///
/// # Errors
///
/// Returns [`DecodeError::Inconsistent`] when any denominator
/// `X_p^{b}·Λ'(X_p^{-1})` is zero — a repeated locator root, which places
/// the word outside the decoding sphere.
pub(crate) fn magnitudes_into<F: FieldKernels>(
    params: &RsParams<F>,
    total_locator: &Polynomial<F>,
    total_evaluator: &Polynomial<F>,
    positions: &[usize],
    scratch: &mut ForneyScratch<F>,
    out: &mut Vec<F::Elem>,
) -> Result<(), DecodeError> {
    out.clear();
    scratch.numerators.clear();
    scratch.denominators.clear();
    scratch.prefix.clear();

    let alpha = <F as Field>::GENERATOR;
    for &position in positions {
        let locator_value = alpha.pow(position as u64);
        let inverse = locator_value.inv();
        scratch.numerators.push(total_evaluator.evaluate(inverse));
        // Denominator folded with X^{b}: e_p = X_p·Ω(y)·(X_p^b·Λ'(y))^{-1};
        // pow(0) is one, so b = 0 needs no special case.
        let derivative = total_locator.evaluate_hasse(inverse, 1);
        let scaled = derivative.mul(locator_value.pow(params.b() as u64));
        scratch.denominators.push(scaled);
    }
    if scratch.denominators.iter().any(|value| value.is_zero()) {
        return Err(DecodeError::Inconsistent);
    }
    batch_invert_into::<F>(&mut scratch.denominators, &mut scratch.prefix);
    for (index, &position) in positions.iter().enumerate() {
        let locator_value = alpha.pow(position as u64);
        out.push(
            scratch.numerators[index]
                .mul(locator_value)
                .mul(scratch.denominators[index]),
        );
    }
    Ok(())
}

/// Scratch for the Forney stage: the numerators, the batch-inversion
/// denominators, and the prefix-product buffer.
#[derive(Debug)]
pub(crate) struct ForneyScratch<F: FieldKernels> {
    numerators: Vec<F::Elem>,
    denominators: Vec<F::Elem>,
    prefix: Vec<F::Elem>,
}

impl<F: FieldKernels> ForneyScratch<F> {
    pub(crate) fn with_capacity(capacity: usize) -> Result<Self, crate::error::ConfigError> {
        Ok(Self {
            numerators: crate::decoder::reserved_elements::<F>(capacity, "forney numerators")?,
            denominators: crate::decoder::reserved_elements::<F>(capacity, "forney denominators")?,
            prefix: crate::decoder::reserved_elements::<F>(capacity, "forney prefix")?,
        })
    }
}

/// Montgomery batch inversion, in place: `values[i] ← values[i]^{-1}` with
/// one inversion and `3(len-1)` multiplications.
///
/// `prefix` is scratch holding the originals while `values` walks the
/// prefix products. Every element must be nonzero; callers zero-test
/// explicitly before inverting (S7) — the inherited `inv(0) == 0` would
/// otherwise poison the whole batch.
pub(crate) fn batch_invert_into<F: FieldKernels>(
    values: &mut [F::Elem],
    prefix: &mut Vec<F::Elem>,
) {
    prefix.clear();
    let mut running = F::Elem::ONE;
    for value in values.iter_mut() {
        prefix.push(*value);
        running = running.mul(*value);
        *value = running;
    }
    // values[i] holds the inclusive prefix product P_i; prefix[i] the
    // original v_i. Walking backwards: v_i^{-1} = P_{i-1}·inv(P_i) and
    // inv(P_{i-1}) = inv(P_i)·v_i.
    let mut inverse = running.inv();
    for index in (0..values.len()).rev() {
        let original = prefix[index];
        let previous = if index == 0 {
            F::Elem::ONE
        } else {
            values[index - 1]
        };
        values[index] = inverse.mul(previous);
        inverse = inverse.mul(original);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fgf::Gf8;

    fn noise_values(count: usize, seed: u64) -> Vec<fgf::gf8::Elem> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                loop {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let value = fgf::gf8::Elem(u8::try_from(state % 256).expect("reduced byte"));
                    if !value.is_zero() {
                        return value;
                    }
                }
            })
            .collect()
    }

    #[test]
    fn batch_invert_agrees_with_elementwise() {
        for count in [1_usize, 2, 3, 8, 17, 32] {
            let values = noise_values(count, 0x0BAD_C0DE + count as u64);
            let mut inverted = values.clone();
            let mut prefix = Vec::new();
            batch_invert_into::<Gf8>(&mut inverted, &mut prefix);
            for (original, inverted) in values.iter().zip(&inverted) {
                assert_eq!(*inverted, original.inv());
            }
        }
    }

    #[test]
    fn batch_invert_of_one_is_the_inverse() {
        let mut values = noise_values(1, 0x5EED);
        let original = values[0];
        let mut prefix = Vec::new();
        batch_invert_into::<Gf8>(&mut values, &mut prefix);
        assert_eq!(values[0], original.inv());
    }
}
