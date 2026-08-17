//! The key-equation solvers: the identity `Λ·S ≡ Ω (mod x^{n-k})` verified
//! by an independent truncated multiply — never by the solver's own
//! cofactors — plus degree bounds and the misuse/geometry errors.

mod common;

use fgf::field::Elem;
use fgf::kernel::FieldKernels;
use fgf::{Gf8, Gf16, Gf32, Gf64};
use syndrome_engine::{
    DecodeError, Euclidean, KeyEqScratch, KeyEquation, KeyEquationSolver, RsParams, syndromes,
};
use univariate::Polynomial;

fn solve<F: FieldKernels>(
    values: &[F::Elem],
    out: &mut KeyEquation<F>,
    scratch: &mut KeyEqScratch<F>,
) -> Result<(), DecodeError> {
    Euclidean.solve(&[values], out, scratch)
}

fn assert_key_equation<F: FieldKernels>(
    params: &RsParams<F>,
    values: &[F::Elem],
    expected_errors: usize,
) {
    let mut out = KeyEquation::<F>::with_capacity(params.redundancy() + 1).expect("out");
    let mut scratch = KeyEqScratch::with_capacity(params.syndrome_count()).expect("scratch");
    solve(values, &mut out, &mut scratch).expect("key equation");

    // Independent verification: multiply the locator by the syndrome series
    // with univariate's truncated product and compare against the evaluator.
    let series = Polynomial::from_coefficients(values).expect("series");
    let product = out
        .locator()
        .multiply_truncated(&series, values.len())
        .expect("product");
    assert_eq!(&product, out.evaluator());
    assert_eq!(out.locator().coefficient(0), F::Elem::ONE);
    assert_eq!(out.locator_degree(), expected_errors);
    assert!(
        out.evaluator().is_zero()
            || out.evaluator().coefficient_count() < out.locator().coefficient_count()
    );
}

fn error_pattern<F: FieldKernels>(n: usize, k: usize, b: usize, errors: usize, seed: u64) {
    let params = RsParams::<F>::new(n, k, b).expect("params");
    let mut word = common::random_codeword(&params, seed);
    let positions = common::distinct_positions(n, errors, seed ^ 0x9E37);
    let mut state = seed ^ 0x85EB;
    common::inject::<F>(&mut word, &positions, &mut state);
    let values = syndromes(&params, &word).expect("syndromes");
    assert_key_equation(&params, &values, errors);
}

#[test]
fn identity_holds_on_error_patterns_across_fields() {
    // At the correction boundary t and below it.
    error_pattern::<Gf8>(15, 9, 1, 1, 0x501);
    error_pattern::<Gf8>(15, 9, 1, 2, 0x502);
    error_pattern::<Gf8>(15, 9, 1, 3, 0x503); // t = 3
    error_pattern::<Gf8>(31, 21, 0, 5, 0x504); // t = 5
    error_pattern::<Gf8>(17, 8, 3, 4, 0x505); // t = 4, odd redundancy
    error_pattern::<Gf16>(40, 28, 1, 6, 0x506);
    error_pattern::<Gf16>(1000, 900, 0, 50, 0x507);
    error_pattern::<Gf32>(120, 100, 1, 10, 0x508);
    error_pattern::<Gf64>(60, 40, 1, 10, 0x509);
}

#[test]
fn identity_holds_on_randomized_patterns() {
    // Fixed-seed sweep over error counts 0..=t on several geometries.
    for errors in 0..=8 {
        error_pattern::<Gf8>(31, 15, 1, errors, 0x600 + errors as u64);
        error_pattern::<Gf16>(60, 44, 1, errors, 0x640 + errors as u64);
    }
}

#[test]
fn no_error_word_yields_the_unit_locator() {
    let params = RsParams::<Gf8>::new(15, 9, 1).expect("params");
    let word = common::random_codeword(&params, 0x701);
    let values = syndromes(&params, &word).expect("syndromes");
    assert_key_equation(&params, &values, 0);
}

#[test]
fn multi_sequence_input_is_rejected_without_touching_scratch() {
    let params = RsParams::<Gf8>::new(15, 9, 1).expect("params");
    let values = vec![fgf::gf8::Elem::ZERO; params.syndrome_count()];
    // Distinct nonzero syndromes so a solver would be tempted to work.
    let mut state = 0x801;
    let values: Vec<_> = values
        .iter()
        .map(|_| common::noise_elem::<Gf8>(&mut state))
        .collect();
    let mut out = KeyEquation::<Gf8>::with_capacity(16).expect("out");
    let mut scratch = KeyEqScratch::with_capacity(params.syndrome_count()).expect("scratch");
    let before = format!("{scratch:?}");
    let error = Euclidean
        .solve(&[&values, &values], &mut out, &mut scratch)
        .expect_err("multi-sequence input");
    assert_eq!(
        error,
        DecodeError::SyndromeGeometry {
            got: 2,
            expected: 1
        }
    );
    assert_eq!(before, format!("{scratch:?}"), "scratch must be untouched");
}
