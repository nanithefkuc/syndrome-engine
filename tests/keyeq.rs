//! The key-equation solvers: the identity `Λ·S ≡ Ω (mod x^{n-k})` verified
//! by an independent truncated multiply — never by the solver's own
//! cofactors — plus degree bounds and the misuse/geometry errors.

mod common;

use fgf::field::Elem;
use fgf::kernel::FieldKernels;
use fgf::{Gf8, Gf16, Gf32, Gf64};
use syndrome_engine::{
    BerlekampMassey, DecodeError, Decoder, Euclidean, KeyEqScratch, KeyEquation, KeyEquationSolver,
    RsParams, syndromes,
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

#[test]
fn dornstetter_cross_check_bm_matches_euclidean_everywhere() {
    // The strongest oracle in the crate: two structurally unrelated
    // solvers — iterative LFSR synthesis versus truncated EEA — must
    // produce the identical normalized (Λ, Ω) pair on every decodable
    // input, across fields, geometries, offsets, and error counts.
    let check = |n: usize, k: usize, b: usize, errors: usize, seed: u64| {
        let params = RsParams::<Gf8>::new(n, k, b).expect("params");
        let mut word = common::random_codeword(&params, seed);
        let mut positions = common::distinct_positions(n, errors, seed ^ 0x9E37);
        positions.sort();
        let mut state = seed ^ 0x85EB;
        common::inject::<Gf8>(&mut word, &positions, &mut state);
        let values = syndromes(&params, &word).expect("syndromes");
        let mut bm = KeyEquation::<Gf8>::with_capacity(params.redundancy() + 1).expect("bm");
        let mut euclidean =
            KeyEquation::<Gf8>::with_capacity(params.redundancy() + 1).expect("eea");
        let mut scratch = KeyEqScratch::with_capacity(values.len()).expect("scratch");
        BerlekampMassey
            .solve(&[&values], &mut bm, &mut scratch)
            .expect("bm solve");
        Euclidean
            .solve(&[&values], &mut euclidean, &mut scratch)
            .expect("eea solve");
        assert_eq!(bm.locator(), euclidean.locator());
        assert_eq!(bm.evaluator(), euclidean.evaluator());
        assert_eq!(bm.locator_degree(), errors);
    };
    // Error counts stay at or below each geometry's correction radius.
    for errors in 0..=5 {
        check(
            31,
            21,
            1,
            errors,
            0x900 + u64::try_from(errors).unwrap_or(0),
        );
    }
    for errors in 0..=4 {
        check(
            21,
            13,
            0,
            errors,
            0x920 + u64::try_from(errors).unwrap_or(0),
        );
        check(17, 8, 3, errors, 0x940 + u64::try_from(errors).unwrap_or(0));
    }
}

#[test]
fn dornstetter_cross_check_across_fields() {
    let check = |n: usize, k: usize, b: usize, errors: usize, seed: u64| {
        let params = RsParams::<Gf16>::new(n, k, b).expect("params");
        let mut word = common::random_codeword(&params, seed);
        let mut positions = common::distinct_positions(n, errors, seed ^ 0x9E37);
        positions.sort();
        let mut state = seed ^ 0x85EB;
        common::inject::<Gf16>(&mut word, &positions, &mut state);
        let values = syndromes(&params, &word).expect("syndromes");
        let mut bm = KeyEquation::<Gf16>::with_capacity(params.redundancy() + 1).expect("bm");
        let mut euclidean =
            KeyEquation::<Gf16>::with_capacity(params.redundancy() + 1).expect("eea");
        let mut scratch = KeyEqScratch::with_capacity(values.len()).expect("scratch");
        BerlekampMassey
            .solve(&[&values], &mut bm, &mut scratch)
            .expect("bm solve");
        Euclidean
            .solve(&[&values], &mut euclidean, &mut scratch)
            .expect("eea solve");
        assert_eq!(bm.locator(), euclidean.locator());
        assert_eq!(bm.evaluator(), euclidean.evaluator());
    };
    for errors in 0..=8 {
        check(
            60,
            44,
            1,
            errors,
            0x960 + u64::try_from(errors).unwrap_or(0),
        );
    }
    let params = RsParams::<Gf64>::new(50, 30, 1).expect("params");
    let mut word = common::random_codeword(&params, 0x980);
    let mut positions = common::distinct_positions(50, 10, 0x981);
    positions.sort();
    let mut state = 0x982;
    common::inject::<Gf64>(&mut word, &positions, &mut state);
    let values = syndromes(&params, &word).expect("syndromes");
    let mut bm = KeyEquation::<Gf64>::with_capacity(41).expect("bm");
    let mut euclidean = KeyEquation::<Gf64>::with_capacity(41).expect("eea");
    let mut scratch = KeyEqScratch::with_capacity(values.len()).expect("scratch");
    BerlekampMassey
        .solve(&[&values], &mut bm, &mut scratch)
        .expect("bm solve");
    Euclidean
        .solve(&[&values], &mut euclidean, &mut scratch)
        .expect("eea solve");
    assert_eq!(bm.locator(), euclidean.locator());
    assert_eq!(bm.evaluator(), euclidean.evaluator());
}

#[test]
fn bm_locator_reproduces_the_syndrome_recurrence() {
    // The LFSR property, checked directly against the supplied sequence:
    // S_r = Σ_{i≥1} Λ_i·S_{r-i} for every r above the evaluator degree.
    let params = RsParams::<Gf8>::new(31, 15, 1).expect("params");
    let mut word = common::random_codeword(&params, 0x9A0);
    let mut positions = common::distinct_positions(31, 8, 0x9A1);
    positions.sort();
    let mut state = 0x9A2;
    common::inject::<Gf8>(&mut word, &positions, &mut state);
    let values = syndromes(&params, &word).expect("syndromes");
    let mut out = KeyEquation::<Gf8>::with_capacity(32).expect("out");
    let mut scratch = KeyEqScratch::with_capacity(values.len()).expect("scratch");
    BerlekampMassey
        .solve(&[&values], &mut out, &mut scratch)
        .expect("bm solve");

    let locator: Vec<_> = out.locator().coefficients().collect();
    let evaluator_degree = out.evaluator().coefficient_count();
    for r in evaluator_degree..values.len() {
        let mut predicted = fgf::gf8::Elem(0);
        for i in 1..locator.len() {
            if r >= i {
                predicted = predicted.add(locator[i].mul(values[r - i]));
            }
        }
        assert_eq!(predicted, values[r], "recurrence fails at r = {r}");
    }
}

#[test]
fn end_to_end_decode_is_byte_identical_across_solvers() {
    // The Dornstetter equivalence promoted from the locator to the whole
    // decode: the pipeline returns identical outcomes with each backend.
    let check = |n: usize, k: usize, b: usize, errors: usize, seed: u64| {
        let params = RsParams::<Gf8>::new(n, k, b).expect("params");
        let sent = common::random_codeword(&params, seed);
        let mut received = sent.clone();
        let mut positions = common::distinct_positions(n, errors, seed ^ 0x9E37);
        positions.sort();
        let mut state = seed ^ 0x85EB;
        common::inject::<Gf8>(&mut received, &positions, &mut state);

        let mut via_bm = received.clone();
        let mut via_euclidean = received.clone();
        let mut via_adaptive = received.clone();

        let decoder = Decoder::new(params, BerlekampMassey);
        let mut scratch = decoder.scratch().expect("scratch");
        let bm = decoder
            .decode_into(&mut via_bm, &mut scratch)
            .expect("bm decode");

        let decoder = Decoder::new(params, Euclidean);
        let mut scratch = decoder.scratch().expect("scratch");
        let euclidean = decoder
            .decode_into(&mut via_euclidean, &mut scratch)
            .expect("eea decode");

        let decoder = Decoder::new(params, syndrome_engine::Adaptive);
        let mut scratch = decoder.scratch().expect("scratch");
        let adaptive = decoder
            .decode_into(&mut via_adaptive, &mut scratch)
            .expect("adaptive decode");

        assert_eq!(bm.positions(), euclidean.positions());
        assert_eq!(bm.positions(), adaptive.positions());
        assert_eq!(bm.magnitudes(), euclidean.magnitudes());
        assert_eq!(bm.magnitudes(), adaptive.magnitudes());
        assert_eq!(via_bm, via_euclidean);
        assert_eq!(via_bm, via_adaptive);
        assert_eq!(via_bm, sent);
    };
    for errors in 0..=6 {
        check(
            31,
            19,
            1,
            errors,
            0x9C0 + u64::try_from(errors).unwrap_or(0),
        );
    }
    check(17, 8, 3, 4, 0x9E0);
}
