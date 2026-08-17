//! The end-to-end bounded-distance decode: positions and magnitudes exact,
//! the re-encode certificate, the Chien root re-check through a different
//! evaluator, totality at and beyond the sphere, and the geometry errors.

mod common;

use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use fgf::{Gf8, Gf16, Gf32, Gf64};
use syndrome_engine::{
    DecodeError, Decoder, Euclidean, KeyEqScratch, KeyEquation, KeyEquationSolver, RsParams,
    syndromes,
};
use univariate::Polynomial;

fn decode_case<F: FieldKernels>(n: usize, k: usize, b: usize, errors: usize, seed: u64) {
    let params = RsParams::<F>::new(n, k, b).expect("params");
    let decoder = Decoder::new(params, Euclidean);
    let mut scratch = decoder.scratch().expect("scratch");
    let sent = common::random_codeword(&params, seed);
    let mut received = sent.clone();
    let mut positions = common::distinct_positions(n, errors, seed ^ 0x9E37);
    positions.sort();
    let mut state = seed ^ 0x85EB;
    let magnitudes = common::inject::<F>(&mut received, &positions, &mut state);

    let outcome = decoder
        .decode_into(&mut received, &mut scratch)
        .expect("decode");
    assert_eq!(outcome.error_count(), errors);
    assert_eq!(outcome.positions(), positions.as_slice());
    assert_eq!(outcome.magnitudes(), magnitudes.as_slice());
    assert_eq!(received, sent, "corrected word must equal the sent word");

    // Re-encode certificate, recomputed independently of the decoder:
    // the corrected word's syndromes are all zero.
    assert!(
        syndromes(&params, &received)
            .expect("syndromes")
            .iter()
            .all(|value| value.is_zero())
    );

    // Chien re-check through a different evaluator: the solved locator
    // vanishes at α^{-p} for every reported position, and the root count
    // equals the locator degree.
    let alpha = <F as Field>::GENERATOR;
    let mut locator = Polynomial::<F>::one().expect("one");
    // Π(1 + X_p x) shares its roots with Π(x + X_p^{-1}); the scale
    // factor Π X_p is irrelevant for the vanishing check.
    for &position in outcome.positions() {
        locator = locator
            .multiply_x_plus(alpha.pow(position as u64).inv())
            .expect("locator");
    }
    for &position in outcome.positions() {
        let inverse = alpha.pow(position as u64).inv();
        assert!(
            locator.evaluate(inverse) == <F as Field>::Elem::ZERO,
            "position {position} must be a locator root"
        );
    }
    assert_eq!(locator.degree(), Some(outcome.error_count()));
}

#[test]
fn decodes_at_and_below_the_boundary() {
    decode_case::<Gf8>(15, 9, 1, 0, 0x1001);
    decode_case::<Gf8>(15, 9, 1, 1, 0x1002);
    decode_case::<Gf8>(15, 9, 1, 3, 0x1003); // t = 3
    decode_case::<Gf8>(31, 21, 0, 5, 0x1004);
    decode_case::<Gf8>(17, 8, 3, 4, 0x1005); // odd redundancy, t = 4
    decode_case::<Gf8>(255, 223, 1, 16, 0x1006);
    decode_case::<Gf16>(40, 28, 1, 6, 0x1007);
    decode_case::<Gf16>(1000, 900, 0, 50, 0x1008);
    decode_case::<Gf32>(120, 100, 1, 10, 0x1009);
    decode_case::<Gf64>(60, 40, 1, 10, 0x100A);
}

#[test]
fn decodes_randomized_patterns_across_the_radius() {
    for errors in 0..=6 {
        decode_case::<Gf8>(31, 19, 1, errors, 0x1100 + errors as u64);
        decode_case::<Gf16>(60, 44, 1, errors, 0x1140 + errors as u64);
    }
}

#[test]
fn beyond_the_sphere_fails_typed_and_leaves_the_word_untouched() {
    let params = RsParams::<Gf16>::new(60, 44, 1).expect("params"); // t = 8
    let decoder = Decoder::new(params, Euclidean);
    let mut scratch = decoder.scratch().expect("scratch");
    let sent = common::random_codeword(&params, 0x1200);
    for errors in [9_usize, 10, 12] {
        let mut received = sent.clone();
        let mut positions = common::distinct_positions(60, errors, 0x1200 + errors as u64);
        positions.sort();
        let mut state = 0x1300 + errors as u64;
        common::inject::<Gf16>(&mut received, &positions, &mut state);
        // Past the radius: a typed detection, or — when the word lands in
        // another codeword's sphere — an undetectable miscorrection to a
        // true codeword. Both are total; neither is a panic or a
        // non-codeword "correction".
        let before = received.clone();
        match decoder.decode_into(&mut received, &mut scratch) {
            Err(
                DecodeError::TooManyErrors { .. }
                | DecodeError::Inconsistent
                | DecodeError::DegreeMismatch { .. },
            ) => assert_eq!(received, before, "failed decode must not touch the word"),
            Err(other) => panic!("unexpected error {other:?}"),
            Ok(_outcome) => assert!(
                syndromes(&params, &received)
                    .expect("syndromes")
                    .iter()
                    .all(|value| value.is_zero()),
                "an accepted correction must yield a true codeword"
            ),
        }
    }
}

#[test]
fn wrong_word_length_is_rejected() {
    let params = RsParams::<Gf8>::new(15, 9, 1).expect("params");
    let decoder = Decoder::new(params, Euclidean);
    let mut scratch = decoder.scratch().expect("scratch");
    let mut word = vec![0_u8; 14];
    let before = word.clone();
    assert_eq!(
        decoder.decode_into(&mut word, &mut scratch).map(|_| ()),
        Err(DecodeError::WordGeometry {
            got: 14,
            expected: 15
        })
    );
    assert_eq!(word, before);
}

#[test]
fn scratch_from_another_geometry_is_rejected() {
    let params = RsParams::<Gf8>::new(15, 9, 1).expect("params");
    let other = RsParams::<Gf8>::new(31, 21, 1).expect("params");
    let decoder = Decoder::new(params, Euclidean);
    let mut foreign = Decoder::new(other, Euclidean).scratch().expect("scratch");
    let mut word = vec![0_u8; 15];
    let before = word.clone();
    assert_eq!(
        decoder.decode_into(&mut word, &mut foreign).map(|_| ()),
        Err(DecodeError::ScratchMismatch {
            expected: 15,
            got: 31
        })
    );
    assert_eq!(word, before);
}

#[test]
fn decode_from_syndromes_recovers_the_pattern() {
    let params = RsParams::<Gf8>::new(21, 13, 1).expect("params");
    let decoder = Decoder::new(params, Euclidean);
    let mut scratch = decoder.scratch().expect("scratch");
    let sent = common::random_codeword(&params, 0x1400);
    let mut received = sent.clone();
    let mut positions = common::distinct_positions(21, 4, 0x1401);
    positions.sort();
    let mut state = 0x1402;
    let magnitudes = common::inject::<Gf8>(&mut received, &positions, &mut state);
    let values = syndromes(&params, &received).expect("syndromes");
    let outcome = decoder
        .decode_syndromes_into(&values, &mut scratch)
        .expect("decode");
    assert_eq!(outcome.positions(), positions.as_slice());
    assert_eq!(outcome.magnitudes(), magnitudes.as_slice());
}

#[test]
fn decode_from_syndromes_checks_the_count() {
    let params = RsParams::<Gf8>::new(15, 9, 1).expect("params");
    let decoder = Decoder::new(params, Euclidean);
    let mut scratch = decoder.scratch().expect("scratch");
    let values = vec![fgf::gf8::Elem(0); 5];
    assert_eq!(
        decoder
            .decode_syndromes_into(&values, &mut scratch)
            .map(|_| ()),
        Err(DecodeError::SyndromeGeometry {
            got: 5,
            expected: 6
        })
    );
}

#[test]
fn zero_redundancy_code_decodes_trivially() {
    let params = RsParams::<Gf8>::new(7, 7, 1).expect("params");
    let decoder = Decoder::new(params, Euclidean);
    let mut scratch = decoder.scratch().expect("scratch");
    let mut word = vec![1_u8, 2, 3, 4, 5, 6, 7];
    let before = word.clone();
    let outcome = decoder
        .decode_into(&mut word, &mut scratch)
        .expect("decode");
    assert_eq!(outcome.error_count(), 0);
    assert_eq!(word, before);
}

#[test]
fn solver_is_total_on_arbitrary_sequences() {
    // Beyond the radius the sequence may or may not admit a within-budget
    // Padé pair — detection is the decoder's guards — but the solver must
    // be total, and any pair it returns must satisfy the identity.
    let mut state = 0x1600_u64;
    for _ in 0..32 {
        let values: Vec<fgf::gf8::Elem> = (0..12)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                fgf::gf8::Elem(state as u8)
            })
            .collect();
        let mut out = KeyEquation::<Gf8>::with_capacity(16).expect("out");
        let mut scratch = KeyEqScratch::with_capacity(values.len()).expect("scratch");
        if let Ok(()) = Euclidean.solve(&[&values], &mut out, &mut scratch) {
            let series = Polynomial::from_coefficients(&values).expect("series");
            let product = out
                .locator()
                .multiply_truncated(&series, values.len())
                .expect("product");
            assert_eq!(&product, out.evaluator());
        }
    }
}
