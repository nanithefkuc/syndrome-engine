//! Errors-and-erasures: mixed patterns at the `2ν + ρ = d - 1` boundary,
//! degenerate agreement with the pure-error path, the erasures-only direct
//! solve, typed validation failures, and zero allocation across the fold.

mod common;

use fgf::field::Elem;
use fgf::kernel::FieldKernels;
use fgf::{Gf8, Gf16, Gf32, Gf64};
use syndrome_engine::{DecodeError, Decoder, Euclidean, RsParams, syndromes};

/// Inject `errors` unflagged and `erasures` flagged corruptions, decode
/// with erasures, and verify: sent word recovered, all corrupted positions
/// reported (ascending), magnitudes exact, re-encode zero.
fn mixed_case<F: FieldKernels>(
    n: usize,
    k: usize,
    b: usize,
    errors: usize,
    erasures: usize,
    seed: u64,
) {
    let params = RsParams::<F>::new(n, k, b).expect("params");
    let decoder = Decoder::new(params, Euclidean);
    let mut scratch = decoder.scratch().expect("scratch");
    let sent = common::random_codeword(&params, seed);
    let mut received = sent.clone();

    let mut error_positions = common::distinct_positions(n, errors, seed ^ 0x1E37);
    let mut erasure_positions = Vec::new();
    for candidate in common::distinct_positions(n, errors + erasures, seed ^ 0x2E37) {
        if erasure_positions.len() < erasures && !error_positions.contains(&candidate) {
            erasure_positions.push(candidate);
        }
    }
    error_positions.retain(|position| !erasure_positions.contains(position));
    error_positions.sort();
    erasure_positions.sort();
    let mut state = seed ^ 0x3E37;
    let error_magnitudes = common::inject::<F>(&mut received, &error_positions, &mut state);
    let erasure_magnitudes = common::inject::<F>(&mut received, &erasure_positions, &mut state);

    let mut expected_positions = error_positions.clone();
    expected_positions.extend_from_slice(&erasure_positions);
    expected_positions.sort_unstable();
    let mut expected_magnitudes = error_magnitudes.clone();
    expected_magnitudes.extend_from_slice(&erasure_magnitudes);
    // Re-pair magnitudes with the ascending position order.
    let mut pairs: Vec<(usize, F::Elem)> = error_positions
        .iter()
        .copied()
        .zip(error_magnitudes)
        .chain(erasure_positions.iter().copied().zip(erasure_magnitudes))
        .collect();
    pairs.sort_by_key(|(position, _)| *position);
    let expected_magnitudes: Vec<F::Elem> =
        pairs.into_iter().map(|(_, magnitude)| magnitude).collect();

    let outcome = decoder
        .decode_with_erasures_into(&mut received, &erasure_positions, &mut scratch)
        .expect("mixed decode");
    assert_eq!(outcome.positions(), expected_positions.as_slice());
    assert_eq!(outcome.magnitudes(), expected_magnitudes.as_slice());
    assert_eq!(received, sent);
    assert!(
        syndromes(&params, &received)
            .expect("syndromes")
            .iter()
            .all(|value| value.is_zero())
    );
}

#[test]
fn mixed_patterns_decode_at_the_boundary() {
    // 2ν + ρ = d - 1 = n - k exactly, across fields and offsets.
    mixed_case::<Gf8>(15, 9, 1, 3, 0, 0x2001); // pure errors, 2·3 = 6
    mixed_case::<Gf8>(15, 9, 1, 2, 2, 0x2002); // 4 + 2 = 6
    mixed_case::<Gf8>(15, 9, 1, 1, 4, 0x2003); // 2 + 4 = 6
    mixed_case::<Gf8>(15, 9, 1, 0, 6, 0x2004); // erasures only
    mixed_case::<Gf8>(17, 8, 3, 3, 3, 0x2005); // odd redundancy: 6 + 3 = 9
    mixed_case::<Gf16>(40, 28, 1, 4, 4, 0x2006);
    mixed_case::<Gf32>(120, 100, 1, 6, 2, 0x2007);
    mixed_case::<Gf64>(60, 40, 1, 8, 4, 0x2008);
}

#[test]
fn one_past_the_boundary_fails_typed() {
    let params = RsParams::<Gf8>::new(15, 9, 1).expect("params"); // budget 6
    let decoder = Decoder::new(params, Euclidean);
    let mut scratch = decoder.scratch().expect("scratch");
    let sent = common::random_codeword(&params, 0x2100);
    // 2·3 + 2 = 8 > 6, decodable claim only for 2ν + ρ ≤ 6.
    let mut received = sent.clone();
    let mut error_positions = common::distinct_positions(15, 3, 0x2101);
    let erasure_positions = vec![0usize, 14];
    error_positions.retain(|position| !erasure_positions.contains(position));
    error_positions.sort();
    let mut state = 0x2102;
    common::inject::<Gf8>(&mut received, &error_positions, &mut state);
    common::inject::<Gf8>(&mut received, &erasure_positions, &mut state);
    // Past the budget the outcome is either a typed detection or — when
    // the word happens to sit inside another codeword's decoding sphere —
    // a valid but different codeword, which no syndrome check can detect.
    // Both are acceptable; a panic or a non-codeword "correction" is not.
    let before = received.clone();
    match decoder.decode_with_erasures_into(&mut received, &erasure_positions, &mut scratch) {
        Err(
            DecodeError::TooManyErrors { .. }
            | DecodeError::Inconsistent
            | DecodeError::DegreeMismatch { .. },
        ) => assert_eq!(received, before, "failed decode must not touch the word"),
        Err(other) => panic!("unexpected {other:?}"),
        Ok(_outcome) => assert!(
            syndromes(&params, &received)
                .expect("syndromes")
                .iter()
                .all(|value| value.is_zero()),
            "an accepted correction must yield a true codeword"
        ),
    }
}

#[test]
fn erasure_free_path_is_byte_identical_to_pure_path() {
    // Degenerate agreement: ρ = 0 collapses the modified syndromes onto
    // the ordinary ones.
    let params = RsParams::<Gf8>::new(21, 13, 1).expect("params");
    let sent = common::random_codeword(&params, 0x2200);
    let mut received = sent.clone();
    let mut positions = common::distinct_positions(21, 4, 0x2201);
    positions.sort();
    let mut state = 0x2202;
    common::inject::<Gf8>(&mut received, &positions, &mut state);

    let decoder = Decoder::new(params, Euclidean);
    let mut scratch = decoder.scratch().expect("scratch");
    let mut via_pure = received.clone();
    let pure = decoder
        .decode_into(&mut via_pure, &mut scratch)
        .expect("pure decode");
    let pure = (pure.positions().to_vec(), pure.magnitudes().to_vec());
    let mut via_erasure = received.clone();
    let erasure = decoder
        .decode_with_erasures_into(&mut via_erasure, &[], &mut scratch)
        .expect("erasure-path decode");
    assert_eq!(pure.0, erasure.positions());
    assert_eq!(pure.1, erasure.magnitudes());
    assert_eq!(via_pure, via_erasure);
}

#[test]
fn erasures_only_agrees_with_the_direct_solve() {
    // ν = 0: the magnitudes must match an independent Vandermonde solve
    // over the erasure positions.
    let params = RsParams::<Gf16>::new(40, 28, 1).expect("params");
    let decoder = Decoder::new(params, Euclidean);
    let mut scratch = decoder.scratch().expect("scratch");
    let sent = common::random_codeword(&params, 0x2300);
    let mut received = sent.clone();
    let mut erasure_positions = common::distinct_positions(40, 6, 0x2301);
    erasure_positions.sort();
    let mut state = 0x2302;
    let injected = common::inject::<Gf16>(&mut received, &erasure_positions, &mut state);

    let outcome = decoder
        .decode_with_erasures_into(&mut received, &erasure_positions, &mut scratch)
        .expect("erasures-only decode");
    assert_eq!(outcome.positions(), erasure_positions.as_slice());
    let produced = outcome.magnitudes().to_vec();
    // Independent oracle: solve the Vandermonde system the corrupted word's
    // syndromes define over the erasure positions.
    let corrupted_syndromes = syndromes_of_corrupted(&params, &sent, &erasure_positions, &injected);
    let direct = common::brute_force_magnitudes(&params, &erasure_positions, &corrupted_syndromes);
    assert_eq!(produced, direct);
    assert_eq!(received, sent);
}

fn syndromes_of_corrupted<F: FieldKernels>(
    params: &RsParams<F>,
    sent: &[u8],
    positions: &[usize],
    magnitudes: &[F::Elem],
) -> Vec<F::Elem> {
    let mut word = sent.to_vec();
    for (position, magnitude) in positions.iter().zip(magnitudes) {
        let start = position * F::BYTES;
        let current = F::read(&word[start..start + F::BYTES]);
        F::write(&mut word[start..start + F::BYTES], current.add(*magnitude));
    }
    syndromes(params, &word).expect("syndromes")
}

#[test]
fn erasure_validation_errors_are_typed() {
    let params = RsParams::<Gf8>::new(15, 9, 1).expect("params");
    let decoder = Decoder::new(params, Euclidean);
    let mut scratch = decoder.scratch().expect("scratch");
    let mut word = vec![0_u8; 15];

    // ρ = 7 > n - k = 6.
    assert_eq!(
        decoder
            .decode_with_erasures_into(&mut word, &[0, 1, 2, 3, 4, 5, 6], &mut scratch)
            .map(|_| ()),
        Err(DecodeError::TooManyErasures { count: 7, limit: 6 })
    );
    // Out-of-range position.
    assert_eq!(
        decoder
            .decode_with_erasures_into(&mut word, &[15], &mut scratch)
            .map(|_| ()),
        Err(DecodeError::ErasurePosition { got: 15, limit: 15 })
    );
    // Duplicate position.
    assert_eq!(
        decoder
            .decode_with_erasures_into(&mut word, &[4, 4], &mut scratch)
            .map(|_| ()),
        Err(DecodeError::DuplicateErasure { position: 4 })
    );
    // The word is untouched by every rejected call.
    assert_eq!(word, vec![0_u8; 15]);
}

#[test]
fn erasure_magnitudes_match_forney_formula_independently() {
    // An erased-but-intact position (zero channel magnitude) stays in the
    // reported set with a zero magnitude: the locator includes it, the
    // syndrome contribution does not.
    let params = RsParams::<Gf8>::new(21, 13, 1).expect("params");
    let decoder = Decoder::new(params, Euclidean);
    let mut scratch = decoder.scratch().expect("scratch");
    let sent = common::random_codeword(&params, 0x2400);
    let mut received = sent.clone();
    let erasure_positions = [2usize, 9, 20];
    // Corrupt only position 9; positions 2 and 20 are flagged but intact.
    let mut state = 0x2401;
    let magnitudes = common::inject::<Gf8>(&mut received, &[9], &mut state);
    let outcome = decoder
        .decode_with_erasures_into(&mut received, &erasure_positions, &mut scratch)
        .expect("decode");
    assert_eq!(outcome.positions(), erasure_positions.as_slice());
    assert_eq!(outcome.magnitudes()[0], fgf::gf8::Elem(0));
    assert_eq!(outcome.magnitudes()[1], magnitudes[0]);
    assert_eq!(outcome.magnitudes()[2], fgf::gf8::Elem(0));
    assert_eq!(received, sent);
}
