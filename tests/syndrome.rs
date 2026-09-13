//! Syndrome computation: the zero-syndrome certificate, the independent
//! Horner oracle, linearity, and geometry rejection.

mod common;

use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use fgf::{Gf8B, Gf16, Gf32, Gf64};
use syndrome_engine::{DecodeError, RsParams, syndromes};

fn zero_syndromes<F: FieldKernels>(n: usize, k: usize, b: usize, seed: u64) {
    let params = RsParams::<F>::new(n, k, b).expect("params");
    let word = common::random_codeword(&params, seed);
    let values = syndromes(&params, &word).expect("syndromes");
    assert_eq!(values.len(), params.syndrome_count());
    assert!(
        values.iter().all(|value| value.is_zero()),
        "intact codeword must have all-zero syndromes"
    );
}

#[test]
fn intact_codewords_have_zero_syndromes() {
    zero_syndromes::<Gf8B>(15, 9, 1, 0x101);
    zero_syndromes::<Gf8B>(31, 21, 0, 0x102);
    zero_syndromes::<Gf8B>(17, 8, 3, 0x103); // odd redundancy: 9 syndromes
    zero_syndromes::<Gf16>(40, 28, 1, 0x104);
    zero_syndromes::<Gf16>(1000, 900, 0, 0x105);
    zero_syndromes::<Gf32>(120, 100, 1, 0x106);
    zero_syndromes::<Gf64>(60, 40, 1, 0x107);
}

#[test]
fn a_flipped_symbol_is_visible() {
    let params = RsParams::<Gf8B>::new(15, 9, 1).expect("params");
    let mut word = common::random_codeword(&params, 0x201);
    word[3] ^= 0x5A;
    let values = syndromes(&params, &word).expect("syndromes");
    assert!(
        values.iter().any(|value| !value.is_zero()),
        "a single flipped symbol must produce a nonzero syndrome"
    );
}

#[test]
fn horner_oracle_agrees_across_fields_and_offsets() {
    let check = |n: usize, k: usize, b: usize, seed: u64| {
        let params = RsParams::<Gf8B>::new(n, k, b).expect("params");
        let word = common::random_codeword(&params, seed);
        let produced = syndromes(&params, &word).expect("syndromes");
        let oracle = common::horner_syndromes(&params, &word);
        assert_eq!(produced, oracle);
    };
    check(15, 9, 1, 0x301);
    check(31, 21, 0, 0x302);
    check(17, 8, 3, 0x303);

    let params = RsParams::<Gf16>::new(300, 250, 1).expect("params");
    let word = common::random_codeword(&params, 0x304);
    assert_eq!(
        syndromes(&params, &word).expect("syndromes"),
        common::horner_syndromes(&params, &word)
    );

    let params = RsParams::<Gf32>::new(90, 70, 2).expect("params");
    let word = common::random_codeword(&params, 0x305);
    assert_eq!(
        syndromes(&params, &word).expect("syndromes"),
        common::horner_syndromes(&params, &word)
    );

    let params = RsParams::<Gf64>::new(50, 30, 1).expect("params");
    let word = common::random_codeword(&params, 0x306);
    assert_eq!(
        syndromes(&params, &word).expect("syndromes"),
        common::horner_syndromes(&params, &word)
    );
}

#[test]
fn syndromes_are_linear_in_the_word() {
    // Fixed-seed randomized words across fields: S(R1 ^ R2) = S(R1) ^ S(R2).
    let linearity = |n: usize, k: usize, b: usize, seed: u64| {
        let params = RsParams::<Gf8B>::new(n, k, b).expect("params");
        let mut state = seed;
        let mut left = vec![0_u8; n * Gf8B::BYTES];
        let mut right = vec![0_u8; n * Gf8B::BYTES];
        for (left, right) in left.iter_mut().zip(right.iter_mut()) {
            *left = common::advance(&mut state) as u8;
            *right = common::advance(&mut state) as u8;
        }
        let mut combined = left.clone();
        for (combined, right) in combined.iter_mut().zip(right.iter()) {
            *combined ^= right;
        }
        let mut expected = syndromes(&params, &left).expect("syndromes");
        for (expected, right) in expected
            .iter_mut()
            .zip(syndromes(&params, &right).expect("syndromes"))
        {
            *expected = expected.add(right);
        }
        assert_eq!(syndromes(&params, &combined).expect("syndromes"), expected);
    };
    for seed in 0x400..0x410 {
        linearity(15, 9, 1, seed);
        linearity(21, 13, 0, seed ^ 0x55);
    }

    let params = RsParams::<Gf16>::new(120, 80, 1).expect("params");
    let mut state = 0x420;
    let mut left = vec![0_u8; 120 * Gf16::BYTES];
    let mut right = vec![0_u8; 120 * Gf16::BYTES];
    for (left, right) in left.iter_mut().zip(right.iter_mut()) {
        let bytes = common::advance(&mut state).to_le_bytes();
        *left = bytes[0];
        *right = bytes[1];
    }
    let mut combined = left.clone();
    for (combined, right) in combined.iter_mut().zip(right.iter()) {
        *combined ^= right;
    }
    let mut expected = syndromes(&params, &left).expect("syndromes");
    for (expected, right) in expected
        .iter_mut()
        .zip(syndromes(&params, &right).expect("syndromes"))
    {
        *expected = expected.add(right);
    }
    assert_eq!(syndromes(&params, &combined).expect("syndromes"), expected);
}

#[test]
fn wrong_word_length_is_rejected_with_the_geometry() {
    let params = RsParams::<Gf8B>::new(15, 9, 1).expect("params");
    let short = vec![0_u8; 14];
    assert_eq!(
        syndromes(&params, &short),
        Err(DecodeError::WordGeometry {
            got: 14,
            expected: 15
        })
    );
    let long = vec![0_u8; 16];
    assert_eq!(
        syndromes(&params, &long),
        Err(DecodeError::WordGeometry {
            got: 16,
            expected: 15
        })
    );
}

#[test]
fn zero_redundancy_yields_no_syndromes() {
    let params = RsParams::<Gf8B>::new(7, 7, 1).expect("params");
    let word = vec![1_u8; 7];
    assert_eq!(params.syndrome_count(), 0);
    assert!(syndromes(&params, &word).expect("syndromes").is_empty());
}
