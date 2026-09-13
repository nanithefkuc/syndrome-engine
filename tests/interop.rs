//! Frozen wire-convention fixtures (S6): the syndrome index convention
//! (offset `b`, primitive element, low-degree-first coefficient order) and
//! the root→position map are pinned by fixed known-answer vectors. A
//! convention change is a failure here requiring a deliberate fixture
//! update, not a silent refactor — a decoder built with one convention
//! cannot read a word encoded against another.
//!
//! The vectors are exact values, not predicates, and pin three conventions
//! at once: `b = 1` narrow-sense, `b = 0`, and the odd-redundancy `b = 3`
//! offset case; plus the byte-level syndrome vector of the `b = 1` word.

mod common;

use fgf::field::Elem;
use fgf::{Gf8B, Gf16};
use syndrome_engine::{Decoder, Euclidean, RsParams, syndromes};

const GF8_15_9_1_SENT: [u8; 15] = [
    0x69, 0x68, 0x67, 0xD7, 0x83, 0xE3, 0xB2, 0x07, 0xA4, 0xCF, 0xFD, 0x29, 0x93, 0x4A, 0x04,
];
const GF8_15_9_1_RECEIVED: [u8; 15] = [
    0x69, 0x68, 0x67, 0xE1, 0x83, 0xE3, 0xB2, 0xCA, 0xA4, 0x97, 0xFD, 0x29, 0x93, 0x4A, 0x04,
];
const GF8_15_9_1_POSITIONS: [usize; 3] = [3, 7, 9];
const GF8_15_9_1_MAGNITUDES: [u8; 3] = [0x36, 0xCD, 0x58];
const GF8_15_9_1_SYNDROMES: [u8; 6] = [0xAC, 0x50, 0xA8, 0xAD, 0x7C, 0xEB];

const GF16_40_28_1_SENT: [u8; 80] = [
    0x47, 0xE8, 0xA5, 0xF5, 0x9C, 0xB5, 0xC6, 0x5A, 0x39, 0xF0, 0x88, 0x60, 0xE7, 0xFD, 0x2B, 0xE6,
    0xBC, 0xBF, 0x07, 0xAF, 0xD7, 0x33, 0x4E, 0xB6, 0x13, 0x5A, 0xBF, 0xCD, 0x1F, 0x95, 0x71, 0x09,
    0x4D, 0xCF, 0x49, 0xE6, 0x45, 0x06, 0xA4, 0x40, 0xA4, 0x66, 0x6D, 0x4F, 0x7F, 0x09, 0xF1, 0x54,
    0xCE, 0xA5, 0x6E, 0xCD, 0x21, 0x44, 0x9A, 0x6D, 0xCC, 0xB1, 0x56, 0x4E, 0x48, 0x08, 0x16, 0x1B,
    0x7B, 0x86, 0xD7, 0x24, 0xBF, 0xB7, 0xDF, 0x72, 0xEE, 0x92, 0x8F, 0x7B, 0x95, 0x69, 0xD0, 0x10,
];
const GF16_40_28_1_RECEIVED: [u8; 80] = [
    0x47, 0xE8, 0xA5, 0xF5, 0x9C, 0xB5, 0xC6, 0x5A, 0x39, 0xF0, 0x88, 0x60, 0xE7, 0xFD, 0x2B, 0xE6,
    0x01, 0x40, 0x07, 0xAF, 0xD7, 0x33, 0x4E, 0xB6, 0x13, 0x5A, 0xBF, 0xCD, 0x1F, 0x95, 0xF9, 0x31,
    0x4D, 0xCF, 0x49, 0xE6, 0x45, 0x06, 0x93, 0xA9, 0x5E, 0xAE, 0x6D, 0x4F, 0x7F, 0x09, 0xF1, 0x54,
    0xCE, 0xA5, 0x6E, 0xCD, 0x21, 0x44, 0x9A, 0x6D, 0xCC, 0xB1, 0x56, 0x4E, 0x48, 0x08, 0x16, 0x1B,
    0x7B, 0x86, 0x96, 0xFF, 0x03, 0xFD, 0xDF, 0x72, 0xEE, 0x92, 0x8F, 0x7B, 0x95, 0x69, 0xD0, 0x10,
];
const GF16_40_28_1_POSITIONS: [usize; 6] = [8, 15, 19, 20, 33, 34];
const GF16_40_28_1_MAGNITUDES: [u16; 6] = [0xFFBD, 0x3888, 0xE937, 0xC8FA, 0xDB41, 0x4ABC];

const GF8_17_8_3_SENT: [u8; 17] = [
    0xFD, 0xF2, 0xB3, 0xC0, 0xB1, 0xDE, 0x34, 0x23, 0xA6, 0x28, 0x9B, 0x2E, 0x75, 0xE6, 0xAC, 0x82,
    0x6F,
];
const GF8_17_8_3_RECEIVED: [u8; 17] = [
    0xFD, 0xF2, 0xF7, 0xC0, 0xF2, 0xDE, 0x22, 0x23, 0xA6, 0x28, 0x9B, 0x2E, 0x75, 0xE6, 0xAC, 0x82,
    0x42,
];
const GF8_17_8_3_POSITIONS: [usize; 4] = [2, 4, 6, 16];
const GF8_17_8_3_MAGNITUDES: [u8; 4] = [0x44, 0x43, 0x16, 0x2D];

#[test]
fn gf8_narrow_sense_fixture_decodes() {
    let params = RsParams::<Gf8B>::new(15, 9, 1).expect("params");
    let decoder = Decoder::new(params, Euclidean);
    let mut scratch = decoder.scratch().expect("scratch");

    // The pinned syndrome vector freezes the index convention at the byte
    // level: offset b = 1, primitive element F::GENERATOR, low degree first.
    let values = syndromes(&params, &GF8_15_9_1_RECEIVED).expect("syndromes");
    let pinned: Vec<u8> = values.iter().map(|value| value.to_raw()).collect();
    assert_eq!(pinned, GF8_15_9_1_SYNDROMES);

    let mut word = GF8_15_9_1_RECEIVED;
    let outcome = decoder
        .decode_into(&mut word, &mut scratch)
        .expect("fixture decodes");
    assert_eq!(outcome.positions(), GF8_15_9_1_POSITIONS);
    let magnitudes: Vec<u8> = outcome.magnitudes().iter().map(|m| m.to_raw()).collect();
    assert_eq!(magnitudes, GF8_15_9_1_MAGNITUDES);
    assert_eq!(word, GF8_15_9_1_SENT);
}

#[test]
fn gf16_narrow_sense_fixture_decodes() {
    let params = RsParams::<Gf16>::new(40, 28, 1).expect("params");
    let decoder = Decoder::new(params, Euclidean);
    let mut scratch = decoder.scratch().expect("scratch");
    let mut word = GF16_40_28_1_RECEIVED;
    let outcome = decoder
        .decode_into(&mut word, &mut scratch)
        .expect("fixture decodes");
    assert_eq!(outcome.positions(), GF16_40_28_1_POSITIONS);
    let magnitudes: Vec<u16> = outcome.magnitudes().iter().map(|m| m.to_raw()).collect();
    assert_eq!(magnitudes, GF16_40_28_1_MAGNITUDES);
    assert_eq!(word, GF16_40_28_1_SENT);
}

#[test]
fn offset_three_odd_redundancy_fixture_decodes() {
    // b = 3 with odd redundancy: the consecutive-root offset is part of the
    // frozen convention, not an afterthought.
    let params = RsParams::<Gf8B>::new(17, 8, 3).expect("params");
    let decoder = Decoder::new(params, Euclidean);
    let mut scratch = decoder.scratch().expect("scratch");
    let mut word = GF8_17_8_3_RECEIVED;
    let outcome = decoder
        .decode_into(&mut word, &mut scratch)
        .expect("fixture decodes");
    assert_eq!(outcome.positions(), GF8_17_8_3_POSITIONS);
    let magnitudes: Vec<u8> = outcome.magnitudes().iter().map(|m| m.to_raw()).collect();
    assert_eq!(magnitudes, GF8_17_8_3_MAGNITUDES);
    assert_eq!(word, GF8_17_8_3_SENT);
}

#[test]
fn fixture_words_are_codewords() {
    // The sent vectors themselves must be zero-syndrome codewords under the
    // frozen convention; otherwise the fixtures pin nothing.
    let params = RsParams::<Gf8B>::new(15, 9, 1).expect("params");
    assert!(
        syndromes(&params, &GF8_15_9_1_SENT)
            .expect("syndromes")
            .iter()
            .all(|value| value.is_zero())
    );
    let params = RsParams::<Gf16>::new(40, 28, 1).expect("params");
    assert!(
        syndromes(&params, &GF16_40_28_1_SENT)
            .expect("syndromes")
            .iter()
            .all(|value| value.is_zero())
    );
    let params = RsParams::<Gf8B>::new(17, 8, 3).expect("params");
    assert!(
        syndromes(&params, &GF8_17_8_3_SENT)
            .expect("syndromes")
            .iter()
            .all(|value| value.is_zero())
    );
}

const GF8_21_13_1_MIXED_SENT: [u8; 21] = [
    0x10, 0x3F, 0x96, 0xAD, 0x57, 0xF9, 0x45, 0x7F, 0xB7, 0xD7, 0x0C, 0xDD, 0x5C, 0x36, 0xCD, 0x06,
    0x08, 0x06, 0xB0, 0x98, 0x68,
];
const GF8_21_13_1_MIXED_RECEIVED: [u8; 21] = [
    0xB6, 0x3F, 0xDE, 0xAD, 0x57, 0xF9, 0x45, 0x7F, 0xB7, 0x20, 0x0C, 0xDD, 0x5C, 0x36, 0xCD, 0x7B,
    0x08, 0x06, 0xB0, 0x98, 0xD2,
];
const GF8_21_13_1_MIXED_ERASED: [usize; 3] = [2, 9, 20];
const GF8_21_13_1_MIXED_POSITIONS: [usize; 5] = [0, 2, 9, 15, 20];
const GF8_21_13_1_MIXED_MAGNITUDES: [u8; 5] = [0xA6, 0x48, 0xF7, 0x7D, 0xBA];

#[test]
fn mixed_errors_and_erasures_fixture_decodes() {
    // Freezes the Forney-syndrome convention: 2 errors + 3 erasures at
    // 2ν + ρ = 7 = d - 1, one unit inside the budget.
    let params = RsParams::<Gf8B>::new(21, 13, 1).expect("params");
    let decoder = Decoder::new(params, Euclidean);
    let mut scratch = decoder.scratch().expect("scratch");
    let mut word = GF8_21_13_1_MIXED_RECEIVED;
    let outcome = decoder
        .decode_with_erasures_into(&mut word, &GF8_21_13_1_MIXED_ERASED, &mut scratch)
        .expect("fixture decodes");
    assert_eq!(outcome.positions(), GF8_21_13_1_MIXED_POSITIONS);
    let magnitudes: Vec<u8> = outcome.magnitudes().iter().map(|m| m.to_raw()).collect();
    assert_eq!(magnitudes, GF8_21_13_1_MIXED_MAGNITUDES);
    assert_eq!(word, GF8_21_13_1_MIXED_SENT);
    // The sent vector is a codeword under the frozen convention.
    assert!(
        syndromes(&params, &GF8_21_13_1_MIXED_SENT)
            .expect("syndromes")
            .iter()
            .all(|value| value.is_zero())
    );
}
