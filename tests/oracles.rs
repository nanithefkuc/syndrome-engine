//! Naive reference oracles, written once and never optimized: a textbook
//! Peterson–Gorenstein–Zierler matrix-solve locator (the O(ν³) algorithm
//! the key equation replaces), a brute-force small-`ν` magnitude solve by
//! direct Vandermonde elimination, and self-tests pinning both against the
//! production solvers. Deliberately structurally different from
//! Berlekamp–Massey / Euclid / Forney.

mod common;

use common::{brute_force_magnitudes, gaussian_solve, pgz_locator};
use fgf::field::Field;
use fgf::kernel::FieldKernels;
use fgf::{Gf8B, Gf16};
use poly_ring::Polynomial;
use syndrome_engine::{
    Euclidean, KeyEqScratch, KeyEquation, KeyEquationSolver, RsParams, syndromes,
};

fn fixture<F: FieldKernels>(n: usize, k: usize, b: usize, errors: usize, seed: u64) {
    let params = RsParams::<F>::new(n, k, b).expect("params");
    let mut word = common::random_codeword(&params, seed);
    let mut positions = common::distinct_positions(n, errors, seed ^ 0x9E37);
    positions.sort();
    let mut state = seed ^ 0x85EB;
    let magnitudes = common::inject::<F>(&mut word, &positions, &mut state);

    let values = syndromes(&params, &word).expect("syndromes");
    let radius = params.correction_radius();

    // PGZ must reproduce the production Euclidean locator exactly.
    let oracle_locator = pgz_locator::<F>(&values, radius);
    let mut out = KeyEquation::<F>::with_capacity(radius + 1).expect("out");
    let mut scratch = KeyEqScratch::with_capacity(values.len()).expect("scratch");
    Euclidean
        .solve(&[&values], &mut out, &mut scratch)
        .expect("solve");
    let produced: Vec<F::Elem> = out.locator().coefficients().collect();
    assert_eq!(produced.len(), oracle_locator.len());
    for (produced, oracle) in produced.iter().zip(&oracle_locator) {
        assert_eq!(*produced, *oracle);
    }

    // The brute-force magnitudes must match what was injected.
    if errors <= 2 {
        let brute = brute_force_magnitudes(&params, &positions, &values);
        for (brute, injected) in brute.iter().zip(&magnitudes) {
            assert_eq!(*brute, *injected);
        }
    }
}

#[test]
fn pgz_and_brute_force_agree_with_production() {
    for errors in 1..=4 {
        fixture::<Gf8B>(15, 7, 1, errors, 0xA00 + errors as u64); // t = 4
        fixture::<Gf8B>(21, 13, 0, errors, 0xA10 + errors as u64);
        fixture::<Gf8B>(17, 8, 3, errors, 0xA20 + errors as u64);
        fixture::<Gf16>(40, 28, 1, errors, 0xA30 + errors as u64);
    }
}

#[test]
fn gaussian_solve_rejects_singular_systems() {
    let matrix = vec![
        vec![fgf::gf8b::Elem::from_raw(1), fgf::gf8b::Elem::from_raw(1)],
        vec![fgf::gf8b::Elem::from_raw(1), fgf::gf8b::Elem::from_raw(1)],
    ];
    let rhs = vec![fgf::gf8b::Elem::from_raw(1), fgf::gf8b::Elem::from_raw(0)];
    let solved = gaussian_solve::<fgf::Gf8B>(matrix, rhs);
    assert!(solved.is_none());
}

#[test]
fn locator_from_positions_matches_solver() {
    // The locator implied by the error positions — Π(1 + X_p x) built with
    // poly-ring — must equal the solved locator, tying the root→position
    // convention to the solver output.
    let params = RsParams::<Gf8B>::new(15, 7, 1).expect("params");
    let mut word = common::random_codeword(&params, 0xB00);
    let mut positions = common::distinct_positions(15, 3, 0xB01);
    positions.sort();
    let mut state = 0xB02;
    common::inject::<Gf8B>(&mut word, &positions, &mut state);
    let values = syndromes(&params, &word).expect("syndromes");
    let mut out = KeyEquation::<Gf8B>::with_capacity(8).expect("out");

    let mut scratch = KeyEqScratch::with_capacity(values.len()).expect("scratch");
    Euclidean
        .solve(&[&values], &mut out, &mut scratch)
        .expect("solve");

    let alpha = <Gf8B as Field>::GENERATOR;
    // The locator implied by the positions: Π(1 + X_p x), monic at the
    // constant term. Build Π(x + X_p^{-1}) and rescale by Π X_p.
    let mut expected = Polynomial::one().expect("one");
    let mut scale = fgf::gf8b::Elem::from_raw(1);
    for &position in &positions {
        let locator = alpha.pow(position as u64);
        scale = scale.mul(locator);
        expected = expected
            .multiply_x_plus(locator.inv())
            .expect("locator product");
    }
    expected.scale_assign(scale);
    assert_eq!(out.locator(), &expected);
}
