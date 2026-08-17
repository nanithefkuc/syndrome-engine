//! Naive reference oracles, written once and never optimized: a textbook
//! Peterson–Gorenstein–Zierler matrix-solve locator (the O(ν³) algorithm
//! the key equation replaces), a brute-force small-`ν` magnitude solve by
//! direct Vandermonde elimination, and self-tests pinning both against the
//! production solvers. Deliberately structurally different from
//! Berlekamp–Massey / Euclid / Forney.

mod common;

use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use fgf::{Gf8, Gf16};
use syndrome_engine::{
    Euclidean, KeyEqScratch, KeyEquation, KeyEquationSolver, RsParams, syndromes,
};
use univariate::Polynomial;

/// Solve a small dense GF(2^m) system by Gauss–Jordan elimination. Returns
/// `None` when the system is singular.
pub fn gaussian_solve<F: FieldKernels>(
    mut matrix: Vec<Vec<F::Elem>>,
    mut rhs: Vec<F::Elem>,
) -> Option<Vec<F::Elem>> {
    let size = rhs.len();
    for column in 0..size {
        let pivot = (column..size).find(|row| !matrix[*row][column].is_zero())?;
        matrix.swap(column, pivot);
        rhs.swap(column, pivot);
        let inverse = matrix[column][column].inv();
        for coefficient in &mut matrix[column] {
            *coefficient = coefficient.mul(inverse);
        }
        rhs[column] = rhs[column].mul(inverse);
        let pivot_row = matrix[column].clone();
        for row in 0..size {
            if row != column && !matrix[row][column].is_zero() {
                let factor = matrix[row][column];
                for (target, source) in matrix[row].iter_mut().zip(&pivot_row) {
                    *target = target.add(factor.mul(*source));
                }
                let source = rhs[column];
                rhs[row] = rhs[row].add(factor.mul(source));
            }
        }
    }
    Some(rhs)
}

/// Textbook Peterson–Gorenstein–Zierler locator: for `ν = t` down to `1`,
/// solve the Hankel system `Σ_{i=1..ν} Λ_i·S_{ν-i+j} = S_{ν+j}` for the
/// locator coefficients; the largest nonsingular `ν` wins.
pub fn pgz_locator<F: FieldKernels>(syndromes: &[F::Elem], radius: usize) -> Vec<F::Elem> {
    for errors in (1..=radius).rev() {
        let mut matrix = Vec::with_capacity(errors);
        for row in 0..errors {
            // Unknown Λ_i at column i-1: coefficient S_{row + errors - i}.
            let coefficients: Vec<F::Elem> =
                (1..=errors).map(|i| syndromes[row + errors - i]).collect();
            matrix.push(coefficients);
        }
        let rhs: Vec<F::Elem> = (0..errors).map(|row| syndromes[errors + row]).collect();
        if let Some(mut locator) = gaussian_solve::<F>(matrix, rhs) {
            locator.insert(0, F::Elem::ONE);
            return locator;
        }
    }
    vec![F::Elem::ONE]
}

/// Brute-force magnitudes for `ν ≤ 2` errors: solve the Vandermonde system
/// `Σ_i e_i·X_i^{b+j} = S_j`, `j = 0..ν-1`, directly.
pub fn brute_force_magnitudes<F: FieldKernels>(
    params: &RsParams<F>,
    positions: &[usize],
    syndromes: &[F::Elem],
) -> Vec<F::Elem> {
    let alpha = <F as Field>::GENERATOR;
    let count = positions.len();
    assert!(count <= 2, "brute-force oracle covers at most two errors");
    let mut matrix = Vec::with_capacity(count);
    for row in 0..count {
        let coefficients: Vec<F::Elem> = positions
            .iter()
            .map(|&position| {
                let exponent = (position * (params.b() + row)) as u64;
                alpha.pow(exponent)
            })
            .collect();
        matrix.push(coefficients);
    }
    let rhs: Vec<F::Elem> = syndromes[..count].to_vec();
    gaussian_solve::<F>(matrix, rhs).expect("distinct positions give a nonsingular Vandermonde")
}

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
        fixture::<Gf8>(15, 7, 1, errors, 0xA00 + errors as u64); // t = 4
        fixture::<Gf8>(21, 13, 0, errors, 0xA10 + errors as u64);
        fixture::<Gf8>(17, 8, 3, errors, 0xA20 + errors as u64);
        fixture::<Gf16>(40, 28, 1, errors, 0xA30 + errors as u64);
    }
}

#[test]
fn gaussian_solve_rejects_singular_systems() {
    let matrix = vec![
        vec![fgf::gf8::Elem(1), fgf::gf8::Elem(1)],
        vec![fgf::gf8::Elem(1), fgf::gf8::Elem(1)],
    ];
    let rhs = vec![fgf::gf8::Elem(1), fgf::gf8::Elem(0)];
    let solved = gaussian_solve::<fgf::Gf8>(matrix, rhs);
    assert!(solved.is_none());
}

#[test]
fn locator_from_positions_matches_solver() {
    // The locator implied by the error positions — Π(1 + X_p x) built with
    // univariate — must equal the solved locator, tying the root→position
    // convention to the solver output.
    let params = RsParams::<Gf8>::new(15, 7, 1).expect("params");
    let mut word = common::random_codeword(&params, 0xB00);
    let mut positions = common::distinct_positions(15, 3, 0xB01);
    positions.sort();
    let mut state = 0xB02;
    common::inject::<Gf8>(&mut word, &positions, &mut state);
    let values = syndromes(&params, &word).expect("syndromes");
    let mut out = KeyEquation::<Gf8>::with_capacity(8).expect("out");

    let mut scratch = KeyEqScratch::with_capacity(values.len()).expect("scratch");
    Euclidean
        .solve(&[&values], &mut out, &mut scratch)
        .expect("solve");

    let alpha = <Gf8 as Field>::GENERATOR;
    // The locator implied by the positions: Π(1 + X_p x), monic at the
    // constant term. Build Π(x + X_p^{-1}) and rescale by Π X_p.
    let mut expected = Polynomial::one().expect("one");
    let mut scale = fgf::gf8::Elem(1);
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
