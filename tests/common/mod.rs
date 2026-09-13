//! Shared test-side oracles: a generator-polynomial encoder, an independent
//! per-point Horner syndrome computation, and the fixed-seed LCG convention
//! from `fgf`'s test suite. Deliberately slow and deliberately structurally
//! different from the production paths.

#![allow(dead_code)]

use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use poly_ring::Polynomial;
use syndrome_engine::RsParams;

/// One LCG step, matching `fgf`'s `noise(len, seed)` convention.
pub fn advance(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// A random field element from the LCG stream.
pub fn noise_elem<F: FieldKernels>(state: &mut u64) -> F::Elem {
    let bytes = advance(state).to_le_bytes();
    F::read(&bytes[..F::BYTES])
}

/// A nonzero random field element (rejection sampling), for error
/// magnitudes and erasure values.
pub fn nonzero_elem<F: FieldKernels>(state: &mut u64) -> F::Elem {
    loop {
        let element = noise_elem::<F>(state);
        if !element.is_zero() {
            return element;
        }
    }
}

/// The generator polynomial `Π_{j<n-k} (x + α^{b+j})`: multiplying any
/// message of degree `< k` by it produces a codeword, since every
/// `α^{b+j}` is a root of the product.
pub fn generator<F: FieldKernels>(params: &RsParams<F>) -> Polynomial<F> {
    let alpha = <F as Field>::GENERATOR;
    let mut polynomial = Polynomial::one().expect("one");
    let mut root = alpha.pow(params.b() as u64);
    for _ in 0..params.redundancy() {
        // Multiply by (x + root), so g vanishes at every consecutive root.
        polynomial = polynomial.multiply_x_plus(root).expect("generator product");
        root = root.mul(alpha);
    }
    polynomial
}

/// A random codeword packed into `n` field elements.
pub fn random_codeword<F: FieldKernels>(params: &RsParams<F>, seed: u64) -> Vec<u8> {
    let mut state = seed;
    let message: Vec<F::Elem> = (0..params.k())
        .map(|_| noise_elem::<F>(&mut state))
        .collect();
    let message = Polynomial::from_coefficients(&message).expect("message");
    let codeword = message.multiply(&generator::<F>(params)).expect("codeword");
    let mut bytes = vec![0_u8; params.n() * F::BYTES];
    for (degree, coefficient) in codeword.coefficients().enumerate() {
        F::write(
            &mut bytes[degree * F::BYTES..(degree + 1) * F::BYTES],
            coefficient,
        );
    }
    bytes
}

/// Per-point Horner syndrome oracle: evaluates `R(x)` at each `α^{b+j}` by
/// an explicit high-to-low coefficient sweep, sharing no code with the
/// subproduct-tree production path.
#[allow(clippy::chunks_exact_to_as_chunks)] // `as_chunks` cannot take a generic parameter's associated const
pub fn horner_syndromes<F: FieldKernels>(params: &RsParams<F>, received: &[u8]) -> Vec<F::Elem> {
    let coefficients: Vec<F::Elem> = received.chunks_exact(F::BYTES).map(F::read).collect();
    let alpha = <F as Field>::GENERATOR;
    let mut point = alpha.pow(params.b() as u64);
    let mut values = Vec::with_capacity(params.syndrome_count());
    for _ in 0..params.syndrome_count() {
        let mut accumulator = F::Elem::ZERO;
        for coefficient in coefficients.iter().rev() {
            accumulator = accumulator.mul(point).add(*coefficient);
        }
        values.push(accumulator);
        point = point.mul(alpha);
    }
    values
}

/// `count` distinct positions below `n` from the LCG stream.
pub fn distinct_positions(n: usize, count: usize, seed: u64) -> Vec<usize> {
    let mut state = seed;
    let mut chosen = Vec::with_capacity(count);
    while chosen.len() < count {
        let position = usize::try_from(advance(&mut state) % (n as u64)).expect("position");
        if !chosen.contains(&position) {
            chosen.push(position);
        }
    }
    chosen
}

/// Flip `positions.len()` symbols of a packed word by random nonzero
/// magnitudes, returning the magnitudes.
pub fn inject<F: FieldKernels>(
    word: &mut [u8],
    positions: &[usize],
    state: &mut u64,
) -> Vec<F::Elem> {
    let mut magnitudes = Vec::with_capacity(positions.len());
    for &position in positions {
        let magnitude = nonzero_elem::<F>(state);
        let start = position * F::BYTES;
        let current = F::read(&word[start..start + F::BYTES]);
        F::write(&mut word[start..start + F::BYTES], current.add(magnitude));
        magnitudes.push(magnitude);
    }
    magnitudes
}

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
