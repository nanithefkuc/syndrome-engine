//! Shared test-side oracles: a generator-polynomial encoder, an independent
//! per-point Horner syndrome computation, and the fixed-seed LCG convention
//! from `fgf`'s test suite. Deliberately slow and deliberately structurally
//! different from the production paths.

#![allow(dead_code)]

use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use syndrome_engine::RsParams;
use univariate::Polynomial;

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
