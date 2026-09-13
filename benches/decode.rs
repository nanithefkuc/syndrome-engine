//! Decode-stage benchmarks: syndrome evaluation, both key-equation
//! backends, the Chien position scan, Forney magnitudes, and the
//! end-to-end decode at the correction boundary. Every case warms its
//! scratch first — the numbers describe the allocation-free steady state
//! that `tests/zero_alloc.rs` proves. Baselines are deliberately not
//! committed.

use criterion::{Criterion, criterion_group, criterion_main};
use fgf::Gf16;
use fgf::field::{Elem, Field};
use poly_ring::{ChienScratch, MultipointScratch, Polynomial, chien_roots_into};
use syndrome_engine::stages::{batch_invert_into, locate_into};
use syndrome_engine::{
    BerlekampMassey, Decoder, Euclidean, KeyEqScratch, KeyEquation, KeyEquationSolver, RsParams,
    syndromes,
};

fn generator<F: fgf::kernel::FieldKernels>(params: &RsParams<F>) -> Polynomial<F> {
    let alpha = <F as Field>::GENERATOR;
    let mut polynomial = Polynomial::one().expect("one");
    let mut root = alpha.pow(params.b() as u64);
    for _ in 0..params.redundancy() {
        polynomial = polynomial.multiply_x_plus(root).expect("generator");
        root = root.mul(alpha);
    }
    polynomial
}

fn received<F: fgf::kernel::FieldKernels>(
    n: usize,
    k: usize,
    b: usize,
    errors: usize,
) -> (RsParams<F>, Vec<u8>) {
    let params = RsParams::<F>::new(n, k, b).expect("params");
    let mut state = 0xBEEF_0000 ^ (u64::try_from(n).expect("n")) ^ (u64::try_from(k).expect("k"));
    let message: Vec<F::Elem> = (0..k)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bytes = state.to_le_bytes();
            F::read(&bytes[..F::BYTES])
        })
        .collect();
    let codeword = Polynomial::from_coefficients(&message)
        .expect("message")
        .multiply(&generator::<F>(&params))
        .expect("codeword");
    let mut bytes = vec![0_u8; n * F::BYTES];
    for (degree, coefficient) in codeword.coefficients().enumerate() {
        F::write(
            &mut bytes[degree * F::BYTES..(degree + 1) * F::BYTES],
            coefficient,
        );
    }
    let mut position = 0_usize;
    for _ in 0..errors {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        position = (position + 7 + (state % 31) as usize) % n;
        let start = position * F::BYTES;
        let current = F::read(&bytes[start..start + F::BYTES]);
        let noise = F::read(&state.to_le_bytes()[..F::BYTES]);
        F::write(&mut bytes[start..start + F::BYTES], current.add(noise));
    }
    (params, bytes)
}

fn bench_syndrome(criterion: &mut Criterion) {
    for &(n, k) in &[(255_usize, 223_usize), (1023, 900)] {
        let params = RsParams::<Gf16>::new(n, k, 1).expect("params");
        let decoder = Decoder::new(params, BerlekampMassey);
        let mut scratch = decoder.scratch().expect("scratch");
        let (_, word) = received::<Gf16>(n, k, 1, 4);
        decoder.syndromes_into(&word, &mut scratch).expect("warm");
        criterion.bench_function(&format!("syndrome/gf16/{n}x{k}"), |bench| {
            bench.iter(|| {
                decoder
                    .syndromes_into(&word, &mut scratch)
                    .expect("syndromes");
            });
        });
    }
}

fn bench_keyeq(criterion: &mut Criterion) {
    for &(n, k) in &[
        (31_usize, 21_usize),
        (48, 32),
        (64, 40),
        (80, 48),
        (96, 56),
        (112, 64),
        (255, 223),
        (1023, 900),
        (4095, 3000),
    ] {
        let (params, word) = received::<Gf16>(n, k, 1, (n - k) / 2);
        let values = syndromes(&params, &word).expect("syndromes");
        let mut bm = KeyEquation::<Gf16>::with_capacity(n - k + 1).expect("bm");
        let mut eea = KeyEquation::<Gf16>::with_capacity(n - k + 1).expect("eea");
        let mut scratch = KeyEqScratch::with_capacity(values.len()).expect("scratch");
        BerlekampMassey
            .solve(&[&values], &mut bm, &mut scratch)
            .expect("warm bm");
        criterion.bench_function(&format!("keyeq_bm/gf16/{n}x{k}"), |bench| {
            bench.iter(|| {
                BerlekampMassey
                    .solve(&[&values], &mut bm, &mut scratch)
                    .expect("bm");
            });
        });
        Euclidean
            .solve(&[&values], &mut eea, &mut scratch)
            .expect("warm eea");
        criterion.bench_function(&format!("keyeq_euclidean/gf16/{n}x{k}"), |bench| {
            bench.iter(|| {
                Euclidean
                    .solve(&[&values], &mut eea, &mut scratch)
                    .expect("eea");
            });
        });
    }
}

fn bench_locate(criterion: &mut Criterion) {
    let (params, word) = received::<Gf16>(1023, 900, 1, 60);
    let values = syndromes(&params, &word).expect("syndromes");
    let mut out = KeyEquation::<Gf16>::with_capacity(200).expect("out");
    let mut scratch = KeyEqScratch::with_capacity(values.len()).expect("scratch");
    BerlekampMassey
        .solve(&[&values], &mut out, &mut scratch)
        .expect("solve");
    let mut position_points = Vec::new();
    syndrome_engine::stages::position_points_into(&params, &mut position_points);
    let mut eval = MultipointScratch::<Gf16>::new();
    let mut values_buffer = Vec::new();
    let mut positions = Vec::new();
    locate_into(
        out.locator(),
        &position_points,
        &mut eval,
        &mut values_buffer,
        &mut positions,
    )
    .expect("warm");
    criterion.bench_function("chien/gf16/1023x900", |bench| {
        bench.iter(|| {
            locate_into(
                out.locator(),
                &position_points,
                &mut eval,
                &mut values_buffer,
                &mut positions,
            )
            .expect("locate");
        });
    });

    // The whole-field `poly-ring` Chien scan for reference, same locator.
    let mut chien = ChienScratch::<Gf16>::new();
    let mut roots = Vec::new();
    chien_roots_into(out.locator(), &mut chien, &mut roots).expect("warm");
    criterion.bench_function("chien_field_scan/gf16/1023x900", |bench| {
        bench.iter(|| {
            chien_roots_into(out.locator(), &mut chien, &mut roots).expect("scan");
        });
    });
}

fn bench_forney(criterion: &mut Criterion) {
    // Batch inversion of 60 denominators — the Forney kernel in isolation.
    let mut values: Vec<fgf::gf16::Elem> = (1..=60)
        .map(|index| fgf::gf16::Elem::from_raw(u16::try_from(index * 37 + 1).expect("nonzero")))
        .collect();
    let mut prefix = Vec::new();
    batch_invert_into::<Gf16>(&mut values, &mut prefix);
    criterion.bench_function("forney/batch_invert/60", |bench| {
        bench.iter(|| {
            batch_invert_into::<Gf16>(&mut values, &mut prefix);
        });
    });
}

fn bench_decode_e2e(criterion: &mut Criterion) {
    for &(n, k) in &[(255_usize, 223_usize), (1023, 900)] {
        let (params, sent) = received::<Gf16>(n, k, 1, 0);
        let decoder = Decoder::new(params, BerlekampMassey);
        let mut scratch = decoder.scratch().expect("scratch");
        let errors = (n - k) / 2;
        let (_, word) = received::<Gf16>(n, k, 1, errors);
        decoder
            .decode_into(&mut word.clone(), &mut scratch)
            .expect("warm");
        criterion.bench_function(&format!("decode_e2e/gf16/{n}x{k}/t{errors}"), |bench| {
            bench.iter_batched(
                || word.clone(),
                |mut received| {
                    decoder
                        .decode_into(&mut received, &mut scratch)
                        .expect("decode");
                },
                criterion::BatchSize::SmallInput,
            );
        });
        let _ = &sent;
    }
}

criterion_group!(
    benches,
    bench_syndrome,
    bench_keyeq,
    bench_locate,
    bench_forney,
    bench_decode_e2e
);
criterion_main!(benches);
