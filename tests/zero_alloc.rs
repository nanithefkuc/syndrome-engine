//! Steady-state zero allocation, proven under a counting global allocator
//! (S5).
//!
//! The counter is scoped to the counting thread: sibling tests running in
//! parallel must not be charged to a measurement.
//!
//! Scope note: the default `Adaptive` solver picks Berlekamp–Massey
//! through the measured crossover (`cost::BM_EUCLIDEAN_CROSSOVER`, see
//! `BENCHMARKS.md`), so every default decode in the parity band is
//! allocation-free — asserted here. Past the threshold it dispatches to the
//! Euclidean backend, which composes `poly-ring`'s allocating
//! `truncated_eea` by design; that path is the latency choice, not the
//! zero-allocation choice.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use fgf::kernel::FieldKernels;
use fgf::{Gf8B, Gf16};
use syndrome_engine::{Decoder, Euclidean, RsParams};

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static COUNT: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && COUNTING.with(Cell::get) {
            COUNT.with(|count| count.set(count.get() + 1));
        }
        pointer
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn count_allocations<F: FnMut()>(mut operation: F) -> usize {
    COUNT.with(|count| count.set(0));
    COUNTING.with(|counting| counting.set(true));
    operation();
    COUNTING.with(|counting| counting.set(false));
    COUNT.with(Cell::get)
}

#[allow(clippy::chunks_exact_to_as_chunks)] // `as_chunks` cannot take a generic parameter's associated const
fn warmed_word<F: FieldKernels>(n: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    let mut bytes = vec![0_u8; n * F::BYTES];
    for chunk in bytes.chunks_exact_mut(F::BYTES) {
        let noise = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        state = noise;
        let raw = noise.to_le_bytes();
        chunk.copy_from_slice(&raw[..F::BYTES]);
    }
    bytes
}

#[test]
fn steady_state_syndrome_pass_allocates_nothing() {
    let params = RsParams::<Gf8B>::new(31, 21, 1).expect("params");
    let decoder = Decoder::new(params, Euclidean);
    let mut scratch = decoder.scratch().expect("scratch");
    let word = warmed_word::<Gf8B>(31, 0x2A00);
    decoder
        .syndromes_into(&word, &mut scratch)
        .expect("warm syndromes");
    // Second warm pass: the ring's subproduct pools settle their internal
    // shapes on the second build, as in the decode test below.
    decoder
        .syndromes_into(&word, &mut scratch)
        .expect("warm syndromes again");

    let allocations = count_allocations(|| {
        decoder
            .syndromes_into(&word, &mut scratch)
            .expect("steady syndromes");
    });
    assert_eq!(allocations, 0, "warm syndrome pass must not allocate");

    let params = RsParams::<Gf16>::new(120, 100, 1).expect("params");
    let decoder = Decoder::new(params, Euclidean);
    let mut scratch = decoder.scratch().expect("scratch");
    let word = warmed_word::<Gf16>(120, 0x2A01);
    decoder
        .syndromes_into(&word, &mut scratch)
        .expect("warm syndromes");
    decoder
        .syndromes_into(&word, &mut scratch)
        .expect("warm syndromes again");
    let allocations = count_allocations(|| {
        decoder
            .syndromes_into(&word, &mut scratch)
            .expect("steady syndromes");
    });
    assert_eq!(allocations, 0, "warm syndrome pass must not allocate");
}

mod common;

#[test]
fn steady_state_decode_allocates_nothing() {
    use syndrome_engine::BerlekampMassey;

    // The default (Adaptive → Berlekamp–Massey) decode is the hot path:
    // warm the scratch with two decodes (see the comment at the second
    // warm call), then the next decode over the same geometry must not
    // allocate.
    let params = RsParams::<Gf8B>::new(31, 21, 1).expect("params");
    let decoder = Decoder::new(params, BerlekampMassey);
    let mut scratch = decoder.scratch().expect("scratch");
    let sent = common::random_codeword(&params, 0x2B00);
    let mut word = sent.clone();
    word[3] ^= 0x5A;
    word[17] ^= 0xA5;
    word[30] ^= 0xFF;
    let outcome = decoder.decode_into(&mut word, &mut scratch).expect("warm");
    assert_eq!(outcome.error_count(), 3);
    // Second warm pass: the first sizes every pool, and `poly-ring`'s
    // subproduct pools settle their internal shapes on the second build.
    // The measured steady state is the converged one.
    decoder
        .decode_into(&mut word, &mut scratch)
        .expect("warm again");
    // The corrected word re-decodes to zero errors through the same
    // scratch; the steady pass must be allocation-free.
    let first = word.clone();
    let allocations = count_allocations(|| {
        let outcome = decoder
            .decode_into(&mut word, &mut scratch)
            .expect("steady decode");
        assert_eq!(outcome.error_count(), 0);
    });
    assert_eq!(allocations, 0, "warm decode must not allocate");
    assert_eq!(word, first);

    // From syndromes, too: the reliability-engine steady state.
    let values = decoder
        .syndromes_into(&word, &mut scratch)
        .expect("syndromes")
        .to_vec();
    let allocations = count_allocations(|| {
        decoder
            .decode_syndromes_into(&values, &mut scratch)
            .expect("steady syndrome decode");
    });
    assert_eq!(
        allocations, 0,
        "warm syndromes-only decode must not allocate"
    );
}

#[test]
fn steady_state_erasure_decode_allocates_nothing() {
    use syndrome_engine::BerlekampMassey;

    let params = RsParams::<Gf8B>::new(31, 21, 1).expect("params");
    let decoder = Decoder::new(params, BerlekampMassey);
    let mut scratch = decoder.scratch().expect("scratch");
    let sent = common::random_codeword(&params, 0x2C00);
    let mut word = sent.clone();
    word[5] ^= 0x3C;
    word[11] ^= 0xC3;
    let erased = [5_usize, 11, 2];
    let warm_allocs = count_allocations(|| {
        decoder
            .decode_with_erasures_into(&mut word, &erased, &mut scratch)
            .expect("warm");
    });
    eprintln!("warm allocs: {warm_allocs}");
    let mid_allocs = count_allocations(|| {
        decoder
            .decode_with_erasures_into(&mut word, &erased, &mut scratch)
            .expect("mid");
    });
    eprintln!("mid allocs: {mid_allocs}");
    let allocations = count_allocations(|| {
        decoder
            .decode_with_erasures_into(&mut word, &erased, &mut scratch)
            .expect("steady erasure decode");
    });
    assert_eq!(allocations, 0, "warm erasure decode must not allocate");
}

#[test]
fn bisect_allocating_stage() {
    use syndrome_engine::BerlekampMassey;
    // Which stage allocates on the second pure decode?
    let params = RsParams::<Gf8B>::new(31, 21, 1).expect("params");
    let decoder = Decoder::new(params, BerlekampMassey);
    let mut scratch = decoder.scratch().expect("scratch");
    let sent = common::random_codeword(&params, 0x2D00);
    let mut word = sent.clone();
    word[3] ^= 0x5A;
    decoder.decode_into(&mut word, &mut scratch).expect("warm");

    let a = count_allocations(|| {
        decoder.syndromes_into(&word, &mut scratch).expect("s");
    });
    let mut word2 = word.clone();
    let b = count_allocations(|| {
        let _ = decoder.decode_into(&mut word2, &mut scratch);
    });
    eprintln!("syndrome stage allocs: {a}, full decode allocs: {b}");
}

#[test]
fn bisect_erasure_stage() {
    use syndrome_engine::BerlekampMassey;

    let params = RsParams::<Gf8B>::new(31, 21, 1).expect("params");
    let decoder = Decoder::new(params, BerlekampMassey);
    let mut scratch = decoder.scratch().expect("scratch");
    let sent = common::random_codeword(&params, 0x2C00);
    let mut word = sent.clone();
    word[5] ^= 0x3C;
    word[11] ^= 0xC3;
    let erased = [5_usize, 11, 2];
    let warm = decoder.decode_with_erasures_into(&mut word, &erased, &mut scratch);
    eprintln!(
        "warm: {:?}",
        warm.as_ref()
            .map(|o| o.error_count())
            .map_err(|e| e.to_string())
    );

    // zero-error steady call
    let a = count_allocations(|| {
        decoder
            .decode_with_erasures_into(&mut word, &erased, &mut scratch)
            .expect("steady");
    });
    eprintln!("erasure steady allocs: {a}");
}
