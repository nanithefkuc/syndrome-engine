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
//! Euclidean backend, which composes `univariate`'s allocating
//! `truncated_eea` by design; that path is the latency choice, not the
//! zero-allocation choice.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use fgf::kernel::FieldKernels;
use fgf::{Gf8, Gf16};
use syndrome_engine::{Decoder, Euclidean, RsParams};

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
}

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && COUNTING.with(Cell::get) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
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
    ALLOCATIONS.store(0, Ordering::Relaxed);
    COUNTING.with(|counting| counting.set(true));
    operation();
    COUNTING.with(|counting| counting.set(false));
    ALLOCATIONS.load(Ordering::Relaxed)
}

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
    let params = RsParams::<Gf8>::new(31, 21, 1).expect("params");
    let decoder = Decoder::new(params, Euclidean);
    let mut scratch = decoder.scratch().expect("scratch");
    let word = warmed_word::<Gf8>(31, 0x2A00);
    decoder
        .syndromes_into(&word, &mut scratch)
        .expect("warm syndromes");

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
    // warm the scratch with one decode, then the next decode over the same
    // geometry must not allocate.
    let params = RsParams::<Gf8>::new(31, 21, 1).expect("params");
    let decoder = Decoder::new(params, BerlekampMassey);
    let mut scratch = decoder.scratch().expect("scratch");
    let sent = common::random_codeword(&params, 0x2B00);
    let mut word = sent.clone();
    word[3] ^= 0x5A;
    word[17] ^= 0xA5;
    word[30] ^= 0xFF;
    let outcome = decoder.decode_into(&mut word, &mut scratch).expect("warm");
    assert_eq!(outcome.error_count(), 3);
    // The corrected word re-decodes to zero errors through the same
    // scratch; both passes must be allocation-free.
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
