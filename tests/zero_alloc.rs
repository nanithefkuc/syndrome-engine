//! Steady-state zero allocation, proven under a counting global allocator
//! (S5).
//!
//! The counter is scoped to the counting thread: sibling tests running in
//! parallel must not be charged to a measurement.
//!
//! Scope note: the Berlekamp–Massey decode is the zero-allocation hot path
//! and is asserted here once it is the default solver; the Euclidean
//! backend composes `univariate`'s allocating `truncated_eea` by design and
//! is the cross-check, not the hot path. The syndrome `_into` pass below is
//! allocation-free with either solver behind it.

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
