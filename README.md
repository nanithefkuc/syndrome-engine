> [!WARNING]
> This library was made with the help of AI. While the library has tests
> to check for regressions, things can break. Audit the code yourself, or with
> your own agent before using.

# syndrome-engine - Bounded-Distance RS/BCH Decoding via the Key Equation

`syndrome-engine` is the bounded-distance error-decoder node of the FEC
stack: the classical Reed–Solomon / BCH unique-decoding pipeline — syndrome
computation, the key-equation solve producing the error-locator `Λ(x)` and
error-evaluator `Ω(x)` (Berlekamp–Massey and Euclidean/Sugiyama backends),
the Chien search that discovers error *positions*, and Forney's formula for
the magnitudes — underneath every consumer that has to correct errors at
unknown positions.

> `syndrome-engine` is a decoder, not a codec and not a polynomial ring.
> Field arithmetic and byte-buffer vector primitives come from `fgf` — never
> re-implemented here. Polynomial evaluation, division, gcd, extended-Euclid,
> formal derivative, and root-finding come from `poly-ring` — never
> re-hosted here. Pure-erasure decoding, wire formats, shard ownership,
> code-parameter selection, and soft-decision reliability processing belong
> to consumers. This crate receives a corrupted word and returns the
> corrections.

Everything here decodes **errors at unknown positions**. The rest of the
stack decodes erasures; this is the missing half. Erasures still enter —
folded into modified (Forney) syndromes — with the unified guarantee
`2ν + ρ ≤ d - 1`. Inside the radius the unique correction is returned;
outside it a typed `DecodeError`, never a panic and never a silent wrong
answer where the failure is detectable: the root count must match the
locator degree and the corrected word must re-encode to all-zero syndromes.

## Usage

```rust
use fgf::Gf8B;
use syndrome_engine::{BerlekampMassey, Decoder, RsParams};

let params = RsParams::<Gf8B>::new(255, 223, 1).unwrap();
let decoder = Decoder::new(params, BerlekampMassey);
let mut scratch = decoder.scratch().unwrap();

// A received word: 255 packed GF(256) symbols, up to t = 16 in error.
let mut received = [0u8; 255];
// ... channel output lands here ...
match decoder.decode_into(&mut received, &mut scratch) {
    Ok(outcome) => {
        for (position, magnitude) in outcome.positions().iter().zip(outcome.magnitudes()) {
            println!("error at {position}");
        }
    }
    Err(error) => { /* TooManyErrors / Inconsistent / ... */ }
}
```

Erasures are known-bad positions; hand them in and the budget widens to
`2·(errors) + (erasures) ≤ n - k`:

```rust
# use fgf::Gf8B;
# use syndrome_engine::{BerlekampMassey, Decoder, RsParams};
# let params = RsParams::<Gf8B>::new(255, 223, 1).unwrap();
# let decoder = Decoder::new(params, BerlekampMassey);
# let mut scratch = decoder.scratch().unwrap();
# let mut received = [0u8; 255];
let erased = [7usize, 42];
let outcome = decoder
    .decode_with_erasures_into(&mut received, &erased, &mut scratch)
    .unwrap();
```

The engine is generic over `fgf`'s binary fields — `Gf8B` (AES `0x11B`),
`Gf16`, `Gf32`, `Gf64` — with the primitive element `F::GENERATOR` and the
syndrome offset `b` frozen per decoder. (Now that `fgf`'s `Gf8B`/`Gf8D`
field split has landed and been re-pinned here, `Gf8D` — the classical
`0x11D` RS field — joins the matrix with a one-line test-matrix addition.)
Both key-equation backends are available behind the `KeyEquationSolver`
trait; they are Dornstetter-equivalent and cross-checked against each other
(and against a textbook Peterson–Gorenstein–Zierler oracle in the test
suite) on every fixture.

Consumers that already hold syndromes (`reliability-engine` recomputes them
far more often than words) skip the evaluation pass with
`decode_syndromes_into`. A `DecodeScratch` is sized once from the geometry
and every steady-state decode allocates nothing — proven by a counting
global allocator in `tests/zero_alloc.rs`, not asserted in prose.

## Cargo features

| Feature | Default | Effect |
| --- | --- | --- |
| `std` | yes | the standard library; `fgf`'s lazy backend cache |
| `simd` | yes | `fgf`'s vector kernels (implies `std`) |
| `parallel` | no | placeholder for block-axis parallelism |
| `internals` | no | unstable benchmarking surface, no compatibility promise |

`--no-default-features` builds the engine `no_std` with scalar field
arithmetic. The runtime dependency set is exactly `{fgf, poly-ring}` — no
`gfm` (the key equation replaces the Peterson matrix solve) and no
`butterfly-fft` (syndrome decoding is coefficient-domain; `poly-ring` is
built without its `fft` feature). CI asserts the tree shape.

## Building

```sh
cargo build
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
```

The MSRV is **1.89** (edition 2024). Benchmarks go through `criterion`;
backend-selection thresholds and the Berlekamp–Massey ↔ Euclidean crossover
are measured and recorded in `BENCHMARKS.md`, never in doc comments.

## License

MIT.
