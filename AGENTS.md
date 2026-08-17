# syndrome-engine

> Given a received word (or its syndromes) from a Reed–Solomon / BCH-style
> code, find the error-and-erasure pattern by the key equation: compute
> syndromes, solve for the error-locator and error-evaluator polynomials,
> find the locator roots, and evaluate the error magnitudes — and never
> construct the code, own a wire format, or do soft-decision reliability
> processing. Field arithmetic comes from `fgf`; all polynomial arithmetic
> from `univariate`. This engine receives a corrupted word and returns the
> corrections.

> `syndrome-engine` is a decoder, not a codec and not a polynomial ring.
> Field arithmetic and byte-buffer vector primitives come from `fgf` — never
> re-implement them here. Polynomial evaluation, division, gcd,
> extended-Euclid, formal derivative, and root-finding come from
> `univariate` — never re-host them here. Pure-erasure decoding, wire
> formats, shard ownership, code-parameter selection, and soft-decision
> reliability processing belong to consumers. This crate receives a
> corrupted word and returns the corrections.

## Non-negotiables

1. **The engine orchestrates; `univariate` computes.** No polynomial ring
   loop lives here. Syndromes are `univariate` evals, the Euclidean key
   equation is `univariate`'s truncated EEA, roots are a `univariate`
   position scan, Forney is `univariate` eval + Hasse derivative. Adding a
   private poly routine is the defect this layering exists to prevent.
2. **Compose `fgf`, never re-host.** Call `fgf::field::Elem` / `fgf::ops::*`
   directly. No hand-rolled field loop; `#![forbid(unsafe_code)]`.
3. **Two solvers, one contract, cross-checked.** Berlekamp–Massey and
   Euclidean produce the same (Λ, Ω) up to normalization on every valid
   input; Dornstetter-equivalence is a permanent property test.
4. **Total over the decoding sphere.** Inside the radius → the unique
   correction. Outside → a typed `DecodeError`, never a panic, never a
   silent wrong answer where detectable. `2ν + ρ ≤ d − 1`.
5. **No `gfm`, no `butterfly-fft`.** The BM/EEA path avoids the matrix solve;
   syndrome decoding is coefficient-domain. CI fails the build if either
   enters the dependency tree.
6. **Steady-state zero allocation.** `DecodeScratch` sized once,
   geometry-checked, every hot path `*_into`. Proven by
   `tests/zero_alloc.rs`. (The Euclidean backend composes `univariate`'s
   allocating `truncated_eea` and is the cross-check, not the hot path.)
7. **`inv(0) == 0` is inherited.** Discrepancy / leading-coefficient /
   Forney-denominator zero-tests are explicit `is_zero()` calls.
8. **Determinism is a wire property.** Syndrome convention and root→position
   map are frozen; `tests/interop.rs` guards them.
9. **Numbers live in `BENCHMARKS.md`.** Doc comments state the decision and
   the mechanism and point there.
10. **Oracles stay independent.** An implementation is never its own test;
    BM and Euclidean check each other and both check against textbook PGZ.

## Working here

- Edition 2024, MSRV 1.89. No toolchain pin; select `+1.89.0` for the MSRV
  job.
- Features: `default = ["std", "simd"]`; `simd` implies `std`; `parallel` is
  an off-by-default no-op placeholder; `internals` exposes this crate's
  unstable surface (never `fgf`'s or `univariate`'s — we do not enable
  them).
- `src/lib.rs` and every `mod.rs` hold declarations only — no function
  bodies, no `impl` blocks. Public items are re-exported at the crate root.
- Errors are hand-rolled in `src/error.rs`: small enums per failure domain,
  struct variants carrying both the offending value and the limit, manual
  `Display`, `std::error::Error` under `std`. Every fallible public function
  documents `# Errors`.
- Test placement follows visibility: in-module `#[cfg(test)]` for private
  state, `tests/` for the public surface. Fixed-seed LCG only (`fgf`'s
  `noise(len, seed)` shape); no `rand`. Exact values, not predicates.
- The full check set:

  ```sh
  cargo fmt --all -- --check
  cargo clippy --all-targets --all-features -- -D warnings
  cargo clippy --all-targets --no-default-features -- -D warnings
  cargo test
  cargo test --all-features
  cargo test --no-default-features
  RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
  cargo build --target aarch64-unknown-linux-gnu --no-default-features
  cargo build --target wasm32-unknown-unknown --no-default-features
  cargo +1.89.0 build --all-features
  ```

- Benchmarks go through `criterion`; baselines are deliberately not
  committed. Measurement hygiene: interleave base/new, take the maximum of
  at least three runs, keep an unchanged 1.00x control.
- Commit subjects are at most ~10 words, shaped `syndrome-engine: short verb
  phrase`. What changed and why lives in the pull request and
  `CHANGELOG.md`.
