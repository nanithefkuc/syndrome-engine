//! Bounded-distance RS / BCH decoding via the key equation.
//!
//! > `syndrome-engine` is a decoder, not a codec and not a polynomial ring.
//! > Field arithmetic and byte-buffer vector primitives come from `fgf` —
//! > never re-implement them here. Polynomial evaluation, division, gcd,
//! > extended-Euclid, formal derivative, and root-finding come from
//! > `univariate` — never re-host them here. Pure-erasure decoding, wire
//! > formats, shard ownership, code-parameter selection, and soft-decision
//! > reliability processing belong to consumers. This crate receives a
//! > corrupted word and returns the corrections.
//!
//! # The pipeline
//!
//! Given a received word `R` of a Reed–Solomon-style code with `n - k`
//! consecutive roots `α^{b}, …, α^{b+n-k-1}` (primitive element `α =
//! F::GENERATOR`, first consecutive root offset `b`), the engine runs the
//! classical bounded-distance pipeline:
//!
//! 1. **Syndromes** `S_j = R(α^{b+j})` — one multipoint evaluation of the
//!    received word through `univariate`.
//! 2. **Key equation** `Λ(x)·S(x) ≡ Ω(x) (mod x^{n-k})` solved for the
//!    error-locator `Λ` and error-evaluator `Ω` — by Berlekamp–Massey
//!    (engine-native LFSR synthesis over the syndrome sequence) or by the
//!    Euclidean/Sugiyama backend composing `univariate`'s truncated EEA. The
//!    two are Dornstetter-equivalent and cross-checked on every fixture.
//! 3. **Chien search** — the locator is evaluated at the inverse error
//!    locators `α^{-p}` of every code position `p`; the vanishing positions
//!    are the errors.
//! 4. **Forney** — each magnitude is one evaluation-and-division:
//!    `e_p = X_p^{1-b}·Ω(X_p^{-1}) / Λ'(X_p^{-1})` with a Montgomery batch
//!    inversion of the denominators.
//!
//!
//! `fgf`'s and every polynomial pass is `univariate`'s. The single exception
//! is Berlekamp–Massey, which synthesizes an LFSR over a scalar *syndrome
//! sequence* — a decoder concept, not a polynomial.
//!
//! # Determinism
//!
//! The syndrome index convention (offset `b`, primitive element, low-degree
//! first coefficient order) and the root→position map (position `p`
//! corresponds to the locator root `α^{-p}`, reported in ascending position
//! order) are a frozen wire property shared with any matching encoder;
//! `tests/interop.rs` pins them against fixed known-answer vectors.
//!
//! # Features
//!
//! | Feature | Effect |
//! | --- | --- |
//! | default (`std`, `simd`) | full engine over `fgf`'s dispatched kernels |
//! | `--no-default-features` | `no_std` core engine, scalar field arithmetic |
//! | `parallel` | off-by-default placeholder for block-axis parallelism |
//! | `internals` | unstable benchmarking surface, no compatibility promise |

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]

extern crate alloc;

mod cost;
mod decoder;
mod error;
mod forney;
mod keyeq;
mod locate;
mod params;
mod syndrome;

pub use cost::{Adaptive, BM_EUCLIDEAN_CROSSOVER, SolverBackend, SolverCostKey, select_solver};
pub use decoder::{DecodeOutcome, DecodeScratch, Decoder};
pub use error::{ConfigError, DecodeError};
pub use keyeq::{BerlekampMassey, Euclidean, KeyEqScratch, KeyEquation, KeyEquationSolver};
pub use params::RsParams;
pub use syndrome::syndromes;

/// The crate's unstable surface for benchmarks and downstream
/// experimentation, gated on the `internals` feature. Nothing here is a
/// compatibility promise.
#[cfg(feature = "internals")]
pub mod stages {
    pub use crate::forney::batch_invert_into;
    pub use crate::locate::{locate_into, position_points_into};
    pub use crate::syndrome::compute_into;
}
