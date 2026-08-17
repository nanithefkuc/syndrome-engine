# Changelog

All notable changes to this project are documented in this file.

## 0.0.0 (2026-08-17)

Initial implementation of the bounded-distance error-decoding engine.

- `RsParams<F>`: frozen code geometry and syndrome index convention
  (`n`, `k`, offset `b` over `fgf`'s binary fields), with typed geometry
  validation.
- Syndrome computation `S_j = R(α^{b+j})` as one `univariate` multipoint
  evaluation of the packed received word; public `syndromes` convenience
  and the scratch-backed `Decoder::syndromes_into`.
- Key equation `Λ·S ≡ Ω (mod x^{n-k})` behind one `KeyEquationSolver`
  trait with two Dornstetter-equivalent backends: `BerlekampMassey`
  (engine-native LFSR synthesis, allocation-free steady state) and
  `Euclidean` (Sugiyama, composing `univariate`'s truncated EEA);
  `Adaptive` selects by the measured crossover in `cost`
  (`BM_EUCLIDEAN_CROSSOVER`, see `BENCHMARKS.md`).
- Chien position search: the locator evaluated at the frozen inverse
  locators `α^{-p}` of all `n` code positions through `univariate`
  multipoint evaluation, ascending position order.
- Forney magnitudes `e_p = X_p^{1-b}·Ω(X_p^{-1})/Λ'(X_p^{-1})` with
  pointwise Hasse derivatives and Montgomery batch inversion of the
  denominators.
- Errors-and-erasures: erasure locator `Γ`, Forney (modified) syndromes,
  suffix key equation, product locator `Λ·Γ` driving Chien and Forney for
  flagged and unflagged corruption in one pass, `2ν + ρ ≤ n - k`.
- `Decoder<F, S>` with `DecodeScratch`: sized once, geometry-checked,
  every steady-state decode `*_into` and allocation-free on the
  Berlekamp–Massey path (proven by a counting global allocator); two
  miscorrection guards (root count vs degree, re-encode to zero
  syndromes) and typed failures with the input left untouched.
- Test oracles: textbook Peterson–Gorenstein–Zierler locator,
  brute-force Vandermonde magnitudes, per-point Horner syndromes, the
  BM↔Euclidean cross-check on every fixture, frozen interop fixtures
  pinning the wire convention, and a `galois` differential (field-level:
  Berlekamp–Massey, locator roots, syndromes, generator polynomials over
  the matching GF(2^8)/0x11B) running under `python3` or `uv`, skipping
  loudly when neither is present.
- Hand-rolled `ConfigError` / `DecodeError` enums, one per failure domain,
  every variant carrying the offending value and the limit.
- Runtime dependency set frozen to `{fgf, univariate}` (rev-pinned,
  `univariate` without default features), asserted by CI.
