# Benchmarks

Measured records only: a crossover or dispatch decision changed without a
re-measurement of this file is a regression with a green test suite. Doc
comments in the source state the decision and the mechanism and point here;
numbers live in this file and nowhere else.

Hardware: Intel Core Ultra 7 258V (x86-64, AVX2/GFNI). rustc 1.93.0
(254b59607 2026-01-19). `fgf` rev `d0e331ce`, `univariate` rev `487997ed`,
backend detected on host. Command line for every table below:

```sh
cargo bench --features internals --bench decode -- --warm-up-time 0.4 --measurement-time 1.2
```

Baselines are deliberately not committed. Measurement hygiene for change
reviews: interleave `base, new, base, new, …` (`--save-baseline before` /
`--baseline before`), take the maximum of at least three runs per key, and
keep an unchanged 1.00× control.

## Berlekamp–Massey ↔ Euclidean crossover (`cost::BM_EUCLIDEAN_CROSSOVER`)

Key-equation solve of a full-radius error pattern over GF(2^16), by
syndrome-sequence length `N = n - k`. Berlekamp–Massey is the scalar LFSR
synthesis over `fgf::field::Elem`; the Euclidean backend runs through
`univariate`'s truncated EEA, whose schoolbook products ride `fgf`'s packed
byte kernels — scalar `O(N²)` against packed `O(N²)` with a larger constant
and internal allocations.

| geometry | `N` | Berlekamp–Massey | Euclidean | faster |
| --- | --- | --- | --- | --- |
| 31×21 | 10 | 496 ns | 923 ns | BM 1.9× |
| 48×32 | 16 | 902 ns | 1.52 µs | BM 1.7× |
| 64×40 | 24 | 1.73 µs | 2.70 µs | BM 1.6× |
| 80×48 | 32 | 2.41 µs | 3.51 µs | BM 1.5× |
| 96×56 | 40 | 3.37 µs | 4.12 µs | BM 1.2× |
| 112×64 | 48 | 5.08 µs | 5.33 µs | parity (BM 1.05×) |
| 1023×900 | 123 | 27.7 µs | 12.9 µs | EEA 2.2× |
| 4095×3000 | 1095 | 1.98 ms | 195 µs | EEA 10× |

Parity sits at `N ≈ 48–55`; `cost::BM_EUCLIDEAN_CROSSOVER = 48` errs toward
the allocation-free backend, so every default decode through the parity
band is steady-state allocation-free (`tests/zero_alloc.rs`). Past the
threshold the packed EEA pulls away quadratically. Re-measure this table
whenever `univariate`'s EEA or `fgf`'s kernels change materially.

## Chien position scan vs whole-field scan

Locating a degree-60 locator over the 1023 code positions (subproduct-tree
multipoint evaluation of the frozen position-point set) against
`univariate::chien_roots_into` scanning all 65 536 field elements:

| path | time (1023×900, GF(2^16)) |
| --- | --- |
| position scan (`locate`) | 261 µs |
| whole-field Chien scan | 8.41 ms |

The position scan is the decoder's classical Chien domain (the `n` code
positions, not `|F|` elements) and is ~32× faster at this geometry; at
`Gf32`/`Gf64` the whole-field scan is not a contender at all. This is the
measured basis for `locate` composing multipoint evaluation rather than the
field scan.

## Stage and end-to-end reference points

GF(2^16), full-radius error patterns, warmed scratch:

| case | time |
| --- | --- |
| syndrome pass, 255×223 | 15.5 µs |
| syndrome pass, 1023×900 | 77.2 µs |
| Forney batch inversion, 60 denominators | 914 ns |
| end-to-end decode, 255×223 (t = 16, BM) | 85.6 µs |
| end-to-end decode, 1023×900 (t = 61, EEA past the crossover) | 496 µs |

The end-to-end numbers include both miscorrection guards (root count,
re-encode to zero syndromes) — the certificate checks are part of the
decoded budget, not an optional extra.
