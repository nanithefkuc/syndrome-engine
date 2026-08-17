# Changelog

All notable changes to this project are documented in this file.

## 0.0.0 (2026-08-17)

Initial scaffold of the bounded-distance error-decoding engine.

- `RsParams<F>`: frozen code geometry and syndrome index convention
  (`n`, `k`, offset `b` over `fgf`'s binary fields), with typed geometry
  validation.
- Hand-rolled `ConfigError` / `DecodeError` enums, one per failure domain,
  every variant carrying the offending value and the limit.
- Runtime dependency set frozen to `{fgf, univariate}` (rev-pinned,
  `univariate` without default features), asserted by CI.
