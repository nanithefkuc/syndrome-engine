//! The forced-backend panel assertion for the `backends` CI job.
//!
//! `SIMD_BACKEND` is the stack-wide downgrade-only override owned by
//! `simdispatch` and consumed through `fgf`; this crate never re-probes.
//! The panel forces a tier and the suite below runs under it; this test
//! asserts the selection actually degraded to at most the requested tier
//! (exactly, for tiers every field implements).

use fgf::kernel::{Backend, backend_for};
use fgf::{Gf8B, Gf16};

#[test]
fn forced_backend_is_selected() {
    let Ok(requested) = std::env::var("SIMD_BACKEND") else {
        eprintln!("SKIP: SIMD_BACKEND not set; running under the detected backend");
        return;
    };
    let expected = match requested.as_str() {
        "v3_gfni_crypto" => Backend::V3GfniCrypto,
        "v3" => Backend::V3,
        "v2" => Backend::V2,
        "scalar" => Backend::Scalar,
        other => panic!("unknown SIMD_BACKEND tier {other}"),
    };
    for selected in [backend_for::<Gf8B>(), backend_for::<Gf16>()] {
        // Downgrade-only: the selected backend is never stronger than the
        // request; a field without kernels at the tier settles lower.
        assert!(
            selected >= expected,
            "selected {selected:?} is stronger than the forced {expected:?}"
        );
        eprintln!("SIMD_BACKEND={requested}: selected {selected:?}");
    }
}
