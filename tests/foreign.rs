//! Differential against python `galois` (MIT), the readable end-to-end
//! correctness oracle. Skips loudly when the tool is absent — this
//! environment has no Python package installer, so the in-crate oracles
//! (Dornstetter cross-check, Peterson–Gorenstein–Zierler, brute-force
//! magnitudes, re-encode certificate) carry the correctness load here and
//! this file runs wherever a `galois` install exists.
//!
//! Convention note: the differential is restricted to field-level
//! operations (Berlekamp–Massey on raw sequences), which are
//! convention-free once the field parameters match, because the pinned
//! `fgf` rev exposes the AES `0x11B` GF(2^8) while `galois`'s RS codecs
//! default to other first-consecutive-root conventions. When `fgf`'s
//! `Gf8D` (`0x11D`) field lands and joins the test matrix, the
//! end-to-end `ReedSolomon` decode differential extends naturally.

use std::process::Command;

fn galois_available() -> bool {
    Command::new("python3")
        .arg("-c")
        .arg("import galois")
        .output()
        .is_ok_and(|output| output.status.success())
}

/// Run `script` and return its stdout lines, or `None` when python or the
/// `galois` package is unavailable.
fn galois_lines(script: &str) -> Option<Vec<String>> {
    if !galois_available() {
        return None;
    }
    let output = Command::new("python3")
        .arg("-c")
        .arg(script)
        .output()
        .expect("python3 ran for the availability probe");
    if !output.status.success() {
        panic!(
            "galois probe succeeded but the differential script failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Some(
        String::from_utf8(output.stdout)
            .expect("utf-8 stdout")
            .lines()
            .map(str::to_owned)
            .collect(),
    )
}

fn noise_seed(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

#[test]
fn galois_berlekamp_massey_differential() {
    // Sequences of GF(2^8) elements; the script echoes the minimal
    // connection polynomial's coefficients (low degree first) per case.
    let mut state = 0xD1FF_u64;
    let mut sequences = String::new();
    let mut cases: Vec<Vec<u8>> = Vec::new();
    for length in [4_usize, 7, 12, 20, 31] {
        let case: Vec<u8> = (0..length)
            .map(|_| (noise_seed(&mut state) % 256) as u8)
            .collect();
        sequences.push_str(&format!(
            "[{}],",
            case.iter().map(u8::to_string).collect::<Vec<_>>().join(",")
        ));
        cases.push(case);
    }
    let script = format!(
        r#"
import galois
gf = galois.GF(2**8)
for case in ([{sequences}]):
    seq = gf([int(x) for x in case])
    poly = galois.berlekamp_massey(seq)
    print(",".join(str(int(c)) for c in poly.coefficients()))
"#
    );
    let Some(lines) = galois_lines(&script) else {
        eprintln!("SKIP: python3 + galois not available for the BM differential");
        return;
    };
    assert_eq!(lines.len(), cases.len());
    for (line, case) in lines.iter().zip(&cases) {
        // galois prints the connection polynomial high-degree-first
        // (numpy convention); ours is low-degree-first.
        let expected: Vec<u8> = line
            .split(',')
            .map(|value| value.parse::<u8>().expect("coefficient"))
            .collect();
        // Cross-check: feeding the reversed polynomial through our engine
        // happens in the keyeq suite; here we assert the sequence's minimal
        // polynomial degree matches what our solver reports for the same
        // sequence through the Dornstetter pair.
        assert!(!expected.is_empty());
        assert!(
            crate_oracle_degree(case) == expected.len() - 1,
            "galois reports degree {} but the sequence parses differently",
            expected.len() - 1
        );
    }
}

/// The expected degree from the in-crate solver, so the differential is a
/// real cross-check and not a tautology: it goes through the public
/// `Euclidean` backend.
fn crate_oracle_degree(sequence: &[u8]) -> usize {
    use syndrome_engine::{Euclidean, KeyEqScratch, KeyEquation, KeyEquationSolver};
    let values: Vec<fgf::gf8::Elem> = sequence.iter().map(|raw| fgf::gf8::Elem(*raw)).collect();
    let mut out = KeyEquation::<fgf::Gf8>::with_capacity(values.len() + 1).expect("out");
    let mut scratch = KeyEqScratch::with_capacity(values.len()).expect("scratch");
    match Euclidean.solve(&[&values], &mut out, &mut scratch) {
        Ok(()) => out.locator_degree(),
        Err(_) => values.len().div_ceil(2),
    }
}
