//! Differential against python `galois` (MIT), the readable independent
//! field-and-solver oracle. The interpreter is discovered as `python3` with
//! `galois` importable, or — on hosts without a system package manager —
//! `uv run --with galois python`. Without either, this file skips loudly;
//! the in-crate oracles (Dornstetter cross-check,
//! Peterson–Gorenstein–Zierler, brute-force magnitudes, re-encode
//! certificate) carry the correctness load regardless.
//!
//! Scope: the differentials are field-level — Berlekamp–Massey on raw
//! sequences, locator root sets, syndrome vectors, and generator
//! polynomials — because the pinned `fgf` rev exposes the AES `0x11B`
//! GF(2^8) with primitive element 3, while `galois`'s `ReedSolomon`
//! constructor will not take that primitive for a custom field (its
//! default alpha is not 3 and forcing it trips an internal assertion).
//! When `fgf`'s `Gf8D` (`0x11D`) lands, `galois`'s default field matches it
//! exactly and the end-to-end `ReedSolomon.decode` differential extends
//! naturally.

mod common;

use syndrome_engine::KeyEquationSolver;

use std::io::Write as _;
use std::process::{Command, Stdio};

/// One line protocol: requests on stdin (`BM`, `ROOTS`, `SYN`, `GEN`),
/// one response line per request on stdout, values space-separated.
const SCRIPT: &str = r#"
import sys
import galois

GF = galois.GF(2**8, irreducible_poly=0x11B)
ALPHA = GF(3)  # matches fgf's Gf8::GENERATOR (0x11B, generator 3)

def powers():
    table = {}
    element = GF(1)
    for exponent in range(255):
        table[int(element)] = exponent
        element = element * ALPHA
    return table

LOG = powers()

for line in sys.stdin:
    parts = line.split()
    kind = parts[0]
    if kind == "BM":
        sequence = GF([int(value) for value in parts[2:]])
        poly = galois.berlekamp_massey(sequence)
        # galois returns the monic reciprocal of the engine's locator: its
        # degree-descending coefficient array, read as-is, is exactly the
        # engine's low-degree-first locator with Lambda(0) = 1 (monic
        # leading == unit constant after the reciprocal).
        coefficients = [int(value) for value in poly.coeffs]
        print("@ " + " ".join(str(value) for value in coefficients))
    elif kind == "ROOTS":
        coefficients = [int(value) for value in parts[2:]]
        poly = galois.Poly(coefficients[::-1], field=GF)
        roots = sorted(int(root) for root in poly.roots())
        # Inverse-locator convention: a root y is alpha^{-p}, so the
        # position is the negative discrete log.
        positions = sorted((255 - LOG[root]) % 255 for root in roots)
        print("@ " + " ".join(str(value) for value in positions))
    elif kind == "SYN":
        n, k, b = int(parts[1]), int(parts[2]), int(parts[3])
        word = [int(value) for value in parts[4:4 + n]]
        syndromes = []
        for j in range(n - k):
            point = ALPHA ** (b + j)
            accumulator = GF(0)
            for coefficient in reversed(word):  # Horner, high degree down
                accumulator = accumulator * point + GF(coefficient)
            syndromes.append(int(accumulator))
        print("@ " + " ".join(str(value) for value in syndromes))
    elif kind == "GEN":
        n, k, b = int(parts[1]), int(parts[2]), int(parts[3])
        generator = galois.Poly([1], field=GF)
        root = ALPHA ** b
        for _ in range(n - k):
            generator = generator * galois.Poly([1, int(root)], field=GF)
            root = root * ALPHA
        print("@ " + " ".join(str(int(value)) for value in generator.coeffs[::-1]))
    sys.stdout.flush()
"#;

fn interpreter() -> Option<Vec<String>> {
    let plain = Command::new("python3")
        .arg("-c")
        .arg("import galois")
        .output();
    if plain.is_ok_and(|output| output.status.success()) {
        return Some(vec!["python3".into()]);
    }
    let uv = Command::new("uv")
        .args([
            "run",
            "--with",
            "galois==0.4.11",
            "python",
            "-c",
            "import galois",
        ])
        .output();
    if uv.is_ok_and(|output| output.status.success()) {
        // Pinned so runner environments resolve the version the
        // coefficient conventions were verified against.
        return Some(vec![
            "uv".into(),
            "run".into(),
            "--with".into(),
            "galois==0.4.11".into(),
            "python".into(),
        ]);
    }
    None
}

fn run_protocol(requests: &str) -> Option<Vec<String>> {
    let mut argv = interpreter()?;
    let script_path = std::env::temp_dir().join("syndrome_engine_galois.py");
    std::fs::write(&script_path, SCRIPT).expect("write protocol script");
    argv.push(
        script_path
            .into_os_string()
            .into_string()
            .expect("utf-8 path"),
    );

    let mut child = Command::new(&argv[0])
        .args(&argv[1..])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("interpreter spawned for the probe");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(requests.as_bytes())
        .expect("write requests");
    let output = child.wait_with_output().expect("collect output");
    assert!(
        output.status.success(),
        "galois differential script failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Some(
        String::from_utf8(output.stdout)
            .expect("utf-8 stdout")
            .lines()
            .filter_map(|line| line.strip_prefix("@ "))
            .map(str::to_owned)
            .collect(),
    )
}

fn advance(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

fn noise_bytes(state: &mut u64, count: usize) -> Vec<u8> {
    (0..count)
        .map(|_| u8::try_from(advance(state) % 256).expect("byte"))
        .collect()
}

fn parse_values(line: &str) -> Vec<u8> {
    line.split_whitespace()
        .map(|value| value.parse::<u8>().expect("galois coefficient"))
        .collect()
}

#[test]
fn galois_field_and_solver_differential() {
    let mut state = 0xD1FF_5EED_u64;

    // Requests, in order: BM sequences, a decode locator for ROOTS, a
    // syndrome word, and generator polynomials.
    let mut requests = String::new();
    let mut sequences: Vec<Vec<u8>> = Vec::new();
    for _ in 0..24 {
        let length = 4 + usize::try_from(advance(&mut state) % 28).expect("length");
        let sequence = noise_bytes(&mut state, length);
        requests.push_str(&format!(
            "BM {length} {}\n",
            sequence
                .iter()
                .map(u8::to_string)
                .collect::<Vec<_>>()
                .join(" ")
        ));
        sequences.push(sequence);
    }

    // A real decode scenario at the correction boundary: codeword, t
    // errors, solve — send the locator. (31, 19) has t = 6.
    let params = syndrome_engine::RsParams::<fgf::Gf8>::new(31, 19, 1).expect("params");
    let sent = common::random_codeword(&params, 0xD200);
    let mut received = sent.clone();
    let mut positions = common::distinct_positions(31, 6, 0xD201);
    positions.sort();
    let mut error_state = 0xD202;
    common::inject::<fgf::Gf8>(&mut received, &positions, &mut error_state);
    let values = syndrome_engine::syndromes(&params, &received).expect("syndromes");
    let mut keyeq = syndrome_engine::KeyEquation::<fgf::Gf8>::with_capacity(16).expect("keyeq");
    let mut scratch = syndrome_engine::KeyEqScratch::with_capacity(values.len()).expect("scratch");
    syndrome_engine::BerlekampMassey
        .solve(
            &[&values
                .iter()
                .map(|v| fgf::gf8::Elem(v.to_raw()))
                .collect::<Vec<_>>()],
            &mut keyeq,
            &mut scratch,
        )
        .expect("bm solve");
    let locator: Vec<u8> = keyeq
        .locator()
        .coefficients()
        .map(|c: fgf::gf8::Elem| c.to_raw())
        .collect();
    requests.push_str(&format!(
        "ROOTS {} {}\n",
        locator.len(),
        locator
            .iter()
            .map(u8::to_string)
            .collect::<Vec<_>>()
            .join(" ")
    ));

    // A syndrome request for a random word under two conventions.
    let word = noise_bytes(&mut state, 31);
    requests.push_str(&format!(
        "SYN 31 19 1 {}\n",
        word.iter().map(u8::to_string).collect::<Vec<_>>().join(" ")
    ));
    let word_b3 = noise_bytes(&mut state, 17);
    requests.push_str(&format!(
        "SYN 17 8 3 {}\n",
        word_b3
            .iter()
            .map(u8::to_string)
            .collect::<Vec<_>>()
            .join(" ")
    ));

    // Generator polynomials for both geometries.
    requests.push_str("GEN 31 19 1\n");
    requests.push_str("GEN 17 8 3\n");

    let Some(responses) = run_protocol(&requests) else {
        eprintln!("SKIP: no python3+galois or uv available for the differential");
        return;
    };
    assert_eq!(responses.len(), sequences.len() + 5);

    // 1. Berlekamp–Massey agrees with the engine on every sequence.
    for (response, sequence) in responses.iter().zip(&sequences) {
        let expected = parse_values(response);
        let elements: Vec<fgf::gf8::Elem> =
            sequence.iter().map(|raw| fgf::gf8::Elem(*raw)).collect();
        let mut out = syndrome_engine::KeyEquation::<fgf::Gf8>::with_capacity(48).expect("out");
        let mut scratch =
            syndrome_engine::KeyEqScratch::with_capacity(elements.len()).expect("scratch");
        // galois may solve sequences the engine's budget rejects; on Ok the
        // locators must match exactly (both normalized at the constant).
        if syndrome_engine::BerlekampMassey
            .solve(&[&elements], &mut out, &mut scratch)
            .is_ok()
        {
            let produced: Vec<u8> = out
                .locator()
                .coefficients()
                .map(|c: fgf::gf8::Elem| c.to_raw())
                .collect();
            assert_eq!(
                produced, expected,
                "galois BM disagrees on sequence {sequence:?}"
            );
        } else {
            // The engine's contract check rejected the pair (its budget or
            // the evaluator degree bound); galois has no such notion, so
            // there is nothing to compare on this sequence.
            eprintln!(
                "NOTE: engine rejected sequence {sequence:?} (galois degree {})",
                expected.len() - 1
            );
        }
    }

    // 2. Locator roots map to exactly the injected positions (the frozen
    //    root→position convention, externally).
    let root_positions: Vec<usize> = responses[sequences.len()]
        .split_whitespace()
        .map(|value| value.parse::<usize>().expect("position"))
        .collect();
    assert_eq!(root_positions, positions);

    // 3. Syndrome vectors agree byte for byte, including the b = 3 offset.
    let mine = syndrome_engine::syndromes(&params, &word).expect("syndromes");
    let theirs: Vec<u8> = responses[sequences.len() + 1]
        .split_whitespace()
        .map(|value| value.parse::<u8>().expect("syndrome"))
        .collect();
    assert_eq!(mine.iter().map(|v| v.to_raw()).collect::<Vec<_>>(), theirs);
    let params_b3 = syndrome_engine::RsParams::<fgf::Gf8>::new(17, 8, 3).expect("params");
    let mine = syndrome_engine::syndromes(&params_b3, &word_b3).expect("syndromes");
    let theirs: Vec<u8> = responses[sequences.len() + 2]
        .split_whitespace()
        .map(|value| value.parse::<u8>().expect("syndrome"))
        .collect();
    assert_eq!(mine.iter().map(|v| v.to_raw()).collect::<Vec<_>>(), theirs);

    // 4. Generator polynomials agree coefficient for coefficient.
    for (response, params) in responses[sequences.len() + 3..sequences.len() + 5]
        .iter()
        .zip([&params, &params_b3])
    {
        let theirs = parse_values(response);
        let mine: Vec<u8> = common::generator(params)
            .coefficients()
            .map(|c: fgf::gf8::Elem| c.to_raw())
            .collect();
        assert_eq!(mine, theirs);
    }
}
