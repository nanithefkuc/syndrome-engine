//! Hand-rolled error types, one enum per failure domain.
//!
//! No `thiserror`: a fourth error convention in the ecosystem would be worse
//! than the boilerplate. Every struct variant carries both the offending
//! value and the limit it violated, so a rejected call names the geometry
//! that failed. `inv(0) == 0` is inherited from `fgf` and is *not* an error
//! anywhere in this crate; zero tests are explicit `is_zero()` checks at the
//! call site.

use core::fmt;

/// Code-geometry and setup failures, rejected before any decoding happens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConfigError {
    /// Block length exceeds the field's multiplicative order `|F*|`, so the
    /// code would have repeated locator powers.
    BlockTooLong {
        /// Requested block length.
        n: usize,
        /// Largest supported block length, `|F| - 1`.
        order: u128,
    },
    /// Dimension is zero or exceeds the block length.
    Dimension {
        /// Requested dimension.
        k: usize,
        /// Block length it was checked against.
        n: usize,
    },
    /// Syndrome offset `b` is not an exponent of a distinct consecutive-root
    /// set, i.e. `b ≥ |F| - 1`.
    Offset {
        /// Requested first-consecutive-root offset.
        b: usize,
        /// Largest supported offset, `|F| - 1`.
        order: u128,
    },
    /// A scratch or decoder buffer could not be reserved.
    AllocationFailed {
        /// Name of the buffer that failed to reserve.
        context: &'static str,
        /// Number of elements requested.
        elements: usize,
        /// Size of one element in bytes.
        element_size: usize,
    },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::BlockTooLong { n, order } => write!(
                formatter,
                "block length {n} exceeds the field's multiplicative order {order}"
            ),
            Self::Dimension { k, n } => {
                write!(
                    formatter,
                    "dimension {k} is zero or exceeds block length {n}"
                )
            }
            Self::Offset { b, order } => write!(
                formatter,
                "syndrome offset {b} is not below the field's multiplicative order {order}"
            ),
            Self::AllocationFailed {
                context,
                elements,
                element_size,
            } => write!(
                formatter,
                "could not reserve {elements} {element_size}-byte elements for {context}"
            ),
        }
    }
}

/// Decoding failures, all detected (never a panic and never a silent wrong
/// answer where the failure is detectable).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecodeError {
    /// Error-and-erasure load exceeds the bounded-distance budget
    /// `2ν + ρ ≤ d - 1`.
    TooManyErrors {
        /// Located error count `ν`.
        errors: usize,
        /// Known erasure count `ρ`.
        erasures: usize,
        /// The budget, `d - 1 = n - k`.
        limit: usize,
    },
    /// A correction was found but the corrected word did not re-encode to
    /// all-zero syndromes, the key equation admitted no normalized solution,
    /// or a Forney denominator vanished. The input is left unchanged.
    Inconsistent,
    /// Scratch was sized for a different geometry than the decoder's.
    ScratchMismatch {
        /// The decoder's block length.
        expected: usize,
        /// The scratch's block length.
        got: usize,
    },
    /// Locator degree and located root count disagree: the locator does not
    /// split into distinct roots over the code's position set, which places
    /// the received word outside the decoding sphere.
    DegreeMismatch {
        /// Locator degree.
        expected: usize,
        /// Number of distinct roots found in the position set.
        got: usize,
    },
    /// Received-word length is not `n` field elements.
    WordGeometry {
        /// Length in field elements as supplied.
        got: usize,
        /// Expected length in field elements, `n`.
        expected: usize,
    },
    /// Supplied syndrome count differs from `n - k`.
    SyndromeGeometry {
        /// Syndrome count as supplied.
        got: usize,
        /// Expected syndrome count, `n - k`.
        expected: usize,
    },
    /// Erasure count exceeds `d - 1 = n - k`.
    TooManyErasures {
        /// Erasure count as supplied.
        count: usize,
        /// The erasure budget, `n - k`.
        limit: usize,
    },
    /// An erasure position is not a code position.
    ErasurePosition {
        /// The offending position.
        got: usize,
        /// The exclusive upper bound, `n`.
        limit: usize,
    },
    /// The same position was erased twice; the erasure locator would have a
    /// repeated root and Forney denominators would vanish.
    DuplicateErasure {
        /// The repeated position.
        position: usize,
    },
}

impl fmt::Display for DecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::TooManyErrors {
                errors,
                erasures,
                limit,
            } => write!(
                formatter,
                "error-and-erasure load 2·{errors} + {erasures} exceeds the budget {limit}"
            ),
            Self::Inconsistent => {
                write!(formatter, "correction failed its consistency check")
            }
            Self::ScratchMismatch { expected, got } => write!(
                formatter,
                "scratch was sized for block length {got}, decoder needs {expected}"
            ),
            Self::DegreeMismatch { expected, got } => write!(
                formatter,
                "locator degree {expected} disagrees with the {got} located roots"
            ),
            Self::WordGeometry { got, expected } => write!(
                formatter,
                "received word holds {got} field elements, expected {expected}"
            ),
            Self::SyndromeGeometry { got, expected } => {
                write!(formatter, "supplied {got} syndromes, expected {expected}")
            }
            Self::TooManyErasures { count, limit } => {
                write!(formatter, "{count} erasures exceed the budget {limit}")
            }
            Self::ErasurePosition { got, limit } => {
                write!(formatter, "erasure position {got} is not below {limit}")
            }
            Self::DuplicateErasure { position } => {
                write!(formatter, "position {position} was erased twice")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ConfigError {}
#[cfg(feature = "std")]
impl std::error::Error for DecodeError {}
