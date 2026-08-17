//! The key-equation result pair and the solver scratch.

use univariate::Polynomial;

use super::berlekamp::BmScratch;
use crate::error::ConfigError;

/// The solved key equation: error locator `Λ` and error evaluator `Ω`.
///
/// Both are `univariate` polynomials in the crate-wide low-degree-first
/// coefficient order. `Λ` is normalized to `Λ(0) = 1`; its roots are the
/// inverse error locators `X_i^{-1} = α^{-p_i}`. `Ω` satisfies
/// `Λ(x)·S(x) ≡ Ω(x) (mod x^{N})` with `deg Ω < deg Λ`.
#[derive(Debug)]
pub struct KeyEquation<F: fgf::kernel::FieldKernels> {
    pub(crate) locator: Polynomial<F>,
    pub(crate) evaluator: Polynomial<F>,
}

impl<F: fgf::kernel::FieldKernels> KeyEquation<F> {
    /// An empty pair with both buffers reserved for `capacity` coefficients.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::AllocationFailed`] when a buffer cannot be
    /// reserved.
    pub fn with_capacity(capacity: usize) -> Result<Self, ConfigError> {
        Ok(Self {
            locator: reserved_polynomial::<F>(capacity, "key-equation locator")?,
            evaluator: reserved_polynomial::<F>(capacity, "key-equation evaluator")?,
        })
    }

    /// The error locator `Λ`, normalized to `Λ(0) = 1`.
    #[must_use]
    pub fn locator(&self) -> &Polynomial<F> {
        &self.locator
    }

    /// The error evaluator `Ω`.
    #[must_use]
    pub fn evaluator(&self) -> &Polynomial<F> {
        &self.evaluator
    }

    /// Degree of the locator; `0` for the no-error locator `Λ = 1`. The
    /// degree is the located error count.
    #[must_use]
    pub fn locator_degree(&self) -> usize {
        self.locator.coefficient_count().saturating_sub(1)
    }
}

/// Caller-owned solver scratch, sized once for the decoder's geometry.
///
/// Holds the EEA operand buffers (`x^{N}` and the syndrome series) and the
/// Berlekamp–Massey register buffers, so a warmed solve allocates nothing on
/// the Berlekamp–Massey path. The Euclidean backend composes `univariate`'s
/// allocating `truncated_eea` and is the cross-check rather than the hot
/// path.
#[derive(Debug)]
pub struct KeyEqScratch<F: fgf::kernel::FieldKernels> {
    /// The `x^{N}` EEA dividend operand.
    pub(crate) x_pow: Polynomial<F>,
    /// The syndrome series as the EEA divisor operand.
    pub(crate) series: Polynomial<F>,
    /// Berlekamp–Massey registers.
    pub(crate) bm: BmScratch<F>,
}

impl<F: fgf::kernel::FieldKernels> KeyEqScratch<F> {
    /// Scratch with buffers reserved for sequences up to `capacity`
    /// syndromes.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::AllocationFailed`] when a buffer cannot be
    /// reserved.
    pub fn with_capacity(capacity: usize) -> Result<Self, ConfigError> {
        Ok(Self {
            x_pow: reserved_polynomial::<F>(capacity + 1, "key-equation x^N operand")?,
            series: reserved_polynomial::<F>(capacity, "key-equation syndrome operand")?,
            bm: BmScratch::with_capacity(capacity)?,
        })
    }
}

fn reserved_polynomial<F: fgf::kernel::FieldKernels>(
    capacity: usize,
    context: &'static str,
) -> Result<Polynomial<F>, ConfigError> {
    let mut polynomial = Polynomial::zero();
    polynomial
        .resize_coefficients(capacity)
        .map_err(|_| ConfigError::AllocationFailed {
            context,
            elements: capacity,
            element_size: F::BYTES,
        })?;
    polynomial.set_zero();
    Ok(polynomial)
}
