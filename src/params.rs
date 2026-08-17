//! Code geometry and the frozen syndrome index convention.

use core::marker::PhantomData;

use fgf::kernel::FieldKernels;

use crate::error::ConfigError;

/// Code geometry and syndrome convention. Immutable after construction; a
/// decoder is built once from it and sizes its scratch from it.
///
/// The two halves of the index convention are the primitive element
/// `F::GENERATOR` and the offset `b`: the syndromes of a received word
/// `R(x) = Σ_p r_p x^p` (coefficient `r_p` at degree `p`, low degree first)
/// are `S_j = R(α^{b+j})` for `j = 0..n-k`, and a position `p` is in error
/// exactly when the error locator vanishes at `α^{-p}`. A decoder built with
/// one `(b, F)` pair cannot read a word encoded against another; this
/// struct is the only place the convention is stated.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct RsParams<F: FieldKernels> {
    n: usize,
    k: usize,
    b: usize,
    field: PhantomData<F>,
}

impl<F: FieldKernels> core::fmt::Debug for RsParams<F> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("RsParams")
            .field("n", &self.n)
            .field("k", &self.k)
            .field("b", &self.b)
            .field("field", &F::NAME)
            .finish()
    }
}

impl<F: FieldKernels> RsParams<F> {
    /// Validate the geometry and construct the parameters.
    ///
    /// The code is the `n`-length code over `F` whose parity checks are the
    /// `n - k` consecutive powers `α^{b}, …, α^{b+n-k-1}`; the correction
    /// radius is `t = (n - k) / 2` and the error-plus-erasure budget is
    /// `2ν + ρ ≤ n - k`.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::BlockTooLong`] if `n > |F*|`,
    /// [`ConfigError::Dimension`] if `k` is zero or exceeds `n`, and
    /// [`ConfigError::Offset`] if `b ≥ |F*|`. A rejected construction
    /// returns nothing; no state exists to corrupt.
    pub fn new(n: usize, k: usize, b: usize) -> Result<Self, ConfigError> {
        let order = F::ORDER - 1;
        if u128::try_from(n).unwrap_or(u128::MAX) > order {
            return Err(ConfigError::BlockTooLong { n, order });
        }
        if k == 0 || k > n {
            return Err(ConfigError::Dimension { k, n });
        }
        if u128::try_from(b).unwrap_or(u128::MAX) >= order {
            return Err(ConfigError::Offset { b, order });
        }
        Ok(Self {
            n,
            k,
            b,
            field: PhantomData,
        })
    }

    /// Block length.
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Dimension.
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }

    /// First consecutive root offset of the syndrome convention.
    #[must_use]
    pub fn b(&self) -> usize {
        self.b
    }

    /// Redundancy `n - k`: the number of parity checks and syndromes, and
    /// the error-plus-erasure budget `d - 1`.
    #[must_use]
    pub fn redundancy(&self) -> usize {
        self.n - self.k
    }

    /// Bounded-distance correction radius `t = (n - k) / 2`.
    #[must_use]
    pub fn correction_radius(&self) -> usize {
        self.redundancy() / 2
    }

    /// Syndrome count, equal to [`Self::redundancy`]: every consecutive root
    /// of the code is evaluated, so the errors-and-erasures boundary
    /// `2ν + ρ = n - k` is reachable even when the redundancy is odd.
    #[must_use]
    pub fn syndrome_count(&self) -> usize {
        self.redundancy()
    }

    /// The `(n, k, b)` triple identifying the geometry a scratch buffer was
    /// sized for.
    #[must_use]
    pub fn geometry(&self) -> (usize, usize, usize) {
        (self.n, self.k, self.b)
    }
}
