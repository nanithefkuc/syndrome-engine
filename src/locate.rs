//! The Chien search over the code's position set.
//!
//! The decoder's Chien search is classical: the (total) locator is
//! evaluated at the inverse error locators `α^{-p}` of every code position
//! `p = 0..n-1`, and the vanishing positions are the errors. The
//! root→position map — position `p` corresponds to the root `α^{-p}` — is
//! the frozen half of the wire convention (S6): positions are reported in
//! ascending order.
//!
//! The arithmetic is `univariate`'s multipoint evaluation over the frozen
//! position-point set. A whole-field scan through `univariate::chien_roots`
//! would visit `|F|` elements — neither the decoder's `n`-position domain
//! nor an affordable pass at `Gf32`/`Gf64` — so the position scan composes
//! the same primitive the syndrome pass uses.

use alloc::vec::Vec;

use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use univariate::{MultipointScratch, Polynomial, evaluate_multipoint_into};

use crate::error::DecodeError;
use crate::params::RsParams;

/// Write the position points `α^{-0}, α^{-1}, …, α^{-(n-1)}` into `points`,
/// stepping by `α^{-1}` from one.
pub fn position_points_into<F: FieldKernels>(params: &RsParams<F>, points: &mut Vec<F::Elem>) {
    points.clear();
    let step = <F as Field>::GENERATOR.inv();
    let mut point = F::Elem::ONE;
    for _ in 0..params.n() {
        points.push(point);
        point = point.mul(step);
    }
}

/// Write the positions whose locator value vanishes into `positions`, in
/// ascending position order.
///
/// # Errors
///
/// Returns [`DecodeError::AllocationFailed`] when the evaluation cannot
/// reserve a buffer.
pub fn locate_into<F: FieldKernels>(
    locator: &Polynomial<F>,
    position_points: &[F::Elem],
    eval: &mut MultipointScratch<F>,
    values: &mut Vec<F::Elem>,
    positions: &mut Vec<usize>,
) -> Result<(), DecodeError> {
    positions.clear();
    evaluate_multipoint_into(locator, position_points, eval, values)?;
    positions.extend(
        values
            .iter()
            .enumerate()
            .filter(|(_, value)| value.is_zero())
            .map(|(position, _)| position),
    );
    Ok(())
}
