use sparseset::SparseSet;

/// Function computing the cosine distance between two sparse vectors by using
/// a cached sparse set helper representing the first of the two vectors
/// Note that it clamps the result to 0 to avoid precision issues with f64.
#[inline]
pub fn sparse_dot_product_distance_with_helper(
    helper: &SparseSet<f64>,
    other: &[(usize, f64)],
) -> f64 {
    let mut product = 0.0;

    for (dim, w2) in other {
        let w1 = helper.get(*dim).unwrap_or(&0.0);
        product += w1 * w2;
    }

    product = 1.0 - product;

    // Precision error
    // TODO: need a larger epsilon?
    if product - f64::EPSILON < 0.0 {
        return 0.0;
    }

    product
}
