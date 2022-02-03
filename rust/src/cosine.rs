use bitintr::Popcnt;
use rayon::prelude::*;
use sparseset::SparseSet;
use tab_hash::Tab64Simple;

lazy_static! {
    // TODO: convert as part of simhash struct
    static ref UNIVERSAL_HASHER: Tab64Simple = Tab64Simple::new();
}

/// Function computing the cosine distance between two normalized sparse vectors
/// by using a cached sparse set helper representing the first of the two vectors
/// Note that it clamps the result to 0 to avoid precision issues with f64.
#[inline]
pub fn sparse_dot_product_distance_with_helper(
    helper: &SparseSet<f64>,
    other: &[(usize, f64)],
) -> f64 {
    let min_nb_words = 1;
    let mut product = 0.0;
    let mut dim_count = 0;

    for (dim, w2) in other {
        let w1 = helper.get(*dim).unwrap_or(&0.0);
        let matching_dim = if w1 > &0.0 { 1 } else { 0 };
        dim_count += matching_dim;
        product += w1 * w2;
    }

    product = if dim_count > min_nb_words {
        1.0 - product
    } else {
        1.0
    };

    // Precision error
    // TODO: need a larger epsilon?
    if product - f64::EPSILON < 0.0 {
        return 0.0;
    }

    product
}

#[inline]
pub fn parallel_sparse_dot_product_distance_with_helper(
    helper: &SparseSet<f64>,
    other: &[(usize, f64)],
) -> f64 {
    let mut product: f64 = other
        .par_iter()
        .map(|(dim, w2)| {
            let w1 = helper.get(*dim).unwrap_or(&0.0);
            w1 * w2
        })
        .sum();

    product = 1.0 - product;

    // Precision error
    // TODO: need a larger epsilon?
    if product - f64::EPSILON < 0.0 {
        return 0.0;
    }

    product
}

// Ref: https://dash.harvard.edu/bitstream/handle/1/38811431/GHOCHE-SENIORTHESIS-2016.pdf
// Ref: http://benwhitmore.altervista.org/simhash-and-solving-the-hamming-distance-problem-explained/
pub fn simhash_64(vector: &[(usize, f64)]) -> u64 {
    let mut histogram: Vec<f64> = vec![0.0; 64];

    for (dim, w) in vector {
        let mut phi = UNIVERSAL_HASHER.hash(*dim as u64);

        for counter in (0..64).rev() {
            // Chosing randomly to add or subtract weight to histogram
            let bit = phi % 2 == 0;

            if bit {
                histogram[counter] += w;
            } else {
                histogram[counter] -= w;
            }

            phi >>= 1;
        }
    }

    let mut hash: u64 = 0;

    for (i, w) in histogram.into_iter().enumerate() {
        if w >= 0.0 {
            hash += 1;
        }
        if i < 63 {
            hash <<= 1;
        }
    }

    hash
}

fn hamming_distance_64(x: u64, y: u64) -> u64 {
    (x ^ y).popcnt()
}

pub fn simhash_distance_64(x: u64, y: u64) -> f64 {
    hamming_distance_64(x, y) as f64 / 64.0
}
