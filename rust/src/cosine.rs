use bitintr::Popcnt;
use sparseset::SparseSet;

/// Function computing the cosine distance between two normalized sparse vectors
/// by using a cached sparse set helper representing the first of the two vectors
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

// Ref: https://dash.harvard.edu/bitstream/handle/1/38811431/GHOCHE-SENIORTHESIS-2016.pdf
// Ref: http://benwhitmore.altervista.org/simhash-and-solving-the-hamming-distance-problem-explained/
pub fn simhash_64(vector: &[(usize, f64)]) -> u64 {
    let mut histogram: Vec<f64> = vec![0.0; 64];

    for (dim, w) in vector {
        let mut phi = *dim as u64; // TODO: need to hash the dimension?

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
    (64 - hamming_distance_64(x, y)) as f64 / 64.0
}

// pub fn simhash_128(vector: &[(usize, f64)]) -> u128 {
//     let mut histogram: Vec<f64> = vec![0.0; 128];

//     for (dim, w) in vector {
//         let mut phi = *dim as u128; // TODO: need to hash the dimension?

//         for counter in (0..128).rev() {
//             // Chosing randomly to add or subtract weight to histogram
//             let bit = phi % 2 == 0;

//             if bit {
//                 histogram[counter] += w;
//             } else {
//                 histogram[counter] -= w;
//             }

//             phi >>= 1;
//         }
//     }

//     let mut hash: u128 = 0;

//     for (i, w) in histogram.into_iter().enumerate() {
//         if w >= 0.0 {
//             hash += 1;
//         }
//         if i < 127 {
//             hash <<= 1;
//         }
//     }

//     hash
// }

// fn hamming_distance_128(x: u128, y: u128) -> u32 {
//     (x ^ y).count_ones()
// }

// pub fn simhash_distance_128(x: u128, y: u128) -> f64 {
//     (128 - hamming_distance_128(x, y)) as f64 / 128.0
// }

// #[test]
// fn test_simhash_64() {
//     let a: Vec<(usize, f64)> = vec![(23, 0.32), (59, 0.003), (4536, 0.01), (89, 0.1)];
//     let b: Vec<(usize, f64)> = vec![(23, 0.23), (59, 0.003), (89, 0.1)];

//     let a_hash = simhash_64(&a);
//     let b_hash = simhash_64(&b);

//     let mut a_helper: SparseSet<f64> = SparseSet::with_capacity(4537);

//     for (dim, w) in a {
//         a_helper.insert(dim, w);
//     }

//     // NOTE: the following does not make sense because my vectors are not normalized!
//     dbg!(a_hash, b_hash);
//     println!("{:?}", hamming_distance_64(a_hash, b_hash));
//     println!("{:?}", simhash_distance_64(a_hash, b_hash));
//     println!(
//         "{:?}",
//         sparse_dot_product_distance_with_helper(&a_helper, &b)
//     );
// }
