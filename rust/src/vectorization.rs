use std::collections::HashMap;

/// Transform tokens into a sparse normalized vector represented as tuples (dim, idf)
/// and sorted by idf so that more discriminant dimensions appear first.
pub fn vectorize(
    vocabulary: &HashMap<String, (usize, f64)>,
    tokens: &[String],
) -> Vec<(usize, f64)> {
    let mut vector: Vec<(usize, f64)> = Vec::new();
    let mut norm = 0.0;

    for token in tokens {
        match vocabulary.get(token) {
            Some((dim, w)) => {
                norm += w * w;
                vector.push((*dim, *w));
            }
            None => continue,
        }
    }

    // Normalizing the vector
    norm = norm.sqrt();

    for (_, w) in vector.iter_mut() {
        *w /= norm;
    }

    vector.sort_unstable_by(|x, y| (y.1, y.0).partial_cmp(&(x.1, x.0)).unwrap());

    vector
}
