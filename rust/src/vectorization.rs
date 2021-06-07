use std::collections::HashMap;

/// Transform tokens into a sparse vector represented as tuples (dim, idf)
/// and sorted by idf so that more discriminant dimensions appear first.
pub fn vectorize(
    vocabulary: &HashMap<String, (usize, f64)>,
    tokens: &[String],
) -> Vec<(usize, f64)> {
    let mut tokens = tokens
        .iter()
        .filter_map(|token| vocabulary.get(token).copied())
        .collect::<Vec<(usize, f64)>>();

    tokens.sort_unstable_by(|x, y| (y.1, y.0).partial_cmp(&(x.1, x.0)).unwrap());

    tokens
}
