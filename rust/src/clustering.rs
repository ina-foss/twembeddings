use std::collections::VecDeque;

use rayon::prelude::*;
use sparseset::SparseSet;

use crate::cosine::sparse_dot_product_distance_with_helper;

type SparseVector = Vec<(usize, f64)>;

// TODO: verify mean/median candidate set size
// TODO: sanity tests and evaluation command
// TODO: start from window directly to easy test
// TODO: use #[cfg] for stats within the function
// TODO: https://dash.harvard.edu/bitstream/handle/1/38811431/GHOCHE-SENIORTHESIS-2016.pdf?sequence=3
pub struct ClusteringBuilder {
    voc_size: usize,
    threshold: f64,
    window: usize,
    query_size: u8,
}

pub struct Clustering {
    threshold: f64,
    window: usize,
    query_size: u8,
    dropped_so_far: usize,
    cosine_helper_set: SparseSet<f64>,
    inverted_index: Vec<VecDeque<usize>>,
    vectors: VecDeque<SparseVector>,
    candidates: SparseSet<bool>,
}

impl ClusteringBuilder {
    pub fn new(voc_size: usize, window: usize) -> ClusteringBuilder {
        ClusteringBuilder {
            voc_size,
            threshold: 0.7,
            window,
            query_size: 5,
        }
    }

    pub fn with_threshold(&mut self, threshold: f64) -> &mut ClusteringBuilder {
        self.threshold = threshold;
        self
    }

    pub fn with_query_size(&mut self, query_size: u8) -> &mut ClusteringBuilder {
        self.query_size = query_size;
        self
    }

    pub fn build(self) -> Clustering {
        let mut inverted_index: Vec<VecDeque<usize>> = Vec::with_capacity(self.voc_size);

        for _ in 0..self.voc_size {
            inverted_index.push(VecDeque::new());
        }

        Clustering {
            threshold: self.threshold,
            window: self.window,
            query_size: self.query_size,
            dropped_so_far: 0,
            cosine_helper_set: SparseSet::with_capacity(self.voc_size),
            inverted_index,
            vectors: VecDeque::with_capacity(self.window),
            candidates: SparseSet::with_capacity(self.window),
        }
    }
}

impl Clustering {
    pub fn nearest_neighbor(&mut self, index: usize, vector: SparseVector) -> Option<(usize, f64)> {
        self.cosine_helper_set.clear();
        self.candidates.clear();

        // Indexing and gathering candidates
        let mut dim_tested: u8 = 0;

        for (dim, w) in vector.iter() {
            self.cosine_helper_set.insert(*dim, *w);

            let deque = &mut self.inverted_index[*dim];

            if dim_tested < self.query_size {
                for &candidate in deque.iter() {
                    self.candidates
                        .insert(candidate - self.dropped_so_far, true);
                }
                dim_tested += 1;
            }

            deque.push_back(index);
        }

        // Finding the nearest neighbor
        // TODO: test par_bridge to avoid collection?
        let best_candidate = self
            .candidates
            .iter()
            .map(|x| x.key())
            .collect::<Vec<usize>>()
            .par_iter()
            .map(|&candidate| {
                let other_vector = &self.vectors[candidate];
                (
                    candidate,
                    sparse_dot_product_distance_with_helper(&self.cosine_helper_set, &other_vector),
                )
            })
            .filter(|x| x.1 < self.threshold)
            .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap());

        // Is the window full already?
        if self.vectors.len() == self.window {
            let to_remove = self.vectors.pop_front().unwrap();

            for (dim, _) in to_remove.into_iter() {
                let deque = &mut self.inverted_index[dim];
                deque.pop_front().unwrap();
            }

            self.dropped_so_far += 1;
        }

        // Adding tweet to the window
        self.vectors.push_back(vector);

        best_candidate
    }
}
