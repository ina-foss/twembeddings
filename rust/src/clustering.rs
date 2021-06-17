use std::collections::VecDeque;

use rayon::prelude::*;
use sparseset::SparseSet;

use crate::cosine::sparse_dot_product_distance_with_helper;

type SparseVector = Vec<(usize, f64)>;

// TODO: verify mean/median candidate set size
// TODO: start from window directly to easy test
// TODO: use #[cfg] for stats within the function
pub struct ClusteringBuilder {
    voc_size: usize,
    threshold: f64,
    window: usize,
    query_size: u8,
    max_candidates_per_dimension: usize,
}

pub struct Clustering {
    threshold: f64,
    window: usize,
    query_size: u8,
    max_candidates_per_dimension: usize,
    dropped_so_far: usize,
    current_thread_id: usize,
    cosine_helper_set: SparseSet<f64>,
    inverted_index: Vec<VecDeque<usize>>,
    // TODO: this should be nice to turn this 3-tuple into a struct but I
    // don't know how to do so without breaking the code regarding ownership
    vectors: VecDeque<(u64, usize, SparseVector)>,
    candidates: SparseSet<bool>,
}

impl ClusteringBuilder {
    pub fn new(voc_size: usize, window: usize) -> Self {
        ClusteringBuilder {
            voc_size,
            threshold: 0.7,
            window,
            query_size: 5,
            max_candidates_per_dimension: 64,
        }
    }

    pub fn with_threshold(&mut self, threshold: f64) -> &mut Self {
        self.threshold = threshold;
        self
    }

    pub fn with_query_size(&mut self, query_size: u8) -> &mut Self {
        self.query_size = query_size;
        self
    }

    pub fn with_max_candidates_per_dimension(&mut self, max: usize) -> &mut Self {
        self.max_candidates_per_dimension = max;
        self
    }

    pub fn build(&self) -> Clustering {
        let mut inverted_index: Vec<VecDeque<usize>> = Vec::with_capacity(self.voc_size);

        for _ in 0..self.voc_size {
            inverted_index.push(VecDeque::new());
        }

        Clustering {
            threshold: self.threshold,
            window: self.window,
            query_size: self.query_size,
            max_candidates_per_dimension: self.max_candidates_per_dimension,
            dropped_so_far: 0,
            current_thread_id: 0,
            cosine_helper_set: SparseSet::with_capacity(self.voc_size),
            inverted_index,
            vectors: VecDeque::with_capacity(self.window),
            candidates: SparseSet::with_capacity(self.window),
        }
    }
}

impl Clustering {
    pub fn nearest_neighbor(
        &mut self,
        index: usize,
        id: u64,
        vector: SparseVector,
    ) -> (Option<(u64, f64)>, usize) {
        self.cosine_helper_set.clear();
        self.candidates.clear();

        // Indexing and gathering candidates
        let mut dim_tested: u8 = 0;

        for (dim, w) in vector.iter() {
            self.cosine_helper_set.insert(*dim, *w);

            let deque = &mut self.inverted_index[*dim];

            if dim_tested < self.query_size {
                for &candidate in deque.iter().rev().take(self.max_candidates_per_dimension) {
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
                let (other_id, other_thread_id, other_vector) = &self.vectors[candidate];

                (
                    *other_id,
                    *other_thread_id,
                    sparse_dot_product_distance_with_helper(&self.cosine_helper_set, &other_vector),
                )
            })
            .filter(|x| x.2 < self.threshold)
            .min_by(|x, y| x.2.partial_cmp(&y.2).unwrap());

        // Is the window full already?
        if self.vectors.len() == self.window {
            let (_, _, to_remove) = self.vectors.pop_front().unwrap();

            for (dim, _) in to_remove.into_iter() {
                let deque = &mut self.inverted_index[dim];
                deque.pop_front().unwrap();
            }

            self.dropped_so_far += 1;
        }

        // Adding tweet to the window
        match best_candidate {
            Some((other_id, other_thread_id, d)) => {
                self.vectors.push_back((id, other_thread_id, vector));
                (Some((other_id, d)), other_thread_id)
            }
            None => {
                let thread_id = self.current_thread_id;
                self.vectors.push_back((id, thread_id, vector));
                self.current_thread_id += 1;
                (None, thread_id)
            }
        }
    }
}
