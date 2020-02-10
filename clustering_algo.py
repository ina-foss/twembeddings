from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances
from scipy.sparse import csr_matrix, vstack, issparse
import numpy as np
import logging

class ClusteringAlgo:

    def __init__(self, threshold=0.65, window_size=300000, batch_size=8, distance="cosine"):
        self.M = None
        self.t = threshold
        self.w = window_size
        self.batch_size = batch_size
        self.zeros_vectors = None
        self.thread_id = 0
        self.distance = distance

    def add_vectors(self, vectors):
        self.M = vectors
        if issparse(vectors):
            self.zeros_vectors = vectors.getnnz(1) == 0
        else:
            self.zeros_vectors = ~vectors.any(axis=1)

    def iter_on_matrix(self, ):
        if self.distance == "precomputed":
            matrix = self.M[~self.zeros_vectors][:, ~self.zeros_vectors]
            for idx in range(0, matrix.shape[0], self.batch_size):
                lim = min(idx + self.batch_size, matrix.shape[0])
                vectors = matrix[idx:lim, max(lim-self.w, 0):lim]
                yield idx, vectors
        else:
            matrix = self.M[~self.zeros_vectors]
            for idx in range(0, matrix.shape[0], self.batch_size):
                if idx % 10000 == 0:
                    logging.info(idx)
                vectors = matrix[idx:min(idx + self.batch_size, matrix.shape[0])]
                yield idx, vectors

    def brute_nn(self, data, tweets):
        nn = NearestNeighbors(n_neighbors=1, algorithm='brute', metric=self.distance)
        if self.distance == "precomputed":
            nn.fit(np.zeros((tweets.shape[1], tweets.shape[1])))
        else:
            nn.fit(data)
        distance, neighbor_exact = nn.kneighbors(tweets)
        return distance.transpose()[0], neighbor_exact.transpose()[0]

    def incremental_clustering(self,):
        if issparse(self.M):
            T = csr_matrix((self.w, self.M.shape[1]))
        else:
            T = np.zeros((self.w, self.M.shape[1]), dtype=self.M.dtype)
        threads = np.zeros(self.w, dtype="int")
        total_threads = []
        # centroïds = {}
        for idx, tweets in self.iter_on_matrix():
            i = idx % self.w
            j = i + tweets.shape[0]
            if idx == 0:
                threads[:j] = np.arange(self.thread_id, self.thread_id + j)
                self.thread_id = self.thread_id + j
            else:
                distances, neighbors = self.brute_nn(T, tweets)
                under_t = np.array(distances) < self.t
                # points that have a close neighbor in the window get the label of that neighbor

                threads[i:j][under_t] = threads[neighbors[under_t]]
                # assign new labels to points that do not have close enough neighbors
                distant_neighbors = neighbors[~under_t]
                new_labels = np.arange(self.thread_id, self.thread_id + len(distant_neighbors))
                threads[i:j][~under_t] = new_labels

                if new_labels.size != 0:
                    self.thread_id = max(new_labels) + 1
            if issparse(self.M):
                T = vstack([T[:i], tweets, T[j:]])  # much faster than T[i:j] = tweets
            elif self.distance != "precomputed":
                T[i:j] = tweets

            total_threads.extend(threads[i:j])

        total_threads = np.array(total_threads)

        total_threads_with_zeros_vectors = np.zeros(self.M.shape[0], dtype="int")
        total_threads_with_zeros_vectors[self.zeros_vectors] = -1
        total_threads_with_zeros_vectors[~self.zeros_vectors] = total_threads
        return total_threads_with_zeros_vectors.tolist()


class ClusteringAlgoSparse:

    def __init__(self, threshold=0.65, window_size=300000, batch_size=8, distance="cosine", tfidf_t=0, min_words_seed=0):
        self.M = None
        self.t = threshold
        self.w = window_size
        self.batch_size = batch_size
        self.zeros_vectors = None
        self.thread_id = 0
        self.distance = distance
        self.tfidf_t = tfidf_t
        self.nnz_length = None
        self.min_words_seed = min_words_seed

    def add_vectors(self, vectors):
        self.M = vectors
        self.zeros_vectors = vectors.getnnz(1) == 0

    def iter_on_matrix(self, matrix):
        mask = self.get_mask(matrix)
        for idx in range(0, self.nnz_length, self.batch_size):
            if idx % 10000 == 0:
                logging.info(idx)
            local_mask = mask[idx:min(idx + self.batch_size, self.nnz_length)]
            vectors = matrix[idx:min(idx + self.batch_size, self.nnz_length)]
            window_mask = mask[max(0, idx-self.w):idx]
            T = matrix[max(0, idx-self.w):idx][window_mask]
            yield idx, vectors, local_mask, T, window_mask

    def brute_nn(self, data, tweets):
        distances = cosine_distances(tweets, data)
        neighbors = distances.argmin(axis=1)
        return distances[range(distances.shape[0]), neighbors], neighbors

    def get_mask(self, matrix):
        mask_min_weight = (matrix.max(axis=1) > self.tfidf_t).T.A.ravel()
        mask_min_words = (matrix.getnnz(1) > self.min_words_seed).ravel()
        return mask_min_weight*mask_min_words

    def incremental_clustering(self,):
        matrix = self.M[~self.zeros_vectors]
        self.nnz_length = matrix.shape[0]
        threads = np.zeros(self.nnz_length, dtype="int")
        for i, tweets, mask, T, window_mask in self.iter_on_matrix(matrix):
            j = i + tweets.shape[0]
            if i == 0:
                threads[:j][mask] = range(len(threads[:j][mask]))
                threads[:j][~mask] = -2
                self.thread_id = self.thread_id + j
            else:
                distances, neighbors = self.brute_nn(T, tweets)
                under_t = np.array(distances) < self.t
                # points that have a close neighbor in the window get the label of that neighbor
                threads[i:j][under_t] = threads[max(0, i-self.w):i][window_mask][neighbors[under_t]]
                # assign new labels to points that do not have close enough neighbors - except those that are ignored
                threads[i:j][~under_t] = -2
                clustered = ~under_t[mask]
                new_labels = range(self.thread_id, self.thread_id + clustered.sum())
                threads[i:j].flat[np.flatnonzero(mask)[clustered]] = new_labels

                if new_labels:
                    self.thread_id = max(new_labels) + 1

        total_threads_with_zeros_vectors = np.zeros(self.M.shape[0], dtype="int")
        total_threads_with_zeros_vectors[self.zeros_vectors] = -1
        total_threads_with_zeros_vectors[~self.zeros_vectors] = threads
        return total_threads_with_zeros_vectors.tolist()
