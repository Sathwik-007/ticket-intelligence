import numpy as np

class CustomKmeansPlusPlus:

    """
    K-Means clustering: Matches sklearn's K-Means assign, fit, fit_predict

    Parameters:
    -----------
    @k            : no.of clusters
    @max_iters    : max no.of EM loops (Expectation and Maximisation)
    @tol          : convergence tolerance of centroids after which the loop shall stop
    @n_init       : choose best max inertia from a couple of runs
    @random_state : seed reproducability
    """
    def __init__(self, k=8, max_iters=100, tol=1e-4, n_init=10, random_state=42):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

        self.centroids = None
        self.inertia_ = None
        self.inertia_history = []
        self.labels = None

    # K-Means++ initialisation
    def _init_centroids_pp(self, X):
        
        rng = np.random.default_rng(self.random_state)

        n = X.shape[0]

        centroids = [X[rng.choice(n)].copy()]

        for _ in range(self.k - 1):
            dists = np.array([np.sum((X - c) ** 2, axis=1) for c in centroids])
            min_dists = np.min(dists, axis=0)

            probs = min_dists / np.sum(min_dists)

            next_idx = rng.choice(n, p=probs)
            centroids.append(X[next_idx])

        self.centroids = np.array(centroids)
        return self.centroids

    def _compute_distances(self, X):
        # shape of X: (N, F), shape of centroids: (C, F)
        diff = X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]
        return np.sum(diff ** 2, axis=2)
        
    def fit(self, X):
        n_samples, n_features = X.shape

        # 1. Initialise centroids
        self.centroids = self._init_centroids_pp(X)

        for _ in range(self.max_iters):
            # 2. Expectation step (Assignment step) EM loop
            sq_diff = self._compute_distances(X)
            self.labels = np.argmin(sq_diff, axis=1)

            # 3. Calculate Error: Inertia (WCSS) Sum of Squared Distances to closest centroid
            min_dists = np.min(sq_diff, axis=1)
            inertia = float(np.sum(min_dists))
            self.inertia_history.append(inertia)

            # 4. Updation of centroids(Maximisation step)
            # Sum of data points with in the centroid / no.of data points in that centroid
            new_centroids = np.zeros_like(self.centroids)
            for label in range(self.k):
                members = X[self.labels == label]

                if len(members) > 0:
                    new_centroids[label] = np.mean(members, axis=0)

                # Edge case: if its an empty cluster, assign farthest point
                else:
                    farthest_point = np.argmax(min_dists, axis=1)
                    new_centroids[label] = X[farthest_point]

            # 5. Check for convergence
            centroid_shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids

            if centroid_shift < self.tol:
                break

        self.inertia_ = self.inertia_history[-1]
        return self

    def predict(self, X):
        sq_diff = self._compute_distances(X)
        # return what clusters the squared differences belong to
        return np.argmin(sq_diff, axis=1)
            
        