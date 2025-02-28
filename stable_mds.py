import sys

import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_random_state


class IndexLoader:
    def __init__(self, n_samples, batch_size):
        self.n_samples = n_samples
        self.indices = np.arange(n_samples)
        self.batch_size = batch_size
        np.random.shuffle(self.indices)
        self.current_index = 0

    def get_batch(self):
        if self.current_index + self.batch_size > self.n_samples:
            np.random.shuffle(self.indices)
            self.current_index = 0

        batch_indices = self.indices[
            self.current_index : self.current_index + self.batch_size
        ]
        self.current_index += self.batch_size
        return batch_indices


@jit(nopython=True, cache=True, fastmath=True)
def numba_euclidean_distances(X):
    """
    Compute the Euclidean distances between the rows of X.
    """
    m, n = X.shape
    distances = np.empty((m, m))

    for i in range(m):
        for j in range(m):
            dist = 0.0
            for k in range(n):
                diff = X[i, k] - X[j, k]
                dist += diff * diff
            distances[i, j] = np.sqrt(dist)

    return distances


@jit(nopython=True, cache=True, fastmath=True)
def compute_loss(X_embedded, distance_matrix):
    low_dim_distances = numba_euclidean_distances(X_embedded)
    n_samples = X_embedded.shape[0]
    loss = 0.0

    for i in range(n_samples):
        for j in range(i, n_samples):
            loss += (low_dim_distances[i, j] - distance_matrix[i, j]) ** 2

    return loss


@jit(nopython=True, cache=True, fastmath=True, parallel=False)
def update_X_embedded_using_gradient_estimation(
    X_embedded, distance_matrix, sampling_indices
):
    n_samples, n_components = X_embedded.shape
    diff = np.zeros((n_components,), dtype=np.float32)
    grad = np.zeros((n_components,), dtype=np.float32)
    step_size = 1.0 / (2 * len(sampling_indices))

    for i in range(n_samples):
        for j in sampling_indices:
            if not i == j:
                sqsum = 0.0
                for k in range(n_components):
                    diff[k] = X_embedded[i, k] - X_embedded[j, k]
                    sqsum += diff[k] ** 2
                norm = max(np.sqrt(sqsum), 1e-5)
                factor = 1.0 - distance_matrix[i, j] / norm

                for k in range(n_components):
                    grad[k] += 2 * diff[k] * factor

        for k in range(n_components):
            X_embedded[i, k] -= step_size * grad[k]
            grad[k] = 0.0

    return X_embedded


@jit(nopython=True, cache=True, fastmath=True, parallel=False)
def update_X_embedded(X_embedded, distance_matrix):
    n_samples, n_components = X_embedded.shape
    diff = np.zeros((n_components,), dtype=np.float32)
    grad = np.zeros((n_components,), dtype=np.float32)
    step_size = 1.0 / (2 * (n_samples - 1))

    for i in range(n_samples):
        for j in range(n_samples):
            if not i == j:
                sqsum = 0.0
                for k in range(n_components):
                    diff[k] = X_embedded[i, k] - X_embedded[j, k]
                    sqsum += diff[k] ** 2
                norm = max(np.sqrt(sqsum), 1e-5)
                factor = 1.0 - distance_matrix[i, j] / norm

                for k in range(n_components):
                    grad[k] += 2 * diff[k] * factor

        for k in range(n_components):
            X_embedded[i, k] -= step_size * grad[k]
            grad[k] = 0.0

    return X_embedded


class StableMDS(BaseEstimator):
    def __init__(
        self,
        n_components=2,
        n_iter=200,
        sampling_size=None,
        normalization=False,
        verbose=0,
        return_loss_values=False,
        random_state=None,
    ):
        self.n_components = n_components
        self.n_iter = n_iter
        self.sampling_size = sampling_size
        self.normalization = normalization
        self.verbose = verbose
        self.return_loss_values = return_loss_values
        self.random_state = random_state

    def fit_transform(self, X):
        X = self._validate_data(
            X,
            accept_sparse=["csr"],
            ensure_min_samples=2,
            dtype=[np.float16, np.float32, np.float64],
        )
        n_samples = X.shape[0]
        loss_values = []

        # Initialize low-dimensional embeddings
        random_state = check_random_state(self.random_state)
        X_embedded = random_state.uniform(size=n_samples * self.n_components).reshape(
            (n_samples, self.n_components)
        )

        # Check the size of data
        max_feasible_number = 2e4
        if n_samples > max_feasible_number:
            raise ValueError("The data is too large and is currently not supported.")

        # Compute a pairwise distance
        if self.normalization == True:
            scaler = MinMaxScaler()
            distance_matrix = scaler.fit_transform(euclidean_distances(X))
        else:
            distance_matrix = euclidean_distances(X)

        # Gradient method for stable MDS
        if self.sampling_size is None:
            # Optimization loop
            for it in range(self.n_iter):
                if self.verbose > 0 or self.return_loss_values == True:
                    if self.verbose == 2 or self.return_loss_values == True:
                        loss = compute_loss(X_embedded, distance_matrix)
                        if np.isnan(loss):
                            break
                        loss_values.append(loss)

                    if self.verbose == 1:
                        print(f"Iteration {it + 1}/{self.n_iter}")
                    if self.verbose == 2:
                        print(f"Iteration {it + 1}/{self.n_iter}, Loss: {loss}")

                X_embedded = update_X_embedded(X_embedded, distance_matrix)

        else:  # Accelerated version
            loader = IndexLoader(n_samples, self.sampling_size)

            # Optimization loop
            for it in range(self.n_iter):
                if self.verbose > 0 or self.return_loss_values == True:
                    if self.verbose == 2 or self.return_loss_values == True:
                        loss = compute_loss(X_embedded, distance_matrix)
                        if np.isnan(loss):
                            break
                        loss_values.append(loss)

                    if self.verbose == 1:
                        print(f"Iteration {it + 1}/{self.n_iter}")
                    if self.verbose == 2:
                        print(f"Iteration {it + 1}/{self.n_iter}, Loss: {loss}")

                sampling_indices = loader.get_batch()
                X_embedded = update_X_embedded_using_gradient_estimation(
                    X_embedded, distance_matrix, sampling_indices
                )

        if self.return_loss_values == True:
            return X_embedded, loss_values
        else:
            return X_embedded


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from utils.data_generator import DataGenerator

    # Settings
    n_samples = 1000
    n_iter = 500
    n_pairs = n_iter * (n_iter - 1) / 2

    #  Dataset loader
    data_generator = DataGenerator(random_state=42)
    X, y = data_generator.make_sphere_dataset(N=n_samples)

    # StableMDS
    smds = StableMDS(n_iter=n_iter, return_loss_values=True, verbose=2)
    Y1, loss_values1 = smds.fit_transform(X)
    Y1 = Y1[::-1]

    # FastMDS
    asmds = StableMDS(
        n_iter=n_iter, return_loss_values=True, sampling_size=100, verbose=2
    )
    Y2, loss_values2 = asmds.fit_transform(X)
    Y2 = Y2[::-1]

    # Plot the loss values
    min_value = min(np.min(loss_values1), np.min(loss_values2))
    plt.plot(
        np.array(loss_values1) - min_value, marker="o", markersize=0.1, color="red"
    )
    plt.plot(
        np.array(loss_values2) - min_value, marker="o", markersize=0.1, color="blue"
    )
    plt.yscale("log")
    plt.title("Loss Values")
    plt.xlabel("Epoch")
    plt.ylabel("Log-scale Loss")
    plt.show()
    plt.close()

    # Plot the embeddings
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(Y1[:, 0], Y1[:, 1], c=y[::-1], cmap=plt.cm.Spectral, s=3)
    plt.title("StableMDS")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.scatter(Y2[:, 0], Y2[:, 1], c=y[::-1], cmap=plt.cm.Spectral, s=3)
    plt.title("FastMDS")
    plt.legend()
    plt.show()
    plt.close()
