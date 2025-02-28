import csv
import random

import numpy as np
from sklearn.datasets import make_swiss_roll

from .spheres import create_sphere_dataset


def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)


class DataGenerator:
    def __init__(self, random_state=42):
        self.random_state = random_state
        fix_seed(self.random_state)

    def _read_csv_data(self, data_path, label_path):
        with open(data_path, "r") as file:
            reader = csv.reader(file)
            X = np.array(list(reader), dtype=np.float32)

        with open(label_path, "r") as file:
            reader = csv.reader(file)
            y = np.array(list(reader), dtype=np.float64)

        return X, y

    def load_mnist(self):
        return self._read_csv_data(
            data_path="./bottleneck/MNIST/data.csv",
            label_path="./bottleneck/MNIST/labels.csv",
        )

    def make_sphere_dataset(self, N=1000):
        N = N // 20
        X, y = create_sphere_dataset(n_samples=N, seed=self.random_state)
        return X, y

    def make_swiss_roll(self, N=10000, noise=0.0):
        X, y = make_swiss_roll(n_samples=N, noise=noise, random_state=self.random_state)
        return X, y

    def make_intersected_loops(self, N):
        assert N % 2 == 0, "N needs to be an even number"

        N = int(N / 2)
        eps = 0.1
        Rell1x, Rell1y = 1, 1
        theta1x, theta1y = 0, 0
        Rell2x, Rell2y = 0.8, 0.6

        angles = np.linspace(0, 2 * np.pi, N)[:, np.newaxis]
        Rcic = 1.0
        cic = np.hstack(
            [
                np.zeros([N, 1]),
                Rcic * np.cos(angles) + eps * np.random.uniform(-1, 1, (N, 1)),
                Rcic * np.sin(angles) + eps * np.random.uniform(-1, 1, (N, 1)),
            ]
        )
        ell1 = np.hstack(
            [
                Rell1x * np.cos(angles) + eps * np.random.uniform(-1, 1, (N, 1)),
                Rell1y * np.sin(angles) + eps * np.random.uniform(-1, 1, (N, 1)),
                np.zeros([N, 1]),
            ]
        )
        R1x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta1x), -np.sin(theta1x)],
                [0, np.sin(theta1x), np.cos(theta1x)],
            ]
        )
        R1y = np.array(
            [
                [np.cos(theta1y), 0, np.sin(theta1y)],
                [0, 1, 0],
                [-np.sin(theta1y), 0, np.cos(theta1y)],
            ]
        )
        ell2 = np.hstack(
            [
                Rell2x * np.cos(angles) + eps * np.random.uniform(-1, 1, (N, 1)),
                Rell2y * np.sin(angles) + eps * np.random.uniform(-1, 1, (N, 1)),
                np.zeros([N, 1]),
            ]
        )
        ell1 = ell1 + np.array([[0.3, 0.3, 0.0]])
        ell2 = ell2 + np.array([[-0.1, -0.1, 0.0]])
        X = np.vstack([cic, cic[0, :] + np.dot(np.dot(ell1, R1x), R1y)])

        # Add univariate position (angles in this case):
        y1 = angles.flatten()  # univariate position along the first loop
        y2 = angles.flatten()  # univariate position along the second loop
        y = np.hstack([y1, y2])  # concatenate into one array

        return X, y
