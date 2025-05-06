"""Utility classes for training PRNNs using JAX"""

import numpy as np
import jax.numpy as jnp
import pandas

class Normalizer:
    """Normalization for strain features.

    Scales strain data to a [-1,1] interval
    """

    def __init__(self, X):
        self.min = X.min(axis=0).values
        self.max = X.max(axis=0).values

    def normalize(self, x):
        return 2.0 * ((x - self.min) / (self.max-self.min)) - 1.0


class Normalize_set:
    """
    Normalize a train, validation and optionally test set based only on the training set samples.
    """
    def __init__(self, dataset, norm_input=True, norm_output=True):
        """
        :param dataset:
        """
        self.norm_input = norm_input
        self.norm_output = norm_output

        if norm_input:
            self.in_normalizer = Normalizer(dataset[:,0])
        if norm_output:
            self.out_normalizer = Normalizer(dataset[:,1])

    def normalize(self, dataset):
        new_dataset = dataset.copy()
        if self.norm_input:
            new_dataset[:,0] = self.in_normalizer.normalize(dataset[:,0])
        if self.norm_output:
            new_dataset[:,1] = self.out_normalizer.normalize(dataset[:,1])
        return new_dataset


class StressStrainDataset:
    """Dataset for handling stress-strain paths in JAX.

    Dataset is loaded with pandas and stress-strain pairs are
    split into paths of 'seq_length' time steps. Normalization
    is optional and the normalizer can be inherited from another
    dataset (e.g. to handle an entirely new test set).
    """

    def __init__(self, filename, features, targets, seq_length, **kwargs):
        df = pandas.read_csv(filename, sep='\\s+', header=None)

        self.seq_length = seq_length

        # Convert to JAX arrays
        self.X = jnp.array(df[features].values)
        self.T = jnp.array(df[targets].values)

        self.normalize_features = kwargs.get('normalize_features', False)

        self._normalizer = None
        if self.normalize_features:
            self._normalizer = kwargs.get('normalizer', Normalizer(df[features]))

    def __len__(self):
        return int(self.X.shape[0]/self.seq_length)

    def get_batch(self, idx):
        """Get a single batch at index idx"""
        start = idx * self.seq_length
        end = start + self.seq_length

        x = self.X[start:end, :]
        t = self.T[start:end, :]

        if self.normalize_features:
            return self._normalizer.normalize(x), t
        else:
            return x, t

    def get_all_batches(self):
        """Get all batches as a list of (x, t) tuples"""
        batches = []
        for i in range(self.__len__()):
            batches.append(self.get_batch(i))
        batches = np.array(batches)
        return batches

    def get_normalizer(self):
        if self.normalize_features:
            return self._normalizer
        else:
            return None

    def get_subset(self, indices):
        """Get a subset of the dataset based on provided indices"""
        x_subset = self.X[indices]
        t_subset = self.T[indices]

        if self.normalize_features:
            x_subset = self._normalizer.normalize(x_subset)

        return x_subset, t_subset

