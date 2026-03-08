"""
transforms.py
-------------
Dimensionality reduction transforms for HEUQ (paper Section 4.1.3, Appendix B).

Both classes follow the sklearn transformer API: fit, transform, fit_transform.

PCATransform
------------
PCA applied as a preprocessing step before training all HEUQ classifiers.
Paper setting: variance_threshold=0.95.
Resulting embedding sizes:
    Private data : 40 components
    Telecom      : 15 components
    Bank         : 10 components
Embedding sizes below 35 (private data) caused overfitting and a considerable
drop in Balanced Accuracy (Appendix B).

GaussianRandomProjection
------------------------
Johnson-Lindenstrauss Gaussian random projection.  Projects from the original
feature space to n_components dimensions while approximately preserving pairwise
distances.  The epsilon parameter controls allowed distortion.

Paper experimental range: epsilon in {0.1, 0.3, 0.5, 0.9}.
Best results (private data): embedding=40, epsilon=0.5 (BA=64.46, Table B.10).
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection as _SklearnGRP


class PCATransform:
    """PCA dimensionality reduction wrapper.

    Fits on training data and applies the same projection to test data.
    Selects n_components to explain at least variance_threshold of total variance.

    Paper setting: variance_threshold=0.95.
    Resulting embedding sizes: 40 (private), 15 (Telecom), 10 (Bank).
    Embedding sizes below 35 caused overfitting on the private dataset (Appendix B).

    Parameters
    ----------
    variance_threshold : float, optional
        Fraction of variance to retain. Default 0.95.
    random_state : int, optional
        Default 42.
    """

    def __init__(self, variance_threshold=0.95, random_state=42):
        self.variance_threshold = variance_threshold
        self.random_state = random_state
        self._pca = None

    def fit(self, X_train):
        """Fit PCA on training data.

        Parameters
        ----------
        X_train : np.ndarray of shape [N, D]

        Returns
        -------
        self
        """
        self._pca = PCA(
            n_components=self.variance_threshold,
            random_state=self.random_state,
        )
        self._pca.fit(X_train)
        return self

    def transform(self, X):
        """Project X into the PCA space.

        Parameters
        ----------
        X : np.ndarray of shape [N, D]

        Returns
        -------
        np.ndarray of shape [N, n_components]
        """
        if self._pca is None:
            raise RuntimeError("PCATransform must be fitted before calling transform.")
        return self._pca.transform(X)

    def fit_transform(self, X_train):
        """Fit on X_train and return the projected training data.

        Parameters
        ----------
        X_train : np.ndarray of shape [N, D]

        Returns
        -------
        np.ndarray of shape [N, n_components]
        """
        self.fit(X_train)
        return self.transform(X_train)

    @property
    def n_components_(self):
        """Number of components selected after fitting."""
        if self._pca is None:
            return None
        return self._pca.n_components_

    @property
    def explained_variance_ratio_(self):
        """Cumulative explained variance ratio after fitting."""
        if self._pca is None:
            return None
        return self._pca.explained_variance_ratio_


class GaussianRandomProjection:
    """Gaussian Random Projection following the Johnson-Lindenstrauss lemma.

    Projects data from a high-dimensional space to n_components dimensions using
    a random Gaussian matrix, approximately preserving pairwise Euclidean distances.

    The Johnson-Lindenstrauss lemma guarantees that a set of n points in
    high-dimensional space can be embedded into O(log n / eps^2) dimensions
    while preserving all pairwise distances within a factor of (1 +/- epsilon).

    Paper experimental settings (Section 4.1.3):
        epsilon range: {0.1, 0.3, 0.5, 0.9}
        Embedding sizes tested:
            Private data : {10, 20, 30, 40}
            Telecom      : {5, 10, 15, 17}
            Bank         : {5, 7, 10}
        Best result (private data): n_components=40, epsilon=0.5 (BA=64.46, Table B.10)

    Parameters
    ----------
    n_components : int
        Target embedding dimension.
    epsilon : float, optional
        Distortion tolerance.  Higher epsilon = lower distortion (less noise).
        Paper range: {0.1, 0.3, 0.5, 0.9}. Default 0.5.
    random_state : int, optional
        Default 42.
    """

    def __init__(self, n_components, epsilon=0.5, random_state=42):
        self.n_components = n_components
        self.epsilon = epsilon
        self.random_state = random_state
        self._proj = None

    def fit(self, X_train):
        """Fit the Gaussian projection matrix on training data.

        Parameters
        ----------
        X_train : np.ndarray of shape [N, D]

        Returns
        -------
        self
        """
        self._proj = _SklearnGRP(
            n_components=self.n_components,
            eps=self.epsilon,
            random_state=self.random_state,
        )
        self._proj.fit(X_train)
        return self

    def transform(self, X):
        """Project X into the low-dimensional space.

        Parameters
        ----------
        X : np.ndarray of shape [N, D]

        Returns
        -------
        np.ndarray of shape [N, n_components]
        """
        if self._proj is None:
            raise RuntimeError(
                "GaussianRandomProjection must be fitted before calling transform."
            )
        return self._proj.transform(X)

    def fit_transform(self, X_train):
        """Fit on X_train and return the projected training data.

        Parameters
        ----------
        X_train : np.ndarray of shape [N, D]

        Returns
        -------
        np.ndarray of shape [N, n_components]
        """
        self.fit(X_train)
        return self.transform(X_train)
