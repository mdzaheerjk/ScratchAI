# PCA (Principal Component Analysis)
Below is **Part 1**: a complete **PCA (Principal Component Analysis) class from scratch using only NumPy**. This implementation includes:

* Mean centering
* Covariance matrix computation
* Eigenvalue & eigenvector calculation
* Sorting principal components
* Explained variance
* Transform
* Inverse transform
* Fit + Transform methods

```python id="6rj1yl"
import numpy as np


class PCA:

    def __init__(self, n_components):
        """
        Parameters
        ----------
        n_components : int
            Number of principal components to keep
        """

        self.n_components = n_components

        self.mean = None
        self.components = None
        self.eigenvalues = None
        self.explained_variance_ratio = None

    ####################################################
    # Fit PCA
    ####################################################

    def fit(self, X):

        n_samples, n_features = X.shape

        ####################################################
        # Mean Centering
        ####################################################

        self.mean = np.mean(X, axis=0)

        X_centered = X - self.mean

        ####################################################
        # Covariance Matrix
        ####################################################

        covariance_matrix = np.cov(
            X_centered,
            rowvar=False
        )

        ####################################################
        # Eigenvalues & Eigenvectors
        ####################################################

        eigenvalues, eigenvectors = np.linalg.eig(
            covariance_matrix
        )

        ####################################################
        # Sort Eigenvalues
        ####################################################

        sorted_indices = np.argsort(
            eigenvalues
        )[::-1]

        eigenvalues = eigenvalues[
            sorted_indices
        ]

        eigenvectors = eigenvectors[
            :,
            sorted_indices
        ]

        ####################################################
        # Store Eigenvalues
        ####################################################

        self.eigenvalues = eigenvalues

        ####################################################
        # Select Top Components
        ####################################################

        self.components = eigenvectors[
            :,
            :self.n_components
        ]

        ####################################################
        # Explained Variance Ratio
        ####################################################

        total_variance = np.sum(
            eigenvalues
        )

        self.explained_variance_ratio = (
            eigenvalues[:self.n_components]
            / total_variance
        )

    ####################################################
    # Transform Data
    ####################################################

    def transform(self, X):

        X_centered = X - self.mean

        return np.dot(
            X_centered,
            self.components
        )

    ####################################################
    # Fit + Transform
    ####################################################

    def fit_transform(self, X):

        self.fit(X)

        return self.transform(X)

    ####################################################
    # Inverse Transform
    ####################################################

    def inverse_transform(self, X_transformed):

        return (
            np.dot(
                X_transformed,
                self.components.T
            )
            + self.mean
        )

    ####################################################
    # Explained Variance
    ####################################################

    def get_explained_variance_ratio(self):

        return self.explained_variance_ratio

    ####################################################
    # Cumulative Explained Variance
    ####################################################

    def get_cumulative_variance(self):

        return np.cumsum(
            self.explained_variance_ratio
        )

    ####################################################
    # Principal Components
    ####################################################

    def get_components(self):

        return self.components

    ####################################################
    # Eigenvalues
    ####################################################

    def get_eigenvalues(self):

        return self.eigenvalues

    ####################################################
    # Mean Vector
    ####################################################

    def get_mean(self):

        return self.mean
```

---

## Methods Summary

| Method                           | Description                         |
| -------------------------------- | ----------------------------------- |
| `fit()`                          | Computes PCA components             |
| `transform()`                    | Projects data into lower dimensions |
| `fit_transform()`                | Fit and transform in one step       |
| `inverse_transform()`            | Reconstruct original features       |
| `get_components()`               | Returns principal components        |
| `get_eigenvalues()`              | Returns eigenvalues                 |
| `get_explained_variance_ratio()` | Returns explained variance ratio    |
| `get_cumulative_variance()`      | Returns cumulative variance         |
| `get_mean()`                     | Returns mean vector                 |

---

## Output Shapes

Suppose:

```text
X.shape = (100, 10)
n_components = 3
```

Then:

```text
components.shape = (10, 3)

transformed.shape = (100, 3)

mean.shape = (10,)
```

This completes the **entire PCA implementation class**. The next part will cover **training, transformation, inverse transformation, visualization, and example usage**.
