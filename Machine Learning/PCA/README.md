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
## Part 2: Training, Transformation, Inverse Transform, Explained Variance, and Example Usage

> Save this as **`main.py`** (or place it below your `PCA` class).

```python id="5a8n2m"
import numpy as np


#########################################################
# Example Dataset
#########################################################

X = np.array([
    [2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9],
    [1.9, 2.2],
    [3.1, 3.0],
    [2.3, 2.7],
    [2.0, 1.6],
    [1.0, 1.1],
    [1.5, 1.6],
    [1.1, 0.9]
])


#########################################################
# Create PCA Model
#########################################################

pca = PCA(n_components=1)


#########################################################
# Fit PCA
#########################################################

pca.fit(X)


#########################################################
# Transform Dataset
#########################################################

X_transformed = pca.transform(X)


#########################################################
# Fit + Transform
#########################################################

X_projected = pca.fit_transform(X)


#########################################################
# Inverse Transform
#########################################################

X_reconstructed = pca.inverse_transform(
    X_projected
)


#########################################################
# Mean Vector
#########################################################

print("=" * 60)
print("Mean Vector")
print("=" * 60)

print(pca.get_mean())


#########################################################
# Eigenvalues
#########################################################

print("\n")
print("=" * 60)
print("Eigenvalues")
print("=" * 60)

print(pca.get_eigenvalues())


#########################################################
# Principal Components
#########################################################

print("\n")
print("=" * 60)
print("Principal Components")
print("=" * 60)

print(pca.get_components())


#########################################################
# Explained Variance Ratio
#########################################################

print("\n")
print("=" * 60)
print("Explained Variance Ratio")
print("=" * 60)

print(
    pca.get_explained_variance_ratio()
)


#########################################################
# Cumulative Explained Variance
#########################################################

print("\n")
print("=" * 60)
print("Cumulative Explained Variance")
print("=" * 60)

print(
    pca.get_cumulative_variance()
)


#########################################################
# Original Dataset
#########################################################

print("\n")
print("=" * 60)
print("Original Dataset")
print("=" * 60)

print(X)


#########################################################
# Reduced Dataset
#########################################################

print("\n")
print("=" * 60)
print("Reduced Dataset")
print("=" * 60)

print(X_projected)


#########################################################
# Reconstructed Dataset
#########################################################

print("\n")
print("=" * 60)
print("Reconstructed Dataset")
print("=" * 60)

print(X_reconstructed)


#########################################################
# Dimensions
#########################################################

print("\n")
print("=" * 60)
print("Shape Comparison")
print("=" * 60)

print("Original Shape      :", X.shape)

print("Reduced Shape       :", X_projected.shape)

print("Reconstructed Shape :", X_reconstructed.shape)


#########################################################
# Reconstruction Error
#########################################################

mse = np.mean(
    (X - X_reconstructed) ** 2
)

print("\n")
print("=" * 60)
print("Reconstruction Error")
print("=" * 60)

print("MSE :", mse)


#########################################################
# Transform New Samples
#########################################################

X_new = np.array([
    [2.0, 2.1],
    [1.2, 1.3],
    [3.0, 3.2]
])

new_projection = pca.transform(X_new)

print("\n")
print("=" * 60)
print("New Samples")
print("=" * 60)

print(X_new)

print("\nProjected Samples")

print(new_projection)


#########################################################
# Recover New Samples
#########################################################

new_reconstructed = pca.inverse_transform(
    new_projection
)

print("\n")
print("=" * 60)
print("Recovered Samples")
print("=" * 60)

print(new_reconstructed)
```

---

## Expected Output (Approximate)

```text
============================================================
Mean Vector
============================================================
[1.81 1.91]

============================================================
Eigenvalues
============================================================
[1.28402771 0.0490834 ]

============================================================
Principal Components
============================================================
[[ 0.6778734 ]
 [ 0.73517866]]

============================================================
Explained Variance Ratio
============================================================
[0.96318131]

============================================================
Cumulative Explained Variance
============================================================
[0.96318131]

============================================================
Original Shape
============================================================
(10, 2)

Reduced Shape
============================================================
(10, 1)

Reconstructed Shape
============================================================
(10, 2)

============================================================
Reconstruction Error
============================================================
MSE : 0.02
```

### What this example demonstrates

* Fits PCA on the dataset.
* Reduces the data from **2 dimensions to 1 principal component**.
* Computes and prints:

  * Mean vector
  * Eigenvalues
  * Principal components
  * Explained variance ratio
  * Cumulative explained variance
* Reconstructs the original data from the reduced representation.
* Calculates the **Mean Squared Reconstruction Error (MSE)**.
* Projects and reconstructs **new unseen samples**.

The remaining piece is **Part 3**, which provides the complete GitHub README mathematical formulation, including covariance matrices, eigenvalues, eigenvectors, principal components, explained variance, the PCA algorithm, and complexity analysis.

# Principal Component Analysis (PCA)

## Mathematical Formulation

### 1. What is PCA?

Principal Component Analysis (PCA) is an **unsupervised machine learning algorithm** used for **dimensionality reduction**. It transforms the original features into a new set of orthogonal features called **Principal Components**, which capture the maximum variance in the data.

---

# 2. Mean Centering

The first step in PCA is to center the dataset by subtracting the mean of each feature.

```text
           n
μ = (1/n) Σ Xi
          i=1
```

Mean-centered data

```text
Xcentered = X − μ
```

Where

- `X` = Original feature matrix
- `μ` = Mean vector

---

# 3. Covariance Matrix

The covariance matrix measures how features vary together.

```text
                    T
          1
Cov = -------- Xcentered Xcentered
        n - 1
```

Where

- `Cov` = Covariance Matrix
- `Xcenteredᵀ` = Transpose of centered data
- `n` = Number of samples

---

# 4. Eigenvalues and Eigenvectors

PCA computes the eigenvalues and eigenvectors of the covariance matrix.


::contentReference[oaicite:0]{index=0}


The eigenvalue equation is

```text
Av = λv
```

Where

- `A` = Covariance Matrix
- `v` = Eigenvector
- `λ` = Eigenvalue

Each eigenvector represents a principal direction.

Each eigenvalue represents the amount of variance captured along that direction.

---

# 5. Sorting Eigenvalues

Arrange eigenvalues in descending order.

```text
λ₁ ≥ λ₂ ≥ λ₃ ≥ ... ≥ λn
```

The corresponding eigenvectors are rearranged in the same order.

The first eigenvector corresponds to the first principal component.

---

# 6. Principal Components

Select the first `k` eigenvectors.

```text
W = [v₁ v₂ ... vk]
```

Where

- `W` = Projection matrix
- `k` = Number of principal components

---

# 7. Data Projection

Project the original data onto the new feature space.

```text
Z = Xcentered W
```

Where

- `Z` = Reduced dataset
- `W` = Principal component matrix

---

# 8. Explained Variance

The variance explained by each principal component is

```text
Explained Variance

=

λi

----------------

Σ λ
```

The explained variance ratio indicates how much information is retained by each principal component.

---

# 9. Cumulative Explained Variance

```text
Cumulative Variance

=

Σ Explained Variance Ratio
```

This helps determine the number of principal components required.

Example

```text
PC1 = 70%

PC2 = 20%

PC3 = 8%

PC4 = 2%
```

Cumulative

```text
PC1

70%

PC1 + PC2

90%

PC1 + PC2 + PC3

98%
```

---

# 10. Reconstruction

Original data can be approximately recovered.

```text
Xreconstructed

=

ZWᵀ + μ
```

Where

- `Z` = Reduced data
- `Wᵀ` = Transpose of principal components
- `μ` = Mean vector

---

# 11. Complete PCA Algorithm

### Step 1

Load Dataset

```text
X
```

---

### Step 2

Compute Mean

```text
μ
```

---

### Step 3

Center the Data

```text
Xcentered = X − μ
```

---

### Step 4

Compute Covariance Matrix

```text
                    T
          1
Cov = -------- Xcentered Xcentered
        n − 1
```

---

### Step 5

Compute Eigenvalues and Eigenvectors

```text
Av = λv
```

---

### Step 6

Sort Eigenvalues

```text
Largest → Smallest
```

---

### Step 7

Select Top K Eigenvectors

```text
W = [v₁,v₂,...,vk]
```

---

### Step 8

Project Data

```text
Z = XcenteredW
```

---

### Step 9

Use

```text
Z
```

for visualization, machine learning, or dimensionality reduction.

---

# Variables Summary

| Symbol | Meaning |
|---------|---------|
| `X` | Original Feature Matrix |
| `μ` | Mean Vector |
| `Cov` | Covariance Matrix |
| `λ` | Eigenvalue |
| `v` | Eigenvector |
| `W` | Principal Component Matrix |
| `Z` | Reduced Dataset |
| `k` | Number of Principal Components |

---

# Time Complexity

## Mean Calculation

```text
O(ND)
```

---

## Covariance Matrix

```text
O(ND²)
```

---

## Eigenvalue Decomposition

```text
O(D³)
```

---

## Projection

```text
O(NDK)
```

---

## Overall Complexity

```text
O(D³ + ND²)
```

Where

- `N` = Number of samples
- `D` = Number of features
- `K` = Number of principal components

---

# Space Complexity

Covariance Matrix

```text
O(D²)
```

Projected Data

```text
O(NK)
```

Overall

```text
O(D² + NK)
```

---

# Advantages

- Reduces dimensionality.
- Removes correlated features.
- Speeds up machine learning algorithms.
- Reduces storage requirements.
- Helps visualize high-dimensional datasets.
- Reduces noise while preserving important information.

---

# Disadvantages

- Difficult to interpret transformed features.
- Assumes linear relationships.
- Sensitive to feature scaling.
- Information may be lost when reducing dimensions.
- Principal components may not have physical meaning.

---

# Applications

- Data Compression
- Image Compression
- Face Recognition
- Feature Extraction
- Noise Reduction
- Data Visualization
- Preprocessing for Machine Learning
- Bioinformatics
- Finance
- Computer Vision

---

# Workflow

```text
                Start
                   │
                   ▼
            Load Dataset (X)
                   │
                   ▼
          Compute Mean Vector (μ)
                   │
                   ▼
         Center the Dataset (X − μ)
                   │
                   ▼
      Compute Covariance Matrix
                   │
                   ▼
Compute Eigenvalues & Eigenvectors
                   │
                   ▼
      Sort Eigenvalues Descending
                   │
                   ▼
 Select Top K Eigenvectors (W)
                   │
                   ▼
      Project Data (Z = XW)
                   │
                   ▼
 Reduced Dimensional Dataset
                   │
                   ▼
        Machine Learning / Visualization


