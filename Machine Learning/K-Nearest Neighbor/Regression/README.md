Below is a **K-Nearest Neighbors (KNN) Regression** implementation **from scratch using only NumPy**. Unlike Linear Regression, KNN Regression **does not learn weights or a model**. It predicts the target value by averaging the target values of the **k nearest neighbors**.

---

# `knn_regressor.py`

```python
import numpy as np


class KNNRegressor:

    def __init__(self, k=3):
        """
        Parameters
        ----------
        k : Number of nearest neighbors
        """
        self.k = k

    def fit(self, X, y):
        """
        Store training data.
        """
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        """
        Compute Euclidean distance.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        """
        Predict target values.
        """
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):

        # Compute distances
        distances = [
            self.euclidean_distance(x, x_train)
            for x_train in self.X_train
        ]

        # Indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]

        # Target values of nearest neighbors
        k_values = [self.y_train[i] for i in k_indices]

        # Average target value
        prediction = np.mean(k_values)

        return prediction

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def r2_score(self, y_true, y_pred):

        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)

        return 1 - (ss_residual / ss_total)
```

---

# Example Usage

```python
import numpy as np

X_train = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6]
])

y_train = np.array([
    2,
    4,
    6,
    8,
    10,
    12
])

X_test = np.array([
    [2.5],
    [5.5]
])

model = KNNRegressor(k=3)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Predictions :", predictions)

print("MSE :", model.mse(np.array([5,11]), predictions))
print("R2 Score :", model.r2_score(np.array([5,11]), predictions))
```

Expected Output

```text
Predictions :
[4.0 10.0]
```

---

# README Mathematical Formulation


# K-Nearest Neighbors (KNN) Regression

## Mathematical Formulation

### 1. Euclidean Distance

For two points

```text
x = (x₁,x₂,...,xn)

y = (y₁,y₂,...,yn)
```

the Euclidean distance is

```text
                ___________________________
               / n
Distance = √  Σ (xi - yi)²
              i=1
```

---

### 2. Compute Distance

For every test sample

```text
Distance(x, Xi)
```

is computed for every training sample.

---

### 3. Sort Distances

Arrange all distances in ascending order.

```text
d₁ ≤ d₂ ≤ d₃ ≤ ...
```

---

### 4. Select K Nearest Neighbors

Choose

```text
K nearest samples
```

Example

```text
K = 5
```

---

### 5. Prediction

For Regression,

```text
                 k
                Σ yi
Prediction = ----------
                  k
```

or

```text
ŷ = Mean(y₁,y₂,...,yk)
```

The predicted value is the average of the target values of the K nearest neighbors.

---

## Variables Summary

| Symbol | Meaning |
|---------|---------|
| `X` | Training Features |
| `x` | Test Sample |
| `k` | Number of Neighbors |
| `d` | Euclidean Distance |
| `y` | Actual Target |
| `ŷ` | Predicted Target |

---

## Prediction Algorithm

### Step 1

Store

```text
X_train

y_train
```

---

### Step 2

For every test sample compute

```text
Distance(x,Xi)
```

using

```text
                ___________________________
               / n
Distance = √  Σ (xi - yi)²
              i=1
```

---

### Step 3

Sort distances

```text
Smallest → Largest
```

---

### Step 4

Choose

```text
K nearest neighbors
```

---

### Step 5

Compute prediction

```text
                 k
                Σ yi
Prediction = ----------
                  k
```

or

```text
ŷ = Mean(y₁,y₂,...,yk)
```

Return the average target value.

---

## Time Complexity

| Operation | Complexity |
|-----------|------------|
| Training | O(1) |
| Prediction | O(N × D) per sample |
| Sorting | O(N log N) |

Where

- `N` = Number of training samples
- `D` = Number of features
- `k` = Number of nearest neighbors


### Notes

* **No training phase** (the algorithm simply stores the training data).
* **No gradient descent**.
* **No loss function during training**.
* **Prediction is based on the average of the target values of the `k` nearest neighbors**.
* Common evaluation metrics for KNN Regression are **Mean Squared Error (MSE)** and **R² Score**.
