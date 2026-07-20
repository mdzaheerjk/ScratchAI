Below is a **K-Nearest Neighbors (KNN) Classification** implementation **from scratch using only NumPy**. Unlike Linear Regression or SVM, **KNN is a lazy learning algorithm**—it does **not** train a model or use gradient descent. Instead, it stores the training data and predicts labels based on the nearest neighbors.

---

# `knn_classifier.py`

```python
import numpy as np


class KNNClassifier:

    def __init__(self, k=3):
        """
        Parameters
        ----------
        k : Number of nearest neighbors
        """
        self.k = k

    def fit(self, X, y):
        """
        Store the training data.
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
        Predict class labels.
        """
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):

        # Compute distance from x to every training sample
        distances = [
            self.euclidean_distance(x, x_train)
            for x_train in self.X_train
        ]

        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]

        # Get labels of k nearest neighbors
        k_labels = [self.y_train[i] for i in k_indices]

        # Majority Voting
        unique_labels, counts = np.unique(
            k_labels,
            return_counts=True
        )

        prediction = unique_labels[np.argmax(counts)]

        return prediction

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
```

---

# Example Usage

```python
import numpy as np

X_train = np.array([
    [1,2],
    [2,3],
    [3,3],
    [6,7],
    [7,8],
    [8,8]
])

y_train = np.array([
    0,
    0,
    0,
    1,
    1,
    1
])

X_test = np.array([
    [2,2],
    [7,7]
])

model = KNNClassifier(k=3)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(predictions)
```

Expected Output

```text
[0 1]
```

---

# README Mathematical Formulation


# K-Nearest Neighbors (KNN) Classification

## Mathematical Formulation

### 1. Euclidean Distance

For two points

```text
x = (x₁,x₂,...,xn)

y = (y₁,y₂,...,yn)
```

The Euclidean distance is

```text
                ___________________________
               / n
Distance = √  Σ (xi - yi)²
              i=1
```

---

### 2. Distance Computation

For every test sample,

```text
Distance(x, Xi)
```

is computed for every training sample.

---

### 3. Sort Distances

Arrange the distances in ascending order.

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
K = 3
```

Take the 3 smallest distances.

---

### 5. Majority Voting

For Classification,

```text
Prediction

=

Most Frequent Label
```

or

```text
ŷ = mode(y₁,y₂,...,yk)
```

---

## Variables Summary

| Symbol | Meaning |
|---------|---------|
| `X` | Training Features |
| `x` | Test Sample |
| `k` | Number of Neighbors |
| `d` | Distance |
| `ŷ` | Predicted Class |
| `y` | Actual Class |

---

## Prediction Algorithm

### Step 1

Store training data

```text
X_train

y_train
```

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

### Step 3

Sort all distances

```text
Smallest → Largest
```

### Step 4

Choose

```text
K nearest neighbors
```

### Step 5

Perform Majority Voting

```text
ŷ = mode(y₁,y₂,...,yk)
```

Return the most frequent class.

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
````

---

### Notes

* **No gradient descent**
* **No loss function**
* **No weight updates**
* Training simply stores the dataset.
* Prediction is based on the **distance** between the test point and all training points, followed by **majority voting** among the `k` nearest neighbors.
