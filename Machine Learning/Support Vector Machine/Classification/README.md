Below is a **Support Vector Machine (SVM) Classification** implementation **from scratch using only NumPy**. It trains a **Linear SVM** using **Gradient Descent** and **Hinge Loss**.

---

# `svm.py`

```python
import numpy as np


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, epochs=1000):
        """
        Parameters:
        learning_rate : Learning rate for gradient descent
        lambda_param  : Regularization parameter
        epochs        : Number of training iterations
        """

        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs

        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Train the SVM model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)
            Labels should be 0 and 1 or -1 and 1
        """

        n_samples, n_features = X.shape

        # Convert labels to -1 and +1
        y = np.where(y <= 0, -1, 1)

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.epochs):

            for idx, x_i in enumerate(X):

                condition = y[idx] * (np.dot(x_i, self.weights) + self.bias) >= 1

                if condition:
                    dw = 2 * self.lambda_param * self.weights
                    db = 0

                else:
                    dw = (
                        2 * self.lambda_param * self.weights
                        - y[idx] * x_i
                    )

                    db = -y[idx]

                # Update Parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

    def decision_function(self, X):
        """
        Compute decision values.
        """
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        """
        Predict class labels.
        """
        linear_output = self.decision_function(X)
        return np.sign(linear_output)

    def accuracy(self, y_true, y_pred):
        """
        Calculate Accuracy.
        """
        y_true = np.where(y_true <= 0, -1, 1)
        return np.mean(y_true == y_pred)
```

---

# Example Usage

```python
import numpy as np

X = np.array([
    [2, 3],
    [1, 2],
    [2, 1],
    [8, 9],
    [9, 8],
    [10, 9]
])

y = np.array([
    -1,
    -1,
    -1,
     1,
     1,
     1
])

model = SVM(
    learning_rate=0.001,
    lambda_param=0.01,
    epochs=1000
)

model.fit(X, y)

predictions = model.predict(X)

print("Weights :", model.weights)
print("Bias :", model.bias)

print("Predictions :", predictions)

print("Accuracy :", model.accuracy(y, predictions))
```

---

# README Mathematical Formulation


# Support Vector Machine (SVM) Classification

## Mathematical Formulation

### 1. Linear Decision Function

For a single feature:

```text
f(x) = wx + b
```

For multiple features:

```text
f(x) = XW + b
```

Prediction:

```text
ŷ = sign(XW + b)
```

Where:
- `X` = Feature matrix
- `W` = Weight vector
- `b` = Bias
- `ŷ` = Predicted class

---

### 2. Hyperplane

The decision boundary is

```text
XW + b = 0
```

The support vectors satisfy

```text
XW + b = ±1
```

---

### 3. Margin

The margin is the distance between the two support vectors.

```text
Margin = 2 / ||W||
```

A larger margin generally leads to better generalization.

---

### 4. Hinge Loss

The hinge loss penalizes points that are inside the margin or misclassified.

```text
Loss = max(0, 1 - y(XW + b))
```

Where

```text
y ∈ {-1, +1}
```

If

```text
y(XW + b) ≥ 1
```

then

```text
Loss = 0
```

Otherwise

```text
Loss > 0
```

---

### 5. Cost Function

The objective function is

```text
                 n
J = λ||W||² + Σ max(0, 1 - yi(XiW + b))
                i=1
```

Where

- `λ` = Regularization parameter
- `||W||²` = L2 Regularization

---

### 6. Gradient (Correct Classification)

If

```text
yi(XiW + b) ≥ 1
```

then

```text
dW = 2λW

db = 0
```

---

### 7. Gradient (Misclassification)

If

```text
yi(XiW + b) < 1
```

then

```text
dW = 2λW − yiXi

db = −yi
```

---

### 8. Gradient Descent Update

Update weights

```text
W = W − αdW
```

Update bias

```text
b = b − αdb
```

Where

- `α` = Learning rate

---

### 9. Prediction

Compute

```text
z = XW + b
```

Then classify

```text
If z ≥ 0

Predict +1

Else

Predict −1
```

---

## Variables Summary

| Symbol | Meaning |
|---------|---------|
| `X` | Feature matrix |
| `W` | Weight vector |
| `b` | Bias |
| `y` | Actual label (-1 or +1) |
| `ŷ` | Predicted label |
| `z` | Decision function (`XW + b`) |
| `λ` | Regularization parameter |
| `α` | Learning rate |
| `||W||` | Euclidean norm of weights |

---

## Gradient Descent Algorithm

### Step 1

Initialize

```text
W = 0

b = 0
```

### Step 2

Compute

```text
z = XW + b
```

### Step 3

Check the margin

```text
yi × z ≥ 1 ?
```

### Step 4

If correctly classified

```text
dW = 2λW

db = 0
```

Otherwise

```text
dW = 2λW − yiXi

db = −yi
```

### Step 5

Update parameters

```text
W = W − αdW

b = b − αdb
```

Repeat until all epochs are completed.

---

## Time Complexity

| Operation | Complexity |
|-----------|------------|
| Training | O(E × N × D) |
| Prediction | O(N × D) |

Where:
- `E` = Number of epochs
- `N` = Number of training samples
- `D` = Number of features
````

This is a complete **Linear SVM classifier from scratch using NumPy**, including the implementation, training loop, prediction logic, and README-ready mathematical derivation.
