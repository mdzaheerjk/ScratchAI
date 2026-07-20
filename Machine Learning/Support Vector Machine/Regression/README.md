Below is a **Support Vector Regression (SVR)** implementation **from scratch using only NumPy**. It uses **Gradient Descent** and the **ε-insensitive loss function**.

---

# `svr.py`

```python
import numpy as np


class SupportVectorRegression:
    def __init__(
        self,
        learning_rate=0.001,
        lambda_param=0.01,
        epsilon=0.1,
        epochs=1000
    ):
        """
        Parameters
        ----------
        learning_rate : Learning rate
        lambda_param  : Regularization parameter
        epsilon       : Epsilon-insensitive margin
        epochs        : Number of iterations
        """

        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.epsilon = epsilon
        self.epochs = epochs

        self.weights = None
        self.bias = None

    def fit(self, X, y):

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):

            for idx, x_i in enumerate(X):

                prediction = np.dot(x_i, self.weights) + self.bias

                error = y[idx] - prediction

                if abs(error) <= self.epsilon:

                    dw = 2 * self.lambda_param * self.weights
                    db = 0

                elif error > self.epsilon:

                    dw = (
                        2 * self.lambda_param * self.weights
                        - x_i
                    )

                    db = -1

                else:

                    dw = (
                        2 * self.lambda_param * self.weights
                        + x_i
                    )

                    db = 1

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

    def predict(self, X):

        return np.dot(X, self.weights) + self.bias

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

X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6]
])

y = np.array([
    3,
    5,
    7,
    9,
    11,
    13
])

model = SupportVectorRegression(
    learning_rate=0.001,
    lambda_param=0.01,
    epsilon=0.1,
    epochs=1000
)

model.fit(X, y)

predictions = model.predict(X)

print("Weights :", model.weights)
print("Bias :", model.bias)

print("Predictions :", predictions)

print("MSE :", model.mse(y, predictions))
print("R2 Score :", model.r2_score(y, predictions))
```

---

# README Mathematical Formulation


# Support Vector Regression (SVR)

## Mathematical Formulation

### 1. Prediction Equation

For a single feature:

```text
y = wx + b
```

For multiple features:

```text
ŷ = XW + b
```

Where:
- `X` = Feature matrix
- `W` = Weight vector
- `b` = Bias
- `ŷ` = Predicted value

---

### 2. Regression Hyperplane

The regression function is

```text
ŷ = XW + b
```

Instead of maximizing the margin between classes, SVR fits a function inside an **ε-insensitive tube**.

---

### 3. ε-Insensitive Loss

Errors smaller than ε are ignored.

```text
Loss = max(0, |y - ŷ| - ε)
```

where

```text
ε = epsilon
```

---

### 4. Objective Function

SVR minimizes

```text
                   n
J = λ||W||² + C Σ max(0, |yi - ŷi| - ε)
                  i=1
```

Where

- `λ` = Regularization parameter
- `C` = Penalty parameter
- `ε` = Width of the insensitive tube

---

### 5. Gradient (Inside ε-Tube)

If

```text
|yi - ŷi| ≤ ε
```

then

```text
dW = 2λW

db = 0
```

---

### 6. Gradient (Prediction Too Small)

If

```text
yi - ŷi > ε
```

then

```text
dW = 2λW − Xi

db = −1
```

---

### 7. Gradient (Prediction Too Large)

If

```text
yi - ŷi < -ε
```

then

```text
dW = 2λW + Xi

db = 1
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

```text
ŷ = XW + b
```

Unlike SVM Classification, SVR predicts **continuous values** rather than class labels.

---

## Variables Summary

| Symbol | Meaning |
|---------|---------|
| `X` | Feature matrix |
| `W` | Weight vector |
| `b` | Bias |
| `ŷ` | Predicted value |
| `y` | Actual value |
| `ε` | Epsilon-insensitive margin |
| `λ` | Regularization parameter |
| `C` | Penalty parameter |
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

Compute prediction

```text
ŷ = XW + b
```

### Step 3

Compute error

```text
Error = y − ŷ
```

### Step 4

If

```text
|Error| ≤ ε
```

```text
dW = 2λW

db = 0
```

Else if

```text
Error > ε
```

```text
dW = 2λW − Xi

db = −1
```

Else

```text
dW = 2λW + Xi

db = 1
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
- `N` = Number of samples
- `D` = Number of features
````

> **Note:** This is an educational implementation of **Linear SVR using gradient descent**. Production-grade SVR (such as `sklearn.svm.SVR`) is typically trained by solving a convex quadratic optimization problem (e.g., using Sequential Minimal Optimization or similar optimization techniques) rather than simple gradient descent.
