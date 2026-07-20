Here's a **Linear Regression algorithm from scratch using only NumPy** (no scikit-learn). It uses **Gradient Descent** to learn the parameters.

```python
import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Train the Linear Regression model using Gradient Descent.

        Parameters:
        X : numpy array of shape (n_samples, n_features)
        y : numpy array of shape (n_samples,)
        """

        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.epochs):

            # Predictions
            y_pred = np.dot(X, self.weights) + self.bias

            # Compute gradients
            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Predict target values.
        """
        return np.dot(X, self.weights) + self.bias

    def mse(self, y_true, y_pred):
        """
        Mean Squared Error
        """
        return np.mean((y_true - y_pred) ** 2)

    def r2_score(self, y_true, y_pred):
        """
        R² Score
        """
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)
```

---

## Example Usage

```python
import numpy as np

# Sample dataset
X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5]
])

y = np.array([3, 5, 7, 9, 11])

# Create model
model = LinearRegression(learning_rate=0.01, epochs=1000)

# Train
model.fit(X, y)

# Predictions
predictions = model.predict(X)

print("Weights:", model.weights)
print("Bias:", model.bias)
print("Predictions:", predictions)

print("MSE:", model.mse(y, predictions))
print("R2 Score:", model.r2_score(y, predictions))
```

### Expected Output (approximately)

```
Weights: [2.]
Bias: 1.

Predictions:
[ 3.
  5.
  7.
  9.
 11.]

MSE: 0.0
R2 Score: 1.0
```

## Mathematical Formulation

### 1. Linear Equation (Prediction)

For a single feature:

```text
y = mx + c
```

For multiple features:

```text
ŷ = XW + b
```

Where:
- `X` = Feature matrix
- `W` = Weight vector
- `b` = Bias (intercept)
- `ŷ` = Predicted values

---

### 2. Mean Squared Error (Cost Function)

```text
              n
MSE = (1 / n) Σ (yi - ŷi)²
             i=1
```

Where:
- `n` = Number of training samples
- `yi` = Actual value
- `ŷi` = Predicted value

---

### 3. Gradient of Weights

```text
dW = (2 / n) × Xᵀ × (ŷ - y)
```

---

### 4. Gradient of Bias

```text
db = (2 / n) × Σ(ŷ - y)
```

---

### 5. Gradient Descent Update

**Update Weights**

```text
W = W - α × dW
```

**Update Bias**

```text
b = b - α × db
```

Where:
- `W` = Weight vector
- `b` = Bias
- `α` = Learning rate
- `dW` = Gradient of weights
- `db` = Gradient of bias

---

## Variables Summary

| Symbol | Meaning |
|--------|---------|
| `X` | Feature matrix |
| `y` | Actual target values |
| `ŷ` | Predicted target values |
| `W` | Weight vector |
| `b` | Bias (intercept) |
| `n` | Number of training samples |
| `α` | Learning rate |
| `dW` | Gradient of weights |
| `db` | Gradient of bias |

---

## Gradient Descent Algorithm

1. Compute predictions

```text
ŷ = XW + b
```

2. Compute the Mean Squared Error (MSE)

```text
              n
MSE = (1 / n) Σ (yi - ŷi)²
             i=1
```

3. Compute gradients

```text
dW = (2 / n) × Xᵀ × (ŷ - y)

db = (2 / n) × Σ(ŷ - y)
```

4. Update parameters

```text
W = W - α × dW

b = b - α × db
```

Repeat these steps until the specified number of epochs is reached or the model converges.

