Below is a **Logistic Regression implementation from scratch using only NumPy**. It uses **Gradient Descent** to learn the model parameters.

```python
import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    # Sigmoid activation function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Train the model
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.epochs):

            # Linear equation (mx + c)
            linear = np.dot(X, self.weights) + self.bias

            # Apply sigmoid
            y_pred = self.sigmoid(linear)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    # Predict probabilities
    def predict_proba(self, X):
        linear = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear)

    # Predict class labels
    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)

    # Accuracy
    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
```

---

# Example Usage

```python
import numpy as np

# Sample dataset
X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6]
])

y = np.array([0, 0, 0, 1, 1, 1])

# Create model
model = LogisticRegression(learning_rate=0.1, epochs=1000)

# Train model
model.fit(X, y)

# Predict
predictions = model.predict(X)

print("Weights:", model.weights)
print("Bias:", model.bias)
print("Predictions:", predictions)

print("Accuracy:", model.accuracy(y, predictions))
```

Example output (values may vary slightly):

```
Weights: [2.18]
Bias: -7.01

Predictions:
[0 0 0 1 1 1]

Accuracy: 1.0
```
## Mathematical Formulation

### 1. Linear Equation

For a single feature:

```text
z = mx + c
```

For multiple features:

```text
z = XW + b
```

Where:
- `X` = Feature matrix
- `W` = Weight vector
- `b` = Bias (intercept)
- `z` = Linear combination of inputs

---

### 2. Sigmoid Function

The sigmoid function converts the linear output into a probability between 0 and 1.

```text
             1
σ(z) = ---------------
        1 + e^(-z)
```

or equivalently,

```text
               1
ŷ = -------------------------
      1 + e^(-(XW + b))
```

Where:
- `ŷ` = Predicted probability
- `σ(z)` = Sigmoid function

---

### 3. Binary Cross-Entropy Loss (Log Loss)

```text
            n
J = -(1/n) Σ [ yi log(ŷi) + (1 - yi) log(1 - ŷi) ]
           i=1
```

Where:
- `J` = Cost function
- `yi` = Actual class label
- `ŷi` = Predicted probability

---

### 4. Gradient of Weights

```text
dW = (1 / n) × Xᵀ × (ŷ - y)
```

---

### 5. Gradient of Bias

```text
db = (1 / n) × Σ(ŷ - y)
```

---

### 6. Gradient Descent Update

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

### 7. Classification Rule

Convert predicted probabilities into class labels using a threshold (typically 0.5).

```text
ŷ ≥ 0.5  →  Class 1

ŷ < 0.5  →  Class 0
```

---

## Variables Summary

| Symbol | Meaning |
|--------|---------|
| `X` | Feature matrix |
| `y` | Actual class labels |
| `ŷ` | Predicted probabilities |
| `z` | Linear combination (`XW + b`) |
| `W` | Weight vector |
| `b` | Bias (intercept) |
| `n` | Number of training samples |
| `α` | Learning rate |
| `dW` | Gradient of weights |
| `db` | Gradient of bias |

---

## Gradient Descent Algorithm

1. Compute the linear equation

```text
z = XW + b
```

2. Apply the sigmoid function

```text
             1
ŷ = ---------------
      1 + e^(-z)
```

3. Compute Binary Cross-Entropy Loss

```text
            n
J = -(1/n) Σ [ yi log(ŷi) + (1 - yi) log(1 - ŷi) ]
           i=1
```

4. Compute gradients

```text
dW = (1 / n) × Xᵀ × (ŷ - y)

db = (1 / n) × Σ(ŷ - y)
```

5. Update parameters

```text
W = W - α × dW

b = b - α × db
```

6. Predict class labels

```text
If ŷ ≥ 0.5 → Predict 1
Else       → Predict 0
```

Repeat these steps until the specified number of epochs is reached or the model converges.
