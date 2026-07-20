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

