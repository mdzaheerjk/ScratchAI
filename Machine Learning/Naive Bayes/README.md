Since you're building ML algorithms **from scratch**, the most common implementation is **Gaussian Naive Bayes**, which works for continuous numerical features.

---

# Gaussian Naive Bayes from Scratch (NumPy)

```python
import numpy as np


class GaussianNaiveBayes:

    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        """
        Train Gaussian Naive Bayes
        """

        n_samples, n_features = X.shape

        self.classes = np.unique(y)

        n_classes = len(self.classes)

        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)

        for idx, cls in enumerate(self.classes):

            X_c = X[y == cls]

            self.mean[idx, :] = np.mean(X_c, axis=0)

            self.var[idx, :] = np.var(X_c, axis=0)

            self.priors[idx] = X_c.shape[0] / float(n_samples)

    def gaussian_pdf(self, class_idx, x):

        mean = self.mean[class_idx]

        var = self.var[class_idx]

        numerator = np.exp(-((x - mean) ** 2) / (2 * var))

        denominator = np.sqrt(2 * np.pi * var)

        return numerator / denominator

    def predict(self, X):

        predictions = []

        for x in X:

            posteriors = []

            for idx, cls in enumerate(self.classes):

                prior = np.log(self.priors[idx])

                conditional = np.sum(
                    np.log(self.gaussian_pdf(idx, x))
                )

                posterior = prior + conditional

                posteriors.append(posterior)

            predictions.append(
                self.classes[np.argmax(posteriors)]
            )

        return np.array(predictions)

    def accuracy(self, y_true, y_pred):

        return np.mean(y_true == y_pred)
```

---

# Example

```python
import numpy as np

X = np.array([
    [1.0,2.1],
    [1.2,1.9],
    [0.8,2.0],
    [5.0,6.1],
    [5.2,5.9],
    [4.8,6.2]
])

y = np.array([
    0,
    0,
    0,
    1,
    1,
    1
])

model = GaussianNaiveBayes()

model.fit(X,y)

predictions = model.predict(X)

print("Predictions :",predictions)

print("Accuracy :",model.accuracy(y,predictions))
```

---

# README Mathematical Formulation


# Gaussian Naive Bayes

## Mathematical Formulation

### 1. Bayes Theorem

```text
                 P(X|Y) P(Y)
P(Y|X) = ----------------------------
                  P(X)
```

Where

- `P(Y|X)` = Posterior Probability
- `P(X|Y)` = Likelihood
- `P(Y)` = Prior Probability
- `P(X)` = Evidence

---

### 2. Naive Assumption

Naive Bayes assumes all features are conditionally independent.

```text
P(X|Y)

= P(x₁,x₂,...,xn|Y)

= P(x₁|Y) × P(x₂|Y) × ... × P(xn|Y)
```

---

### 3. Posterior Probability

```text
Posterior

= Prior × Likelihood
```

or

```text
P(Y|X)

∝ P(Y)

× Π P(xi|Y)
```

---

### 4. Prior Probability

```text
P(Y)

= Number of samples in class Y

  ----------------------------

      Total number of samples
```

---

### 5. Gaussian Probability Density Function

For continuous features,

```text
                     -(x-μ)²
                   ------------
                      2σ²
P(x|Y) = (1 / √(2πσ²)) e
```

Where

- `μ` = Mean
- `σ²` = Variance

---

### 6. Log Probability

To avoid numerical underflow,

```text
log(P(Y|X))

=

log(P(Y))

+

Σ log(P(xi|Y))
```

---

### 7. Prediction

For every class,

```text
Posterior

=

log(Prior)

+

Σ log(Likelihood)
```

Choose the class having the maximum posterior.

```text
Prediction

=

argmax Posterior
```

---

## Variables Summary

| Symbol | Meaning |
|---------|---------|
| `X` | Feature Matrix |
| `Y` | Target Class |
| `μ` | Mean |
| `σ²` | Variance |
| `P(Y)` | Prior Probability |
| `P(X|Y)` | Likelihood |
| `P(Y|X)` | Posterior Probability |

---

## Training Algorithm

### Step 1

Separate data according to each class.

### Step 2

For every class compute

```text
Mean (μ)

Variance (σ²)

Prior Probability
```

### Step 3

Store

```text
μ

σ²

Prior
```

---

## Prediction Algorithm

### Step 1

Compute Gaussian Probability

```text
                     -(x-μ)²
                   ------------
                      2σ²
P(x|Y) = (1 / √(2πσ²)) e
```

### Step 2

Compute Posterior

```text
Posterior

=

log(Prior)

+

Σ log(Likelihood)
```

### Step 3

Choose

```text
Prediction

=

argmax Posterior
```

---

## Time Complexity

| Operation | Complexity |
|-----------|------------|
| Training | O(N × D) |
| Prediction | O(C × D) per sample |

Where

- `N` = Number of training samples
- `D` = Number of features
- `C` = Number of classes
````

This implementation corresponds to **Gaussian Naive Bayes**, the standard version for continuous numerical features. It estimates the **mean**, **variance**, and **prior probability** for each class during training, then uses **Bayes' theorem** with the **Gaussian probability density function** to classify new samples.
