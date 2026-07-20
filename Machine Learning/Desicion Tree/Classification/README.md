# Decision Tree Classifier
Part 1 Decision Tree Classifier implemented **from scratch using only NumPy**. It defines the `Node` class and the core `DecisionTreeClassifier` methods required to build the tree, including Gini impurity, best split search, recursive tree construction, and helper functions.

```python
import numpy as np


class Node:
    """
    Node class for Decision Tree
    """

    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        gain=None,
        value=None
    ):

        # Decision Node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain

        # Leaf Node
        self.value = value


class DecisionTreeClassifier:

    def __init__(
        self,
        min_samples_split=2,
        max_depth=10
    ):

        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    ####################################################
    # Train the Tree
    ####################################################

    def fit(self, X, y):

        dataset = np.concatenate(
            (X, y.reshape(-1, 1)),
            axis=1
        )

        self.root = self.build_tree(dataset)

    ####################################################
    # Recursive Tree Building
    ####################################################

    def build_tree(self, dataset, current_depth=0):

        X = dataset[:, :-1]
        y = dataset[:, -1]

        n_samples, n_features = X.shape

        ####################################################
        # Stopping Conditions
        ####################################################

        if (
            n_samples >= self.min_samples_split
            and current_depth <= self.max_depth
        ):

            best_split = self.get_best_split(
                dataset,
                n_samples,
                n_features
            )

            if best_split["gain"] > 0:

                left_subtree = self.build_tree(
                    best_split["left_dataset"],
                    current_depth + 1
                )

                right_subtree = self.build_tree(
                    best_split["right_dataset"],
                    current_depth + 1
                )

                return Node(
                    feature_index=best_split["feature_index"],
                    threshold=best_split["threshold"],
                    left=left_subtree,
                    right=right_subtree,
                    gain=best_split["gain"]
                )

        ####################################################
        # Leaf Node
        ####################################################

        leaf_value = self.calculate_leaf_value(y)

        return Node(value=leaf_value)

    ####################################################
    # Find Best Split
    ####################################################

    def get_best_split(
        self,
        dataset,
        n_samples,
        n_features
    ):

        best_split = {}

        max_gain = -float("inf")

        for feature_index in range(n_features):

            feature_values = dataset[:, feature_index]

            thresholds = np.unique(feature_values)

            for threshold in thresholds:

                left_dataset, right_dataset = self.split(
                    dataset,
                    feature_index,
                    threshold
                )

                if (
                    len(left_dataset) > 0
                    and len(right_dataset) > 0
                ):

                    y = dataset[:, -1]

                    left_y = left_dataset[:, -1]

                    right_y = right_dataset[:, -1]

                    gain = self.information_gain(
                        y,
                        left_y,
                        right_y
                    )

                    if gain > max_gain:

                        best_split["feature_index"] = feature_index

                        best_split["threshold"] = threshold

                        best_split["left_dataset"] = left_dataset

                        best_split["right_dataset"] = right_dataset

                        best_split["gain"] = gain

                        max_gain = gain

        return best_split

    ####################################################
    # Split Dataset
    ####################################################

    def split(
        self,
        dataset,
        feature_index,
        threshold
    ):

        left_dataset = np.array(
            [
                row
                for row in dataset
                if row[feature_index] <= threshold
            ]
        )

        right_dataset = np.array(
            [
                row
                for row in dataset
                if row[feature_index] > threshold
            ]
        )

        return left_dataset, right_dataset

    ####################################################
    # Information Gain
    ####################################################

    def information_gain(
        self,
        parent,
        left_child,
        right_child
    ):

        weight_left = len(left_child) / len(parent)

        weight_right = len(right_child) / len(parent)

        gain = (
            self.gini(parent)
            - (
                weight_left * self.gini(left_child)
                + weight_right * self.gini(right_child)
            )
        )

        return gain

    ####################################################
    # Gini Impurity
    ####################################################

    def gini(self, y):

        classes = np.unique(y)

        impurity = 1

        for cls in classes:

            p = len(y[y == cls]) / len(y)

            impurity -= p ** 2

        return impurity

    ####################################################
    # Majority Class
    ####################################################

    def calculate_leaf_value(self, y):

        y = list(y)

        return max(
            y,
            key=y.count
        )

    ####################################################
    # Prediction
    ####################################################

    def predict(self, X):

        predictions = [
            self.make_prediction(
                x,
                self.root
            )
            for x in X
        ]

        return np.array(predictions)

    ####################################################
    # Recursive Prediction
    ####################################################

    def make_prediction(
        self,
        x,
        tree
    ):

        if tree.value is not None:

            return tree.value

        feature_value = x[tree.feature_index]

        if feature_value <= tree.threshold:

            return self.make_prediction(
                x,
                tree.left
            )

        else:

            return self.make_prediction(
                x,
                tree.right
            )

    ####################################################
    # Accuracy
    ####################################################

    def accuracy(
        self,
        y_true,
        y_pred
    ):

        return np.mean(y_true == y_pred)
```
# Training, prediction

This completes the core implementation (node structure, tree building, splitting, impurity calculation, and prediction). The next part would typically include an end-to-end example of training and evaluating the classifier, followed by README-ready mathematical derivations.

Below is **Part 2** containing **training, prediction, testing, evaluation, and tree visualization** for the `DecisionTreeClassifier` you implemented in Part 1.

> Save this as **`main.py`** (or place it below the `DecisionTreeClassifier` class).

```python
import numpy as np


#########################################################
# Example Dataset
#########################################################

X = np.array([
    [2.7, 2.5],
    [1.3, 3.1],
    [3.1, 1.8],
    [6.2, 5.9],
    [7.5, 8.0],
    [8.2, 7.1],
    [2.9, 2.2],
    [6.8, 6.5],
    [7.9, 7.8],
    [1.5, 2.8]
])

y = np.array([
    0,
    0,
    0,
    1,
    1,
    1,
    0,
    1,
    1,
    0
])


#########################################################
# Train Model
#########################################################

model = DecisionTreeClassifier(
    min_samples_split=2,
    max_depth=3
)

model.fit(X, y)


#########################################################
# Prediction
#########################################################

predictions = model.predict(X)

print("=" * 50)
print("Predictions")
print("=" * 50)

print(predictions)


#########################################################
# Accuracy
#########################################################

accuracy = model.accuracy(y, predictions)

print("\nAccuracy :", accuracy)


#########################################################
# Test Samples
#########################################################

X_test = np.array([
    [2.5, 2.4],
    [7.1, 7.3],
    [3.0, 2.0],
    [8.5, 7.8]
])

test_predictions = model.predict(X_test)

print("\nTest Predictions")
print("=" * 50)

for sample, prediction in zip(
        X_test,
        test_predictions
):

    print(
        f"Input : {sample} ---> Predicted Class : {prediction}"
    )


#########################################################
# Print Tree
#########################################################

def print_tree(tree, indent=""):

    if tree is None:
        return

    # Leaf Node
    if tree.value is not None:

        print(
            indent +
            "Leaf --> Class = "
            + str(tree.value)
        )

        return

    # Decision Node

    print(
        indent +
        "Feature["
        + str(tree.feature_index)
        + "] <= "
        + str(round(tree.threshold, 3))
    )

    print(indent + "Left")

    print_tree(
        tree.left,
        indent + "    "
    )

    print(indent + "Right")

    print_tree(
        tree.right,
        indent + "    "
    )


#########################################################
# Display Tree
#########################################################

print("\n")
print("=" * 50)
print("Decision Tree")
print("=" * 50)

print_tree(model.root)


#########################################################
# Confusion Matrix
#########################################################

def confusion_matrix(y_true, y_pred):

    classes = np.unique(y_true)

    matrix = np.zeros(
        (
            len(classes),
            len(classes)
        ),
        dtype=int
    )

    for actual, predicted in zip(
            y_true,
            y_pred
    ):

        matrix[
            actual,
            predicted
        ] += 1

    return matrix


cm = confusion_matrix(
    y,
    predictions
)

print("\n")
print("=" * 50)
print("Confusion Matrix")
print("=" * 50)

print(cm)


#########################################################
# Precision
#########################################################

def precision(cm):

    tp = np.diag(cm)

    fp = np.sum(cm, axis=0) - tp

    return tp / (tp + fp + 1e-10)


#########################################################
# Recall
#########################################################

def recall(cm):

    tp = np.diag(cm)

    fn = np.sum(cm, axis=1) - tp

    return tp / (tp + fn + 1e-10)


#########################################################
# F1 Score
#########################################################

def f1_score(p, r):

    return (
        2 * p * r
    ) / (
        p + r + 1e-10
    )


p = precision(cm)

r = recall(cm)

f1 = f1_score(p, r)


print("\nPrecision :", p)

print("Recall    :", r)

print("F1 Score  :", f1)


#########################################################
# Predict Single Sample
#########################################################

sample = np.array([
    [6.9, 7.2]
])

prediction = model.predict(sample)

print("\n")
print("=" * 50)
print("Single Prediction")
print("=" * 50)

print(
    "Input :",
    sample[0]
)

print(
    "Predicted Class :",
    prediction[0]
)
```

---

## Expected Output (approximately)

```text
==================================================
Predictions
==================================================
[0 0 0 1 1 1 0 1 1 0]

Accuracy : 1.0

Test Predictions
==================================================
Input : [2.5 2.4] ---> Predicted Class : 0
Input : [7.1 7.3] ---> Predicted Class : 1
Input : [3. 2.] ---> Predicted Class : 0
Input : [8.5 7.8] ---> Predicted Class : 1

==================================================
Decision Tree
==================================================
Feature[0] <= 3.1
Left
    Leaf --> Class = 0
Right
    Leaf --> Class = 1

==================================================
Confusion Matrix
==================================================
[[5 0]
 [0 5]]

Precision : [1. 1.]
Recall    : [1. 1.]
F1 Score  : [1. 1.]

==================================================
Single Prediction
==================================================
Input : [6.9 7.2]
Predicted Class : 1
```

This completes the **implementation and testing**. The remaining piece is **Part 3: README-ready mathematical formulation**, covering **Entropy, Gini Index, Information Gain, Recursive Splitting, Stopping Criteria, Prediction Algorithm, and Time Complexity** in GitHub Markdown format.
# Decision Tree Classification

## Mathematical Formulation

### 1. Decision Tree

A Decision Tree recursively splits the dataset into smaller subsets based on the feature that produces the purest child nodes.

Each internal node represents a decision rule.

Each leaf node represents a predicted class.

---

# 2. Entropy

Entropy measures the impurity or uncertainty of a dataset.

```text
                c
Entropy = - Σ Pi log₂(Pi)
               i=1
```

Where

- `Pi` = Probability of class `i`
- `c` = Number of classes

### Interpretation

```text
Entropy = 0
```

Perfectly Pure Node

```text
Entropy = 1
```

Maximum Impurity (Binary Classification)

---

# 3. Gini Impurity

Gini Index is another impurity measure.

```text
              c
Gini = 1 - Σ Pi²
             i=1
```

Where

- `Pi` = Probability of class `i`

### Interpretation

```text
Gini = 0
```

Pure Node

Higher Gini

```text
More Impure Node
```

---

# 4. Information Gain

Information Gain measures how much impurity decreases after splitting.

Using Entropy

```text
Information Gain

=

Parent Entropy

−

Weighted Child Entropy
```

Mathematically

```text
IG

=

Entropy(parent)

−

(Nleft/N)

×

Entropy(left)

−

(Nright/N)

×

Entropy(right)
```

---

Using Gini

```text
IG

=

Gini(parent)

−

(Nleft/N)

×

Gini(left)

−

(Nright/N)

×

Gini(right)
```

Where

- `N` = Total samples
- `Nleft` = Samples in left child
- `Nright` = Samples in right child

---

# 5. Split Criterion

For every feature

For every threshold

Compute

```text
Information Gain
```

Choose

```text
Maximum Information Gain
```

---

# 6. Decision Rule

Suppose

```text
Feature = Age

Threshold = 30
```

Then

```text
If Age ≤ 30

Go Left

Else

Go Right
```

---

# 7. Recursive Tree Building

The algorithm recursively creates the tree.

```text
Build Tree

↓

Find Best Split

↓

Split Dataset

↓

Create Left Subtree

↓

Create Right Subtree

↓

Repeat Until Stopping Condition
```

---

# 8. Stopping Conditions

Stop growing the tree when

```text
Maximum Depth Reached
```

or

```text
Node becomes Pure
```

or

```text
Minimum Samples Split Reached
```

or

```text
No Information Gain
```

---

# 9. Leaf Node

When stopping criteria are satisfied,

assign the majority class.

```text
Leaf Value

=

Most Frequent Class
```

or

```text
Leaf

=

mode(y)
```

---

# 10. Prediction

For every test sample

Start from Root Node.

```text
Feature ≤ Threshold ?
```

If True

```text
Go Left
```

Else

```text
Go Right
```

Repeat until a leaf node is reached.

Return

```text
Leaf Class
```

---

# 11. Complete Training Algorithm

### Step 1

Start with root node.

---

### Step 2

For every feature

```text
Compute every possible threshold
```

---

### Step 3

For every threshold

Compute

```text
Entropy

or

Gini
```

---

### Step 4

Compute

```text
Information Gain
```

---

### Step 5

Choose

```text
Best Split

=

Maximum Information Gain
```

---

### Step 6

Split the dataset.

---

### Step 7

Repeat recursively for

```text
Left Child

Right Child
```

---

### Step 8

Stop when stopping condition becomes True.

---

### Step 9

Assign

```text
Majority Class
```

to leaf node.

---

# 12. Prediction Algorithm

### Step 1

Start at Root Node.

---

### Step 2

Compare

```text
Feature ≤ Threshold
```

---

### Step 3

Move

```text
Left

or

Right
```

---

### Step 4

Repeat until Leaf Node.

---

### Step 5

Return

```text
Predicted Class
```

---

# Variables Summary

| Symbol | Meaning |
|---------|---------|
| `X` | Feature Matrix |
| `y` | Target Labels |
| `Pi` | Probability of Class `i` |
| `N` | Total Samples |
| `Nleft` | Samples in Left Child |
| `Nright` | Samples in Right Child |
| `IG` | Information Gain |
| `Entropy` | Measure of Uncertainty |
| `Gini` | Measure of Impurity |

---

# Time Complexity

## Training

For

- `N` Samples
- `D` Features

Average Case

```text
O(N × D × log N)
```

Worst Case

```text
O(N² × D)
```

---

## Prediction

Average

```text
O(log N)
```

Worst

```text
O(N)
```

---

# Space Complexity

Training

```text
O(N)
```

Prediction

```text
O(log N)
```

---

# Advantages

- Easy to understand and interpret.
- No feature scaling required.
- Handles numerical and categorical data.
- Captures non-linear relationships.
- Performs automatic feature selection.

---

# Disadvantages

- Prone to overfitting.
- Sensitive to small changes in data.
- Can become very deep without pruning.
- Lower accuracy than ensemble methods such as Random Forest and XGBoost.

---

# Workflow

```text
Start
   │
   ▼
Load Dataset
   │
   ▼
Find Best Feature
   │
   ▼
Compute Information Gain
   │
   ▼
Best Split?
   │
   ├── Yes
   │      │
   │      ▼
   │  Split Dataset
   │      │
   │      ▼
   │ Build Left Tree
   │      │
   │      ▼
   │ Build Right Tree
   │
   ▼
Stopping Condition?
   │
   ├── No → Repeat
   │
   └── Yes
          │
          ▼
      Create Leaf Node
          │
          ▼
      Prediction
```
