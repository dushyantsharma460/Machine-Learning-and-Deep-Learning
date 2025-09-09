from sklearn.linear_model import Perceptron         # Perceptron: The model you are training.
from sklearn.datasets import make_classification        # make_classification: Creates a fake dataset for testing (instead of using real data).
from sklearn.model_selection import train_test_split        # train_test_split: Splits data into training (to learn) and testing (to check accuracy).


# Creates a dataset with:
# 1000 samples (rows).
# 10 features (columns / inputs per sample).
# 2 classes (binary classification: 0 or 1).
# random_state=42 → ensures the same random dataset every time (reproducibility).

x,y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# Training set (75% by default → 750 samples).
# Test set (25% by default → 250 samples).
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)


# max_iter=1000 → Train for maximum 1000 passes over the data.
# eta0=0.1 → Learning rate (how big the weight updates are).
# random_state=42 → For reproducibility.
# tol=1e-3 → Training stops early if improvement is very small.
# shuffle=True → Shuffles data before each epoch for better training.

clf = Perceptron(
    max_iter=1000,
    eta0=0.1,
    random_state=42,
    tol=1e-3,
    shuffle=True
)
# During training, the algorithm checks improvement in loss/error after each epoch.
# If the improvement is smaller than tol, training stops early. So even if you change eta0 it don't effect too

# Example 
# Suppose after epoch 1, accuracy improves from 85% → 86%. (Improvement = 1%).
# After epoch 2, accuracy improves from 86% → 86.0005% (Improvement = 0.0005).
# If tol=1e-3 (0.001), this improvement is smaller than tolerance → training stops.

# By doing tol=None it will effect. If tol=None → it will run full max_iter, even if improvements are tiny.
# If data is not perfectly separable → perceptron cannot reach 100% accuracy, no matter how many iterations you run. It will keep updating weights, but some points will always be misclassified.

clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)
print(f"The accuracy is {accuracy}")
