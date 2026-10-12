"""
QSkill AI/ML Internship - Task 1: Iris Flower Classification
Classify iris flowers (Setosa, Versicolor, Virginica) from petal/sepal measurements.
"""
import os
os.makedirs("outputs", exist_ok=True)
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

# 1. Load dataset and explore -------------------------------------------------
iris = load_iris(as_frame=True)
df = iris.frame
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nClass counts:\n", df["species"].value_counts())
print("\nSummary stats:\n", df.describe().round(2))

features = iris.feature_names
sns.pairplot(df, vars=features, hue="species", corner=True)
plt.savefig("outputs/iris_pairplot.png", dpi=120, bbox_inches="tight"); plt.close()

df[features].hist(figsize=(8, 6), bins=20)
plt.tight_layout(); plt.savefig("outputs/iris_histograms.png", dpi=120); plt.close()

# 2. Train/test split (stratified keeps class balance) --------------------------
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Preprocess: data is clean; scaling only helps KNN / Logistic Regression ----
scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

# 4. Train three simple classifiers ---------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
}
results, preds = [], {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    p = model.predict(X_test_s)
    preds[name] = p
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, p),
        "Precision (macro)": precision_score(y_test, p, average="macro"),
    })

# 5. Evaluate -----------------------------------------------------------------------
res = pd.DataFrame(results).round(3)
print("\nModel comparison:\n", res.to_string(index=False))

best = res.sort_values("Accuracy", ascending=False).iloc[0]["Model"]
print(f"\nDetailed report for: {best}")
print(classification_report(y_test, preds[best], target_names=iris.target_names))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, p) in zip(axes, preds.items()):
    ConfusionMatrixDisplay(confusion_matrix(y_test, p),
                           display_labels=iris.target_names).plot(ax=ax, colorbar=False)
    ax.set_title(name)
plt.tight_layout(); plt.savefig("outputs/iris_confusion_matrices.png", dpi=120); plt.close()
