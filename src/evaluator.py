import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

# ============================================================
# WHAT THIS FILE DOES:
# Trains a classifier to predict which model wrote a given text
# based on its stylometric features.
# If accuracy is high → models have distinct stylistic signatures!
# ============================================================

os.makedirs("results", exist_ok=True)

FEATURES = [
    "lexical_diversity",
    "avg_sentence_length",
    "adj_ratio",
    "verb_ratio",
    "noun_ratio",
    "punct_frequency",
    "syntactic_depth",
    "sentiment_polarity",
    "readability"
]

def load_features(path="data/features.csv"):
    """Load stylometric features from CSV."""
    return pd.read_csv(path)

def train_and_evaluate(df):
    """
    Train a Random Forest classifier on stylometric features.
    Use Leave-One-Out cross validation because dataset is small (12 texts).
    """
    X = df[FEATURES].values
    y = df["model"].values

    # Encode model names to numbers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Leave-One-Out: train on 11, test on 1 — repeat 12 times
    # Best strategy for small datasets
    loo = LeaveOneOut()
    scores = cross_val_score(clf, X, y_encoded, cv=loo, scoring="accuracy")

    print("=" * 50)
    print("COMPARATIVE EVALUATION RESULTS")
    print("=" * 50)
    print(f"Dataset size:     {len(df)} texts")
    print(f"Number of models: {df['model'].nunique()}")
    print(f"Features used:    {len(FEATURES)}")
    print()
    print(f"Accuracy (Leave-One-Out): {scores.mean()*100:.1f}%")
    print(f"Chance level:             {100/df['model'].nunique():.1f}%")
    print()

    if scores.mean() > (1 / df["model"].nunique()):
        print("✅ RESULT: Classifier performs ABOVE chance level!")
        print("   → Models have measurable stylistic differences.")
    else:
        print("❌ RESULT: Classifier performs at chance level.")
        print("   → Models may not have distinct stylistic signatures.")

    return clf, le, scores

def plot_confusion_matrix(clf, df, le):
    """
    Plot confusion matrix using Leave-One-Out predictions.
    More honest than training on full dataset.
    """
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import confusion_matrix

    X = df[FEATURES].values
    y = df["model"].values
    y_encoded = le.fit_transform(y)

    # Collect LOO predictions
    loo = LeaveOneOut()
    y_true = []
    y_pred = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        clf.fit(X_train, y_train)
        y_true.append(y_test[0])
        y_pred.append(clf.predict(X_test)[0])

    cm = confusion_matrix(y_true, y_pred)
    labels = le.classes_

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Model")
    ax.set_ylabel("True Model")
    ax.set_title("Confusion Matrix — Leave-One-Out Cross Validation")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black",
                    fontsize=14)

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=150)
    print("✅ Saved: results/confusion_matrix.png")

def plot_feature_importance(clf, df, le):
    """
    Show which stylometric features are most useful
    for identifying each model's writing style.
    """
    X = df[FEATURES].values
    y_encoded = le.transform(df["model"].values)
    clf.fit(X, y_encoded)

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(FEATURES)),
           importances[indices],
           color="#4C72B0")
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels([FEATURES[i] for i in indices], rotation=45)
    ax.set_title("Feature Importance — Which Features Best Identify Each Model?",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Importance Score")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/feature_importance.png", dpi=150)
    print("✅ Saved: results/feature_importance.png")

if __name__ == "__main__":
    df = load_features()

    # Train and evaluate
    clf, le, scores = train_and_evaluate(df)

    # Plot results
    plot_confusion_matrix(clf, df, le)
    plot_feature_importance(clf, df, le)

    print("\n🎉 Evaluation complete! Check results/ folder.")