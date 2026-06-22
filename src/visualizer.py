import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
import numpy as np
import os

# ============================================================
# WHAT THIS FILE DOES:
# Reads features.csv and creates 4 visualizations:
# 1. Bar chart - lexical diversity per model
# 2. Bar chart - average sentence length per model
# 3. Bar chart - punctuation frequency per model
# 4. PCA plot - overall stylistic distance between models
# NOTE: Now averages across repeats for bar charts
# ============================================================

os.makedirs("results", exist_ok=True)

def load_features(path="data/features.csv"):
    """Load the stylometric features from CSV."""
    return pd.read_csv(path)

def plot_bar(df, feature, title, ylabel, filename):
    """
    Bar chart: compare one feature across 3 models.
    Averages across repeats and genres.
    Error bars show standard deviation.
    """
    models = df["model"].unique()
    genres = df["genre"].unique()
    x = np.arange(len(models))
    width = 0.2
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, genre in enumerate(genres):
        means = []
        stds = []
        for m in models:
            # Average across repeats for same model+genre
            vals = df[(df["model"] == m) & (df["genre"] == genre)][feature].values
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        ax.bar(x + i * width, means, width,
               label=genre, color=colors[i],
               yerr=stds, capsize=3)  # error bars show variation across repeats

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.legend(title="Genre")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"results/{filename}")
    print(f"✅ Saved: results/{filename}")

def plot_pca(df):
    """
    PCA plot: reduce all features to 2D.
    Now shows multiple points per model (one per text).
    Models clustered together = similar style.
    """
    features = [
        "lexical_diversity",
        "avg_sentence_length",
        "adj_ratio",
        "verb_ratio",
        "noun_ratio",
        "punct_frequency"
    ]

    X = df[features].values
    labels = df["model"].values
    genres = df["genre"].values

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    color_map = {
        "qwen":   "#4C72B0",
        "qwen72": "#DD8452",
        "qwen3":  "#55A868"
    }

    marker_map = {
        "narration":     "o",
        "argumentation": "s",
        "dialogue":      "^",
        "description":   "D"
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    for i, (x, y) in enumerate(X_pca):
        model = labels[i]
        genre = genres[i]
        ax.scatter(
            x, y,
            color=color_map[model],
            marker=marker_map[genre],
            s=120,
            alpha=0.8,
            zorder=3
        )

    # Legend for models
    model_patches = [
        mpatches.Patch(color=c, label=m)
        for m, c in color_map.items()
    ]

    # Legend for genres
    genre_patches = [
        plt.Line2D([0], [0],
                   marker=mk, color="gray",
                   label=g, markersize=8,
                   linestyle="None")
        for g, mk in marker_map.items()
    ]

    ax.legend(
        handles=model_patches + genre_patches,
        loc="best", fontsize=8
    )

    variance = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({variance[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({variance[1]*100:.1f}% variance)")
    ax.set_title(
        "PCA of Stylometric Features — Stylistic Distance Between Models",
        fontsize=13, fontweight="bold"
    )
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/pca_styles.png", dpi=150)
    print("✅ Saved: results/pca_styles.png")

if __name__ == "__main__":
    df = load_features()

    plot_bar(df,
        feature="lexical_diversity",
        title="Lexical Diversity by Model and Genre",
        ylabel="Lexical Diversity (unique/total words)",
        filename="lexical_diversity.png"
    )

    plot_bar(df,
        feature="avg_sentence_length",
        title="Average Sentence Length by Model and Genre",
        ylabel="Average words per sentence",
        filename="sentence_length.png"
    )

    plot_bar(df,
        feature="punct_frequency",
        title="Punctuation Frequency by Model and Genre",
        ylabel="Punctuation marks per word",
        filename="punctuation.png"
    )

    plot_pca(df)

    print("\n🎉 All charts saved in results/ folder!")