import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
import numpy as np
import os

# ============================================================
# WHAT THIS FILE DOES:
# Reads features.csv and creates 3 visualizations:
# 1. Bar charts for each feature (9 total)
# 2. PCA plot - stylistic distance between models (alternative)
# 3. t-SNE plot - stylistic distance between models (alternative)
# NOTE: Now averages across repeats for bar charts
# ============================================================

os.makedirs("results", exist_ok=True)

def load_features(path="data/features.csv"):
    """Load the stylometric features from CSV."""
    return pd.read_csv(path)

#===========================================================
#def plot_bar(df, feature, title, ylabel, filename):
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
#============================================

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
        "punct_frequency",
        "syntactic_depth",
        "sentiment_polarity",
        "readability"
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

def plot_tsne(df):
    """
    t-SNE plot: alternative to PCA, often shows clearer clusters.
    Unlike PCA, t-SNE preserves local structure between points.
    """
    from sklearn.manifold import TSNE

    features = [
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

    X = df[features].values
    labels = df["model"].values
    genres = df["genre"].values

    # Reduce 6 features → 2 dimensions using t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=10)
    X_tsne = tsne.fit_transform(X)

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

    for i, (x, y) in enumerate(X_tsne):
        model = labels[i]
        genre = genres[i]
        ax.scatter(
            x, y,
            color=color_map[model],
            marker=marker_map[genre],
            s=150,
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

    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title(
        "t-SNE of Stylometric Features — Aesthetic Landscape",
        fontsize=13, fontweight="bold"
    )
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/tsne_styles.png", dpi=150)
    print("✅ Saved: results/tsne_styles.png")

def plot_all_features(df):
    """
    Plot all 9 stylometric features in one figure (3x3 grid).
    Easier to compare all features at once.
    """
    features = [
        ("lexical_diversity",   "Lexical Diversity",         "unique/total words"),
        ("avg_sentence_length", "Avg Sentence Length",       "words/sentence"),
        ("punct_frequency",     "Punctuation Frequency",     "punct/total words"),
        ("sentiment_polarity",  "Sentiment Polarity",        "-1 neg, +1 pos"),
        ("adj_ratio",           "Adjective Ratio",           "adj/total words"),
        ("verb_ratio",          "Verb Ratio",                "verbs/total words"),
        ("noun_ratio",          "Noun Ratio",                "nouns/total words"),
        ("syntactic_depth",     "Syntactic Depth",           "subclause/total words"),
        ("readability",         "Readability (Flesch)",      "0=hard, 100=easy"),
    ]

    models = df["model"].unique()
    genres = df["genre"].unique()
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()

    for idx, (feature, title, ylabel) in enumerate(features):
        ax = axes[idx]
        x = np.arange(len(models))
        width = 0.2

        for i, genre in enumerate(genres):
            means = []
            stds = []
            for m in models:
                vals = df[(df["model"] == m) & (df["genre"] == genre)][feature].values
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            ax.bar(x + i * width, means, width,
                   label=genre, color=colors[i],
                   yerr=stds, capsize=2)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    # Shared legend at the bottom
    handles = [
        mpatches.Patch(color=colors[i], label=genre)
        for i, genre in enumerate(genres)
    ]
    fig.legend(handles=handles, title="Genre",
               loc="lower center", ncol=4,
               fontsize=10, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle("Stylometric Features by Model and Genre",
                 fontsize=16, fontweight="bold", y=1.01)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig("results/all_features.png", dpi=150, bbox_inches="tight")
    print("✅ Saved: results/all_features.png")

if __name__ == "__main__":
    df = load_features()

    # All 9 features in one figure
    plot_all_features(df)

    # PCA
    plot_pca(df)

    # t-SNE
    plot_tsne(df)

    print("\n🎉 All charts saved in results/ folder!")