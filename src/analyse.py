import json
import nltk
import pandas as pd
from collections import Counter

# Download NLTK data (only running once time)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# ============================================================
# WHAT THIS FILE DOES:
# Reads corpus.json and measures 5 stylometric features
# for each text, then saves results to data/features.csv
# ============================================================

def load_corpus(path="data/corpus.json"):
    """Load the collected texts from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def lexical_diversity(text):
    """
    Measure how rich the vocabulary is.
    Formula: unique words / total words
    High score = more varied vocabulary
    """
    words = nltk.word_tokenize(text.lower())
    words = [w for w in words if w.isalpha()]  # remove punctuation
    if len(words) == 0:
        return 0
    return len(set(words)) / len(words)

def avg_sentence_length(text):
    """
    Measure average number of words per sentence.
    High score = longer, more complex sentences
    """
    sentences = nltk.sent_tokenize(text)
    if len(sentences) == 0:
        return 0
    lengths = [len(nltk.word_tokenize(s)) for s in sentences]
    return sum(lengths) / len(lengths)

def pos_ratios(text):
    """
    Measure ratio of adjectives, verbs, and nouns.
    - Many adjectives = descriptive, colorful writing
    - Many verbs = action-driven writing
    - Many nouns = informational writing
    """
    words = nltk.word_tokenize(text)
    tags = nltk.pos_tag(words)
    total = len(tags)
    if total == 0:
        return 0, 0, 0

    adj   = sum(1 for _, t in tags if t.startswith('JJ')) / total
    verb  = sum(1 for _, t in tags if t.startswith('VB')) / total
    noun  = sum(1 for _, t in tags if t.startswith('NN')) / total

    return adj, verb, noun

def punctuation_frequency(text):
    """
    Measure how often punctuation is used per word.
    High score = more dramatic or expressive writing
    """
    words = nltk.word_tokenize(text)
    punct = sum(1 for w in words if not w.isalpha())
    total = len(words)
    if total == 0:
        return 0
    return punct / total

def analyze_corpus(corpus):
    """Run all measurements on every text and return a DataFrame."""
    rows = []
    for entry in corpus:
        text = entry["text"]
        text = text.replace("\x00", "").strip()
        if not text:
            continue
        adj, verb, noun = pos_ratios(text)
        rows.append({
            "model":               entry["model"],
            "genre":               entry["genre"],
            "lexical_diversity":   round(lexical_diversity(text), 3),
            "avg_sentence_length": round(avg_sentence_length(text), 3),
            "adj_ratio":           round(adj, 3),
            "verb_ratio":          round(verb, 3),
            "noun_ratio":          round(noun, 3),
            "punct_frequency":     round(punctuation_frequency(text), 3),
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    corpus = load_corpus()
    df = analyze_corpus(corpus)

    # Save to CSV
    df.to_csv("data/features.csv", index=False)

    # Print results nicely
    print("\n📊 STYLOMETRIC FEATURES PER TEXT:")
    print(df.to_string(index=False))

    # Print average per model
    print("\n📊 AVERAGE PER MODEL:")
    print(df.groupby("model")[
        ["lexical_diversity", "avg_sentence_length",
         "adj_ratio", "verb_ratio", "punct_frequency"]
    ].mean().round(3).to_string())