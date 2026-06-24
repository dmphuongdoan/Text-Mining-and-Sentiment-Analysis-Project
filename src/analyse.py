import json
import nltk # Natural Language Toolkit for text processing
import pandas as pd
from textblob import TextBlob # For sentiment analysis
from textstat import textstat # For readability metrics
from collections import Counter

# Download NLTK data (only running once time)
nltk.download('punkt') # Tokenizer for splitting text into words and sentences
nltk.download('averaged_perceptron_tagger') # Part-of-speech tagger for identifying nouns, verbs, adjectives, etc.
nltk.download('punkt_tab') # Download punkt sentence tokenizer data
nltk.download('averaged_perceptron_tagger_eng') # Download English part-of-speech tagger data

# ============================================================
# WHAT THIS FILE DOES:
# Reads corpus.json and measures 9 stylometric features
# for each text, then saves results to data/features.csv
#
# Stylometric features include:
# 1. Lexical diversity (unique words / total words)
# 2. Average sentence length (words per sentence)
# 3. Adjective ratio (adjectives / total words)
# 4. Verb ratio (verbs / total words)
# 5. Noun ratio (nouns / total words)
# 6. Punctuation frequency (punctuation / total words)
# 7. Syntactic Depth (average parse tree depth per sentence)
# 8. Sentiment polarity (TextBlob sentiment analysis)
# 9. Readability (Flesch Reading Ease)

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
    words = nltk.word_tokenize(text.lower()) # tokenize text into words and convert to lowercase
    words = [w for w in words if w.isalpha()]  # remove punctuation
    if len(words) == 0:
        return 0
    return len(set(words)) / len(words) # unique words divided by total words = lexical diversity

def avg_sentence_length(text):
    """
    Measure average number of words per sentence.
    High score = longer, more complex sentences
    """
    sentences = nltk.sent_tokenize(text) # split text into sentences using NLTK's sentence tokenizer
    if len(sentences) == 0:
        return 0
    lengths = [len(nltk.word_tokenize(s)) for s in sentences] # list of sentence lengths in words
    return sum(lengths) / len(lengths) # average sentence length = total words divided by number of sentences

def pos_ratios(text):
    """
    Measure ratio of adjectives, verbs, and nouns.
    - Many adjectives = descriptive, colorful writing
    - Many verbs = action-driven writing
    - Many nouns = informational writing
    """
    words = nltk.word_tokenize(text) # tokenize text into words
    tags = nltk.pos_tag(words) # tag each word with its part of speech (e.g., noun, verb, adjective)
    total = len(tags)
    if total == 0:
        return 0, 0, 0

    adj   = sum(1 for _, t in tags if t.startswith('JJ')) / total # Adjectives in NLTK are tagged as 'JJ', 'JJR', 'JJS' (base, comparative, superlative)
    verb  = sum(1 for _, t in tags if t.startswith('VB')) / total # Verbs in NLTK are tagged as 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ' (base, past, gerund, past participle, present, present participle)
    noun  = sum(1 for _, t in tags if t.startswith('NN')) / total # Nouns in NLTK are tagged as 'NN', 'NNS' (singular, plural)

    return adj, verb, noun

def punctuation_frequency(text):
    """
    Measure how often punctuation is used per word.
    High score = more dramatic or expressive writing
    """
    words = nltk.word_tokenize(text)
    punct = sum(1 for w in words if not w.isalpha()) # Count tokens that are not purely alphabetic (i.e., punctuation)
    total = len(words)
    if total == 0:
        return 0
    return punct / total # punctuation frequency = number of punctuation tokens divided by total tokens

def syntactic_depth(text):
    """
    Estimate syntactic complexity by counting
    subordinating conjunctions and relative pronouns.
    High score = more complex nested sentence structures.
    e.g. 'which', 'that', 'because', 'although', 'if'
    """
    words = nltk.word_tokenize(text)
    tags = nltk.pos_tag(words)
    total = len(words)
    if total == 0:
        return 0
    # IN = subordinating conjunction
    # WDT = relative pronoun (which, that)
    # WP = wh-pronoun (who, what)
    depth_markers = sum(
        1 for _, t in tags
        if t in ['IN', 'WDT', 'WP']
    )
    return depth_markers / total

def sentiment_polarity(text):
    """
    Measure sentiment of the text using TextBlob.
    -1.0 = very negative
     0.0 = neutral
    +1.0 = very positive
    """
    return TextBlob(text).sentiment.polarity

def readability(text):
    """
    Measure how easy the text is to read.
    90-100 = very easy (children's book)
    60-70  = standard (normal prose)
    0-30   = very difficult (academic)
    """
    return textstat.flesch_reading_ease(text)

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
            "syntactic_depth":     round(syntactic_depth(text), 3),
            "sentiment_polarity":  round(sentiment_polarity(text), 3),
            "readability":         round(readability(text), 3)
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
         "adj_ratio", "verb_ratio", "punct_frequency", "syntactic_depth", "sentiment_polarity", "readability"]
    ].mean().round(3).to_string())