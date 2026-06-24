from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import json
import pandas as pd
import sys
sys.path.append("src")
from analyse import lexical_diversity, avg_sentence_length, pos_ratios, punctuation_frequency, syntactic_depth, sentiment_polarity, readability

# ============================================================
# WHAT THIS FILE DOES:
# Takes texts from qwen and asks qwen3 to rewrite them in the style of qwen72. Then measures if stylometric features shift toward qwen72's profile.
# ============================================================

load_dotenv()
client = InferenceClient(token=os.getenv("HF_TOKEN"))

def transfer_style(original_text, target_style_description):
    """Ask qwen72 to rewrite a text in a different style."""
    prompt = f"""Rewrite the following text in a more formal, academic style with longer sentences and less punctuation. Keep the same meaning but change the writing style significantly.

Original text:
{original_text}

Rewritten text:"""

    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model="Qwen/Qwen2.5-72B-Instruct",  # ← change to qwen72
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].message.content

def measure_features(text):
    """Measure stylometric features of a text."""
    adj, verb, noun = pos_ratios(text)
    return {
        "lexical_diversity":   round(lexical_diversity(text), 3),
        "avg_sentence_length": round(avg_sentence_length(text), 3),
        "adj_ratio":           round(adj, 3),
        "verb_ratio":          round(verb, 3),
        "noun_ratio":          round(noun, 3),
        "punct_frequency":     round(punctuation_frequency(text), 3),
        "syntactic_depth":     round(syntactic_depth(text), 3),
        "sentiment_polarity":  round(sentiment_polarity(text), 3),
        "readability":         round(readability(text), 3)
    }

def run_style_transfer():
    """
    Take 4 texts from qwen (one per genre),
    transfer to qwen72 style, compare features.
    """
    with open("data/corpus.json", "r") as f:
        corpus = json.load(f)

    genres = ["narration", "argumentation", "dialogue", "description"]
    results = []

    for genre in genres:
        original = next(
            e for e in corpus
            if e["model"] == "qwen" and e["genre"] == genre
        )

        qwen72_texts = [
            e for e in corpus
            if e["model"] == "qwen72" and e["genre"] == genre
        ]

        print(f"\n🔄 Transferring style: qwen → qwen72 style ({genre})...")

        # Transfer style
        transferred = transfer_style(original["text"], "qwen72")

        # Check if transfer was successful
        if not transferred or not transferred.strip():
            print(f"⚠️ Skipping {genre} — empty response")
            continue

        # Clean text
        transferred = transferred.replace("\x00", "").strip()

        # Measure features
        original_features    = measure_features(original["text"])
        transferred_features = measure_features(transferred)
        qwen72_features = {
            k: round(sum(measure_features(t["text"])[k]
                     for t in qwen72_texts) / len(qwen72_texts), 3)
            for k in original_features
        }

        results.append({
            "genre":            genre,
            "original":         original_features,
            "transferred":      transferred_features,
            "qwen72_avg":       qwen72_features,
            "original_text":    original["text"][:100],
            "transferred_text": transferred[:100]
        })

        print(f"✅ Done: {genre}")

    return results

def print_comparison(results):
    """Print a comparison table of features before and after transfer."""
    print("\n" + "="*60)
    print("STYLE TRANSFER RESULTS")
    print("="*60)

    for r in results:
        print(f"\n📝 Genre: {r['genre'].upper()}")
        print(f"{'Feature':<25} {'Original':>10} {'Transferred':>12} {'qwen72 avg':>12}")
        print("-" * 60)
        for feature in r["original"]:
            orig = r["original"][feature]
            trans = r["transferred"][feature]
            target = r["qwen72_avg"][feature]

            # Check if transferred moved toward qwen72
            orig_dist  = abs(orig - target)
            trans_dist = abs(trans - target)
            arrow = "✅" if trans_dist < orig_dist else "❌"

            print(f"{feature:<25} {orig:>10} {trans:>12} {target:>12} {arrow}")

if __name__ == "__main__":
    results = run_style_transfer()
    print_comparison(results)

    # Save results
    with open("data/style_transfer.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✅ Saved: data/style_transfer.json")