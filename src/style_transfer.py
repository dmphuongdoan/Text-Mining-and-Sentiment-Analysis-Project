from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import json
import sys
sys.path.append("src")
from analyse import (lexical_diversity, avg_sentence_length, pos_ratios,
                     punctuation_frequency, syntactic_depth,
                     sentiment_polarity, readability)

# ============================================================
# WHAT THIS FILE DOES:
# Style Transfer experiment:
# - Takes texts written by qwen (Qwen2.5-7B)
# - Asks qwen (Qwen2.5-7B) to rewrite them in qwen72's style
# - Measures whether stylometric features shift toward qwen72's profile
#
# Research question:
# "Can the smaller model (Qwen2.5-7B) emulate the larger model's
#  (Qwen2.5-72B) stylistic fingerprint when given explicit instructions?"
# ============================================================
load_dotenv()
client = InferenceClient(token=os.getenv("HF_TOKEN"))

def transfer_style(original_text):
    """
    Ask qwen3 to rewrite a qwen text in qwen72's style.
    qwen72 style = formal, academic, long sentences, less punctuation.
    """
    prompt = f"""Rewrite the following text to match this academic style:
- Use longer, more complex sentences
- Reduce punctuation marks (fewer dashes, asterisks, ellipses)
- Use formal, sophisticated vocabulary
- Keep the same meaning but change the writing style significantly

Original text:
{original_text}

Rewritten text:"""

    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model="Qwen/Qwen2.5-7B-Instruct",
        max_tokens=500,
        temperature=0.7,
    )
    return response.choices[0].message.content

def measure_features(text):
    """Measure all 9 stylometric features of a text."""
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
        "readability":         round(readability(text), 3),
    }


N_REPEATS = 3  # Run 3 times and average results

def run_style_transfer():
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

        original_features = measure_features(original["text"])
        qwen72_features = {
            k: round(sum(measure_features(t["text"])[k]
                     for t in qwen72_texts) / len(qwen72_texts), 3)
            for k in original_features
        }

        # Run N_REPEATS times and average
        all_transferred = []
        for repeat in range(N_REPEATS):
            print(f"\n🔄 qwen rewrites in qwen72 style ({genre}) — run {repeat+1}/{N_REPEATS}...")
            transferred = transfer_style(original["text"])

            if not transferred or not transferred.strip():
                print(f"⚠️ Skipping run {repeat+1} — empty response")
                continue

            transferred = transferred.replace("\x00", "").strip()
            all_transferred.append(measure_features(transferred))
            print(f"✅ Done")

        if not all_transferred:
            print(f"⚠️ Skipping {genre} — all runs failed")
            continue

        # Average features across repeats
        avg_transferred = {
            k: round(sum(r[k] for r in all_transferred) / len(all_transferred), 3)
            for k in original_features
        }

        results.append({
            "genre":          genre,
            "original":       original_features,
            "transferred":    avg_transferred,
            "qwen72_avg":     qwen72_features,
            "n_runs":         len(all_transferred)
        })

        print(f"✅ Genre {genre} done ({len(all_transferred)}/{N_REPEATS} runs)")

    return results

def print_comparison(results):
    print("\n" + "="*65)
    print("STYLE TRANSFER RESULTS")
    print("qwen text → rewritten by qwen (self) → target: qwen72 style")
    print("="*65)

    total_success = 0
    total_features = 0

    for r in results:
        print(f"\n📝 Genre: {r['genre'].upper()}")
        print(f"{'Feature':<25} {'qwen':>10} {'qwen rewrite':>14} {'qwen72 target':>14}")
        print("-" * 65)

        genre_success = 0
        for feature in r["original"]:
            orig   = r["original"][feature]
            trans  = r["transferred"][feature]
            target = r["qwen72_avg"][feature]

            orig_dist  = abs(orig - target)
            trans_dist = abs(trans - target)
            arrow = "✅" if trans_dist < orig_dist else "❌"

            if trans_dist < orig_dist:
                genre_success += 1
            total_features += 1

            print(f"{feature:<25} {orig:>10} {trans:>14} {target:>14} {arrow}")

        total_success += genre_success
        print(f"  → {genre_success}/9 features shifted toward qwen72 style")

    print(f"\n📊 Overall: {total_success}/{total_features} features "
      f"({total_success/total_features*100:.1f}%) shifted toward qwen72")
    print(f"(averaged over {N_REPEATS} runs per genre)")

if __name__ == "__main__":
    results = run_style_transfer()
    print_comparison(results)

    with open("data/style_transfer.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n✅ Saved: data/style_transfer.json")