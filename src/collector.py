from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import json
import time

load_dotenv()
client = InferenceClient(token=os.getenv("HF_TOKEN"))

# 4 prompts covering 4 different writing genres
# Important: Keep these the same for all models -> fair comparison
PROMPTS = {
    "narration":     "Write a short story about an orphaned kid who discovers a magic mirror. Keep it under 150 words.",
    "argumentation": "Argue for and against the use of artificial intelligence in art and creativity. Keep it under 150 words.",
    "dialogue":      "Write a conversation between two old friends meeting after 10 years. Keep it under 150 words.",
    "description":   "Describe a sunset over the ocean in the Mediterranean region. Keep it under 150 words."
}

# 3 Qwen models for comparison
# Why Qwen Models?
# 1. Availability: fully supported on HuggingFace free-tier API
# 2. Quality: ranks among top open-source LLMs globally
# 3. Comparative value: 3 variants (7B, 72B, Qwen3-8B) to study
#    whether model size and generation affect stylistic fingerprints
MODELS = {
    "qwen":   "Qwen/Qwen2.5-7B-Instruct",
    "qwen72": "Qwen/Qwen2.5-72B-Instruct",
    "qwen3":  "Qwen/Qwen3-8B",
}

# Number of repeats per prompt per model
# 3 repeats × 4 genres × 3 models = 36 texts total
N_REPEATS = 3

def generate_text(prompt, model):
    """Send a prompt to a model and return the response."""
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=500,
        temperature = 0.7)
    return response.choices[0].message.content

def collect_all():
    """Loop through all models, prompts and repeats, save to JSON."""
    results = []
    for model_name, model_id in MODELS.items():
        for genre, prompt in PROMPTS.items():
            for repeat in range(N_REPEATS):
                print(f"Collecting: {model_name} - {genre} (run {repeat+1}/{N_REPEATS})...")
                try:
                    text = generate_text(prompt, model_id)
                    results.append({
                        "model": model_name,
                        "genre": genre,
                        "repeat": repeat + 1,
                        "prompt": prompt,
                        "text": text
                    })
                    print(f"✅ Done: {model_name} - {genre} #{repeat+1}")
                    time.sleep(1)  # pause to avoid rate limiting
                except Exception as e:
                    print(f"❌ Error: {e}")

    with open("data/corpus.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Finished! Collected {len(results)} texts.")
    print("Saved to data/corpus.json")

if __name__ == "__main__":
    collect_all()