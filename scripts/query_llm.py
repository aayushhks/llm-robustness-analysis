import os
import csv
import time

# OpenAI
try:
    import openai
except ImportError:
    openai = None

# HuggingFace Transformers
try:
    from transformers import pipeline
except ImportError:
    pipeline = None

PROMPT_FILE = "/Users/aayushks/aiml_proj/llm-robustness-analysis/data/adversarial_prompts.csv"
OUTPUT_FILE = "/Users/aayushks/aiml_proj/llm-robustness-analysis/data/adversarial_prompts.csv"

# === CONFIG ===
MODEL_PROVIDER = "openai"   # Options: "openai" or "hf"
OPENAI_MODEL = "gpt-3.5-turbo"
HUGGINGFACE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # Example: any pipeline-compatible model

# HuggingFace pipeline cache
HF_PIPE = None

def read_prompts(filename):
    prompts = []
    with open(filename, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append({"prompt": row["prompt"], "category": row["category"]})
    return prompts

def query_openai(prompt):
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    try:
        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI error for prompt: {prompt}\n{e}")
        return "ERROR"

def query_huggingface(prompt):
    global HF_PIPE
    if pipeline is None:
        return "ERROR: transformers library not installed"
    if HF_PIPE is None:
        HF_PIPE = pipeline("text-generation", model=HUGGINGFACE_MODEL)
    try:
        output = HF_PIPE(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
        return output[0]["generated_text"].strip()
    except Exception as e:
        print(f"HuggingFace error for prompt: {prompt}\n{e}")
        return "ERROR"

def write_outputs(results, filename):
    with open(filename, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "category", "model_provider", "model_output"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

def main():
    prompts = read_prompts(PROMPT_FILE)
    results = []
    print(f"Running {len(prompts)} prompts using {MODEL_PROVIDER}...")
    for i, item in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] Querying: {item['prompt']}")
        if MODEL_PROVIDER == "openai":
            output = query_openai(item["prompt"])
        elif MODEL_PROVIDER == "hf":
            output = query_huggingface(item["prompt"])
        else:
            output = "ERROR: Invalid model provider"
        results.append({
            "prompt": item["prompt"],
            "category": item["category"],
            "model_provider": MODEL_PROVIDER,
            "model_output": output
        })
        time.sleep(1)  # Adjust for rate limits
    write_outputs(results, OUTPUT_FILE)
    print("Done! Outputs saved to:", OUTPUT_FILE)

if __name__ == "__main__":
    main()