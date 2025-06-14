from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

# Configuration settings
MODEL_PATH = "./flan_t5_finetune_v3"  # Path to the fine-tuned FLAN-T5 model
EVAL_FILE  = "eval_split.jsonl"  # Evaluation dataset file
device     = "mps" if torch.backends.mps.is_available() else "cpu"  # Use MPS if available (for Apple Silicon)

# Load the fine-tuned model and tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)

def extract_and_clean(prompt: str) -> str:
    """
    Generate and clean function calls from a given prompt.
    
    Args:
        prompt (str): Input text prompt to generate function calls from
        
    Returns:
        str: Cleaned and normalized function calls, one per line
    """
    # Tokenize input and prepare for model
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(device)
    
    # Generate function calls
    out_ids = model.generate(**inputs, max_new_tokens=128)
    
    # Decode the generated tokens
    raw = tokenizer.decode(
        out_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    # Extract and normalize function calls
    # Matches patterns like: function_name() or #comment function_name()
    calls = re.findall(r'(?:#\w+\s*)?[A-Za-z_]+\([^)]*\)', raw)
    return "\n".join(calls)

# Load evaluation dataset
print("Loading evaluation dataset...")
ds = load_dataset("json", data_files={"eval": EVAL_FILE}, split="eval")
prompts    = [ex["input_text"]  for ex in ds]
references = [ex["target_text"] for ex in ds]

# Generate predictions for all prompts
print("Generating predictions...")
predictions = [extract_and_clean(p) for p in prompts]

# Calculate exact-match accuracy
matches  = [pred == ref for pred, ref in zip(predictions, references)]
accuracy = sum(matches) / len(matches)
print(f"\nEvaluation Results:")
print(f"Exact‑match accuracy on eval split: {accuracy:.2%}")
print(f"Total samples evaluated: {len(matches)}")
print(f"Correct predictions: {sum(matches)}")
print(f"Failed predictions: {len(matches) - sum(matches)}\n")

# Analyze and display failure cases
print("Failure Analysis:")
print("=" * 40)
for p, ref, pred, ok in zip(prompts, references, predictions, matches):
    if not ok:
        print("\nPROMPT:     ", p)
        print("EXPECTED:   ", ref)
        print("PREDICTION: ", pred)
        print("-" * 40)
