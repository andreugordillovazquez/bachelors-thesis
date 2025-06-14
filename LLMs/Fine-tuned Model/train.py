import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# ─── 1) Dataset Loading ───────────────────────────────────
print("Loading training datasets...")
# Load the three phases of training data
ds1 = load_dataset("json", data_files="phase1_dataset.jsonl", split="train")  # Initial training data
ds2 = load_dataset("json", data_files="phase2_dataset.jsonl", split="train")  # Secondary training data
ds3 = load_dataset("json", data_files="phase3_dataset.jsonl", split="train")  # Additional training data

# ─── 2) Dataset Preparation ───────────────────────────────
print("Preparing combined dataset...")
# Combine all datasets and shuffle to ensure random distribution
full = concatenate_datasets([ds1, ds2, ds3])
full = full.shuffle(seed=42)  # Fixed seed for reproducibility

# ─── 3) Train/Validation Split ───────────────────────────
print("Creating train/validation split...")
# Split the dataset with 90% training, 10% validation
splits = full.train_test_split(test_size=0.1, seed=42)
train_ds = splits["train"]
val_ds = splits["eval"]

# ─── 4) Tokenization and Preprocessing ───────────────────
print("Initializing tokenizer and preprocessing data...")
# Initialize FLAN-T5 tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

def preprocess(ex):
    """
    Preprocess a single example by tokenizing input and target texts.
    
    Args:
        ex (dict): Dictionary containing 'input_text' and 'target_text'
        
    Returns:
        dict: Tokenized inputs with attention masks and labels
    """
    # Tokenize input and target sequences with padding and truncation
    inp = tokenizer(ex["input_text"], padding="max_length", truncation=True, max_length=128)
    tgt = tokenizer(ex["target_text"], padding="max_length", truncation=True, max_length=128)
    return {
        "input_ids": inp.input_ids,
        "attention_mask": inp.attention_mask,
        "labels": tgt.input_ids
    }

# Apply preprocessing to both training and validation datasets
train_tok = train_ds.map(preprocess, batched=True, remove_columns=["input_text", "target_text"])
val_tok = val_ds.map(preprocess, batched=True, remove_columns=["input_text", "target_text"])

# ─── 5) Model and Data Collator Setup ────────────────────
print("Setting up model and data collator...")
# Use MPS (Metal Performance Shaders) if available for Apple Silicon, otherwise CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Loading model from previous checkpoint (flan_t5_finetune_v2)...")
model = AutoModelForSeq2SeqLM.from_pretrained("./flan_t5_finetune_v2").to(device)

# Configure LoRA
print("Configuring LoRA...")
lora_config = LoraConfig(
    r=16,                     # LoRA attention dimension
    lora_alpha=32,            # LoRA alpha parameter
    target_modules=["q", "v"], # Target attention modules
    lora_dropout=0.05,        # Dropout probability for LoRA layers
    bias="none",              # Bias type
    task_type=TaskType.SEQ_2_SEQ_LM  # Task type for sequence-to-sequence
)

# Prepare model for LoRA training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Print trainable parameters info

collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# ─── 6) Training Configuration ───────────────────────────
print("Configuring training arguments...")
training_args = Seq2SeqTrainingArguments(
    output_dir="./flan_t5_finetune_v3_lora",  # Directory for saving checkpoints
    per_device_train_batch_size=16,           # Increased batch size due to LoRA's efficiency
    per_device_eval_batch_size=16,            # Increased batch size for evaluation
    gradient_accumulation_steps=1,            # Reduced due to increased batch size
    evaluation_strategy="steps",              # Evaluate every n steps
    eval_steps=200,                          # Evaluation frequency
    logging_steps=100,                       # Logging frequency
    save_steps=0,                            # Disable checkpoint saving during training
    save_total_limit=1,                      # Maximum number of checkpoints to keep
    load_best_model_at_end=True,             # Load the best model at the end of training
    metric_for_best_model="eval_loss",       # Metric to use for best model selection
    greater_is_better=False,                 # Lower loss is better
    num_train_epochs=20,                     # Total number of training epochs
    learning_rate=1e-3,                      # Increased learning rate for LoRA
    predict_with_generate=True,              # Use generation for prediction
    fp16=False,                              # Disable mixed precision training
    ddp_find_unused_parameters=False,        # Disable unused parameter detection
)

# ─── 7) Trainer Setup ───────────────────────────────────
print("Initializing trainer...")
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    data_collator=collator,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Stop if no improvement for 3 evaluations
)

# ─── 8) Training and Model Saving ───────────────────────
print("Starting training...")
trainer.train()
trainer.save_model("./flan_t5_finetune_v3")
