import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, F1."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

def main():
    print(" Starting PhishGuard training...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Using device: {device.upper()} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU only'})")

    # === Load dataset ===
    # Check for training data (created by scripts/prepare_training_data.py)
    train_path = "data/train.csv"
    val_path = "data/validation.csv"
    
    if not os.path.exists(train_path):
        print(" Training data not found!")
        print("\n Next steps:")
        print("   1. Run: python scripts/prepare_training_data.py")
        print("   2. Then run training again\n")
        exit(1)
    
    print(f"Loading training set: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f" Training samples: {len(train_df):,}")
    print(f" Label distribution:\n{train_df['label'].value_counts()}")
    
    # Combine subject and body into a single text column
    train_df["text"] = train_df["subject"].astype(str) + " . " + train_df["body"].astype(str)
    
    # Convert labels to integers (0=benign, 1=phishing)
    train_df["label_int"] = train_df["label"].map({"benign": 0, "phishing": 1})
    
    # Load validation set if it exists
    if os.path.exists(val_path):
        print(f"\n Loading validation set: {val_path}")
        val_df = pd.read_csv(val_path)
        print(f" Validation samples: {len(val_df):,}")
        print(f" Label distribution:\n{val_df['label'].value_counts()}")
        val_df["text"] = val_df["subject"].astype(str) + " . " + val_df["body"].astype(str)
        val_df["label_int"] = val_df["label"].map({"benign": 0, "phishing": 1})
        use_separate_val = True
    else:
        print("\n  No separate validation set found, will split training data")
        use_separate_val = False
    
    # Create temporary files for loading
    temp_train = "data/temp_train.csv"
    train_df[["text", "label_int"]].to_csv(temp_train, index=False)
    
    if use_separate_val:
        temp_val = "data/temp_val.csv"
        val_df[["text", "label_int"]].to_csv(temp_val, index=False)
    
    text_col, label_col = "text", "label_int"

    # === Tokenizer & model ===
    model_name = "roberta-base"  # Using RoBERTa-base (125M parameters)
    print(f"\n Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(batch):
        return tokenizer(batch[text_col], truncation=True, padding="max_length", max_length=256)

    # Load and tokenize datasets
    if use_separate_val:
        # Use separate validation set (no splitting needed)
        train_dataset = load_dataset("csv", data_files=temp_train)["train"]
        val_dataset = load_dataset("csv", data_files=temp_val)["train"]
        
        train_dataset = train_dataset.map(tokenize_fn, batched=True)
        val_dataset = val_dataset.map(tokenize_fn, batched=True)
        
        train_dataset = train_dataset.rename_column(label_col, "labels")
        val_dataset = val_dataset.rename_column(label_col, "labels")
        
        train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        
        eval_dataset = val_dataset
        
        print(f"\n Dataset Split:")
        print(f"   Training: {len(train_dataset):,} emails (from train.csv)")
        print(f"   Validation: {len(eval_dataset):,} emails (from validation.csv)")
        print(f"   Validation set is completely separate - never seen in training!")
    else:
        # Fallback: split training data if no validation set
        dataset = load_dataset("csv", data_files=temp_train)
        dataset = dataset.map(tokenize_fn, batched=True)
        dataset = dataset.rename_column(label_col, "labels")
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        
        full_dataset = dataset["train"].shuffle(seed=42)
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        
        train_dataset = full_dataset.select(range(train_size))
        eval_dataset = full_dataset.select(range(train_size, total_size))
        
        print(f"\n Dataset Split:")
        print(f"   Total: {total_size:,} emails")
        print(f"   Training: {len(train_dataset):,} emails (80%)")
        print(f"   Validation: {len(eval_dataset):,} emails (20%)")

    # === Model ===
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    # === Training args ===
    training_args = TrainingArguments(
            output_dir="models/phishguard-model",
            logging_dir="logs",
            logging_steps=50,
            num_train_epochs=5,              # Increased from 2 to 5 for better accuracy
            per_device_train_batch_size=8 if device == "cuda" else 2,
            per_device_eval_batch_size=8 if device == "cuda" else 2,
            warmup_ratio=0.1,
            weight_decay=0.01,
            fp16=(device == "cuda"),
            learning_rate=2e-5,
            eval_strategy="epoch",           # Evaluate after each epoch
            save_strategy="epoch",           # Save after each epoch
            load_best_model_at_end=True,     # Keep best model
            metric_for_best_model="eval_loss",  # Use validation loss
            save_total_limit=2,              # Keep only 2 best checkpoints
            report_to="none",
        )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, compute_metrics=compute_metrics)

    print(" Training started...")
    trainer.train()

    # === Save model ===
    model.save_pretrained("models/phishguard-model")
    tokenizer.save_pretrained("models/phishguard-model")
    print(" Model trained and saved to 'models/phishguard-model/'")


    if device == "cuda":
        print(f" GPU memory used: {torch.cuda.memory_allocated(0)/1e6:.2f} MB")

if __name__ == "__main__":
    main()
