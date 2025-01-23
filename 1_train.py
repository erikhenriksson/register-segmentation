import os
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import numpy as np
from sklearn.metrics import f1_score, classification_report
import torch

# Your existing labels configuration
labels_structure = {
    "MT": [],
    "LY": [],
    "SP": ["it"],
    "ID": [],
    "NA": ["ne", "sr", "nb"],
    "HI": ["re"],
    "IN": ["en", "ra", "dtp", "fi", "lt"],
    "OP": ["rv", "ob", "rs", "av"],
    "IP": ["ds", "ed"],
}
labels = [k for k in labels_structure.keys()] + [
    item for row in labels_structure.values() for item in row
]

# Enable TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def load_tsv(file_path):
    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=["labels", "text"],
        na_values=None,
        keep_default_na=False,
    )
    return [
        {
            "text": row["text"],
            "labels": row["labels"].split() if pd.notna(row["labels"]) else [],
        }
        for _, row in df.iterrows()
    ]


# Rest of your functions remain the same
def convert_to_label_ids(example):
    label_array = np.zeros(len(labels))
    for label in example["labels"]:
        if label in labels:
            label_array[labels.index(label)] = 1
    return {"labels": label_array}


# Load datasets from TSV
train_data = load_tsv("en_core/train.tsv")
dev_data = load_tsv("en_core/dev.tsv")
test_data = load_tsv("en_core/test.tsv")

# Convert to HuggingFace datasets
train_dataset = Dataset.from_list(train_data)
dev_dataset = Dataset.from_list(dev_data)
test_dataset = Dataset.from_list(test_data)

# Rest of your code remains identical
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")


def tokenize_function(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=2048
    )


# Tokenize and format datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_dev = dev_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

tokenized_train.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)
tokenized_dev.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)
tokenized_test.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)


def compute_metrics(eval_pred):
    predictions, labels_ids = eval_pred
    predictions = (predictions > 0).astype(int)

    micro_f1 = f1_score(labels_ids, predictions, average="micro")
    macro_f1 = f1_score(labels_ids, predictions, average="macro")
    weighted_f1 = f1_score(labels_ids, predictions, average="weighted")

    report = classification_report(
        labels_ids, predictions, target_names=labels, zero_division=0, output_dict=True
    )

    print("\nClassification Report:")
    print(
        classification_report(
            labels_ids, predictions, target_names=labels, zero_division=0
        )
    )

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report,
    }


# Initialize model
model = AutoModelForSequenceClassification.from_pretrained(
    "answerdotai/ModernBERT-base",
    problem_type="multi_label_classification",
    num_labels=len(labels),
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=500,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    load_best_model_at_end=True,
    metric_for_best_model="micro_f1",
    greater_is_better=True,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=1,
    max_grad_norm=1.0,
    learning_rate=5e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    tf32=True,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_dev,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

# Train
trainer.train()

# Save model and tokenizer
trainer.save_model("./best_model")
tokenizer.save_pretrained("./best_model")

# Evaluate on test set
print("\nFinal Test Set Evaluation:")
test_results = trainer.evaluate(tokenized_test)

# Print metrics
print("\nFinal Test Metrics:")
for metric, value in test_results.items():
    if metric != "classification_report":
        print(f"{metric}: {value:.4f}")

print("\nBest model saved to ./best_model")
