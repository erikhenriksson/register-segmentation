import json
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DebertaV2Config,
    DebertaV2ForSequenceClassification,
    DebertaV2Tokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

models = {
    "deberta": "microsoft/deberta-v3-large",
    "modernbert": "answerdotai/ModernBERT-large",
    "bge-m3": "BAAI/bge-m3-retromae",
}

model_type = sys.argv[1] if len(sys.argv) > 1 else ""
dataset = sys.argv[2] if len(sys.argv) > 2 else ""
if model_type not in models:
    print(f"Invalid model type: {model_type}")
    sys.exit(1)
if not dataset:
    print(f"Invalid dataset: {dataset}")
    sys.exit(1)

labels_structure = {
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


def create_extended_deberta_multilabel(model_name, num_labels, max_length=2048):
    config = DebertaV2Config.from_pretrained(model_name)
    config.update(
        {
            "output_hidden_states": False,
            "max_position_embeddings": max_length,
            "attention_probs_dropout_prob": 0.0,
            "hidden_dropout_prob": 0.0,
            "num_labels": num_labels,
            "problem_type": "multi_label_classification",
            # Add torch dtype
            "dtype": torch.bfloat16,
        }
    )

    model = DebertaV2ForSequenceClassification.from_pretrained(
        model_name, config=config
    )
    return model


# Custom Focal Loss for multi-label classification
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=1.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    """
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
    """

    def forward(self, pred, target):
        sigmoid_pred = torch.sigmoid(pred)
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pt = torch.where(target == 1, sigmoid_pred, 1 - sigmoid_pred)
        focal_weight = torch.pow(1 - pt, self.gamma)

        if self.alpha != 1:
            alpha_weight = torch.where(target == 1, self.alpha, 1)
            focal_weight = focal_weight * alpha_weight

        focal_loss = focal_weight * bce_loss
        return focal_loss.mean()


# Custom Trainer with Focal Loss
class FocalLossTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(alpha=0.5, gamma=1.0)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.focal_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss


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


def convert_to_label_ids(example):
    label_array = np.zeros(len(labels))
    for label in example["labels"]:
        if label in labels:
            label_array[labels.index(label)] = 1
    return {"labels": label_array}


# Load datasets from TSV
train_data = load_tsv(f"{dataset}/train.tsv")
dev_data = load_tsv(f"{dataset}/dev.tsv")
test_data = load_tsv(f"{dataset}/test.tsv")

# Convert to HuggingFace datasets
train_dataset = Dataset.from_list(train_data)
dev_dataset = Dataset.from_list(dev_data)
test_dataset = Dataset.from_list(test_data)

# Convert labels to multi-hot encoding
train_dataset = train_dataset.map(convert_to_label_ids)
dev_dataset = dev_dataset.map(convert_to_label_ids)
test_dataset = test_dataset.map(convert_to_label_ids)

# Load model
num_labels = len(labels)
model_name = models[model_type]
if model_type == "deberta":
    model = create_extended_deberta_multilabel(model_name, num_labels)
else:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        problem_type="multi_label_classification",
        num_labels=num_labels,
        # torch_dtype=torch.float16 if model_type == "modernbert" else torch.bfloat16,
    )

# Load tokenizer
if model_type == "deberta":
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name)


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


def optimize_threshold(predictions, labels_ids):
    best_f1 = 0
    best_threshold = 0.5
    thresholds = np.arange(0.3, 0.75, 0.05)

    for threshold in thresholds:
        predictions_binary = (
            (torch.sigmoid(torch.tensor(predictions)) > threshold).numpy().astype(int)
        )
        f1 = f1_score(labels_ids, predictions_binary, average="micro")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold


def compute_metrics(eval_pred):
    predictions, labels_ids = eval_pred
    best_threshold = optimize_threshold(predictions, labels_ids)
    predictions = (
        (torch.sigmoid(torch.tensor(predictions)) > best_threshold).numpy().astype(int)
    )

    micro_f1 = f1_score(labels_ids, predictions, average="micro")
    macro_f1 = f1_score(labels_ids, predictions, average="macro")
    weighted_f1 = f1_score(labels_ids, predictions, average="weighted")

    report = classification_report(
        labels_ids, predictions, target_names=labels, zero_division=0, output_dict=True
    )

    print(f"\nOptimal threshold: {best_threshold:.3f}")
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
        "optimal_threshold": best_threshold,
        "classification_report": report,
    }


# Training arguments
training_args = TrainingArguments(
    output_dir=f"./results/{model_type}/{dataset}",
    eval_strategy="steps",
    eval_steps=500,
    # per_device_train_batch_size=8,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=32,
    num_train_epochs=30,
    load_best_model_at_end=True,
    metric_for_best_model="micro_f1",
    greater_is_better=True,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=1,
    max_grad_norm=1.0,
    learning_rate=5e-5,
    warmup_ratio=0.05,
    weight_decay=0.01,
    tf32=True,
    group_by_length=True,
    bf16=True if model_type != "modernbert" else False,
    fp16=True if model_type == "modernbert" else False,
)

# Initialize trainer with Focal Loss
trainer = FocalLossTrainer(
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

# Get optimal threshold from test results
optimal_threshold = test_results["eval_optimal_threshold"]

print(f"\nOptimal test threshold: {optimal_threshold:.3f}")

# Print metrics
print("\nFinal Test Metrics:")
for metric, value in test_results.items():
    if metric != "classification_report" and isinstance(value, (int, float)):
        print(f"{metric}: {value:.4f}")

print("\nBest model saved to ./best_model")


test_pred_output = trainer.predict(tokenized_test)
test_pred_probs = (
    torch.sigmoid(torch.tensor(test_pred_output.predictions)).numpy().tolist()
)
test_true_labels = test_pred_output.label_ids.tolist()

# Save as JSON
with open(f"./results/{model_type}/{dataset}/test_predictions.json", "w") as f:
    json.dump({"pred_probs": test_pred_probs, "labels": test_true_labels}, f)
