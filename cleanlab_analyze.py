import json
import numpy as np
from cleanlab import Datalab


def load_and_process_jsonl(file_path):
    with open(file_path, "r") as f:
        records = [json.loads(line) for line in f]

    index_labels = []
    pred_probs = []
    texts = []
    labels_str = []

    for record in records:
        labels = [i for i, val in enumerate(record["labels"]) if val == 1]
        index_labels.append(labels)
        pred_probs.append(record["pred_probs"])
        texts.append(record["text"])
        labels_str.append(record["labels_str"])

    return index_labels, np.array(pred_probs), texts, labels_str


def analyze_and_filter_data(file_path, output_path, percentile=10):
    labels, pred_probs, texts, labels_str = load_and_process_jsonl(file_path)

    lab = Datalab(
        data={"labels": labels, "text": texts},
        label_name="labels",
        task="multilabel",
    )

    # Get label issues and scores
    lab.find_issues(pred_probs=pred_probs)
    label_issues = lab.get_issues("label")
    scores = label_issues["label_score"].values

    # Calculate statistics
    stats = {
        "mean": np.mean(scores),
        "std": np.std(scores),
        "min": np.min(scores),
        "max": np.max(scores),
        "percentiles": {
            "5th": np.percentile(scores, 5),
            "25th": np.percentile(scores, 25),
            "50th": np.percentile(scores, 50),
            "75th": np.percentile(scores, 75),
            "95th": np.percentile(scores, 95),
        },
    }

    # Get threshold for filtering (lower percentile = worse scores)
    threshold = np.percentile(scores, percentile)

    # Get indices of examples to keep (scores above threshold)
    keep_indices = scores > threshold

    # Filter data
    filtered_texts = [text for text, keep in zip(texts, keep_indices) if keep]
    filtered_labels = [label for label, keep in zip(labels_str, keep_indices) if keep]

    # Write filtered data to TSV
    with open(output_path, "w", encoding="utf-8") as f:
        for label, text in zip(filtered_labels, filtered_texts):
            f.write(f"{label}\t{text}\n")

    return stats, sum(~keep_indices)  # Return statistics and number of removed examples


if __name__ == "__main__":
    file_path = "test_predictions.jsonl"
    output_path = "filtered_data.tsv"
    stats, removed_count = analyze_and_filter_data(file_path, output_path)

    print(f"Statistics of label scores:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v:.4f}")
        else:
            print(f"{key}: {value:.4f}")
    print(f"\nRemoved {removed_count} examples")
