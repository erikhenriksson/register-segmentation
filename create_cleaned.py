import os
import numpy as np
from skmultilearn.model_selection import IterativeStratification

# Define labels structure
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


def convert_to_label_ids(labels_str):
    label_array = np.zeros(len(labels))
    for label in labels_str.split():
        if label in labels:
            label_array[labels.index(label)] = 1
    return label_array


def split_data(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, "r") as f:
        rows = [line.strip().split("\t") for line in f]

    X = []
    y = []

    for row in rows:
        if len(row) > 1:
            labels_str, text = row
            X.append(text)
            y.append(convert_to_label_ids(labels_str))

    y = np.array(y)

    # First split: 70% train, 30% dev+test
    stratifier = IterativeStratification(
        n_splits=2, order=1, sample_distribution_per_fold=[0.3, 0.7]
    )
    train_idx, dev_test_idx = next(stratifier.split(X, y))

    # Second split: split the 30% into dev (1/3) and test (2/3)
    X_dev_test = [X[i] for i in dev_test_idx]
    y_dev_test = y[dev_test_idx]

    stratifier = IterativeStratification(
        n_splits=2, order=1, sample_distribution_per_fold=[2 / 3, 1 / 3]
    )
    dev_idx, test_idx = next(stratifier.split(X_dev_test, y_dev_test))

    # Get final splits
    X_train = [X[i] for i in train_idx]
    X_dev = [X_dev_test[i] for i in dev_idx]
    X_test = [X_dev_test[i] for i in test_idx]

    y_train = y[train_idx]
    y_dev = y_dev_test[dev_idx]
    y_test = y_dev_test[test_idx]

    def labels_to_string(label_array):
        return " ".join([labels[i] for i, val in enumerate(label_array) if val == 1])

    splits = {
        "train": (X_train, y_train),
        "dev": (X_dev, y_dev),
        "test": (X_test, y_test),
    }

    for split_name, (texts, labels_array) in splits.items():
        output_file = os.path.join(output_dir, f"{split_name}.tsv")
        with open(output_file, "w") as f:
            for text, label_array in zip(texts, labels_array):
                labels_str = labels_to_string(label_array)
                f.write(f"{labels_str}\t{text}\n")
        print(f"{split_name}: {len(texts)} examples")


if __name__ == "__main__":
    split_data("filtered_data.tsv", "en_core_cleaned")
