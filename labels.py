import numpy as np

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
"""
labels = [k for k in labels_structure.keys()] + [
    item for row in labels_structure.values() for item in row
]
"""
labels = [
    "LY",
    "SP",
    "ID",
    "NA",
    "HI",
    "IN",
    "OP",
    "IP",
]


def convert_to_label_ids(example):
    label_array = np.zeros(len(labels))
    for label in example["labels"]:
        if label in labels:
            label_array[labels.index(label)] = 1
    return {"labels": label_array}
