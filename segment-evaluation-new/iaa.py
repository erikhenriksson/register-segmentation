import json
import numpy as np
from sklearn.metrics import cohen_kappa_score


def read_scores_file(filename):
    scores = {"label": [], "segment": []}
    with open(filename, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            scores["label"].append(data["label_score"])
            scores["segment"].append(data["segment_score"])
    return scores


def quadratic_weighted_kappa(rater1, rater2):
    # Convert to numpy arrays
    rater1 = np.array(rater1)
    rater2 = np.array(rater2)

    return cohen_kappa_score(rater1, rater2, weights="quadratic")


def print_stats(scores1, scores2, name):
    """Print descriptive statistics for a score type"""
    s1 = np.array(scores1)
    s2 = np.array(scores2)

    print(f"\n{name} Statistics:")
    print(
        f"Annotator 1 - Mean: {np.mean(s1):.2f}, Std: {np.std(s1):.2f}, Min: {np.min(s1)}, Max: {np.max(s1)}"
    )
    print(
        f"Annotator 2 - Mean: {np.mean(s2):.2f}, Std: {np.std(s2):.2f}, Min: {np.min(s2)}, Max: {np.max(s2)}"
    )
    print(
        f"Score distribution Annotator 1:",
        dict(zip(*np.unique(s1, return_counts=True))),
    )
    print(
        f"Score distribution Annotator 2:",
        dict(zip(*np.unique(s2, return_counts=True))),
    )


def calculate_iaa(file1, file2):
    # Read both files
    scores1 = read_scores_file(file1)
    scores2 = read_scores_file(file2)

    # Take only the minimum length to ensure we compare same number of items
    min_len = min(len(scores1["label"]), len(scores2["label"]))

    print(f"\nTotal items compared: {min_len}")

    # Print segment by segment comparison
    print("\nSegment by segment comparison:")
    for i, k in zip(scores1["segment"][:min_len], scores2["segment"][:min_len]):
        print(i, k)

    # Print statistics for both types of scores
    print_stats(scores1["label"][:min_len], scores2["label"][:min_len], "Label Scores")
    print_stats(
        scores1["segment"][:min_len], scores2["segment"][:min_len], "Segment Scores"
    )

    # Calculate weighted kappa for both types of scores
    label_kappa = quadratic_weighted_kappa(
        scores1["label"][:min_len], scores2["label"][:min_len]
    )

    segment_kappa = quadratic_weighted_kappa(
        scores1["segment"][:min_len], scores2["segment"][:min_len]
    )

    return {"label_score_iaa": label_kappa, "segment_score_iaa": segment_kappa}


# Usage
if __name__ == "__main__":
    results = calculate_iaa("evaluations_erik.jsonl", "evaluations_Saara.jsonl")
    print(f"\nFinal IAA Scores:")
    print(f"Label Score IAA: {results['label_score_iaa']:.3f}")
    print(f"Segment Score IAA: {results['segment_score_iaa']:.3f}")
