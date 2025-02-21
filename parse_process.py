import json
from collections import Counter
from typing import Dict, List, Any
from tqdm import tqdm
import numpy as np

LABELS = ["LY", "SP", "ID", "NA", "HI", "IN", "OP", "IP"]
THRESHOLD = 0.70


def split_feats(feats_str: str) -> List[str]:
    """Split the feats string into individual features."""
    if not feats_str or feats_str == "_":
        return []
    return [feat for feat_pair in feats_str.split("|") for feat in feat_pair.split("=")]


def get_dominant_register(probs: List[float]) -> str:
    """Get the dominant register if it's above threshold, else None."""
    if not probs:
        return None
    max_prob = max(probs)
    if max_prob > THRESHOLD:
        # Check if there's only one register above threshold
        above_threshold = [p > THRESHOLD for p in probs]
        if sum(above_threshold) == 1:
            return LABELS[probs.index(max_prob)]
    return None


def extract_features(
    parsed: Dict[str, Any], normalize: bool = True
) -> Dict[str, float]:
    """Extract all linguistic features from parsed text and normalize by length."""
    features = Counter()

    # Get all tokens from all sentences
    all_tokens = []
    for sent in parsed["sentences"]:
        all_tokens.extend(sent["tokens"])

    n_tokens = len(all_tokens)
    if n_tokens == 0:
        return {}

    # Extract all features
    for token in all_tokens:
        # POS tags
        features[f'pos_{token["upos"]}'] += 1

        # Dependency relations
        features[f'dep_{token["deprel"]}'] += 1

        # Morphological features
        for feat in split_feats(token["feats"]):
            features[f"morph_{feat}"] += 1

    # Normalize if requested
    if normalize:
        for key in features:
            features[key] = features[key] / n_tokens

    return dict(features)


def process_file(input_path: str, is_segment: bool = False) -> List[Dict[str, Any]]:
    """Process either full text or segment file."""
    processed_data = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(
            f, desc=f"Processing {'segments' if is_segment else 'full texts'}"
        ):
            entry = json.loads(line)

            # Get probabilities and check for dominant register
            probs = entry.get("probs" if is_segment else "text_probs")
            if not probs:
                continue

            register = get_dominant_register(probs)
            if not register:
                continue

            # Extract features
            features = extract_features(entry["parsed"])
            if not features:  # Skip empty documents
                continue

            # Count total tokens across all sentences
            n_tokens = sum(len(sent["tokens"]) for sent in entry["parsed"]["sentences"])

            processed_data.append(
                {
                    "id": entry["id"],
                    "register": register,
                    "features": features,
                    "text_length": n_tokens,
                }
            )

    return processed_data


def main():
    # Input files
    FULL_TEXT_FILE = "full_text_parsed.jsonl"
    SEGMENT_FILE = "segment_parsed.jsonl"
    OUTPUT_FILE = "processed_features.json"

    # Process both files
    full_text_data = process_file(FULL_TEXT_FILE, is_segment=False)
    segment_data = process_file(SEGMENT_FILE, is_segment=True)

    # Combine all features to get complete feature set
    all_features = set()
    for data in [full_text_data, segment_data]:
        for entry in data:
            all_features.update(entry["features"].keys())

    # Convert to sorted list for consistency
    feature_list = sorted(list(all_features))

    # Add missing features with 0 values
    for data in [full_text_data, segment_data]:
        for entry in data:
            for feature in feature_list:
                if feature not in entry["features"]:
                    entry["features"][feature] = 0.0

    # Save processed data
    output_data = {
        "features": feature_list,
        "full_texts": full_text_data,
        "segments": segment_data,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(
        f"Processed {len(full_text_data)} full texts and {len(segment_data)} segments"
    )
    print(f"Total features extracted: {len(feature_list)}")
    print(f"Data saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
