import torch
import numpy as np
from dataclasses import dataclass
import json
import sys
import glob
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Tuple, Dict
import torch.nn.functional as F


@dataclass
class Config:
    window_size: int = 4096  # Half of max context size
    overlap: int = 2048  # 50% overlap
    threshold: float = 0.70
    labels: List[str] = None

    def __post_init__(self):
        if self.labels is None:
            self.labels = ["LY", "SP", "ID", "NA", "HI", "IN", "OP", "IP"]


class Segmenter:
    def __init__(self, model_path: str, config: Config):
        self.config = config
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            output_hidden_states=True,
        ).to("cuda")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "answerdotai/ModernBERT-large", use_fast=True
        )

    def get_predictions(self, text: str) -> torch.Tensor:
        """Get model predictions for a text segment."""
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits)  # For multilabel classification

        return probs.cpu()

    def find_segment_boundary(
        self, text: str, start: int, end: int
    ) -> Tuple[int, List[float]]:
        """Use binary search to find the exact point where register changes."""
        if end - start <= 100:  # Minimum segment size
            return start, self.get_predictions(text[start:end])[0].tolist()

        mid = (start + end) // 2
        left_probs = self.get_predictions(text[start:mid])[0]
        right_probs = self.get_predictions(text[mid:end])[0]

        # Check if predictions are significantly different
        diff = torch.abs(left_probs - right_probs)
        if torch.any(diff > 0.3):  # Threshold for significant change
            return mid, left_probs.tolist()

        # If no significant change, try both halves
        left_boundary, left_p = self.find_segment_boundary(text, start, mid)
        if left_boundary != start:
            return left_boundary, left_p

        right_boundary, right_p = self.find_segment_boundary(text, mid, end)
        return right_boundary, right_p

    def segment_text(
        self, text: str
    ) -> Tuple[List[float], List[Tuple[str, List[float], List[float]]]]:
        """Segment text using sliding windows and binary search."""
        segments = []
        current_pos = 0
        text_length = len(text)

        # Get overall text prediction
        text_probs = self.get_predictions(text)[0]

        while current_pos < text_length:
            end_pos = min(current_pos + self.config.window_size, text_length)
            window_text = text[current_pos:end_pos]

            # Get predictions for current window
            window_probs = self.get_predictions(window_text)[0]

            # If predictions meet threshold, find exact boundary
            if torch.any(window_probs >= self.config.threshold):
                boundary, probs = self.find_segment_boundary(text, current_pos, end_pos)

                if boundary > current_pos:
                    segment_text = text[current_pos:boundary]
                    segments.append(
                        (
                            segment_text,
                            [window_probs.tolist()],
                            [],  # Placeholder for embeddings if needed
                        )
                    )
                    current_pos = boundary
                else:
                    current_pos += self.config.overlap
            else:
                current_pos += self.config.overlap

        # Add any remaining text
        if current_pos < text_length:
            final_text = text[current_pos:]
            final_probs = self.get_predictions(final_text)[0]
            segments.append((final_text, [final_probs.tolist()], []))

        return text_probs, segments

    def print_result(self, result: Dict):
        """Print segmentation results."""
        print(f"\nDocument ID: {result['id']}")
        print(f"True Label: {result['label']}")
        print("Overall Probabilities:")
        for label, prob in zip(self.config.labels, result["text_probs"]):
            if prob >= self.config.threshold:
                print(f"  {label}: {prob:.4f}")

        print("\nSegments:")
        for i, segment in enumerate(result["segments"]):
            print(f"\nSegment {i+1}:")
            print(f"Length: {len(segment['text'])} chars")
            print("Probabilities:")
            for label, prob in zip(self.config.labels, segment["probs"][0]):
                if prob >= self.config.threshold:
                    print(f"  {label}: {prob:.4f}")


def get_last_processed_id(output_path):
    """Get the ID of the last processed document."""
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            last_line = None
            for line in f:
                if line.strip():  # Skip empty lines
                    last_line = line
            if last_line:
                return json.loads(last_line)["id"]
    except FileNotFoundError:
        pass
    return -1


def main(model_path, dataset_path, output_path):
    """Main function to process documents and generate segments."""
    config = Config()
    all_data = []
    for tsv_file in glob.glob(f"{dataset_path}/*.tsv"):
        df = pd.read_csv(
            tsv_file,
            sep="\t",
            header=None,
            names=["label", "text"],
            na_values="",
            keep_default_na=False,
        )
        all_data.append(df)
    combined_df = pd.concat(all_data, ignore_index=True)

    last_id = get_last_processed_id(output_path)
    segmenter = Segmenter(model_path=model_path, config=config)

    print("last_id:", last_id)

    with open(output_path, "a", encoding="utf-8") as f:
        for i, row in combined_df.iterrows():
            if i <= last_id:
                continue

            text = row["text"]
            text_probs, segments = segmenter.segment_text(text)

            result = {
                "id": i,
                "label": row["label"],
                "text_probs": [round(x, 8) for x in text_probs.tolist()],
                "segments": [
                    {
                        "text": text,
                        "probs": [
                            [round(x, 8) for x in prob_array.tolist()]
                            for prob_array in probs
                        ],
                    }
                    for text, probs, emb in segments
                ],
            }

            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            segmenter.print_result(result)
            f.flush()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: <model_path> <dataset_path> <output_path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
