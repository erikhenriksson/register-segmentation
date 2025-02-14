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

        # Keep track of previous window's active registers
        prev_active_registers = None
        current_segment_start = 0

        while current_pos < text_length:
            end_pos = min(current_pos + self.config.window_size, text_length)
            window_text = text[current_pos:end_pos]

            # Get predictions for current window
            window_probs = self.get_predictions(window_text)[0]

            # Determine active registers in this window
            active_registers = set(
                i for i, p in enumerate(window_probs) if p >= self.config.threshold
            )

            if prev_active_registers is None:
                # First window - initialize
                prev_active_registers = active_registers
                current_segment_start = current_pos
            elif active_registers != prev_active_registers:
                # Register composition changed - find exact boundary
                boundary, probs = self.find_segment_boundary(
                    text, current_segment_start, end_pos
                )

                # Add segment up to boundary
                segment_text = text[current_segment_start:boundary]
                segments.append(
                    (segment_text, [self.get_predictions(segment_text)[0].tolist()], [])
                )

                # Start new segment
                current_segment_start = boundary
                current_pos = boundary
                prev_active_registers = active_registers
            else:
                # Same registers - just move window
                current_pos += self.config.overlap

        # Add final segment if needed
        if current_segment_start < text_length:
            final_text = text[current_segment_start:]
            final_probs = self.get_predictions(final_text)[0]
            segments.append((final_text, [final_probs.tolist()], []))

        return text_probs, segments

    def print_result(self, result: Dict):
        """Print segmentation results with hierarchical register information."""
        print(f"\nText [{result['id']}]")
        print(f"True label: {result['label']}")

        # Get document-level registers
        doc_registers = [
            self.config.labels[i]
            for i, p in enumerate(result["text_probs"])
            if p >= self.config.threshold
        ]
        print(f"Predicted registers: {', '.join(doc_registers)}")
        print("Segments:")

        for i, seg in enumerate(result["segments"], 1):
            # Create hierarchical register string
            register_chain = []
            for prob_level in seg["probs"]:
                level_registers = [
                    self.config.labels[i]
                    for i, p in enumerate(prob_level)
                    if p >= self.config.threshold
                ]
                if level_registers:  # Only add non-empty register lists
                    register_chain.append(" ".join(level_registers))

            # Join with '>' to show hierarchy
            register_str = " > ".join(register_chain)

            print(f"\nSegment {i} [{register_str}]:")
            print(seg["text"])
            print("---")


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
                            [round(x, 8) for x in prob_array] for prob_array in probs
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
