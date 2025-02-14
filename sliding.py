import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set
from nltk.tokenize import PunktSentenceTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import pandas as pd
import glob
import sys
from torch.nn.functional import sigmoid


@dataclass
class Config:
    labels: List[str] = None
    threshold: float = 0.70
    window_size: int = 8  # Number of sentences in each window
    stride: int = 4  # Stride between windows

    def __post_init__(self):
        if self.labels is None:
            self.labels = ["LY", "SP", "ID", "NA", "HI", "IN", "OP", "IP"]


class Segmenter:
    def __init__(self, model_path: str, config: Config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            "answerdotai/ModernBERT-large", use_fast=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            output_hidden_states=True,
        ).to("cuda")
        self.model.eval()
        self.sent_tokenizer = PunktSentenceTokenizer()

    def predict(self, text: str) -> torch.Tensor:
        """Predict register probabilities for a text segment."""
        inputs = self.tokenizer(
            text, max_length=512, truncation=True, padding=True, return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = sigmoid(
                outputs.logits
            )  # Changed from softmax to sigmoid for multilabel

        return probs[0].cpu()

    def get_window_text(
        self, sentences: List[str], start_idx: int, window_size: int
    ) -> str:
        """Get concatenated text for a window of sentences."""
        end_idx = min(start_idx + window_size, len(sentences))
        return " ".join(sentences[start_idx:end_idx])

    def get_active_registers(self, probs: torch.Tensor) -> Set[int]:
        """Get indices of active registers based on threshold."""
        return {i for i, p in enumerate(probs) if p >= self.config.threshold}

    def jaccard_similarity(self, set1: Set[int], set2: Set[int]) -> float:
        """Calculate Jaccard similarity between two sets of register indices."""
        if not set1 and not set2:
            return 1.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def find_best_boundary(
        self,
        sentences: List[str],
        start_context: List[str],
        end_context: List[str],
        uncertain_region: List[str],
    ) -> Tuple[int, List[np.ndarray]]:
        """Find the best boundary point in the uncertain region."""
        best_score = float(
            "inf"
        )  # Changed to inf because we're looking for minimum similarity
        best_boundary = 0
        best_probs = None

        # Try each possible boundary point
        for i in range(len(uncertain_region) + 1):
            left_text = " ".join(start_context + uncertain_region[:i])
            right_text = " ".join(uncertain_region[i:] + end_context)

            left_probs = self.predict(left_text)
            right_probs = self.predict(right_text)

            # Get active registers for each side
            left_registers = self.get_active_registers(left_probs)
            right_registers = self.get_active_registers(right_probs)

            # Score based on Jaccard similarity (lower is better - we want distinct segments)
            score = self.jaccard_similarity(left_registers, right_registers)

            if score < best_score:
                best_score = score
                best_boundary = i
                best_probs = [(left_probs.numpy(), right_probs.numpy())]

        return best_boundary, best_probs

    def segment_text(
        self, text: str
    ) -> Tuple[torch.Tensor, List[Tuple[str, List[np.ndarray]]]]:
        """Segment text into regions of different registers."""
        # Tokenize into sentences
        sentences = self.sent_tokenizer.tokenize(text)

        # Get initial windows
        windows = []
        window_probs = []

        for i in range(0, len(sentences), self.config.stride):
            window_text = self.get_window_text(sentences, i, self.config.window_size)
            if not window_text.strip():
                continue

            probs = self.predict(window_text)
            windows.append((i, i + self.config.window_size, probs))
            window_probs.append(probs)

        # Find register transitions
        segments = []
        current_start = 0
        current_registers = self.get_active_registers(window_probs[0])

        for i in range(1, len(windows)):
            window_registers = self.get_active_registers(window_probs[i])

            # Check if registers changed significantly
            similarity = self.jaccard_similarity(current_registers, window_registers)
            if similarity < 0.5:  # Threshold for considering it a transition
                # Found a transition
                start_idx = windows[i - 1][0]
                end_idx = windows[i][1]

                # Get context for boundary search
                start_context = sentences[:start_idx]
                end_context = sentences[end_idx:]
                uncertain_region = sentences[start_idx:end_idx]

                boundary, boundary_probs = self.find_best_boundary(
                    sentences, start_context, end_context, uncertain_region
                )

                # Add segment
                segment_text = " ".join(sentences[current_start : start_idx + boundary])
                segments.append((segment_text, boundary_probs))

                current_start = start_idx + boundary
                current_registers = window_registers

        # Add final segment
        final_text = " ".join(sentences[current_start:])
        final_probs = [self.predict(final_text).numpy()]
        segments.append((final_text, final_probs))

        # Calculate document-level probabilities
        text_probs = torch.mean(torch.stack([p for _, _, p in windows]), dim=0)

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
            registers = [
                self.config.labels[i]
                for i, p in enumerate(
                    seg["probs"][0][0]
                )  # Changed to handle tuple of arrays
                if p >= self.config.threshold
            ]
            register_str = ", ".join(registers) if registers else "Unknown"

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
                        "probs": [[round(float(x), 8) for x in probs]],
                    }
                    for text, probs in segments
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
