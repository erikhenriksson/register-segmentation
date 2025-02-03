import sys
import json
import glob
import pandas as pd
import torch
import numpy as np
from typing import List, Dict, Tuple
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dataclasses import dataclass
from scipy.spatial.distance import cosine

LABELS = ["LY", "SP", "ID", "NA", "HI", "IN", "OP", "IP"]


@dataclass
class SegmenterConfig:
    max_length: int = 2048
    classification_threshold: float = 0.35
    min_sentences: int = 3
    max_sentences: int = 20
    window_sentences: int = 5
    stride: int = 2
    merge_threshold: float = 0.3


class TextSegmenter:
    def __init__(self, model_path: str, config: SegmenterConfig = None):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, output_hidden_states=True
        ).to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
        self.config = config or SegmenterConfig()

    def truncate_text(self, text: str) -> str:
        tokens = self.tokenizer(text, truncation=False, return_tensors="pt")[
            "input_ids"
        ][0]
        if len(tokens) > self.config.max_length:
            text = self.tokenizer.decode(
                tokens[: self.config.max_length], skip_special_tokens=True
            )
        return text

    def get_model_outputs(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (probabilities, embedding) tuple from a single model forward pass"""
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        ).to("cuda")
        outputs = self.model(**inputs)

        # Get probabilities
        probs = torch.sigmoid(outputs.logits).detach().cpu().numpy()[0][:8]

        # Get embedding
        last_hidden_state = outputs.hidden_states[-1].detach().cpu().numpy()
        attention_mask = inputs["attention_mask"].cpu().numpy()
        embedding = (
            np.sum(last_hidden_state * attention_mask[..., None], axis=1)
            / np.sum(attention_mask, axis=1)[..., None]
        )

        return probs, embedding[0]

    def compute_register_distance(
        self, probs1: np.ndarray, probs2: np.ndarray
    ) -> float:
        return cosine(probs1, probs2)

    def estimate_local_complexity(
        self, sentences: List[str], center_idx: int, context_size: int = 10
    ) -> float:
        start = max(0, center_idx - context_size // 2)
        end = min(len(sentences), center_idx + context_size // 2)

        if end - start < context_size // 2:
            return 0.5

        window_probs = []
        for i in range(
            start,
            end - self.config.window_sentences + 1,
            self.config.window_sentences // 2,
        ):
            window_text = " ".join(sentences[i : i + self.config.window_sentences])
            probs, _ = self.get_model_outputs(window_text)
            window_probs.append(probs)

        distances = []
        for i in range(len(window_probs)):
            for j in range(i + 1, len(window_probs)):
                distances.append(
                    self.compute_register_distance(window_probs[i], window_probs[j])
                )

        return sum(distances) / len(distances) if distances else 0.5

    def get_depth_scores(self, sentences: List[str]) -> List[Tuple[int, float]]:
        window_probs = []
        positions = []

        for i in range(
            0, len(sentences) - self.config.window_sentences + 1, self.config.stride
        ):
            window_text = " ".join(sentences[i : i + self.config.window_sentences])
            probs, _ = self.get_model_outputs(window_text)
            window_probs.append(probs)
            positions.append(i)

        depth_scores = []
        for i in range(1, len(window_probs) - 1):
            left_dist = self.compute_register_distance(
                window_probs[i], window_probs[i - 1]
            )
            right_dist = self.compute_register_distance(
                window_probs[i], window_probs[i + 1]
            )
            between_dist = self.compute_register_distance(
                window_probs[i - 1], window_probs[i + 1]
            )

            depth = left_dist + right_dist - between_dist
            complexity = self.estimate_local_complexity(sentences, positions[i])
            weighted_depth = depth * (1 + complexity)

            depth_scores.append((positions[i], weighted_depth))

        return depth_scores

    def segment_text(self, text: str) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        text = self.truncate_text(text)
        sentences = sent_tokenize(text)

        if len(sentences) <= self.config.min_sentences:
            probs, embedding = self.get_model_outputs(text)
            return [(text, probs, embedding)]

        depth_scores = self.get_depth_scores(sentences)

        boundaries = []
        for i in range(1, len(depth_scores) - 1):
            pos, depth = depth_scores[i]
            prev_pos, prev_depth = depth_scores[i - 1]
            next_pos, next_depth = depth_scores[i + 1]

            if (
                depth > prev_depth
                and depth > next_depth
                and depth > self.config.merge_threshold
            ):
                boundaries.append(pos + self.config.window_sentences // 2)

        segments = []
        start = 0
        for boundary in sorted(boundaries):
            if boundary - start >= self.config.min_sentences:
                segment_text = " ".join(sentences[start:boundary])
                probs, embedding = self.get_model_outputs(segment_text)
                segments.append((segment_text, probs, embedding))
                start = boundary

        final_segment = " ".join(sentences[start:])
        final_probs, final_embedding = self.get_model_outputs(final_segment)
        segments.append((final_segment, final_probs, final_embedding))

        return segments

    def print_result(self, result: Dict):
        print(f"\nText [{result['id']}]")
        print(f"True label: {result['label']}")
        registers = [
            LABELS[i]
            for i, p in enumerate(result["text_probs"])
            if p >= self.config.classification_threshold
        ]
        print(f"Predicted registers: {', '.join(registers)}")
        print("Segments:")
        for i, seg in enumerate(result["segments"], 1):
            registers = [
                LABELS[i]
                for i, p in enumerate(seg["probs"])
                if p >= self.config.classification_threshold
            ]
            print(f"\nSegment {i} [{', '.join(registers)}]:")
            print(seg["text"])
            print("---")


def get_last_processed_id(output_path):
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            last_line = None
            for line in f:
                last_line = line
            if last_line:
                return json.loads(last_line)["id"]
    except FileNotFoundError:
        pass
    return -1


def main(model_path, dataset_path, output_path):
    config = SegmenterConfig()
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
    segmenter = TextSegmenter(model_path=model_path, config=config)

    print("last_id:", last_id)

    with open(output_path, "a", encoding="utf-8") as f:
        for i, row in combined_df.iterrows():
            if i <= last_id:
                continue

            text = row["text"]
            full_probs, text_embedding = segmenter.get_model_outputs(text)
            segments = segmenter.segment_text(text)

            result = {
                "id": i,
                "label": row["label"],
                "text_probs": [round(x, 4) for x in full_probs.tolist()],
                "text_embedding": text_embedding.tolist(),
                "segments": [
                    {
                        "text": text,
                        "probs": [round(x, 4) for x in probs.tolist()],
                        "embedding": emb.tolist(),
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
