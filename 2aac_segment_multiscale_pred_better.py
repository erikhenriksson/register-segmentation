print("Importing basic libraries...")
import sys
import json
import glob
from dataclasses import dataclass
from typing import List, Tuple, Dict

print("Importing pandas and numpy")
import pandas as pd
import numpy as np

print("Importing nltk")
from nltk.tokenize import PunktSentenceTokenizer

print("Importing torch and transformers")
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["LY", "SP", "ID", "NA", "HI", "IN", "OP", "IP"]


@dataclass
class MultiScaleConfig:
    max_length: int = 8192
    min_tokens: int = 128  # Minimum token count per segment
    classification_threshold: float = 0.70
    min_register_diff: float = 0.075
    scale_weights = {"short": 0.1, "long": 0.15, "whole": 0.75}


class MultiScaleSegmenter:
    def __init__(self, model_path: str, config: MultiScaleConfig = None):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            output_hidden_states=True,
        ).to("cuda")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "answerdotai/ModernBERT-large", use_fast=True
        )
        self.config = config or MultiScaleConfig()

        # Cache for offset mapping (for token/character conversion)
        self.offset_mapping = None

        # Cache for predictions on text segments
        self._prediction_cache = {}

    def get_register_probs(self, text: str) -> Tuple[np.ndarray, torch.Tensor]:
        """Get register probabilities and embedding for text by running full model."""
        # Check cache first
        if text in self._prediction_cache:
            return self._prediction_cache[text]

        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
            last_hidden_state = outputs.hidden_states[-1]

        attention_mask = inputs["attention_mask"].unsqueeze(
            -1
        )  # Expand for broadcasting
        mean_embedding = (last_hidden_state * attention_mask).sum(
            dim=1
        ) / attention_mask.sum(dim=1)

        # Cache the results
        self._prediction_cache[text] = (probs, mean_embedding)
        return probs, mean_embedding

    def prepare_document(self, text: str):
        """Store offset mapping for token/character conversion."""
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
        self.offset_mapping = inputs["offset_mapping"][0].cpu().tolist()

    def get_text_for_span(self, text: str, start_token: int, end_token: int) -> str:
        """Get the original text corresponding to a token span."""
        char_start = self.offset_mapping[start_token][0]
        char_end = self.offset_mapping[end_token - 1][1]
        return text[char_start:char_end]

    def compute_register_distinctness(
        self, probs1: np.ndarray, probs2: np.ndarray, parent_probs: np.ndarray = None
    ) -> float:
        """Compute how distinct two spans are using Hellinger distance."""
        regs1 = set(np.where(probs1 >= self.config.classification_threshold)[0])
        regs2 = set(np.where(probs2 >= self.config.classification_threshold)[0])
        parent_regs = (
            set(np.where(parent_probs >= self.config.classification_threshold)[0])
            if parent_probs is not None
            else set()
        )

        if not (regs1 and regs2):
            return 0.0, [], []
        if regs1 == parent_regs == regs2:
            return 0.0, [], []
        if regs1 == regs2:
            return 0.0, [], []

        # For each probability, treat as binary distribution [p, 1-p]
        p1_stack = np.column_stack([probs1, 1 - probs1])
        p2_stack = np.column_stack([probs2, 1 - probs2])

        # Compute Hellinger distance for each label
        h_distances = np.sqrt(
            np.sum((np.sqrt(p1_stack) - np.sqrt(p2_stack)) ** 2, axis=1)
        ) / np.sqrt(2)

        return np.mean(h_distances) / (len(regs1) + len(regs2) - 1), regs1, regs2

    def evaluate_split(
        self,
        text: str,
        left_spans: List[Tuple[int, int]],
        right_spans: List[Tuple[int, int]],
        window_size: int = 0,
    ) -> float:
        """Evaluate split using window_size groups on each side of boundary."""
        if len(left_spans) < window_size or len(right_spans) < window_size:
            return None, [], []

        left_window = (left_spans[-window_size][0], left_spans[-1][1])
        right_window = (right_spans[0][0], right_spans[window_size - 1][1])

        left_text = self.get_text_for_span(text, left_window[0], left_window[1])
        right_text = self.get_text_for_span(text, right_window[0], right_window[1])
        parent_text = self.get_text_for_span(text, left_window[0], right_window[1])

        left_probs, _ = self.get_register_probs(left_text)
        right_probs, _ = self.get_register_probs(right_text)
        parent_probs, _ = self.get_register_probs(parent_text)

        return self.compute_register_distinctness(left_probs, right_probs, parent_probs)

    def find_best_split(
        self,
        text: str,
        sentences: List[str],
        sent_spans: List[Tuple[int, int]],
        depth: int,
        side: str,
    ) -> Tuple[int, float]:
        """Find best split point using multi-scale analysis."""
        best_score = 0
        best_split = None
        best_regs_left = []
        best_regs_right = []

        for i in range(1, len(sentences)):
            scores = []
            left_spans = sent_spans[:i]
            right_spans = sent_spans[i:]

            left_length = left_spans[-1][-1] - left_spans[0][0]
            right_length = right_spans[-1][-1] - right_spans[0][0]

            if (
                left_length < self.config.min_tokens
                or right_length < self.config.min_tokens
            ):
                continue

            # Always do whole segment comparison
            score_whole, whole_regs_left, whole_regs_right = self.evaluate_split(
                text, left_spans, right_spans
            )
            if score_whole == 0:
                continue

            # Get minimun segment length in tokens
            min_tokens = min(left_length, right_length)

            scores.append(
                score_whole * self.config.scale_weights["whole"] * min_tokens / 8192
            )

            # Short window (2+2)
            score_short, _, _ = self.evaluate_split(
                text, left_spans, right_spans, window_size=3
            )
            if score_short is not None:
                scores.append(score_short * self.config.scale_weights["short"])

            # Long window (4+4)
            score_long, _, _ = self.evaluate_split(
                text, left_spans, right_spans, window_size=9
            )
            if score_long is not None:
                scores.append(score_long * self.config.scale_weights["long"])

            total_score = np.sum(scores) if scores else 0.0
            if total_score > best_score:
                best_score = total_score
                best_split = i
                best_regs_left = whole_regs_left
                best_regs_right = whole_regs_right

        print(
            f"Depth: {depth}, Side: {side}, Best split: {best_split}, Best score: {best_score}, Best regs left: {[LABELS[int(x)] for x in best_regs_left]}, Best regs right: {[LABELS[int(x)] for x in best_regs_right]}"
        )
        return best_split, best_score

    def segment_recursive(
        self,
        text: str,
        sentences: List[str],
        sent_spans: List[Tuple[int, int]],
        parent_probs: List[np.ndarray] = None,
        depth: int = 0,
        side: str = "root",
        full_text_probs: np.ndarray = None,
    ) -> List[Tuple[str, List[np.ndarray], torch.Tensor]]:
        """Recursively segment text using binary splitting."""
        # Initialize parent_probs if None
        if parent_probs is None:
            parent_probs = []
            if full_text_probs is None:
                # Get full text probabilities only once at the root
                full_text_probs, _ = self.get_register_probs(text)
            parent_probs = [full_text_probs]  # Always include full text probs

        total_tokens = sent_spans[-1][1] - sent_spans[0][0]

        # Get probabilities for current segment
        span_text = " ".join(sentences)
        current_probs, current_embedding = self.get_register_probs(span_text)

        # If too small to split further
        if total_tokens < 2 * self.config.min_tokens:
            all_probs = parent_probs + [current_probs]
            return [(span_text, all_probs, current_embedding)]

        split_idx, score = self.find_best_split(
            text, sentences, sent_spans, depth, side
        )

        if score < self.config.min_register_diff or split_idx is None:
            all_probs = parent_probs + [current_probs]
            return [(span_text, all_probs, current_embedding)]

        # For splits, only pass the parent probabilities without current level
        left_segments = self.segment_recursive(
            text,
            sentences[:split_idx],
            sent_spans[:split_idx],
            parent_probs,  # Don't add current_probs here
            depth + 1,
            "left",
            full_text_probs,
        )
        right_segments = self.segment_recursive(
            text,
            sentences[split_idx:],
            sent_spans[split_idx:],
            parent_probs,  # Don't add current_probs here
            depth + 1,
            "right",
            full_text_probs,
        )

        return left_segments + right_segments

    def segment_text(
        self, text: str
    ) -> List[Tuple[str, List[np.ndarray], torch.Tensor]]:
        """Main entry point for text segmentation."""
        self._prediction_cache = {}
        text = self.truncate_text(text)
        sent_detector = PunktSentenceTokenizer()
        sent_char_spans = list(sent_detector.span_tokenize(text))
        sentences = [text[s:e] for s, e in sent_char_spans]

        self.prepare_document(text)
        offset_mapping = np.array(self.offset_mapping)

        sent_spans = []
        for char_start, char_end in sent_char_spans:
            token_start = None
            token_end = None
            for i, (tok_start, tok_end) in enumerate(offset_mapping):
                if tok_start == tok_end == 0:
                    continue
                if token_start is None and tok_end > char_start:
                    token_start = i
                if tok_start < char_end:
                    token_end = i + 1
                else:
                    break
            if token_start is None or token_end is None:
                token_start, token_end = 0, 0
            sent_spans.append((int(token_start), int(token_end)))

        if not sent_spans:
            probs, embedding = self.get_register_probs(text)
            return [(text, [probs], embedding)]

        # Get document-level predictions first
        text_probs, text_embedding = self.get_register_probs(text)
        segments = self.segment_recursive(
            text, sentences, sent_spans, full_text_probs=text_probs
        )

        if len(segments) == 1:
            return [(text, [text_probs], text_embedding)]

        return segments

    def truncate_text(self, text: str) -> str:
        """Truncate text to max_length tokens."""
        tokens = self.tokenizer(text, truncation=False, return_tensors="pt")[
            "input_ids"
        ][0]
        if len(tokens) > self.config.max_length:
            text = self.tokenizer.decode(
                tokens[: self.config.max_length], skip_special_tokens=True
            )
        return text

    def print_result(self, result: Dict):
        """Print segmentation results with hierarchical register information."""
        print(f"\nText [{result['id']}]")
        print(f"True label: {result['label']}")

        # Get document-level registers
        doc_registers = [
            LABELS[i]
            for i, p in enumerate(result["text_probs"])
            if p >= self.config.classification_threshold
        ]
        print(f"Predicted registers: {', '.join(doc_registers)}")
        print("Segments:")

        for i, seg in enumerate(result["segments"], 1):
            # Create hierarchical register string
            register_chain = []
            for prob_level in seg["probs"]:
                level_registers = [
                    LABELS[i]
                    for i, p in enumerate(prob_level)
                    if p >= self.config.classification_threshold
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
    config = MultiScaleConfig()
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
    segmenter = MultiScaleSegmenter(model_path=model_path, config=config)

    print("last_id:", last_id)

    with open(output_path, "a", encoding="utf-8") as f:
        for i, row in combined_df.iterrows():
            if i <= last_id:
                continue

            text = row["text"]
            text_probs, text_embedding = segmenter.get_register_probs(text)
            segments = segmenter.segment_text(text)

            result = {
                "id": i,
                "label": row["label"],
                "text_probs": [round(x, 8) for x in text_probs.tolist()],
                "text_embedding": text_embedding.tolist(),
                "segments": [
                    {
                        "text": text,
                        "probs": [
                            [round(x, 8) for x in prob_array.tolist()]
                            for prob_array in probs
                        ],
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
