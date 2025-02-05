import sys
import json
import glob
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["LY", "SP", "ID", "NA", "HI", "IN", "OP", "IP"]


@dataclass
class MultiScaleConfig:
    max_length: int = 8192
    min_tokens: int = 64  # Minimum token count per segment
    classification_threshold: float = 0.70
    min_register_diff: float = 0.0
    scale_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.scale_weights is None:
            self.scale_weights = {"individual": 1 / 3, "pairs": 1 / 3, "whole": 1 / 3}


class MultiScaleSegmenter:
    def __init__(self, model_path: str, config: MultiScaleConfig = None):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            # torch_dtype=torch.bfloat16,
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
            logits = self.model(**inputs).logits
            probs = torch.sigmoid(logits).cpu().numpy()[0]

            print(probs)
            pooled_output = []

        # Cache the results
        self._prediction_cache[text] = (probs, pooled_output)
        return probs, pooled_output

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

    def merge_short_sentences(
        self, sentences: List[str], sent_spans: List[Tuple[int, int]]
    ) -> Tuple[List[str], List[Tuple[int, int]]]:
        """Merge short sentences into groups that meet the minimum token requirement."""
        if not sentences:
            return [], []

        merged_sentences = []
        merged_spans = []
        current_group = [sentences[0]]
        current_span = [sent_spans[0]]

        for sent, span in zip(sentences[1:], sent_spans[1:]):
            potential_span = (current_span[0][0], span[1])
            token_count = potential_span[1] - potential_span[0]
            current_token_count = current_span[-1][1] - current_span[0][0]

            if (
                token_count < self.config.min_tokens
                or current_token_count < self.config.min_tokens
            ):
                current_group.append(sent)
                current_span.append(span)
            else:
                merged_sentences.append(" ".join(current_group))
                merged_spans.append((current_span[0][0], current_span[-1][1]))
                current_group = [sent]
                current_span = [span]

        if current_group:
            current_token_count = current_span[-1][1] - current_span[0][0]
            if current_token_count < self.config.min_tokens and merged_sentences:
                last_group = merged_sentences.pop().split()
                last_span = merged_spans.pop()
                merged_sentences.append(" ".join(last_group + current_group))
                merged_spans.append((last_span[0], current_span[-1][1]))
            else:
                merged_sentences.append(" ".join(current_group))
                merged_spans.append((current_span[0][0], current_span[-1][1]))

        return merged_sentences, merged_spans

    def get_text_for_span(self, text: str, start_token: int, end_token: int) -> str:
        """Get the original text corresponding to a token span."""
        if not self.offset_mapping:
            raise ValueError("Must call prepare_document first")

        char_start = self.offset_mapping[start_token][0]
        char_end = self.offset_mapping[end_token - 1][1]
        return text[char_start:char_end]

    def evaluate_split_individual(
        self,
        text: str,
        left_spans: List[Tuple[int, int]],
        right_spans: List[Tuple[int, int]],
    ) -> float:
        """Evaluate split by comparing regions right around the boundary."""
        if not left_spans or not right_spans:
            return 0.0

        # Get windows around boundary (10% of total length)
        INDIVIDUAL_WINDOW_PCT = 0.1

        total_left_tokens = left_spans[-1][1] - left_spans[0][0]
        total_right_tokens = right_spans[-1][1] - right_spans[0][0]

        window_left_tokens = int(total_left_tokens * INDIVIDUAL_WINDOW_PCT)
        window_right_tokens = int(total_right_tokens * INDIVIDUAL_WINDOW_PCT)

        left_start = max(left_spans[-1][1] - window_left_tokens, left_spans[0][0])
        right_end = min(right_spans[0][0] + window_right_tokens, right_spans[-1][1])

        left_text = self.get_text_for_span(text, left_start, left_spans[-1][1])
        right_text = self.get_text_for_span(text, right_spans[0][0], right_end)
        boundary_text = self.get_text_for_span(text, left_start, right_end)

        left_prob, _ = self.get_register_probs(left_text)
        right_prob, _ = self.get_register_probs(right_text)
        local_parent_probs, _ = self.get_register_probs(boundary_text)

        return self.compute_register_distinctness(
            left_prob, right_prob, local_parent_probs
        )

    def evaluate_split_pairs(
        self,
        text: str,
        left_spans: List[Tuple[int, int]],
        right_spans: List[Tuple[int, int]],
    ) -> float:
        """Evaluate split using larger windows around the boundary."""
        if not left_spans or not right_spans:
            return 0.0

        # Get larger windows (25% of each segment)
        PAIRS_WINDOW_PCT = 0.25

        total_left_tokens = left_spans[-1][1] - left_spans[0][0]
        total_right_tokens = right_spans[-1][1] - right_spans[0][0]

        window_left_tokens = int(total_left_tokens * PAIRS_WINDOW_PCT)
        window_right_tokens = int(total_right_tokens * PAIRS_WINDOW_PCT)

        left_start = max(left_spans[-1][1] - window_left_tokens, left_spans[0][0])
        right_end = min(right_spans[0][0] + window_right_tokens, right_spans[-1][1])

        left_text = self.get_text_for_span(text, left_start, left_spans[-1][1])
        right_text = self.get_text_for_span(text, right_spans[0][0], right_end)
        parent_text = self.get_text_for_span(text, left_start, right_end)

        left_probs, _ = self.get_register_probs(left_text)
        right_probs, _ = self.get_register_probs(right_text)
        parent_probs, _ = self.get_register_probs(parent_text)

        return self.compute_register_distinctness(left_probs, right_probs, parent_probs)

    def evaluate_split_whole(
        self,
        text: str,
        left_spans: List[Tuple[int, int]],
        right_spans: List[Tuple[int, int]],
    ) -> float:
        """Evaluate split comparing whole segments."""
        if not left_spans or not right_spans:
            return 0.0

        left_text = self.get_text_for_span(text, left_spans[0][0], left_spans[-1][1])
        right_text = self.get_text_for_span(text, right_spans[0][0], right_spans[-1][1])
        parent_text = self.get_text_for_span(text, left_spans[0][0], right_spans[-1][1])

        left_probs, _ = self.get_register_probs(left_text)
        right_probs, _ = self.get_register_probs(right_text)
        parent_probs, _ = self.get_register_probs(parent_text)

        return self.compute_register_distinctness(left_probs, right_probs, parent_probs)

    def compute_register_distinctness(
        self, probs1: np.ndarray, probs2: np.ndarray, parent_probs: np.ndarray = None
    ) -> float:
        """Compute how distinct two spans are in terms of their register probabilities."""
        regs1 = set(np.where(probs1 >= self.config.classification_threshold)[0])
        regs2 = set(np.where(probs2 >= self.config.classification_threshold)[0])
        parent_regs = set(
            np.where(parent_probs >= self.config.classification_threshold)[0]
        )

        if not (regs1 and regs2):
            return 0.0

        if regs1 == parent_regs == regs2:
            return 0.0

        if regs1 == regs2:
            seg_diff = 0.0

        max_prob1 = max(probs1)
        max_prob2 = max(probs2)
        max_prob_parent = max(parent_probs) if parent_probs is not None else 0.0

        diff_score = 0.0
        diff_registers = (regs1 - regs2) | (regs2 - regs1)
        for reg_idx in diff_registers:
            diff_score += abs(probs1[reg_idx] - probs2[reg_idx])
        seg_diff = diff_score * (max_prob1 + max_prob2) / 2

        parent_diff = (max_prob1 - max_prob_parent + max_prob2 - max_prob_parent) / 2
        lambda_weight = 0.5
        combined_score = lambda_weight * seg_diff + (1 - lambda_weight) * parent_diff

        return combined_score

    def evaluate_split_window(
        self,
        text: str,
        left_spans: List[Tuple[int, int]],
        right_spans: List[Tuple[int, int]],
        window_size: int,
    ) -> float:
        """Evaluate split using window_size groups on each side of boundary."""
        if len(left_spans) < window_size or len(right_spans) < window_size:
            return None

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
        self, text: str, sentences: List[str], sent_spans: List[Tuple[int, int]]
    ) -> Tuple[int, float]:
        """Find best split point using multi-scale analysis."""
        best_score = 0
        best_split = None

        for i in range(1, len(sentences)):
            scores = []
            left_spans = sent_spans[:i]
            right_spans = sent_spans[i:]

            # Always do whole segment comparison
            score_whole = self.evaluate_split_whole(text, left_spans, right_spans)
            scores.append(score_whole)

            # Short window (2+2)
            score_short = self.evaluate_split_window(
                text, left_spans, right_spans, window_size=2
            )
            if score_short is not None:
                scores.append(score_short)

            # Long window (4+4)
            score_long = self.evaluate_split_window(
                text, left_spans, right_spans, window_size=4
            )
            if score_long is not None:
                scores.append(score_long)

            total_score = np.mean(scores) if scores else 0.0

            if total_score > best_score:
                best_score = total_score
                best_split = i

        return best_split, best_score

    def segment_recursive(
        self, text: str, sentences: List[str], sent_spans: List[Tuple[int, int]]
    ) -> List[Tuple[str, np.ndarray, torch.Tensor]]:
        """Recursively segment text using binary splitting."""
        total_tokens = sent_spans[-1][1] - sent_spans[0][0]

        if total_tokens < 2 * self.config.min_tokens:
            span_text = " ".join(sentences)
            probs, embedding = self.get_register_probs(span_text)
            return [(span_text, probs, embedding)]

        split_idx, score = self.find_best_split(text, sentences, sent_spans)

        if score < self.config.min_register_diff or split_idx is None:
            span_text = " ".join(sentences)
            probs, embedding = self.get_register_probs(span_text)
            return [(span_text, probs, embedding)]

        left_segments = self.segment_recursive(
            text, sentences[:split_idx], sent_spans[:split_idx]
        )
        right_segments = self.segment_recursive(
            text, sentences[split_idx:], sent_spans[split_idx:]
        )

        return left_segments + right_segments

    def segment_text(self, text: str) -> List[Tuple[str, np.ndarray, torch.Tensor]]:
        """Main entry point for text segmentation."""
        # Clear the prediction cache for each new document
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
            return [(text, probs, embedding)]

        # Merge short sentences into adequately-sized groups
        sentences, sent_spans = self.merge_short_sentences(sentences, sent_spans)
        segments = self.segment_recursive(text, sentences, sent_spans)

        if len(segments) == 1:
            probs, embedding = self.get_register_probs(text)
            segments = [(text, probs, embedding)]

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
        """Print segmentation results."""
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
            segments = segmenter.segment_text(text)

            # Get document-level predictions
            text_probs, text_embedding = segmenter.get_register_probs(text)

            result = {
                "id": i,
                "label": row["label"],
                "text_probs": [round(x, 4) for x in text_probs.tolist()],
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
