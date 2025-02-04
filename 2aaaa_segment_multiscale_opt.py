from dataclasses import dataclass
from typing import List, Tuple, Dict
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from nltk.tokenize import PunktSentenceTokenizer
import sys
import glob
import json
import pandas as pd
import torch

LABELS = ["LY", "SP", "ID", "NA", "HI", "IN", "OP", "IP"]


@dataclass
class MultiScaleConfig:
    max_length: int = 2048
    min_sentences: int = 3
    classification_threshold: float = 0.70
    min_register_diff: float = 0.0
    scale_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.scale_weights is None:
            self.scale_weights = {"individual": 1 / 3, "pairs": 1 / 3, "whole": 1 / 3}


class FastMultiScaleSegmenter:
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

        # Cache and state variables
        self.probs_cache = {}
        self.token_embeddings = None
        self.attention_mask = None
        self.tokens = None
        self.offset_mapping = None

    def safe_mean_pooling(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Safe mean pooling with handling of edge cases."""
        epsilon = 1e-10
        attention_mask = attention_mask.unsqueeze(-1)
        mask_sum = attention_mask.sum(dim=0) + epsilon
        mask_sum = torch.maximum(
            mask_sum,
            torch.tensor(epsilon, device=mask_sum.device, dtype=hidden_states.dtype),
        )
        weighted_sum = (hidden_states * attention_mask).sum(dim=0)
        pooled = weighted_sum / mask_sum
        pooled = torch.clamp(pooled, min=-100.0, max=100.0)
        return pooled.to(dtype=torch.float16)

    def prepare_document(self, text: str):
        """Process document once and cache results."""
        # Clear cache for new document
        self.probs_cache = {}

        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=True,
        )

        inputs = {
            k: v.to("cuda") if k != "offset_mapping" else v for k, v in inputs.items()
        }
        self.tokens = inputs["input_ids"][0]
        outputs = self.model(**inputs)
        self.token_embeddings = outputs.hidden_states[-1][0].detach()
        self.attention_mask = inputs["attention_mask"][0]
        self.offset_mapping = inputs["offset_mapping"][0].cpu().tolist()

    def get_span_embedding(self, start_token: int, end_token: int) -> torch.Tensor:
        """Get embedding for token span."""
        if self.token_embeddings is None:
            raise ValueError("Must call prepare_document first")
        span_embeddings = self.token_embeddings[start_token:end_token]
        span_mask = self.attention_mask[start_token:end_token]
        return self.safe_mean_pooling(span_embeddings, span_mask)

    def get_register_probs(
        self, text: str = None, start_token: int = None, end_token: int = None
    ) -> np.ndarray:
        """Get register probabilities with caching."""
        if text is not None and (start_token is None or end_token is None):
            self.prepare_document(text)
            start_token = 0
            end_token = len(self.token_embeddings)

        cache_key = (start_token, end_token)
        if cache_key in self.probs_cache:
            return self.probs_cache[cache_key]

        span_embedding = self.get_span_embedding(start_token, end_token)
        with torch.no_grad():
            hidden = self.model.head(span_embedding.unsqueeze(0))
            logits = self.model.classifier(hidden)
            probs = torch.sigmoid(logits).detach().cpu().numpy()[0][:8]

        self.probs_cache[cache_key] = probs
        return probs

    def compute_register_distinctness(
        self, probs1: np.ndarray, probs2: np.ndarray, parent_probs: np.ndarray = None
    ) -> float:
        """Compute register distinctness between segments."""
        regs1 = set(np.where(probs1 >= self.config.classification_threshold)[0])
        regs2 = set(np.where(probs2 >= self.config.classification_threshold)[0])
        parent_regs = set(
            np.where(parent_probs >= self.config.classification_threshold)[0]
        )

        if not (regs1 and regs2):
            return 0.0

        max_prob1 = max(probs1)
        max_prob2 = max(probs2)
        max_prob_parent = max(parent_probs) if parent_probs is not None else 0.0

        if max_prob1 <= max_prob_parent and max_prob2 <= max_prob_parent:
            return 0

        seg_diff = 0.0
        if regs1 != regs2:
            diff_registers = (regs1 - regs2) | (regs2 - regs1)
            diff_score = sum(
                abs(probs1[reg_idx] - probs2[reg_idx]) for reg_idx in diff_registers
            )
            seg_diff = diff_score * (max_prob1 + max_prob2) / 2

        parent_diff = min(max_prob1 - max_prob_parent, max_prob2 - max_prob_parent)
        return 0.5 * seg_diff + 0.5 * parent_diff

    def evaluate_split_individual(
        self,
        text: str,
        left_sents: List[str],
        right_sents: List[str],
        left_spans: List[Tuple[int, int]],
        right_spans: List[Tuple[int, int]],
    ) -> float:
        """Optimized individual sentence evaluation using sampling."""
        max_samples = 5
        scores = []

        left_indices = np.random.choice(
            len(left_spans), min(max_samples, len(left_spans)), replace=False
        )
        right_indices = np.random.choice(
            len(right_spans), min(max_samples, len(right_spans)), replace=False
        )

        for left_idx in left_indices:
            left_span = left_spans[left_idx]
            left_prob = self.get_register_probs(
                start_token=left_span[0], end_token=left_span[1]
            )

            for right_idx in right_indices:
                right_span = right_spans[right_idx]
                right_prob = self.get_register_probs(
                    start_token=right_span[0], end_token=right_span[1]
                )
                local_parent_probs = self.get_register_probs(
                    start_token=left_span[0], end_token=right_span[1]
                )
                scores.append(
                    self.compute_register_distinctness(
                        left_prob, right_prob, local_parent_probs
                    )
                )

        return np.mean(scores) if scores else 0.0

    def evaluate_split_pairs(
        self,
        text: str,
        left_sents: List[str],
        right_sents: List[str],
        left_spans: List[Tuple[int, int]],
        right_spans: List[Tuple[int, int]],
    ) -> float:
        """Optimized pair evaluation using sampling."""
        if len(left_spans) < 2 or len(right_spans) < 2:
            return 0.0

        left_pair_spans = [
            (left_spans[i][0], left_spans[i + 1][1])
            for i in range(0, len(left_spans) - 1, 2)
        ]
        right_pair_spans = [
            (right_spans[i][0], right_spans[i + 1][1])
            for i in range(0, len(right_spans) - 1, 2)
        ]

        max_pairs = 3
        if len(left_pair_spans) > max_pairs:
            left_pair_spans = [
                left_pair_spans[i]
                for i in np.random.choice(
                    len(left_pair_spans), max_pairs, replace=False
                )
            ]
        if len(right_pair_spans) > max_pairs:
            right_pair_spans = [
                right_pair_spans[i]
                for i in np.random.choice(
                    len(right_pair_spans), max_pairs, replace=False
                )
            ]

        scores = []
        for left_span in left_pair_spans:
            left_probs = self.get_register_probs(
                start_token=left_span[0], end_token=left_span[1]
            )
            for right_span in right_pair_spans:
                right_probs = self.get_register_probs(
                    start_token=right_span[0], end_token=right_span[1]
                )
                local_parent_probs = self.get_register_probs(
                    start_token=left_span[0], end_token=right_span[1]
                )
                scores.append(
                    self.compute_register_distinctness(
                        left_probs, right_probs, local_parent_probs
                    )
                )

        return np.mean(scores) if scores else 0.0

    def evaluate_split_whole(
        self,
        text: str,
        left_sents: List[str],
        right_sents: List[str],
        left_spans: List[Tuple[int, int]],
        right_spans: List[Tuple[int, int]],
    ) -> float:
        """Evaluate whole segments."""
        left_start = left_spans[0][0]
        left_end = left_spans[-1][1]
        right_start = right_spans[0][0]
        right_end = right_spans[-1][1]

        left_probs = self.get_register_probs(start_token=left_start, end_token=left_end)
        right_probs = self.get_register_probs(
            start_token=right_start, end_token=right_end
        )
        parent_probs = self.get_register_probs(
            start_token=left_start, end_token=right_end
        )

        return self.compute_register_distinctness(left_probs, right_probs, parent_probs)

    def find_best_split(
        self, text: str, sentences: List[str], sent_spans: List[Tuple[int, int]]
    ) -> Tuple[int, float]:
        """Find best split using stride-based sampling."""
        best_score = 0
        best_split = None

        min_idx = self.config.min_sentences
        max_idx = len(sentences) - self.config.min_sentences + 1
        stride = max(1, (max_idx - min_idx) // 10)

        # First pass: check strided positions
        for i in range(min_idx, max_idx, stride):
            left_sents = sentences[:i]
            right_sents = sentences[i:]
            left_spans = sent_spans[:i]
            right_spans = sent_spans[i:]

            score_individual = self.evaluate_split_individual(
                text, left_sents, right_sents, left_spans, right_spans
            )
            score_pairs = self.evaluate_split_pairs(
                text, left_sents, right_sents, left_spans, right_spans
            )
            score_whole = self.evaluate_split_whole(
                text, left_sents, right_sents, left_spans, right_spans
            )

            total_score = (
                self.config.scale_weights["individual"] * score_individual
                + self.config.scale_weights["pairs"] * score_pairs
                + self.config.scale_weights["whole"] * score_whole
            )

            if total_score > best_score:
                best_score = total_score
                best_split = i

        # Second pass: refine locally if we found a good split
        if best_split is not None:
            local_range = range(
                max(min_idx, best_split - stride), min(max_idx, best_split + stride)
            )
            for i in local_range:
                left_sents = sentences[:i]
                right_sents = sentences[i:]
                left_spans = sent_spans[:i]
                right_spans = sent_spans[i:]

                total_score = (
                    self.config.scale_weights["individual"]
                    * self.evaluate_split_individual(
                        text, left_sents, right_sents, left_spans, right_spans
                    )
                    + self.config.scale_weights["pairs"]
                    * self.evaluate_split_pairs(
                        text, left_sents, right_sents, left_spans, right_spans
                    )
                    + self.config.scale_weights["whole"]
                    * self.evaluate_split_whole(
                        text, left_sents, right_sents, left_spans, right_spans
                    )
                )

                if total_score > best_score:
                    best_score = total_score
                    best_split = i

        return best_split, best_score

    def combine_short_sentences(
        self,
        sentences: List[str],
        sent_spans: List[Tuple[int, int]],
        min_chars: int = 100,
    ) -> Tuple[List[str], List[Tuple[int, int]]]:
        """Combine sentences into larger blocks while preserving token spans."""
        result = []
        result_spans = []
        buffer = []
        buffer_spans = []

        for sentence, span in zip(sentences, sent_spans):
            if len(sentence) >= min_chars:
                # If there's a buffer, save it first
                if buffer:
                    combined_buffer = " ".join(buffer)
                    result.append(combined_buffer)
                    # Use the start of the first buffered span and end of the last
                    result_spans.append((buffer_spans[0][0], buffer_spans[-1][1]))
                    buffer = []
                    buffer_spans = []

                # Add the current long sentence
                result.append(sentence)
                result_spans.append(span)
            else:
                buffer.append(sentence)
                buffer_spans.append(span)

                # Check if buffer is now large enough
                if len(" ".join(buffer)) >= min_chars:
                    combined_buffer = " ".join(buffer)
                    result.append(combined_buffer)
                    # Use the start of the first buffered span and end of the last
                    result_spans.append((buffer_spans[0][0], buffer_spans[-1][1]))
                    buffer = []
                    buffer_spans = []

        # Handle any remaining buffer
        if buffer:
            combined_buffer = " ".join(buffer)
            result.append(combined_buffer)
            result_spans.append((buffer_spans[0][0], buffer_spans[-1][1]))

        # Final pass to ensure no tiny segments remain
        final_result = []
        final_spans = []
        i = 0
        while i < len(result):
            if len(result[i]) < min_chars:
                if i < len(result) - 1:
                    # Combine with next segment
                    result[i + 1] = result[i] + " " + result[i + 1]
                    # Update span to cover both segments
                    result_spans[i + 1] = (result_spans[i][0], result_spans[i + 1][1])
                    result.pop(i)
                    result_spans.pop(i)
                elif i > 0:
                    # Combine with previous segment
                    result[i - 1] += " " + result[i]
                    # Update span to cover both segments
                    result_spans[i - 1] = (result_spans[i - 1][0], result_spans[i][1])
                    result.pop(i)
                    result_spans.pop(i)
                else:
                    break
            else:
                final_result.append(result[i])
                final_spans.append(result_spans[i])
                i += 1

        return final_result, final_spans

    def segment_text(self, text: str) -> List[Tuple[str, np.ndarray]]:
        """Main entry point for text segmentation."""
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

        # Combine sentences and their spans
        combined_sentences, combined_sent_spans = self.combine_short_sentences(
            sentences, sent_spans
        )

        if not combined_sent_spans:
            return [(text, self.get_register_probs())]

        segments = self.segment_recursive(text, combined_sentences, combined_sent_spans)

        if len(segments) == 1:
            segments = [(text, self.get_register_probs())]

        return segments

    def segment_recursive(
        self, text: str, sentences: List[str], sent_spans: List[Tuple[int, int]]
    ) -> List[Tuple[str, np.ndarray]]:
        """Recursively segment text using binary splitting."""
        if len(sentences) < 2 * self.config.min_sentences:
            start_token = sent_spans[0][0]
            end_token = sent_spans[-1][1]
            span_text = " ".join(sentences)
            return [
                (
                    span_text,
                    self.get_register_probs(
                        start_token=start_token, end_token=end_token
                    ),
                )
            ]

        split_idx, score = self.find_best_split(text, sentences, sent_spans)

        if score < self.config.min_register_diff or split_idx is None:
            start_token = sent_spans[0][0]
            end_token = sent_spans[-1][1]
            span_text = " ".join(sentences)
            return [
                (
                    span_text,
                    self.get_register_probs(
                        start_token=start_token, end_token=end_token
                    ),
                )
            ]

        left_segments = self.segment_recursive(
            text, sentences[:split_idx], sent_spans[:split_idx]
        )
        right_segments = self.segment_recursive(
            text, sentences[split_idx:], sent_spans[split_idx:]
        )

        return left_segments + right_segments

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
                last_line = line
            if last_line:
                return json.loads(last_line)["id"]
    except FileNotFoundError:
        pass
    return -1


def main(model_path, dataset_path, output_path):
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
    segmenter = FastMultiScaleSegmenter(model_path=model_path, config=config)

    print("last_id:", last_id)

    with open(output_path, "a", encoding="utf-8") as f:
        for i, row in combined_df.iterrows():
            if i <= last_id:
                continue

            text = row["text"]
            segments = segmenter.segment_text(text)

            # If single segment, use its probs for both text_probs and segment probs
            if len(segments) == 1:
                text_probs = segments[0][1]
            else:
                # Otherwise get full document probs
                text_probs = segmenter.get_register_probs(text)

            result = {
                "id": i,
                "label": row["label"],
                "text_probs": [round(x, 4) for x in text_probs.tolist()],
                "segments": [
                    {
                        "text": text,
                        "probs": [round(x, 4) for x in probs.tolist()],
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
