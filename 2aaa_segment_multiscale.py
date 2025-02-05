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
            torch_dtype=torch.float16,
            output_hidden_states=True,
        ).to("cuda")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "answerdotai/ModernBERT-large", use_fast=True
        )
        self.config = config or MultiScaleConfig()

        # Cache for token representations and offset mappings
        self.token_embeddings = None
        self.attention_mask = None
        self.tokens = None
        self.offset_mapping = None

        # Cache for register probabilities and embeddings
        self._prob_cache = {}
        self._embedding_cache = {}

    def safe_mean_pooling(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Safe mean pooling that handles edge cases to prevent infinite values"""
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

    def get_span_embedding(self, start_token: int, end_token: int) -> torch.Tensor:
        """Get mean-pooled embedding for token span using cached embeddings."""
        cache_key = (start_token, end_token)
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        if self.token_embeddings is None:
            raise ValueError("Must call prepare_document before get_span_embedding")

        span_embeddings = self.token_embeddings[start_token:end_token]
        span_mask = self.attention_mask[start_token:end_token]
        embedding = self.safe_mean_pooling(span_embeddings, span_mask)

        self._embedding_cache[cache_key] = embedding
        return embedding

    def get_register_probs(
        self, text: str = None, start_token: int = None, end_token: int = None
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Get register probabilities and embedding for a text span."""
        if self.token_embeddings is None:
            if text is None:
                raise ValueError(
                    "Must either provide text or call prepare_document first"
                )
            self.prepare_document(text)
            start_token = 0
            end_token = len(self.token_embeddings)

        if start_token is None or end_token is None:
            start_token = 0
            end_token = len(self.token_embeddings)

        cache_key = (start_token, end_token)
        if cache_key in self._prob_cache:
            return self._prob_cache[cache_key], self._embedding_cache[cache_key]

        span_embedding = self.get_span_embedding(start_token, end_token)

        with torch.no_grad():
            hidden = self.model.head(span_embedding.unsqueeze(0))
            logits = self.model.classifier(hidden)
            probs = torch.sigmoid(logits).detach().cpu().numpy()[0][:8]

        self._prob_cache[cache_key] = probs
        return probs, span_embedding

    def prepare_document(self, text: str):
        """Run model once for whole document and cache results."""
        self._prob_cache = {}
        self._embedding_cache = {}

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
            # Calculate tokens in current group plus next sentence
            potential_span = (current_span[0][0], span[1])
            token_count = potential_span[1] - potential_span[0]
            current_token_count = current_span[-1][1] - current_span[0][0]

            # If adding next sentence keeps us under minimum, or current group is under minimum
            if (
                token_count < self.config.min_tokens
                or current_token_count < self.config.min_tokens
            ):
                # Add to current group
                current_group.append(sent)
                current_span.append(span)
            else:
                # Current group is large enough and adding more would be too large
                merged_sentences.append(" ".join(current_group))
                merged_spans.append((current_span[0][0], current_span[-1][1]))
                current_group = [sent]
                current_span = [span]

        # Handle the last group - if it's too small and we have previous groups, merge with last group
        if current_group:
            current_token_count = current_span[-1][1] - current_span[0][0]
            if current_token_count < self.config.min_tokens and merged_sentences:
                # Merge with previous group
                last_group = merged_sentences.pop().split()
                last_span = merged_spans.pop()
                merged_sentences.append(" ".join(last_group + current_group))
                merged_spans.append((last_span[0], current_span[-1][1]))
            else:
                # Add as new group
                merged_sentences.append(" ".join(current_group))
                merged_spans.append((current_span[0][0], current_span[-1][1]))

        return merged_sentences, merged_spans

    def get_span_window(
        self, spans: List[Tuple[int, int]], percentage: float, from_end: bool = True
    ) -> Tuple[int, int]:
        """Get a window of spans based on percentage of total token length.

        Args:
            spans: List of token spans
            percentage: What percentage of total length to include (0.0 to 1.0)
            from_end: If True, get window from end of spans, else from start
        """
        if not spans:
            return None

        # Calculate total token length
        total_tokens = spans[-1][1] - spans[0][0]
        window_tokens = int(total_tokens * percentage)

        if from_end:
            # Work backwards from end
            end_token = spans[-1][1]
            start_token = end_token
            for span in reversed(spans):
                start_token = span[0]
                if end_token - start_token >= window_tokens:
                    break
            return (start_token, end_token)
        else:
            # Work forwards from start
            start_token = spans[0][0]
            end_token = start_token
            for span in spans:
                end_token = span[1]
                if end_token - start_token >= window_tokens:
                    break
            return (start_token, end_token)

    def evaluate_split_individual(
        self,
        text: str,
        left_sents: List[str],
        right_sents: List[str],
        left_spans: List[Tuple[int, int]],
        right_spans: List[Tuple[int, int]],
    ) -> float:
        """Evaluate split by comparing regions right around the boundary."""
        if not left_spans or not right_spans:
            return 0.0

        # Get windows around boundary (10% of each segment)
        INDIVIDUAL_WINDOW_PCT = 0.1

        boundary_left = self.get_span_window(
            left_spans, INDIVIDUAL_WINDOW_PCT, from_end=True
        )
        boundary_right = self.get_span_window(
            right_spans, INDIVIDUAL_WINDOW_PCT, from_end=False
        )

        if not boundary_left or not boundary_right:
            return 0.0

        left_prob, _ = self.get_register_probs(
            start_token=boundary_left[0], end_token=boundary_left[1]
        )
        right_prob, _ = self.get_register_probs(
            start_token=boundary_right[0], end_token=boundary_right[1]
        )
        local_parent_probs, _ = self.get_register_probs(
            start_token=boundary_left[0], end_token=boundary_right[1]
        )

        return self.compute_register_distinctness(
            left_prob, right_prob, local_parent_probs
        )

    def evaluate_split_pairs(
        self,
        text: str,
        left_sents: List[str],
        right_sents: List[str],
        left_spans: List[Tuple[int, int]],
        right_spans: List[Tuple[int, int]],
    ) -> float:
        """Evaluate split using larger windows around the boundary."""
        if not left_spans or not right_spans:
            return 0.0

        # Get larger windows around boundary (25% of each segment)
        PAIRS_WINDOW_PCT = 0.25

        left_window = self.get_span_window(left_spans, PAIRS_WINDOW_PCT, from_end=True)
        right_window = self.get_span_window(
            right_spans, PAIRS_WINDOW_PCT, from_end=False
        )

        if not left_window or not right_window:
            return 0.0

        left_probs, _ = self.get_register_probs(
            start_token=left_window[0], end_token=left_window[1]
        )
        right_probs, _ = self.get_register_probs(
            start_token=right_window[0], end_token=right_window[1]
        )
        parent_probs, _ = self.get_register_probs(
            start_token=left_window[0], end_token=right_window[1]
        )

        return self.compute_register_distinctness(left_probs, right_probs, parent_probs)

    def evaluate_split_whole(
        self,
        text: str,
        left_sents: List[str],
        right_sents: List[str],
        left_spans: List[Tuple[int, int]],
        right_spans: List[Tuple[int, int]],
    ) -> float:
        """Evaluate split comparing whole segments."""
        left_start = left_spans[0][0]
        left_end = left_spans[-1][1]
        right_start = right_spans[0][0]
        right_end = right_spans[-1][1]

        left_probs, _ = self.get_register_probs(
            start_token=left_start, end_token=left_end
        )
        right_probs, _ = self.get_register_probs(
            start_token=right_start, end_token=right_end
        )
        parent_probs, _ = self.get_register_probs(
            start_token=left_start, end_token=right_end
        )

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
        # Reject if no registers above threshold
        if not (regs1 and regs2):
            return 0.0

        # Reject if both segments have exactly same registers as parent
        if regs1 == parent_regs == regs2:
            return 0.0

        # Reject if both segments have exactly same registers
        if regs1 == regs2:
            seg_diff = 0.0

        max_prob1 = max(probs1)
        max_prob2 = max(probs2)
        max_prob_parent = max(parent_probs) if parent_probs is not None else 0.0

        #if max_prob1 <= max_prob_parent and max_prob2 <= max_prob_parent:
        #    return 0

        
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
        """Evaluate split using window_size groups on each side of boundary.

        Args:
            window_size: Number of groups to use on each side of boundary
        Returns:
            Score or None if not enough groups available
        """
        if len(left_spans) < window_size or len(right_spans) < window_size:
            return None

        # Get window_size groups from each side
        left_window = (left_spans[-window_size][0], left_spans[-1][1])
        right_window = (right_spans[0][0], right_spans[window_size - 1][1])

        left_probs, _ = self.get_register_probs(
            start_token=left_window[0], end_token=left_window[1]
        )
        right_probs, _ = self.get_register_probs(
            start_token=right_window[0], end_token=right_window[1]
        )
        parent_probs, _ = self.get_register_probs(
            start_token=left_window[0], end_token=right_window[1]
        )

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

            # Average available scores
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

        # Base case: if total tokens is less than 2 * min_tokens, don't split further
        if total_tokens < 2 * self.config.min_tokens:
            start_token = sent_spans[0][0]
            end_token = sent_spans[-1][1]
            span_text = " ".join(sentences)
            probs, embedding = self.get_register_probs(
                start_token=start_token, end_token=end_token
            )
            return [(span_text, probs, embedding)]

        split_idx, score = self.find_best_split(text, sentences, sent_spans)

        if score < self.config.min_register_diff or split_idx is None:
            start_token = sent_spans[0][0]
            end_token = sent_spans[-1][1]
            span_text = " ".join(sentences)
            probs, embedding = self.get_register_probs(
                start_token=start_token, end_token=end_token
            )
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
            probs, embedding = self.get_register_probs()
            return [(text, probs, embedding)]

        # Merge short sentences into adequately-sized groups
        sentences, sent_spans = self.merge_short_sentences(sentences, sent_spans)

        segments = self.segment_recursive(text, sentences, sent_spans)

        if len(segments) == 1:
            probs, embedding = self.get_register_probs()
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
