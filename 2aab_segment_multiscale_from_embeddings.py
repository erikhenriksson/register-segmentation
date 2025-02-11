import sys
import json
import glob
from dataclasses import dataclass
from typing import List, Tuple, Dict

print("imported basic libraries")
import pandas as pd
import numpy as np

print("imported pandas and numpy")
from nltk.tokenize import PunktSentenceTokenizer

print("imported PunktSentenceTokenizer")
import torch

print("imported torch")
from transformers import AutoModelForSequenceClassification, AutoTokenizer

print("imported transformer libraries")
LABELS = ["LY", "SP", "ID", "NA", "HI", "IN", "OP", "IP"]


@dataclass
class MultiScaleConfig:
    max_length: int = 8192
    min_tokens: int = 64  # Minimum token count per segment
    classification_threshold: float = 0.70
    min_register_diff: float = 0
    scale_weights = {"short": 0, "long": 0, "whole": 0.25}


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

        # Extract classification head components
        self.head = self.model.head
        self.classifier = self.model.classifier

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

    def predict_from_embeddings(self, embeddings: torch.Tensor) -> np.ndarray:
        """Get register probabilities using cached model head and classifier"""
        with torch.no_grad():
            hidden = self.head(embeddings.unsqueeze(0))
            logits = self.classifier(hidden)
            probs = torch.sigmoid(logits).detach().cpu().numpy()[0][:8]
        return probs

    def get_span_embedding(self, start_token: int, end_token: int) -> torch.Tensor:
        """Get mean-pooled embedding for token span using cached embeddings."""
        if self.token_embeddings is None:
            raise ValueError("Must call prepare_document before get_span_embedding")

        span_embeddings = self.token_embeddings[start_token:end_token]
        span_mask = self.attention_mask[start_token:end_token]
        embedding = self.safe_mean_pooling(span_embeddings, span_mask)
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

        span_embedding = self.get_span_embedding(start_token, end_token)
        probs = self.predict_from_embeddings(span_embedding)

        return probs, span_embedding

    def prepare_document(self, text: str):
        """Run model once for whole document and cache results."""
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
        with torch.no_grad():
            outputs = self.model(**inputs)
            self.token_embeddings = outputs.hidden_states[-1][0].detach()
        self.attention_mask = inputs["attention_mask"][0]
        self.offset_mapping = inputs["offset_mapping"][0].cpu().tolist()

    def get_span_window(
        self, spans: List[Tuple[int, int]], percentage: float, from_end: bool = True
    ) -> Tuple[int, int]:
        """Get a window of spans based on percentage of total token length."""
        if not spans:
            return None

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

    def evaluate_split_whole(
        self,
        text: str,
        left_spans: List[Tuple[int, int]],
        right_spans: List[Tuple[int, int]],
    ) -> float:
        """Evaluate split comparing whole segments."""
        if not left_spans or not right_spans:
            return 0.0

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

        # compute lengths of segments
        left_length = left_end - left_start
        right_length = right_end - right_start
        parent_length = right_end - left_start

        return self.compute_register_distinctness(
            left_probs,
            right_probs,
            parent_probs,
            left_length,
            right_length,
            parent_length,
        )

    def compute_register_distinctness(
        self,
        probs1: np.ndarray,
        probs2: np.ndarray,
        parent_probs: np.ndarray = None,
        left_length: int = 0,
        right_length: int = 0,
        parent_length: int = 0,
    ) -> float:
        """Compute how distinct two spans are in terms of their register probabilities,
        normalized by number of predicted labels."""
        regs1 = set(np.where(probs1 >= self.config.classification_threshold)[0])
        regs2 = set(np.where(probs2 >= self.config.classification_threshold)[0])
        parent_regs = (
            set(np.where(parent_probs >= self.config.classification_threshold)[0])
            if parent_probs is not None
            else set()
        )

        # Reject if no registers above threshold
        if not (regs1 and regs2):
            return 0.0

        # Reject if both segments have exactly same registers as parent
        if regs1 == parent_regs == regs2:
            return 0.0

        # Reject if both segments have exactly same registers
        if regs1 == regs2:
            return 0.0

        max_prob1 = max(probs1)
        max_prob2 = max(probs2)

        diff_score = 0.0
        diff_registers = (regs1 - regs2) | (regs2 - regs1)
        for reg_idx in diff_registers:
            diff_score += abs(probs1[reg_idx] - probs2[reg_idx])

        diff_1 = diff_score * max_prob1
        diff_2 = diff_score * max_prob2

        scaled_diff_1 = diff_1 / (2 ** len(regs1))
        scaled_diff_2 = diff_2 / (2 ** len(regs2))

        length_penalized_diff_1 = scaled_diff_1 * (left_length / 8192)
        length_penalized_diff_2 = scaled_diff_2 * (right_length / 8192)

        return (length_penalized_diff_1 + length_penalized_diff_2) / 2

        seg_diff = diff_score * (max_prob1 + max_prob2) / 2

        # Normalize by total number of predicted labels
        total_labels = 2 ** (len(regs1) + len(regs2))

        return seg_diff / total_labels

    def compute_register_distinctness_prev(
        self,
        probs1: np.ndarray,
        probs2: np.ndarray,
        parent_probs: np.ndarray,
        left_length: int,
        right_length: int,
        parent_length,
    ) -> float:
        """Compute how distinct two spans are in terms of their register probabilities."""
        # Get active registers and their probabilities
        regs1 = set(np.where(probs1 >= self.config.classification_threshold)[0])
        regs2 = set(np.where(probs2 >= self.config.classification_threshold)[0])
        parent_regs = set(
            np.where(parent_probs >= self.config.classification_threshold)[0]
        )

        if not (regs1 and regs2):  # If either span has no active registers
            return 0.0
        if regs1 == regs2:  # If both spans have identical active registers
            return 0.0

        # if len(regs1) > 1 or len(regs2) > 1:
        #    return 0
        """
        max_seg1 = max(probs1)
        max_seg2 = max(probs2)
        max_parent = max(parent_probs)

        return min(max_seg1 - max_parent, max_seg2 - max_parent) / (
            (len(regs1) - 1 + len(regs2) - 1)
        )
        """
        # Average of above-threshold probabilities - rewards fewer, stronger signals
        score1 = sum(probs1[list(regs1)]) / len(regs1) if regs1 else 0
        score2 = sum(probs2[list(regs2)]) / len(regs2) if regs2 else 0
        parent_score = (
            sum(parent_probs[list(parent_regs)]) / len(parent_regs)
            if parent_regs
            else 0
        )

        # Improvement over parent times number of different registers
        """
        score = (
            (((score1 + score2) / 2) - parent_score)
            # * len(regs1 ^ regs2)
            / (len(list(regs1)) - 1 + len(list(regs2)) - 1)
        )
        """

        score1 = (score1 - parent_score) / (2 ** len(regs1))
        score2 = (score2 - parent_score) / (2 ** len(regs2))

        score1 = score1 * (left_length / 8192)
        score2 = score2 * (right_length / 8192)

        return (score1 + score2) / 2

        # Length penalty: multiply by average length ratio
        # avg_length_ratio = ((left_length + right_length) / 2) / parent_length
        # score = score * avg_length_ratio

        return score

    def compute_register_distinctness_old(
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
            return 0.0

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
        print(combined_score)
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

        left_probs, _ = self.get_register_probs(
            start_token=left_window[0], end_token=left_window[1]
        )
        right_probs, _ = self.get_register_probs(
            start_token=right_window[0], end_token=right_window[1]
        )
        parent_probs, _ = self.get_register_probs(
            start_token=left_window[0], end_token=right_window[1]
        )

        # compute lengths of segments
        left_length = left_window[1] - left_window[0]
        right_length = right_window[1] - right_window[0]
        parent_length = right_window[1] - left_window[0]

        return self.compute_register_distinctness(
            left_probs,
            right_probs,
            parent_probs,
            left_length,
            right_length,
            parent_length,
        )

    def find_best_split(
        self, text: str, sentences: List[str], sent_spans: List[Tuple[int, int]]
    ) -> Tuple[int, float]:
        """Find best split point using multi-scale analysis."""
        best_score = 0
        best_split = None

        for i in range(1, len(sentences)):
            # Check minimum token counts for both potential segments
            left_spans = sent_spans[:i]
            right_spans = sent_spans[i:]

            left_tokens = left_spans[-1][1] - left_spans[0][0]
            right_tokens = right_spans[-1][1] - right_spans[0][0]

            # Skip this split point if either segment would be too small
            if (
                left_tokens < self.config.min_tokens
                or right_tokens < self.config.min_tokens
            ):
                continue

            scores = []

            # Always do whole segment comparison
            score_whole = self.evaluate_split_whole(text, left_spans, right_spans)
            if score_whole == 0:
                continue
            scores.append(score_whole * self.config.scale_weights["whole"])

            # Short window (2+2)
            score_short = self.evaluate_split_window(
                text, left_spans, right_spans, window_size=2
            )
            if score_short is not None:
                scores.append(score_short * self.config.scale_weights["short"])

            # Long window (4+4)
            score_long = self.evaluate_split_window(
                text, left_spans, right_spans, window_size=4
            )
            if score_long is not None:
                scores.append(score_long * self.config.scale_weights["long"])

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

        # If no valid split found (including due to minimum token constraints)
        if split_idx is None or score < self.config.min_register_diff:
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
