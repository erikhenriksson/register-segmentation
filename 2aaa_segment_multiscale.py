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
    max_length: int = 2048
    min_sentences: int = 3
    classification_threshold: float = 0.70  # Changed to match working code
    min_register_diff: float = 0.0
    scale_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.scale_weights is None:
            self.scale_weights = {"individual": 1 / 3, "pairs": 1 / 3, "whole": 1 / 3}


class MultiScaleSegmenter:
    def __init__(self, model_path: str, config: MultiScaleConfig = None):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # Use float16 like working code
            output_hidden_states=True,
        ).to("cuda")
        self.model.eval()  # Set model to eval mode
        self.tokenizer = AutoTokenizer.from_pretrained(
            "answerdotai/ModernBERT-large", use_fast=True
        )
        self.config = config or MultiScaleConfig()

        # Cache for token representations and offset mappings
        self.token_embeddings = None
        self.attention_mask = None
        self.tokens = None
        self.offset_mapping = (
            None  # Will hold the (start, end) positions for each token
        )

    def safe_mean_pooling(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Safe mean pooling that handles edge cases to prevent infinite values"""
        epsilon = 1e-10
        attention_mask = attention_mask.unsqueeze(-1)
        mask_sum = attention_mask.sum(dim=0) + epsilon

        # Ensure the mask sum is at least epsilon to prevent explosion
        mask_sum = torch.maximum(
            mask_sum,
            torch.tensor(epsilon, device=mask_sum.device, dtype=hidden_states.dtype),
        )

        # Compute weighted sum and divide by mask sum
        weighted_sum = (hidden_states * attention_mask).sum(dim=0)
        pooled = weighted_sum / mask_sum

        # Clip any extreme values
        pooled = torch.clamp(pooled, min=-100.0, max=100.0)

        # Ensure output is in float16
        return pooled.to(dtype=torch.float16)

    def prepare_document(self, text: str):
        """Run model once for whole document and cache results, including offset mappings."""
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,  # Request offset mapping
            add_special_tokens=True,
        )
        # Move tensors (except offset mapping) to CUDA
        inputs = {
            k: v.to("cuda") if k != "offset_mapping" else v for k, v in inputs.items()
        }

        self.tokens = inputs["input_ids"][0]
        outputs = self.model(**inputs)
        self.token_embeddings = outputs.hidden_states[-1][0].detach()
        self.attention_mask = inputs["attention_mask"][0]

        # Save offset mapping; note that offset_mapping is on CPU and remains as a list/tensor of tuples.
        # The offset mapping shape is (batch_size, seq_len, 2). We take the first batch.
        self.offset_mapping = inputs["offset_mapping"][0].cpu().tolist()

    def get_span_embedding(self, start_token: int, end_token: int) -> torch.Tensor:
        """Get mean-pooled embedding for token span using cached embeddings."""
        if self.token_embeddings is None:
            raise ValueError("Must call prepare_document before get_span_embedding")

        # Use safe mean pooling on the span
        span_embeddings = self.token_embeddings[start_token:end_token]
        span_mask = self.attention_mask[start_token:end_token]
        return self.safe_mean_pooling(span_embeddings, span_mask)

    def get_register_probs(
        self, text: str = None, start_token: int = None, end_token: int = None
    ) -> np.ndarray:
        """Get register probabilities for a text span."""
        if self.token_embeddings is None:
            if text is None:
                raise ValueError(
                    "Must either provide text or call prepare_document first"
                )
            self.prepare_document(text)

        # If no span specified, use entire sequence
        if start_token is None or end_token is None:
            start_token = 0
            end_token = len(self.token_embeddings)

        # Get span embedding
        span_embedding = self.get_span_embedding(start_token, end_token)

        # Important: Use model.head before classifier like in working code
        with torch.no_grad():
            hidden = self.model.head(span_embedding.unsqueeze(0))
            logits = self.model.classifier(hidden)
            probs = torch.sigmoid(logits).detach().cpu().numpy()[0][:8]

        return probs

    def compute_register_distinctness(
        self, probs1: np.ndarray, probs2: np.ndarray, parent_probs: np.ndarray = None
    ) -> float:
        # Get registers above threshold for each segment and parent.
        regs1 = set(np.where(probs1 >= self.config.classification_threshold)[0])
        regs2 = set(np.where(probs2 >= self.config.classification_threshold)[0])
        parent_regs = set(
            np.where(parent_probs >= self.config.classification_threshold)[0]
        )

        # Both segments must have at least one register
        if not (regs1 and regs2):
            return 0.0

        # Compute max probabilities for segments and parent.
        max_prob1 = max(probs1)
        max_prob2 = max(probs2)
        max_prob_parent = max(parent_probs) if parent_probs is not None else 0.0

        # Ensure that at least one segment improves over the parent.
        if max_prob1 <= max_prob_parent and max_prob2 <= max_prob_parent:
            return 0

        # Alternative 1: Differences between segments.
        if regs1 == regs2:
            seg_diff = 0.0
        else:
            diff_score = 0.0
            diff_registers = (regs1 - regs2) | (regs2 - regs1)
            for reg_idx in diff_registers:
                diff_score += abs(probs1[reg_idx] - probs2[reg_idx])
            seg_diff = diff_score * (max_prob1 + max_prob2) / 2

        # Alternative 2: Improvement over parent.
        parent_diff = min(max_prob1 - max_prob_parent, max_prob2 - max_prob_parent)

        # Combine both perspectives.
        lambda_weight = 0.5  # Adjust this hyperparameter as needed.
        combined_score = lambda_weight * seg_diff + (1 - lambda_weight) * parent_diff

        return combined_score

    def evaluate_split_individual(
        self,
        text: str,
        left_sents: List[str],
        right_sents: List[str],
        left_spans: List[Tuple[int, int]],
        right_spans: List[Tuple[int, int]],
    ) -> float:
        """Evaluate split at individual sentence level using cached embeddings."""
        # Get predictions for each sentence using cached embeddings
        left_probs = [
            self.get_register_probs(start_token=span[0], end_token=span[1])
            for span in left_spans
        ]
        right_probs = [
            self.get_register_probs(start_token=span[0], end_token=span[1])
            for span in right_spans
        ]
        parent_probs = self.get_register_probs(
            start_token=left_spans[0][0], end_token=right_spans[-1][1]
        )

        scores = []
        for l_prob in left_probs:
            for r_prob in right_probs:
                scores.append(
                    self.compute_register_distinctness(l_prob, r_prob, parent_probs)
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
        """Evaluate split using pairs of sentences."""
        if len(left_spans) < 2 or len(right_spans) < 2:
            return 0.0

        # Create pairs by combining token spans
        left_pair_spans = [
            (left_spans[i][0], left_spans[i + 1][1])
            for i in range(0, len(left_spans) - 1, 2)
        ]
        right_pair_spans = [
            (right_spans[i][0], right_spans[i + 1][1])
            for i in range(0, len(right_spans) - 1, 2)
        ]

        # Get predictions for each pair using cached embeddings
        left_probs = [
            self.get_register_probs(start_token=span[0], end_token=span[1])
            for span in left_pair_spans
        ]
        right_probs = [
            self.get_register_probs(start_token=span[0], end_token=span[1])
            for span in right_pair_spans
        ]

        parent_probs = self.get_register_probs(
            start_token=left_spans[0][0], end_token=right_spans[-1][1]
        )

        scores = []
        for l_prob in left_probs:
            for r_prob in right_probs:
                scores.append(
                    self.compute_register_distinctness(l_prob, r_prob, parent_probs)
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
        """Evaluate split comparing whole segments using cached embeddings."""
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
        """Find best split point using multi-scale analysis."""
        best_score = 0
        best_split = None

        for i in range(
            self.config.min_sentences, len(sentences) - self.config.min_sentences + 1
        ):
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

        return best_split, best_score

    def segment_text(self, text: str) -> List[Tuple[str, np.ndarray]]:
        """Main entry point for text segmentation."""
        text = self.truncate_text(text)

        # Use NLTK’s span tokenizer to get character-level spans of sentences.
        sent_detector = PunktSentenceTokenizer()
        sent_char_spans = list(sent_detector.span_tokenize(text))
        sentences = [text[s:e] for s, e in sent_char_spans]

        # Prepare document once and get token indices, embeddings, and offset mappings.
        self.prepare_document(text)

        # Convert offset mapping to a numpy array for easier handling (each element is [start, end]).
        offset_mapping = np.array(self.offset_mapping)  # shape: (seq_len, 2)

        # Map sentence character spans to token index spans.
        sent_spans = []
        for char_start, char_end in sent_char_spans:
            token_start = None
            token_end = None
            # Iterate over token offsets to find tokens overlapping with the sentence span.
            for i, (tok_start, tok_end) in enumerate(offset_mapping):
                # Some tokenizers (like BERT’s fast tokenizers) assign (0, 0) to special tokens.
                if tok_start == tok_end == 0:
                    continue
                # Identify the first token that ends after the sentence starts.
                if token_start is None and tok_end > char_start:
                    token_start = i
                # Continue as long as the token starts before the sentence ends.
                if tok_start < char_end:
                    token_end = i + 1
                else:
                    break
            # Fallback: if no tokens were found (should not occur in well-formed cases)
            if token_start is None or token_end is None:
                token_start, token_end = 0, 0
            sent_spans.append((int(token_start), int(token_end)))

        # If no valid spans found, return single segment with whole document
        if not sent_spans:
            return [(text, self.get_register_probs())]

        segments = self.segment_recursive(text, sentences, sent_spans)

        # If only one segment, ensure it matches whole document prediction
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


# (The rest of the code, including get_last_processed_id and main, remains unchanged.)


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
    segmenter = MultiScaleSegmenter(model_path=model_path, config=config)

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
