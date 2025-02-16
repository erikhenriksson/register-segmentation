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

print("Importing torch")
import torch

print("Importing transformers")
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["LY", "SP", "ID", "NA", "HI", "IN", "OP", "IP"]


@dataclass
class MultiScaleConfig:
    max_length: int = 8192
    min_tokens: int = 0  # Minimum token count per segment
    classification_threshold: float = 0.70
    min_register_diff: float = 0.0001
    scale_weights = {"short": 0, "long": 0, "whole": 1}
    use_embeddings = True


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

        # Cache for token representations and offset mappings
        self.token_embeddings = None
        self.attention_mask = None
        self.tokens = None

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

        span_embeddings = self.token_embeddings[start_token:end_token]
        span_mask = self.attention_mask[start_token:end_token]
        embedding = self.safe_mean_pooling(span_embeddings, span_mask)
        return embedding

    def get_register_probs_batch(
        self, texts: List[str]
    ) -> Tuple[List[np.ndarray], List[torch.Tensor]]:
        """Get register probabilities and embeddings for a batch of texts."""
        # Check cache first and collect uncached texts
        uncached_texts = []
        uncached_indices = []
        cached_results = []

        for i, text in enumerate(texts):
            if text in self._prediction_cache:
                cached_results.append(self._prediction_cache[text])
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if not uncached_texts:
            probs = [result[0] for result in cached_results]
            embeddings = [result[1] for result in cached_results]
            return probs, embeddings

        # Tokenize all uncached texts at once
        inputs = self.tokenizer(
            uncached_texts,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            batch_probs = torch.sigmoid(outputs.logits).cpu().numpy()
            last_hidden_state = outputs.hidden_states[-1]

        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        batch_embeddings = (last_hidden_state * attention_mask).sum(
            dim=1
        ) / attention_mask.sum(dim=1)

        # Cache the results and prepare final output
        all_probs = []
        all_embeddings = []
        cached_idx = 0
        uncached_idx = 0

        for i in range(len(texts)):
            if i in uncached_indices:
                probs = batch_probs[uncached_idx]
                embedding = batch_embeddings[uncached_idx]
                self._prediction_cache[texts[i]] = (probs, embedding)
                all_probs.append(probs)
                all_embeddings.append(embedding)
                uncached_idx += 1
            else:
                all_probs.append(cached_results[cached_idx][0])
                all_embeddings.append(cached_results[cached_idx][1])
                cached_idx += 1

        return all_probs, all_embeddings

    def get_register_probs(
        self, text: str, start_token: int = None, end_token: int = None
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Get register probabilities and embedding for text"""

        if self.use_embeddings and self.token_embeddings is not None:
            """
            span_embedding = self.get_span_embedding(start_token, end_token)
            probs = self.predict_from_embeddings(span_embedding)

            return probs, span_embedding
            """
            span_embeddings = self.token_embeddings[
                start_token : end_token + 1
            ]  # +1 to include end_idx
            mean_pooled = span_embeddings.mean(dim=0)
            pooled_batch = mean_pooled.unsqueeze(0)  # Add batch dimension

            # Apply the full classification pipeline
            pooled_output = self.model.head(pooled_batch)  # Project features
            pooled_output = self.model.drop(pooled_output)  # Apply dropout
            logits = self.model.classifier(pooled_output)  # Get class logits

            probabilities = torch.sigmoid(logits)  # Get probabilities
            return probabilities.squeeze(0)  # Remove batch dimension

        # Check cache first
        if text in self._prediction_cache:
            return self._prediction_cache[text]

        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
            last_hidden_state = outputs.hidden_states[-1]
            if not self.token_embeddings:
                self.token_embeddings = outputs.hidden_states[-1][0].detach()
                self.attention_mask = inputs["attention_mask"][0]
                self.offset_mapping = inputs["offset_mapping"][0].cpu().tolist()

        attention_mask = inputs["attention_mask"].unsqueeze(
            -1
        )  # Expand for broadcasting
        mean_embedding = (last_hidden_state * attention_mask).sum(
            dim=1
        ) / attention_mask.sum(dim=1)

        # Cache the results
        self._prediction_cache[text] = (probs, mean_embedding)
        return probs, mean_embedding

    def get_text_for_span(self, text: str, start_token: int, end_token: int) -> str:
        """Get the original text corresponding to a token span with bounds checking."""
        if not self.offset_mapping:
            return text

        # Ensure indices are within bounds
        max_idx = len(self.offset_mapping) - 1
        start_token = max(0, min(start_token, max_idx))
        end_token = max(start_token + 1, min(end_token, max_idx + 1))

        try:
            char_start = self.offset_mapping[start_token][0]
            char_end = self.offset_mapping[end_token - 1][1]
            return text[char_start:char_end]
        except (IndexError, TypeError):
            # Fallback to using the original text if we encounter any issues
            print(
                f"Warning: Failed to get precise span for tokens {start_token}:{end_token}. Using approximate spans."
            )
            # Approximate the character positions
            char_length = len(text)
            approx_start = int((start_token / max_idx) * char_length)
            approx_end = int((end_token / max_idx) * char_length)
            return text[approx_start:approx_end]

    def compute_register_distinctness(
        self, probs1: np.ndarray, probs2: np.ndarray, parent_probs: np.ndarray = None
    ) -> float:
        """Compute how distinct two spans are using cosine distance."""
        # Get active registers using threshold
        regs1 = set(np.where(probs1 >= self.config.classification_threshold)[0])
        regs2 = set(np.where(probs2 >= self.config.classification_threshold)[0])
        parent_regs = (
            set(np.where(parent_probs >= self.config.classification_threshold)[0])
            if parent_probs is not None
            else set()
        )

        # Same early return conditions as before
        if not (regs1 and regs2):
            return 0.0, [], []
        if regs1 == parent_regs == regs2:
            return 0.0, [], []
        if regs1 == regs2:
            return 0.0, [], []

        # Compute cosine distance
        similarity = np.dot(probs1, probs2) / (
            np.linalg.norm(probs1) * np.linalg.norm(probs2)
        )
        distance = 1 - similarity

        print(distance)

        return distance, regs1, regs2

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
        print(left_text)
        print(right_text)
        print(parent_text)
        print(".--------.")
        left_probs, _ = self.get_register_probs(
            left_text, left_window[0], left_window[1]
        )
        right_probs, _ = self.get_register_probs(
            right_text, right_window[0], right_window[1]
        )
        parent_probs, _ = self.get_register_probs(
            parent_text, left_window[0], right_window[1]
        )

        print(left_probs)
        print(right_probs)
        print(parent_probs)

        """
        # Batch the three predictions together
        batch_probs, _ = self.get_register_probs_batch(
            [left_text, right_text, parent_text]
        )
        left_probs, right_probs, parent_probs = batch_probs
        """

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
            score_short, short_regs_left, short_regs_right = self.evaluate_split(
                text, left_spans, right_spans, window_size=3
            )
            if (
                score_short
                and short_regs_left == whole_regs_left
                and short_regs_right == whole_regs_right
            ):
                scores.append(
                    score_short * self.config.scale_weights["short"] * min_tokens / 8192
                )

            # Long window (4+4)
            score_long, long_regs_left, long_regs_right = self.evaluate_split(
                text, left_spans, right_spans, window_size=9
            )
            if (
                score_long
                and long_regs_left == whole_regs_left
                and long_regs_right == whole_regs_right
            ):
                scores.append(
                    score_long * self.config.scale_weights["long"] * min_tokens / 8192
                )

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
        prob_chain: List[np.ndarray] = [],
        depth: int = 0,
        side: str = "root",
    ) -> List[Tuple[str, List[np.ndarray], torch.Tensor]]:
        """Recursively segment text using binary splitting."""

        # Get probabilities for current segment
        span_text = self.get_text_for_span(text, sent_spans[0][0], sent_spans[-1][-1])
        current_probs, current_embedding = self.get_register_probs(
            span_text, sent_spans[0][0], sent_spans[-1][-1]
        )

        new_chain = prob_chain + [current_probs]

        split_idx, score = self.find_best_split(
            text, sentences, sent_spans, depth, side
        )

        if score < self.config.min_register_diff or split_idx is None:
            return [(span_text, new_chain, current_embedding)]

        # For splits, only pass the parent probabilities without current level
        left_segments = self.segment_recursive(
            text,
            sentences[:split_idx],
            sent_spans[:split_idx],
            new_chain,
            depth + 1,
            "left",
        )
        right_segments = self.segment_recursive(
            text,
            sentences[split_idx:],
            sent_spans[split_idx:],
            new_chain,
            depth + 1,
            "right",
        )

        return left_segments + right_segments

    def segment_text(
        self, text: str
    ) -> List[Tuple[str, List[np.ndarray], torch.Tensor]]:
        """Main entry point for text segmentation with improved error handling."""
        sent_detector = PunktSentenceTokenizer()
        sent_char_spans = list(sent_detector.span_tokenize(text))
        sentences = [text[s:e] for s, e in sent_char_spans]

        if not self.offset_mapping:
            # Initialize tokenizer and get offset mapping if not already done
            inputs = self.tokenizer(
                text,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
                add_special_tokens=True,
                return_offsets_mapping=True,
            )
            self.offset_mapping = inputs["offset_mapping"][0].cpu().tolist()

        offset_mapping = np.array(self.offset_mapping)
        max_token_idx = len(offset_mapping) - 1

        sent_spans = []
        for char_start, char_end in sent_char_spans:
            token_start = None
            token_end = None

            # Find token boundaries
            for i, (tok_start, tok_end) in enumerate(offset_mapping):
                if tok_start == tok_end == 0:
                    continue
                if token_start is None and tok_end > char_start:
                    token_start = i
                if tok_start < char_end:
                    token_end = i + 1
                else:
                    break

            # Ensure valid token spans
            if token_start is None or token_end is None:
                token_start, token_end = 0, min(
                    50, max_token_idx
                )  # Use reasonable default
            else:
                # Ensure spans are within bounds
                token_start = min(token_start, max_token_idx)
                token_end = min(token_end, max_token_idx + 1)

            if token_start >= token_end:
                token_end = min(token_start + 1, max_token_idx + 1)

            sent_spans.append((int(token_start), int(token_end)))

        return self.segment_recursive(text, sentences, sent_spans)

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
    files = ["dev.tsv", "test.tsv", "train.tsv"]
    for tsv_file in files:
        try:
            df = pd.read_csv(
                f"{dataset_path}/{tsv_file}",
                sep="\t",
                header=None,
                names=["label", "text"],
                na_values="",
                keep_default_na=False,
            )
            all_data.append(df)
        except:
            pass

    combined_df = pd.concat(all_data, ignore_index=True)

    last_id = get_last_processed_id(output_path)
    segmenter = MultiScaleSegmenter(model_path=model_path, config=config)

    print("last_id:", last_id)

    with open(output_path, "a", encoding="utf-8") as f:
        for i, row in combined_df.iterrows():
            if i <= last_id:
                continue

            text = row["text"]
            text = segmenter.truncate_text(text)
            segmenter.token_embeddings = None
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
