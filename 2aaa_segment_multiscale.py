import sys
import json
import glob
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
from nltk.tokenize import sent_tokenize
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["LY", "SP", "ID", "NA", "HI", "IN", "OP", "IP"]


@dataclass
class MultiScaleConfig:
    max_length: int = 2048
    min_sentences: int = 3
    classification_threshold: float = 0.4
    min_register_diff: float = 0.15
    scale_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.scale_weights is None:
            self.scale_weights = {"individual": 0.4, "pairs": 0.3, "whole": 0.3}


class MultiScaleSegmenter:
    def __init__(self, model_path: str, config: MultiScaleConfig = None):
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, output_hidden_states=True
        )
        # Extract classification head weights
        self.classifier = model.classifier.to("cuda")
        self.model = model.to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
        self.config = config or MultiScaleConfig()
        # Cache for token representations
        self.token_embeddings = None
        self.attention_mask = None
        self.token_to_char_map = None

    def prepare_document(self, text: str):
        """Run model once for whole document and cache results."""
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,  # Get char offsets for tokens
        ).to("cuda")

        # Save token to character mapping
        self.token_to_char_map = inputs.offset_mapping[0].cpu().numpy()

        # Get token-level representations
        outputs = self.model(**inputs)
        self.token_embeddings = outputs.hidden_states[-1][
            0
        ].detach()  # Remove batch dim
        self.attention_mask = inputs.attention_mask[0]

    def get_span_embedding(self, start_char: int, end_char: int) -> torch.Tensor:
        """Get mean-pooled embedding for a text span using cached embeddings."""
        if self.token_embeddings is None:
            raise ValueError("Must call prepare_document before get_span_embedding")

        # Ensure valid span indices
        if start_char is None or end_char is None:
            raise ValueError("start_char and end_char must not be None")

        # Find tokens that overlap with character span, accounting for special tokens
        token_mask = np.zeros(len(self.token_to_char_map), dtype=bool)
        content_token_idx = 0  # Track position in content tokens

        for i, (token_start, token_end) in enumerate(self.token_to_char_map):
            # Skip special tokens (they have 0,0 offset)
            if token_start == 0 and token_end == 0:
                continue

            if token_end > start_char and token_start < end_char:
                # Map back to full token sequence including special tokens
                # Add 1 to account for [CLS] token at start
                token_mask[i] = True

        # Convert to tensor - special tokens have already been included in mask
        token_mask = torch.tensor(token_mask, device="cuda")
        token_mask = token_mask & self.attention_mask.bool()

        # Mean pool relevant token embeddings
        masked_embeddings = self.token_embeddings[token_mask]
        if len(masked_embeddings) == 0:
            raise ValueError(f"No tokens found for span [{start_char}, {end_char}]")
        return torch.mean(masked_embeddings, dim=0)

        # Convert to tensor
        token_mask = torch.tensor(token_mask, device="cuda")
        token_mask = token_mask & self.attention_mask.bool()

        # Mean pool relevant token embeddings
        masked_embeddings = self.token_embeddings[token_mask]
        return torch.mean(masked_embeddings, dim=0)

    def get_register_probs(
        self, text: str, start_char: int = None, end_char: int = None
    ) -> np.ndarray:
        """Get register probabilities for a text span."""
        # First time called - process whole document
        if self.token_embeddings is None:
            self.prepare_document(text)
            # For whole text, use attention mask to get valid token mean
            valid_tokens = self.attention_mask.bool()
            mean_embedding = torch.mean(self.token_embeddings[valid_tokens], dim=0)
            logits = self.classifier(mean_embedding.unsqueeze(0))
            return torch.sigmoid(logits).detach().cpu().numpy()[0][:8]

        # If no span specified, use entire text the same way as above
        if start_char is None or end_char is None:
            valid_tokens = self.attention_mask.bool()
            mean_embedding = torch.mean(self.token_embeddings[valid_tokens], dim=0)
            logits = self.classifier(mean_embedding.unsqueeze(0))
            return torch.sigmoid(logits).detach().cpu().numpy()[0][:8]

        # For specific spans, mean pool only the relevant tokens
        span_embedding = self.get_span_embedding(start_char, end_char)
        logits = self.classifier(span_embedding.unsqueeze(0))
        return torch.sigmoid(logits).detach().cpu().numpy()[0][:8]

    def compute_register_distinctness(
        self, probs1: np.ndarray, probs2: np.ndarray
    ) -> float:
        """Compute register distinctness between two probability vectors."""
        regs1 = set(np.where(probs1 >= self.config.classification_threshold)[0])
        regs2 = set(np.where(probs2 >= self.config.classification_threshold)[0])

        total_diff = 0
        for reg_idx in range(len(probs1)):
            if reg_idx in regs1 and reg_idx not in regs2:
                total_diff += abs(probs1[reg_idx] - probs2[reg_idx])
            elif reg_idx not in regs1 and reg_idx in regs2:
                total_diff += abs(probs1[reg_idx] - probs2[reg_idx])

        return total_diff

    def evaluate_split_individual(
        self, left_sents: List[str], right_sents: List[str]
    ) -> float:
        """Evaluate split at individual sentence level."""
        left_probs = [self.get_register_probs(sent) for sent in left_sents]
        right_probs = [self.get_register_probs(sent) for sent in right_sents]

        scores = []
        for l_prob in left_probs:
            for r_prob in right_probs:
                scores.append(self.compute_register_distinctness(l_prob, r_prob))

        return np.mean(scores) if scores else 0.0

    def evaluate_split_pairs(
        self, left_sents: List[str], right_sents: List[str]
    ) -> float:
        """Evaluate split using sentence pairs."""
        left_pairs = [
            " ".join(left_sents[i : i + 2]) for i in range(0, len(left_sents) - 1, 2)
        ]
        right_pairs = [
            " ".join(right_sents[i : i + 2]) for i in range(0, len(right_sents) - 1, 2)
        ]

        if not left_pairs or not right_pairs:
            return 0.0

        left_probs = [self.get_register_probs(pair) for pair in left_pairs]
        right_probs = [self.get_register_probs(pair) for pair in right_pairs]

        scores = []
        for l_prob in left_probs:
            for r_prob in right_probs:
                scores.append(self.compute_register_distinctness(l_prob, r_prob))

        return np.mean(scores) if scores else 0.0

    def evaluate_split_whole(
        self, left_sents: List[str], right_sents: List[str]
    ) -> float:
        """Evaluate split comparing whole segments."""
        left_text = " ".join(left_sents)
        right_text = " ".join(right_sents)

        left_probs = self.get_register_probs(left_text)
        right_probs = self.get_register_probs(right_text)

        return self.compute_register_distinctness(left_probs, right_probs)

    def find_best_split(self, sentences: List[str]) -> Tuple[int, float]:
        """Find best split point using multi-scale analysis."""
        best_score = -float("inf")
        best_split = None

        # Adjust weights based on segment length
        total_len = len(sentences)
        weights = self.config.scale_weights.copy()
        if total_len > 20:
            weights["whole"] *= 1.5
            weights["individual"] *= 0.7
        elif total_len < 8:
            weights["individual"] *= 1.5
            weights["whole"] *= 0.7

        for i in range(
            self.config.min_sentences, len(sentences) - self.config.min_sentences + 1
        ):
            left_sents = sentences[:i]
            right_sents = sentences[i:]

            score_individual = self.evaluate_split_individual(left_sents, right_sents)
            score_pairs = self.evaluate_split_pairs(left_sents, right_sents)
            score_whole = self.evaluate_split_whole(left_sents, right_sents)

            total_score = (
                weights["individual"] * score_individual
                + weights["pairs"] * score_pairs
                + weights["whole"] * score_whole
            )

            if total_score > best_score:
                best_score = total_score
                best_split = i

        return best_split, best_score

    def segment_recursive(
        self, text: str, sentences: List[str], sent_spans: List[Tuple[int, int]]
    ) -> List[Tuple[str, np.ndarray]]:
        """Recursively segment text using binary splitting."""
        if len(sentences) < 2 * self.config.min_sentences:
            start_char = sent_spans[0][0]
            end_char = sent_spans[-1][1]
            span_text = text[start_char:end_char]
            return [(span_text, self.get_register_probs(text, start_char, end_char))]

        split_idx, score = self.find_best_split(sentences)

        if score < self.config.min_register_diff or split_idx is None:
            start_char = sent_spans[0][0]
            end_char = sent_spans[-1][1]
            span_text = text[start_char:end_char]
            return [(span_text, self.get_register_probs(text, start_char, end_char))]

        left_segments = self.segment_recursive(
            text, sentences[:split_idx], sent_spans[:split_idx]
        )
        right_segments = self.segment_recursive(
            text, sentences[split_idx:], sent_spans[split_idx:]
        )

        return left_segments + right_segments

    def segment_text(self, text: str) -> List[Tuple[str, np.ndarray]]:
        """Main entry point for text segmentation."""
        text = self.truncate_text(text)

        # Get sentence boundaries with character offsets
        sentences = []
        sent_spans = []
        curr_pos = 0

        for sent in sent_tokenize(text):
            start = text.find(sent, curr_pos)
            end = start + len(sent)
            sentences.append(sent)
            sent_spans.append((start, end))
            curr_pos = end

        # Prepare document once
        self.prepare_document(text)

        return self.segment_recursive(text, sentences, sent_spans)

    def truncate_text(self, text: str) -> str:
        tokens = self.tokenizer(text, truncation=False, return_tensors="pt")[
            "input_ids"
        ][0]
        if len(tokens) > self.config.max_length:
            text = self.tokenizer.decode(
                tokens[: self.config.max_length], skip_special_tokens=True
            )
        return text

    def print_result(self, result: Dict):
        """Print segmentation results in the same format as original code."""
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
    segmenter = MultiScaleSegmenter(model_path=model_path, config=config)

    print("last_id:", last_id)

    with open(output_path, "a", encoding="utf-8") as f:
        for i, row in combined_df.iterrows():
            if i <= last_id:
                continue

            text = row["text"]
            full_probs = segmenter.get_register_probs(text)
            segments = segmenter.segment_text(text)

            result = {
                "id": i,
                "label": row["label"],
                "text_probs": [round(x, 4) for x in full_probs.tolist()],
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
